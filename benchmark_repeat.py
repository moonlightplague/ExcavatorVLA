"""Fixed-attempt Newton MPM benchmark, including failures in acceptance rates.

Run with the policy Python. Each worker consumes its assigned consecutive seed
range exactly once. This deliberately bypasses the success-target shell retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def interval(successes, attempts):
    if not attempts:
        return None
    z = 1.96
    p = successes / attempts
    scale = 1 + z*z/attempts
    center = (p + z*z/(2*attempts))/scale
    half = z*math.sqrt(p*(1-p)/attempts + z*z/(4*attempts*attempts))/scale
    return [max(0., center-half), min(1., center+half)]


def summarize(root, expected_attempts, target_rate=.5):
    results, episodes, reasons = [], [], {}
    for path in sorted(Path(root).glob("worker_*/result.json")):
        result = json.loads(path.read_text())
        run = Path(result["run_dir"])
        row = dict(worker=path.parent.name, **result)
        results.append(row)
        for meta_path in sorted(run.glob("episode_*/meta.json")):
            meta = json.loads(meta_path.read_text())
            quality = meta.get("repeat_episode_quality", {})
            receipt = meta_path.parent / "export.json"
            exported = receipt.exists() and json.loads(receipt.read_text()).get("validated", False)
            metrics = meta.get("repeat_metrics") or {}
            scoop_results = meta.get("scoop_results", [])
            expected_scoops = meta.get("repeat_baseline", {}).get("config", {}).get("scoops", 10)
            cycles = metrics.get("scoop_cycles", [])
            mass = metrics.get("particle_mass_kg")
            cycle_payloads = [dict(scoop_index=cycle.get("scoop_index"),
                payload_particles=cycle.get("payload_particles"),
                scooped_particles=cycle.get("scooped_particles"),
                payload_mass_kg=(cycle["payload_particles"] * mass
                    if cycle.get("payload_particles") is not None and mass is not None else None),
                runtime_seconds=cycle.get("runtime_seconds"), success=cycle.get("success"))
                for cycle in cycles]
            baseline = meta.get("repeat_baseline", {})
            episodes.append(dict(worker=path.parent.name, episode=meta_path.parent.name,
                success=bool(meta.get("success")), exported=bool(exported),
                failure_reason=meta.get("failure_reason"), scoops=meta.get("scoops_completed", 0),
                returns=meta.get("repeat_returns_completed", 0), source_audit=meta.get("source_sampling_audit", {}).get("ok"),
                quality=quality, metrics={k:metrics.get(k) for k in ("payload_particles", "payload_mass_kg",
                    "scooped_particles", "spillage_particles", "spillage_fraction", "runtime_seconds", "simulated_runtime_seconds")},
                cycle_payloads=cycle_payloads,
                sand_amount=(meta.get("scene_randomization") or {}).get("applied", {}).get("sand_amount_multiplier"),
                mpm_initial=(baseline.get("soil") or {}).get("mpm"), mpm_final=baseline.get("mpm_final"),
                planning_resolution=baseline.get("planning_resolution"),
                strict_success=bool(meta.get("success")) and len(scoop_results) == expected_scoops
                    and all(r.get("strict_success", r.get("success")) for r in scoop_results)))
        for index in ("rejected_episodes.jsonl", "failed_episodes.jsonl", "planning_diagnostics.jsonl"):
            path = run / index
            for line in path.read_text().splitlines() if path.exists() else []:
                reason = json.loads(line).get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
    attempts = sum(r["attempts"] for r in results)
    accepted = sum(e["success"] for e in episodes)
    exported = sum(e["exported"] for e in episodes)
    export_interval = interval(exported, attempts)
    return dict(schema="repeat_fixed_attempt_benchmark_v2", expected_attempts=expected_attempts,
                attempts=attempts, complete=attempts == expected_attempts,
                accepted=accepted, exported=exported,
                acceptance_rate=accepted/attempts if attempts else None,
                export_acceptance_rate=exported/attempts if attempts else None,
                target_rate=target_rate, target_comparison="strictly_greater_than",
                target_met=attempts == expected_attempts and exported/attempts > target_rate if attempts else False,
                export_rate_above_target_95=bool(attempts == expected_attempts and export_interval
                                                and export_interval[0] > target_rate),
                acceptance_wilson_95=interval(accepted, attempts),
                export_acceptance_wilson_95=export_interval,
                strict_equivalent_episodes=sum(e["strict_success"] for e in episodes),
                failures=reasons, workers=results, episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1")
    parser.add_argument("--workers-per-gpu", type=int, default=1,
                        help="Independent simulator processes per GPU; every worker gets disjoint seeds.")
    parser.add_argument("--seed-start", type=int, default=71000)
    parser.add_argument("--policy-python", default=sys.executable)
    parser.add_argument("--scoops-per-episode", type=int, choices=(10,), default=10,
                        help="This benchmark measures complete ten-cycle episodes.")
    parser.add_argument("--soil-backend", choices=("newton_mpm",), default="newton_mpm")
    parser.add_argument("--target-rate", type=float, default=.5,
                        help="An observed export rate must be strictly above this value.")
    parser.add_argument("--require-target", action="store_true",
                        help="Exit unsuccessfully when the completed cohort misses the target rate.")
    parser.add_argument("--require-confidence", action="store_true",
                        help="Also require the 95%% Wilson lower bound to exceed the target rate.")
    args, extra = parser.parse_known_args()
    if not math.isfinite(args.target_rate) or not 0 <= args.target_rate < 1:
        parser.error("--target-rate must be finite and in [0, 1)")
    gpus = [int(v) for v in args.gpu_ids.split(",")]
    if args.workers_per_gpu < 1 or any(gpu < 0 for gpu in gpus):
        parser.error("Use positive --workers-per-gpu and nonnegative GPU IDs")
    worker_gpus = gpus * args.workers_per_gpu
    if args.attempts < len(worker_gpus) or args.seed_start < 0 or len(set(gpus)) != len(gpus):
        parser.error("Use distinct GPUs, nonnegative seeds, and at least one attempt per worker")
    if any(flag in extra for flag in ("--scene-seed", "--fixed-scene-profile")):
        parser.error("Fixed-layout replay cannot benchmark acceptance across independent scenes")
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    started = time.monotonic()
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    commands = []
    seed = args.seed_start
    for i, gpu in enumerate(worker_gpus):
        count = args.attempts//len(worker_gpus) + (i < args.attempts % len(worker_gpus))
        directory = output / f"worker_{i}"
        directory.mkdir()
        command = [str(ROOT/"run_isaac_python.sh"), str(ROOT/"baseline/collect_repeat.py"),
                   "--scoops-per-episode", str(args.scoops_per_episode), "--soil-backend", args.soil_backend,
                   "--scene-profile", "reachable", "--acceptance-profile", "episode",
                   "--sand-amount", "3", "--bed-wall-margin", "1.0", "--dump-settle-seconds", "1.5",
                   "--release-clearance", ".45",
                   "--width", "512", "--height", "512", *extra,
                   "--active-gpu", str(gpu), "--physics-gpu", str(gpu),
                   "--scene-seed-start", str(seed), "--success-count", str(count), "--max-attempts", str(count),
                   "--dataset-root", str(directory/"raw"), "--stream-export-dir", str(output/"export_repeat"),
                   "--policy-python", args.policy_python]
        environment = os.environ.copy()
        environment.update(EXCAVATOR_AUTO_COLLECT_RESULT_FILE=str(directory/"result.json"),
                           EXCAVATOR_KIT_LOG_FILE=str(directory/"kit.log"),
                           EXCAVATOR_AUTO_RUN_ID_SUFFIX=f"benchmark_{args.seed_start}_w{i}",
                           REPEAT_POLICY_ENV=json.dumps({k:os.environ.get(k) for k in
                               ("PATH", "PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD")}))
        commands.append((command, environment, directory, count, seed, gpu))
        seed += count
    manifest = dict(schema="repeat_fixed_attempt_protocol_v2", commands=[dict(command=c, attempts=n, seed_start=s, gpu=g)
        for c, _e, _d, n, s, g in commands],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((ROOT/"baseline").glob("*.py")),
                      *(ROOT/"ExcavatorVLA-isaac/scripts/excavator_app"/name for name in
                        ("newton_mpm_soil.py", "soil_rock_display.py", "sand_site_runtime.py", "excavator_runtime.py",
                         "auto_dataset_collect.py"))]},
        target_rate=args.target_rate, target_comparison="strictly_greater_than",
        require_confidence=args.require_confidence, workers_per_gpu=args.workers_per_gpu,
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        mpm_environment={k:os.environ.get(k) for k in ("EXCAVATOR_MPM_PROFILE", "EXCAVATOR_MPM_EXECUTION",
            "EXCAVATOR_MPM_DISPLAY", "EXCAVATOR_MPM_HZ", "EXCAVATOR_MPM_MAX_PARTICLES",
            "EXCAVATOR_MPM_VOXEL_SIZE", "EXCAVATOR_MPM_PARTICLE_SPACING", "EXCAVATOR_MPM_GRID_CAPACITY")})
    (output/"commands.json").write_text(json.dumps(manifest, indent=2)+"\n")
    processes = []
    handles = []
    def stop(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    try:
        for command, environment, directory, _count, _seed, _gpu in commands:
            handle = (directory/"worker.log").open("w")
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=ROOT/"ExcavatorVLA-isaac", env=environment,
                              stdout=handle, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            summary = summarize(output, args.attempts, args.target_rate)
            live_exports = sum(bool(json.loads(path.read_text()).get("validated"))
                for path in output.glob("worker_*/raw/run_*/episode_*/export.json"))
            print(f"[BENCHMARK] finished_attempts={summary['attempts']}/{args.attempts} "
                  f"accepted={summary['accepted']} exported={live_exports} "
                  f"active_workers={sum(p.poll() is None for p in processes)}", flush=True)
            time.sleep(30)
        result = summarize(output, args.attempts, args.target_rate)
        result["process_exit_codes"] = [p.returncode for p in processes]
        result["elapsed_seconds"] = time.monotonic() - started
        result["exported_episodes_per_hour"] = 3600 * result["exported"] / result["elapsed_seconds"]
        result["infrastructure_valid"] = all(code == 0 for code in result["process_exit_codes"])
        result["target_met"] = result["target_met"] and result["infrastructure_valid"]
        result["export_rate_above_target_95"] = result["export_rate_above_target_95"] and result["infrastructure_valid"]
        (output/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("workers", "episodes")}, indent=2))
        if not result["complete"] or any(p.returncode for p in processes):
            raise SystemExit(1)
        if (args.require_target or args.require_confidence) and not result["target_met"]:
            raise SystemExit(2)
        if args.require_confidence and not result["export_rate_above_target_95"]:
            raise SystemExit(2)
    finally:
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
