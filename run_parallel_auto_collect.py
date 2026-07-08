#!/usr/bin/env python3
"""
Launch multiple headless ExcavatorVLA auto-collect workers from one terminal.

This script intentionally uses one Isaac process per worker. Isaac/Kit/RTX/PhysX
state is process-global, so multiprocessing is the safer form of parallelism.

Examples
--------
Windows:
    py run_parallel_auto_collect.py --workers 2 --success-count 100 --max-attempts 300 --graphics-api d3d12

Linux:
    python3 run_parallel_auto_collect.py --workers 4 --success-count 500 --max-attempts 2000 --python /isaac-sim/python.sh
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCENE = PROJECT_ROOT / "run_vla_train_scene.py"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "excavator_auto_dataset"


def parse_int_list(text):
    values = []
    for item in str(text or "").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or [0]


def split_total(total, workers):
    total = int(total or 0)
    workers = max(1, int(workers or 1))
    if total <= 0:
        return [0 for _ in range(workers)]
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


def make_unique_dir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    for index in range(2, 10000):
        candidate = Path(f"{path}_{index:02d}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"Unable to create unique directory near {path}")


def default_python():
    for env_name in ["ISAAC_SIM_PYTHON", "ISAAC_PYTHON"]:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    if platform.system().lower().startswith("win"):
        candidate = Path(r"D:\450\apps\isaacsim\python.bat")
        if candidate.exists():
            return str(candidate)
    candidate = Path("/isaac-sim/python.sh")
    if candidate.exists():
        return str(candidate)
    return sys.executable


def shell_safe_command(command):
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def build_python_command(python_exe, args):
    suffix = Path(str(python_exe)).suffix.lower()
    if platform.system().lower().startswith("win") and suffix in [".bat", ".cmd"]:
        return ["cmd.exe", "/c", str(python_exe)] + list(args)
    return [str(python_exe)] + list(args)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def discover_worker_runs(dataset_root, suffix, started_at):
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        return []
    pattern = re.compile(rf"^run_\d{{8}}_\d{{6}}_{re.escape(suffix)}(?:_\d+)?$")
    runs = []
    for path in dataset_root.iterdir():
        if not path.is_dir() or not pattern.match(path.name):
            continue
        try:
            if path.stat().st_mtime + 5.0 < float(started_at):
                continue
        except Exception:
            pass
        summary = read_json(path / "summary.json") or {}
        runs.append(
            {
                "path": str(path),
                "name": path.name,
                "mtime": path.stat().st_mtime,
                "summary": summary,
            }
        )
    runs.sort(key=lambda row: row.get("mtime", 0.0))
    return runs


def parse_args():
    parser = argparse.ArgumentParser(description="Run multiple headless ExcavatorVLA auto-collect workers.")
    parser.add_argument("--workers", type=int, default=2, help="Number of Isaac worker processes.")
    parser.add_argument("--success-count", type=int, default=100, help="Total target trainable episodes across workers.")
    parser.add_argument("--success-per-worker", type=int, default=0, help="Override per-worker success target.")
    parser.add_argument("--max-attempts", type=int, default=0, help="Total max attempts across workers. 0 lets each worker use runtime default.")
    parser.add_argument("--max-attempts-per-worker", type=int, default=0, help="Override per-worker max attempts.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="Shared dataset root.")
    parser.add_argument("--python", default=default_python(), help="Isaac Python launcher, e.g. python.bat or /isaac-sim/python.sh.")
    parser.add_argument("--graphics-api", choices=("auto", "d3d12", "vulkan"), default="auto")
    parser.add_argument("--renderer", default="RaytracedLighting")
    parser.add_argument("--active-gpus", default="0", help="Comma-separated renderer GPU ids, round-robin assigned.")
    parser.add_argument("--physics-gpus", default="", help="Comma-separated physics GPU ids. Defaults to active GPU assignment.")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--log-mode", choices=("data", "debug", "profile"), default="data")
    parser.add_argument("--capture-resolution", default="", help="Optional EXCAVATOR_DATASET_CAMERA_CAPTURE_RESOLUTION, e.g. 1024.")
    parser.add_argument("--camera-frequency", default="", help="Optional EXCAVATOR_DATASET_CAMERA_FREQUENCY override.")
    parser.add_argument("--sand-amount", type=float, default=None)
    parser.add_argument("--random-sand-amount", action="store_true")
    parser.add_argument("--stagger-seconds", type=float, default=5.0, help="Delay between worker launches.")
    parser.add_argument("--heartbeat-seconds", type=float, default=0.0, help="Worker auto-collect heartbeat print interval.")
    parser.add_argument("--export", action="store_true", help="Allow every worker to run LeRobot export. Default is disabled.")
    parser.add_argument("--wait-export", action="store_true", help="Wait for worker export if --export is used.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Extra args after -- are passed to run_vla_train_scene.py.")
    return parser.parse_args()


def worker_targets(args):
    workers = max(1, int(args.workers))
    if int(args.success_per_worker or 0) > 0:
        success = [int(args.success_per_worker) for _ in range(workers)]
    else:
        success = split_total(args.success_count, workers)

    if int(args.max_attempts_per_worker or 0) > 0:
        attempts = [int(args.max_attempts_per_worker) for _ in range(workers)]
    elif int(args.max_attempts or 0) > 0:
        attempts = split_total(args.max_attempts, workers)
    else:
        attempts = [0 for _ in range(workers)]
    return success, attempts


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    parallel_dir = make_unique_dir(dataset_root / f"parallel_{timestamp}")
    active_gpus = parse_int_list(args.active_gpus)
    physics_gpus = parse_int_list(args.physics_gpus) if str(args.physics_gpus or "").strip() else active_gpus
    success_targets, attempt_targets = worker_targets(args)
    extra_args = list(args.extra_args or [])
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    workers = []
    manifest = {
        "created_at": time.time(),
        "project_root": str(PROJECT_ROOT),
        "train_scene": str(TRAIN_SCENE),
        "dataset_root": str(dataset_root),
        "parallel_dir": str(parallel_dir),
        "python": str(args.python),
        "workers_requested": int(args.workers),
        "success_count": int(args.success_count),
        "max_attempts": int(args.max_attempts),
        "export_enabled": bool(args.export),
        "worker_records": [],
    }

    for worker_index, success_target in enumerate(success_targets):
        if success_target <= 0:
            continue
        suffix = f"w{worker_index:02d}"
        active_gpu = active_gpus[worker_index % len(active_gpus)]
        physics_gpu = physics_gpus[worker_index % len(physics_gpus)]
        max_attempts = int(attempt_targets[worker_index] or 0)
        log_path = parallel_dir / f"worker_{worker_index:02d}.log"
        env = os.environ.copy()
        env["EXCAVATOR_AUTO_RUN_ID_SUFFIX"] = suffix
        env["EXCAVATOR_PARALLEL_WORKER_INDEX"] = str(worker_index)
        env["EXCAVATOR_PARALLEL_WORKER_COUNT"] = str(len([v for v in success_targets if v > 0]))
        if args.capture_resolution:
            env["EXCAVATOR_DATASET_CAMERA_CAPTURE_RESOLUTION"] = str(args.capture_resolution)
        if args.camera_frequency:
            env["EXCAVATOR_DATASET_CAMERA_FREQUENCY"] = str(args.camera_frequency)
        if float(args.heartbeat_seconds or 0.0) > 0.0:
            env["EXCAVATOR_AUTO_COLLECT_HEARTBEAT_SECONDS"] = str(float(args.heartbeat_seconds))

        scene_args = [
            str(TRAIN_SCENE),
            "--headless",
            "--auto-collect",
            "--success-count",
            str(int(success_target)),
            "--max-attempts",
            str(max_attempts),
            "--dataset-root",
            str(dataset_root),
            "--log-mode",
            str(args.log_mode),
            "--graphics-api",
            str(args.graphics_api),
            "--renderer",
            str(args.renderer),
            "--active-gpu",
            str(int(active_gpu)),
            "--physics-gpu",
            str(int(physics_gpu)),
        ]
        if args.multi_gpu:
            scene_args.append("--multi-gpu")
        if args.sand_amount is not None:
            scene_args += ["--sand-amount", str(float(args.sand_amount))]
        if args.random_sand_amount:
            scene_args.append("--random-sand-amount")
        if args.export:
            scene_args.append("--wait-export" if args.wait_export else "--no-wait-export")
        else:
            scene_args += ["--disable-export", "--no-wait-export"]
        scene_args += extra_args

        command = build_python_command(args.python, scene_args)
        record = {
            "worker_index": worker_index,
            "suffix": suffix,
            "success_target": int(success_target),
            "max_attempts": max_attempts,
            "active_gpu": int(active_gpu),
            "physics_gpu": int(physics_gpu),
            "log_path": str(log_path),
            "command": command,
            "command_text": shell_safe_command(command),
            "started_at": None,
            "exit_code": None,
            "runs": [],
        }
        manifest["worker_records"].append(record)
        if args.dry_run:
            print(f"[DRY RUN] worker={worker_index} cmd={record['command_text']}")
            continue

        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        record["started_at"] = time.time()
        print(
            "[PARALLEL START]",
            f"worker={worker_index}",
            f"target={success_target}",
            f"max_attempts={max_attempts if max_attempts > 0 else 'runtime_default'}",
            f"gpu={active_gpu}",
            f"log={log_path}",
            flush=True,
        )
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        workers.append({"proc": proc, "log_file": log_file, "record": record})
        with open(parallel_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        if worker_index < len(success_targets) - 1 and float(args.stagger_seconds or 0.0) > 0.0:
            time.sleep(float(args.stagger_seconds))

    if args.dry_run:
        with open(parallel_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[DRY RUN] manifest={parallel_dir / 'manifest.json'}")
        return 0

    if not workers:
        print("[PARALLEL DONE] no workers started; success target may be 0.")
        return 0

    interrupted = False
    try:
        while True:
            alive = 0
            for item in workers:
                proc = item["proc"]
                record = item["record"]
                code = proc.poll()
                if code is None:
                    alive += 1
                    continue
                if record.get("exit_code") is None:
                    record["exit_code"] = int(code)
                    record["ended_at"] = time.time()
                    record["duration_s"] = round(record["ended_at"] - float(record.get("started_at") or record["ended_at"]), 3)
                    record["runs"] = discover_worker_runs(dataset_root, record["suffix"], float(record.get("started_at") or 0.0))
                    print(
                        "[PARALLEL WORKER DONE]",
                        f"worker={record['worker_index']}",
                        f"exit={code}",
                        f"runs={len(record['runs'])}",
                        f"log={record['log_path']}",
                        flush=True,
                    )
                    try:
                        item["log_file"].close()
                    except Exception:
                        pass
                    with open(parallel_dir / "manifest.json", "w", encoding="utf-8") as f:
                        json.dump(manifest, f, indent=2)
            if alive <= 0:
                break
            print(f"[PARALLEL STATUS] alive={alive}/{len(workers)} manifest={parallel_dir / 'manifest.json'}", flush=True)
            time.sleep(10.0)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[PARALLEL STOP] Ctrl+C received; terminating workers...", flush=True)
        for item in workers:
            proc = item["proc"]
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        deadline = time.time() + 20.0
        for item in workers:
            proc = item["proc"]
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
    finally:
        for item in workers:
            record = item["record"]
            proc = item["proc"]
            if record.get("exit_code") is None:
                code = proc.poll()
                record["exit_code"] = None if code is None else int(code)
                record["ended_at"] = time.time()
                record["duration_s"] = round(record["ended_at"] - float(record.get("started_at") or record["ended_at"]), 3)
            record["runs"] = discover_worker_runs(dataset_root, record["suffix"], float(record.get("started_at") or 0.0))
            try:
                item["log_file"].close()
            except Exception:
                pass
        manifest["ended_at"] = time.time()
        manifest["interrupted"] = bool(interrupted)
        manifest["exit_codes"] = [record.get("exit_code") for record in manifest["worker_records"]]
        with open(parallel_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    failed = [code for code in manifest["exit_codes"] if code not in [0]]
    print(f"[PARALLEL DONE] manifest={parallel_dir / 'manifest.json'} failed_workers={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
