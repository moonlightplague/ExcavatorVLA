"""Prepare new evaluation scenes with the unmodified collection planner.

Runs the normal repeat collector through scene preparation and first-cycle
planning, then captures the setup BEFORE recording or executing any dig action.
Only scene/plan feasibility selects trials; policy outcomes never select trials.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ISAAC = ROOT / "ExcavatorVLA-isaac"
TASK = "Excavate one scoop from the visible sand pile, carry the material, and dump it into the truck bed."
SELECTOR = "repeat_flat_fill_v2"


def write(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def install_capture(rt, controller, output, count, seed_start, seed_stride):
    """Wrap only preparation bookkeeping; preserve the real planner and gates."""
    output = Path(output)
    scenes = []
    original_plan = rt.auto_collect_find_plan
    original_diagnostic = rt.auto_collect_record_planning_diagnostic
    captured_attempts = set()

    def seed_context(attempt):
        seed = seed_start + (int(attempt) - 1) * seed_stride
        return dict(scene_seed=seed, scene_seed_namespace=f"collection-eval-{seed_start}",
                    scene_seed_source="evaluation_manifest", scene_seed_worker=str(seed_stride),
                    scene_sampling_schema="collection_random_scene_v1")

    def save_progress(phase):
        write(output / "progress.json", dict(
            phase=phase, prepared_scenes=len(scenes), expected_scenes=count,
            attempts=int(rt.STATE.get("auto_collect_attempts", 0)),
            last_stage=rt.STATE.get("auto_collect_progress", {})))

    async def capture_plan(attempt, ranked_targets=None):
        save_progress("planning")
        target, sequence, details = await original_plan(attempt, ranked_targets=ranked_targets)
        if sequence:
            selected = dict(controller.selected or {})
            if not selected.get("radius_ok") or not selected.get("dump_preflight_ok"):
                raise RuntimeError("Collection did not perform both reach and dump IK preflight")
            record = rt.STATE["auto_scene_last_randomization"]
            applied = record["applied"]
            import numpy as np
            initial = np.degrees(rt.get_real_joint_positions()).tolist()
            profile = dict(sand_xy=applied["sand_center"][:2],
                           truck_translation_xyz=applied["truck_translation_xyz"],
                           truck_yaw_deg=applied["truck_yaw_deg"],
                           unload_landing_xyz=selected["landing"])
            scene = dict(scene_seed=seed_context(attempt)["scene_seed"],
                         preparation_attempt=int(attempt), profile=profile,
                         initial_joints_deg=initial, dig_target_xyz=np.asarray(target).tolist(),
                         sand_amount=float(applied["sand_amount_multiplier"]), task=TASK,
                         selection=selected, collection_randomization=record,
                         collection_initial_pose=rt.STATE.get("auto_collect_initial_pose_id"),
                         collection_plan_stages=[str(stage[0]) for stage in sequence],
                         selector=SELECTOR, soil_backend="physx", expert_actions_executed=False)
            scene = rt.json_sanitize(scene)
            scenes.append(scene)
            captured_attempts.add(int(attempt))
            write(output / "scenes.json", dict(schema_version="collection_eval_scenes_v1",
                  selector=SELECTOR, soil_backend="physx", config=asdict(controller.config),
                  scenes=scenes))
            if len(scenes) == count:
                rt.STATE["auto_collect_stop_requested"] = True
            save_progress("completed" if len(scenes) == count else "preparing")
            print(f"[EVAL SCENE] prepared={len(scenes)}/{count} seed={scene['scene_seed']} "
                  f"cell={selected['cell_id']} landing={selected['landing']}", flush=True)
            # Do not let the collector start recording, execute a scoop, or
            # retain a trajectory. Its normal diagnostic path closes this attempt.
            return target, None, [{"reason": "evaluation_setup_captured_no_actions_executed"}]
        return target, sequence, details

    def diagnostic(attempt, target, details, reason, initial_info=None):
        captured = int(attempt) in captured_attempts
        row = dict(attempt=int(attempt), scene_seed=seed_context(attempt)["scene_seed"],
                   accepted=captured, reason="setup_captured" if captured else str(reason))
        with (output / "attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if not captured:
            original_diagnostic(attempt, target, details, reason, initial_info=initial_info)
        else:
            rt.STATE["auto_collect_last_result"] = "evaluation_setup_captured"
        save_progress("completed" if len(scenes) == count else "preparing")

    rt.auto_scene_seed_context = seed_context
    rt.auto_collect_find_plan = capture_plan
    rt.auto_collect_record_planning_diagnostic = diagnostic
    return scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=500)
    args = parser.parse_args()
    if min(args.count, args.seed_stride, args.max_attempts) < 1 or args.seed_start < 0:
        parser.error("Counts/stride must be positive and seed nonnegative")
    if args.output.exists():
        parser.error("Preparation output already exists")
    args.output.mkdir(parents=True)
    os.environ["EXCAVATOR_SOIL_BACKEND"] = "physx"
    os.environ["EXCAVATOR_SAND_WALLS_DURING_SETTLE"] = "1"
    os.environ["EXCAVATOR_PROJECT_ROOT"] = str(ISAAC)
    os.environ["EXCAVATOR_AUTO_RUN_ID_SUFFIX"] = f"eval_prepare_{args.seed_start}"
    for directory in (ROOT, ISAAC, ISAAC / "scripts"):
        sys.path.insert(0, str(directory))
    from baseline.repeat_runtime import RepeatConfig, install
    from excavator_app import bootstrap
    original_bootstrap = bootstrap.run_excavator_with_sand
    prepared = []

    def boot():
        rt = original_bootstrap()
        if rt.AUTO_SCENE_FIXED_PROFILE is not None or rt.AUTO_SCENE_REPLAY_SEED is not None:
            raise ValueError("Preparation must use the collection sampler, not fixed deployment replay")
        controller = install(rt, RepeatConfig(scoops=5))
        prepared.append(install_capture(rt, controller, args.output, args.count,
                                        args.seed_start, args.seed_stride))
        return rt

    bootstrap.run_excavator_with_sand = boot
    launcher_path = ISAAC / "run_vla_train_scene.py"
    spec = importlib.util.spec_from_file_location("_collection_eval_launcher", launcher_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    # The collection launcher uses physical Vulkan indices. Each evaluation
    # preparation worker is CUDA-isolated, so select rendering by its visible
    # CUDA device, exactly as the deployment evaluator does. Keep the upstream
    # launcher/source unchanged and limit this adapter to the current process.
    from isaacsim import SimulationApp
    original_init = SimulationApp.__init__

    def mapped_init(self, launch_config=None, *parameters, **kwargs):
        config = dict(launch_config or {})
        config.pop("active_gpu", None)
        config.update(active_cuda_gpus=[0], physics_gpu=0, multi_gpu=False, limit_cpu_threads=8)
        original_init(self, config, *parameters, **kwargs)

    SimulationApp.__init__ = mapped_init
    sys.argv = [str(launcher_path), "--headless", "--auto-collect", "--no-bridge", "--disable-export",
                "--log-mode", "data_multi", "--scoops-per-episode", "5", "--max-scoops-per-episode", "6",
                "--dataset-hz", "10", "--sand-amount", "1.0", "--active-gpu", "0", "--physics-gpu", "0",
                "--success-count", str(args.count), "--max-attempts", str(args.max_attempts),
                "--dataset-root", str(args.output / "preparation_logs")]
    launcher.main()
    if not prepared or len(prepared[0]) != args.count:
        raise RuntimeError(f"Prepared {len(prepared[0]) if prepared else 0}/{args.count} scenes; inspect logs")


if __name__ == "__main__":
    main()
