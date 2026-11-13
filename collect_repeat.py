#!/usr/bin/env python3
"""Launch the repeat baseline with Isaac Python and optional streaming export."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ISAAC_REPO = ROOT / "ExcavatorVLA-isaac"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, epilog=(
        "Other arguments are forwarded to run_vla_train_scene.py, e.g. --success-count 1 "
        "--max-attempts 5 --scene-seed 2 --active-gpu 0 --physics-gpu 0 --dataset-root PATH."
    ))
    parser.add_argument("--scoops-per-episode", type=int, default=10)
    parser.add_argument("--sand-amount", type=float, default=float(os.environ.get("EXCAVATOR_SAND_AMOUNT", "3")))
    parser.add_argument("--width", type=int, default=512, help="Headless application width; recorded cameras remain 256×256.")
    parser.add_argument("--height", type=int, default=512, help="Headless application height.")
    parser.add_argument("--bed-long-cells", type=int, default=5)
    parser.add_argument("--bed-short-cells", type=int, default=3)
    parser.add_argument("--bed-wall-margin", type=float, default=1.)
    parser.add_argument("--return-plan-seconds", type=float, default=12.)
    parser.add_argument("--dump-settle-seconds", type=float, default=1.5)
    parser.add_argument("--dig-target-spacing", type=float, default=.28)
    parser.add_argument("--min-dig-height", type=float, default=.15)
    parser.add_argument("--min-curl-height", type=float, default=1.05)
    parser.add_argument("--min-delivered-kg", type=float, default=None,
                        help="Minimum measured new source delivery in every dig; defaults to 50 kg for coarse MPM, 0 otherwise.")
    parser.add_argument("--scene-profile", choices=("broad", "reachable"), default="reachable")
    parser.add_argument("--acceptance-profile", choices=("strict", "episode", "coarse_mass"), default="episode")
    parser.add_argument('--min-retained-delivery-ratio', type=float, default=.8)
    parser.add_argument("--episode-min-score", type=float, default=65.)
    parser.add_argument("--episode-max-spill-ratio", type=float, default=.50)
    parser.add_argument("--scene-seed-start", type=int, default=None,
                        help="Reproducible consecutive scene seeds; each failed attempt consumes a seed.")
    parser.add_argument("--release-clearance", type=float, default=.45,
                        help="Minimum opening-center height above the truck wall, in meters.")
    parser.add_argument("--planning-only", action="store_true",
                        help="Probe scene/first-plan feasibility and record workspace_probe.jsonl without executing scoops.")
    parser.add_argument("--truck-reach-min", type=float, default=6.7)
    parser.add_argument("--truck-reach-max", type=float, default=7.0)
    parser.add_argument("--truck-approach", choices=("side", "rear"), default="side")
    parser.add_argument("--mpm-profile", choices=("coarse", "fine"),
                        default=os.environ.get("EXCAVATOR_MPM_PROFILE", "coarse"),
                        help="Shared Newton soil resolution/display profile; independent of the point budget.")
    parser.add_argument("--mpm-hz", type=int, choices=(60, 120, 240),
                        default=int(os.environ.get("EXCAVATOR_MPM_HZ", "60")))
    parser.add_argument("--mpm-particle-budget", type=int,
                        default=int(os.environ.get("EXCAVATOR_MPM_MAX_PARTICLES", "32000")))
    parser.add_argument("--stream-export-dir", default="",
                        help="Export each finalized successful episode, then retain only raw metrics.")
    parser.add_argument("--policy-python", default="", help="Python with LeRobot export dependencies.")
    parser.add_argument("--soil-backend", choices=("physx", "newton_mpm"),
                        default=os.environ.get("EXCAVATOR_SOIL_BACKEND", "newton_mpm"))
    args, forwarded = parser.parse_known_args(argv)
    if not 2 <= args.scoops_per_episode <= 64:
        parser.error("--scoops-per-episode must be between 2 and 64")
    if bool(args.stream_export_dir) != bool(args.policy_python):
        parser.error("--stream-export-dir and --policy-python must be supplied together")
    if args.scene_seed_start is not None and args.scene_seed_start < 0:
        parser.error("--scene-seed-start must be nonnegative")
    if not 1 <= args.mpm_particle_budget <= 40000:
        parser.error("--mpm-particle-budget must be between 1 and 40000")
    if not math.isfinite(args.sand_amount) or args.sand_amount <= 0:
        parser.error("--sand-amount must be finite and positive")
    if min(args.width, args.height) <= 0:
        parser.error("--width and --height must be positive")
    if args.soil_backend not in ("physx", "newton_mpm") or args.mpm_hz not in (60, 120, 240):
        parser.error("Invalid soil backend or MPM frequency in environment defaults")
    if args.mpm_profile not in ("coarse", "fine"):
        parser.error("EXCAVATOR_MPM_PROFILE must be coarse or fine")
    if args.min_delivered_kg is None:
        args.min_delivered_kg = 50. if args.soil_backend == "newton_mpm" and args.mpm_profile == "coarse" else 0.
    if not math.isfinite(args.min_delivered_kg) or args.min_delivered_kg < 0:
        parser.error("--min-delivered-kg must be finite and nonnegative")
    if args.acceptance_profile == 'coarse_mass' and (args.soil_backend != 'newton_mpm' or args.mpm_profile != 'coarse'):
        parser.error('--acceptance-profile coarse_mass requires Newton MPM with the coarse profile')
    if not math.isfinite(args.min_retained_delivery_ratio) or not 0 <= args.min_retained_delivery_ratio <= 1:
        parser.error('--min-retained-delivery-ratio must be finite and between zero and one')
    if args.acceptance_profile == 'coarse_mass' and args.min_delivered_kg <= 0:
        parser.error('coarse_mass requires a positive --min-delivered-kg')
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from baseline.bed_fill import FillConfig
    from baseline.repeat_runtime import RepeatConfig, install
    from baseline.tuning import configure_seed_sequence, install_workspace_probe

    config = RepeatConfig(args.scoops_per_episode, args.return_plan_seconds, args.dump_settle_seconds,
                          FillConfig(args.bed_long_cells, args.bed_short_cells, args.bed_wall_margin),
                          args.dig_target_spacing, args.min_dig_height, args.min_curl_height,
                          stream_export_dir=args.stream_export_dir, policy_python=args.policy_python,
                          scene_profile=args.scene_profile, acceptance_profile=args.acceptance_profile,
                          episode_min_score=args.episode_min_score,
                          episode_max_spill_ratio=args.episode_max_spill_ratio,
                          release_clearance_m=args.release_clearance,
                          truck_reach_min_m=args.truck_reach_min, truck_reach_max_m=args.truck_reach_max,
                          truck_approach=args.truck_approach, min_delivered_kg=args.min_delivered_kg,
                          min_retained_delivery_ratio=args.min_retained_delivery_ratio)
    os.environ["EXCAVATOR_SOIL_BACKEND"] = args.soil_backend
    os.environ["EXCAVATOR_MPM_PROFILE"] = args.mpm_profile
    os.environ["EXCAVATOR_MPM_HZ"] = str(args.mpm_hz)
    os.environ["EXCAVATOR_MPM_MAX_PARTICLES"] = str(args.mpm_particle_budget)
    for path in (ISAAC_REPO, ISAAC_REPO / "scripts"):
        sys.path.insert(0, str(path))
    # Bootstrap is simulator-free until called by the upstream launcher after
    # SimulationApp starts. Install after its module reloads, before collection.
    from excavator_app import bootstrap
    original_bootstrap = bootstrap.run_excavator_with_sand

    def boot_repeat():
        rt = original_bootstrap()
        install(rt, config)
        configure_seed_sequence(rt, args.scene_seed_start)
        if args.planning_only:
            install_workspace_probe(rt)
        return rt

    bootstrap.run_excavator_with_sand = boot_repeat
    spec = importlib.util.spec_from_file_location("_repeat_isaac_launcher", ISAAC_REPO / "run_vla_train_scene.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    sys.argv = [str(ISAAC_REPO / "run_vla_train_scene.py"), *forwarded,
                "--headless", "--auto-collect", "--no-bridge", "--disable-export",
                "--log-mode", "data_multi", "--scoops-per-episode", str(config.scoops),
                "--max-scoops-per-episode", str(config.scoops + 1), "--dataset-hz", "10",
                "--sand-amount", str(args.sand_amount), "--width", str(args.width), "--height", str(args.height)]
    launcher.main()


if __name__ == "__main__":
    main()
