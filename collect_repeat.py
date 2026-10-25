#!/usr/bin/env python3
"""Launch the repeat baseline with Isaac Python; export with collect_repeat.sh."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
ISAAC_REPO = ROOT / "ExcavatorVLA-isaac"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Other arguments are forwarded to run_vla_train_scene.py, e.g. --success-count 1 "
        "--max-attempts 5 --scene-seed 2 --active-gpu 0 --physics-gpu 0 --dataset-root PATH."
    ))
    parser.add_argument("--scoops-per-episode", type=int, default=3)
    parser.add_argument("--bed-long-cells", type=int, default=5)
    parser.add_argument("--bed-short-cells", type=int, default=3)
    parser.add_argument("--bed-wall-margin", type=float, default=.60)
    parser.add_argument("--return-plan-seconds", type=float, default=12.)
    parser.add_argument("--dump-settle-seconds", type=float, default=1.)
    parser.add_argument("--dig-target-spacing", type=float, default=.55)
    parser.add_argument("--min-dig-height", type=float, default=.15)
    parser.add_argument("--min-curl-height", type=float, default=1.05)
    parser.add_argument("--soil-backend", choices=("physx", "newton_mpm"),
                        default=os.environ.get("EXCAVATOR_SOIL_BACKEND", "physx"))
    args, forwarded = parser.parse_known_args(argv)
    if not 2 <= args.scoops_per_episode <= 64:
        parser.error("--scoops-per-episode must be between 2 and 64")
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from baseline.bed_fill import FillConfig
    from baseline.repeat_runtime import RepeatConfig, install

    config = RepeatConfig(args.scoops_per_episode, args.return_plan_seconds, args.dump_settle_seconds,
                          FillConfig(args.bed_long_cells, args.bed_short_cells, args.bed_wall_margin),
                          args.dig_target_spacing, args.min_dig_height, args.min_curl_height)
    os.environ["EXCAVATOR_SOIL_BACKEND"] = args.soil_backend
    for path in (ISAAC_REPO, ISAAC_REPO / "scripts"):
        sys.path.insert(0, str(path))
    # Bootstrap is simulator-free until called by the upstream launcher after
    # SimulationApp starts. Install after its module reloads, before collection.
    from excavator_app import bootstrap
    original_bootstrap = bootstrap.run_excavator_with_sand

    def boot_repeat():
        rt = original_bootstrap()
        install(rt, config)
        return rt

    bootstrap.run_excavator_with_sand = boot_repeat
    spec = importlib.util.spec_from_file_location("_repeat_isaac_launcher", ISAAC_REPO / "run_vla_train_scene.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    sys.argv = [str(ISAAC_REPO / "run_vla_train_scene.py"), *forwarded,
                "--headless", "--auto-collect", "--no-bridge", "--disable-export",
                "--log-mode", "data_multi", "--scoops-per-episode", str(config.scoops),
                "--max-scoops-per-episode", str(config.scoops + 1), "--dataset-hz", "10"]
    launcher.main()


if __name__ == "__main__":
    main()
