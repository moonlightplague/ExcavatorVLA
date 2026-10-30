#!/usr/bin/env python3
"""Audit successful repeat episodes and optional standard LeRobot publication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_episode(directory):
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    require(meta.get("success") and meta.get("status") == "trainable", f"{directory}: not trainable")
    config = meta["repeat_baseline"]["config"]
    count = config["scoops"]
    require(count >= 2, "Repeat episodes need at least two cycles")
    require(meta.get("scoops_completed") == count, "Wrong completed scoop count")
    require(meta.get("all_scoops_success"), "A scoop failed the existing quality gate")
    require(meta.get("repeat_returns_completed") == count, "Incomplete physical returns")
    require(meta.get("source_sampling_audit", {}).get("ok"), "Source sampling audit failed")
    if meta["repeat_baseline"].get("version") == "repeat_flat_fill_v2":
        metrics = meta.get("repeat_metrics", {})
        require(metrics.get("metrics_schema_version") == "repeat_episode_metrics_v1", "Missing repeat metrics")
        require(json.loads((directory / "repeat_metrics.json").read_text()) == metrics,
                "Persisted repeat metrics disagree with metadata")
        require(metrics.get("success") and metrics.get("available"), "Repeat metrics unavailable or unsuccessful")
        require(metrics["samples"] == meta["final_metrics"]["samples"], "Metric sample count mismatch")
        for key in ("runtime_seconds", "simulated_runtime_seconds", "cycle_time_seconds", "cycle_time_wall_seconds"):
            require(np.isfinite(metrics[key]) and metrics[key] > 0, f"Invalid {key}")
        require(metrics["spillage_particles"] + metrics["retained_particles"] <= metrics["scooped_particles"],
                "Inconsistent source particle accounting")
        require(len(metrics["scoop_cycles"]) == count and all(row["success"] for row in metrics["scoop_cycles"]),
                "Missing scoop cycle measurements")
        plans = [meta.get("plan", [])] + [row.get("plan", []) for row in meta.get("scoop_plans", [])]
        for plan in plans:
            contact = next((index for index, stage in enumerate(plan) if stage.get("phase") == "approach_contact"), None)
            exit_cut = next((index for index, stage in enumerate(plan) if stage.get("phase") == "pull_exit_cut"), len(plan) - 1)
            if contact is not None:
                require(not any(str(stage.get("phase", "")).startswith("clearance_route")
                                for stage in plan[contact:exit_cut + 1]), "Clearance detour interrupts the cut")
    if "min_curl_height_m" in config:
        minimum = config["min_curl_height_m"] + meta["repeat_baseline"].get("ground_top_z", 0)
        plans = [meta.get("plan", [])] + [row.get("plan", []) for row in meta.get("scoop_plans", [])]
        for plan in plans:
            curl = [step for step in plan if step.get("phase") == "curl_to_hold_material"]
            require(curl and all(step["target_point"][2] >= minimum - .02 for step in curl),
                    "Planner bypassed the curl endpoint clearance")
    rows = read_jsonl(directory / "trajectory.jsonl")
    require(len(rows) >= 2, "Too few trajectory rows")
    times = np.array([row["t"] for row in rows])
    require(np.isfinite(times).all() and np.allclose(np.diff(times), .1, atol=1e-5), "Source is not uniform 10 Hz")
    require([row["i"] for row in rows] == list(range(len(rows))), "Non-contiguous frame indices")
    for key, width in (("observation.state", 28), ("action", 4)):
        values = np.array([row[key] for row in rows])
        require(values.shape == (len(rows), width) and np.isfinite(values).all(), f"Invalid {key}")
    indices = [int(row["observation.scoop_index"]) for row in rows]
    require(sorted(set(indices)) == list(range(count)), "Missing scoop observations")
    require(np.all(np.diff(indices) >= 0), "Scoop observation indices went backwards")
    for camera in range(3):
        paths = [row[f"observation.images.{camera}"] for row in rows]
        require(len(set(paths)) == len(rows), "Reused camera frame")
        require(all((directory / path).is_file() for path in paths), "Missing camera image")
    cycles = meta["repeat_cycles"]
    require(len(cycles) == count, "Missing cycle reports")
    returns = []
    for index, cycle in enumerate(cycles):
        require(cycle["scoop_index"] == index and cycle["returned"], "Incomplete return cycle")
        begin, end = cycle["return_start"], cycle["sample_end"]
        require(0 <= begin < end <= len(rows), "Return has no recorded frames")
        segment = rows[begin:end]
        # The stock scheduled camera writer labels samples from control.mode,
        # so the motion's descriptive label is not necessarily copied to rows.
        # Episode-local bounds identify the return; phase 0 identifies movement.
        physical = [row for row in segment if row.get("phase.index") == 0]
        require(physical, "Return trajectory is missing")
        require(all(row["phase.index"] == 0 for row in physical), "Return must use the existing pre_dig phase")
        require(any(np.max(np.abs(row["action"])) > .01 for row in physical), "Return has no physical actions")
        require(cycle["bed_after_return"]["particle_count"] > 0, "Truck received no soil")
        returns.append({"scoop_index": index, "frames": len(physical),
                        "landing": cycle["landing"]["landing"],
                        "bed_particles": cycle["bed_after_return"]["particle_count"],
                        "surface_std_m": cycle["bed_after_return"]["surface_std_m"],
                        "occupied_fraction": cycle["bed_after_return"]["occupied_fraction"]})
    require(all(row.get("success") and row.get("bin_gain_from_pile_particles", 0) > 0
                for row in meta["scoop_results"]), "Cycle did not transfer source soil to truck")
    return {"episode": str(directory), "frames": len(rows), "scoops": count, "returns": returns,
            "score": meta["score_summary"]["score"], "task": rows[0]["task"]}


def validate_export(directory):
    import imageio_ffmpeg
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = Path(directory)
    info = json.loads((directory / "meta/info.json").read_text())
    require(info["fps"] == 10, "Export fps is not 10 Hz")
    require(info["features"]["observation.state"]["shape"] == [28], "Export state is not canonical 28D")
    require(info["features"]["action"]["shape"] == [4], "Export actions are not 4D")
    tables = [pq.read_table(path) for path in sorted((directory / "data").rglob("*.parquet"))]
    require(bool(tables), "Missing exported Parquet")
    table = pa.concat_tables(tables)
    require(table.num_rows == info["total_frames"], "Export frame count mismatch")
    for key, width in (("observation.state", 28), ("action", 4)):
        values = np.array(table[key].to_pylist())
        require(values.shape == (table.num_rows, width) and np.isfinite(values).all(), f"Invalid exported {key}")
    require("observation.return_to_pile" in table.column_names, "Missing physical return supervision")
    flags = np.array(table["observation.return_to_pile"].to_pylist())
    stages = np.array(table["observation.stage_current_id"].to_pylist())
    require(set(flags.tolist()) == {0, 1}, "Invalid or empty physical return flags")
    require(np.all(stages[flags == 1] == 0), "Returns were exported as another stage")
    camera_frames = {}
    for camera in range(3):
        feature = f"observation.images.{camera}"
        require(info["features"][feature]["shape"] == [256, 256, 3], f"Unexpected {feature} shape")
        videos = sorted((directory / "videos" / feature).rglob("*.mp4"))
        require(bool(videos), f"Missing {feature} video")
        frames = sum(imageio_ffmpeg.count_frames_and_secs(str(video))[0] for video in videos)
        require(frames == table.num_rows, f"{feature} video/Parquet mismatch")
        camera_frames[feature] = frames
    return {"directory": str(directory), "episodes": info["total_episodes"],
            "frames": table.num_rows, "camera_frames": camera_frames,
            "state_dimensions": 28, "action_dimensions": 4, "return_frames": int(flags.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    episodes = []
    for path in sorted(args.run_dir.glob("episode_*/meta.json")):
        if json.loads(path.read_text()).get("status") == "trainable":
            episodes.append(validate_episode(path.parent))
    require(bool(episodes), "No trainable repeated episodes found")
    result = {"passed": True, "raw_episodes": episodes}
    if args.export_dir:
        result["export"] = validate_export(args.export_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
