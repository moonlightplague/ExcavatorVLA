#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STAGE_NAMES = [
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "curl_to_hold_material",
    "pull_exit_cut",
    "secure_load",
    "lift_carry",
    "loaded_transit",
    "unload_to_bin",
]


# The checkpoint uses the legacy v3 state with phase_index removed.  The v4
# runtime also has 28 values, but those values have different meanings and
# cannot be converted to the checkpoint schema by deleting one dimension.
V4_STATE_MARKERS = {
    "swing_tracking_error",
    "previous_swing_action",
    "dig_target_from_tip_forward",
    "bucket_fill_fraction",
}

LEGACY_STATE_NAMES_27D = [
    "base_x",
    "base_y",
    "base_yaw",
    "swing",
    "boom",
    "arm",
    "bucket",
    "bucket_load_estimate",
    "bucket_tip_x",
    "bucket_tip_y",
    "bucket_tip_z",
    "bucket_load_x",
    "bucket_load_y",
    "bucket_load_z",
    "swing_velocity",
    "boom_velocity",
    "arm_velocity",
    "bucket_velocity",
    "dig_target_local_x",
    "dig_target_local_y",
    "dig_target_local_z",
    "unload_landing_local_x",
    "unload_landing_local_y",
    "unload_landing_local_z",
    "truck_heading_relative_sin",
    "truck_heading_relative_cos",
    "bucket_load_rate",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_stage_names(runtime_path: Path | None) -> list[str]:
    if runtime_path is None or not runtime_path.exists():
        return DEFAULT_STAGE_NAMES.copy()

    try:
        tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "DATASET_PHASE_NAMES"
                ):
                    names = ast.literal_eval(node.value)
                    names = [str(x) for x in names]
                    if len(names) == 10:
                        return names
    except Exception as exc:
        print(f"[WARN] Could not parse DATASET_PHASE_NAMES: {exc}")

    return DEFAULT_STAGE_NAMES.copy()


def resolve_phase_dimension(
    state_names: list[str],
    states: np.ndarray,
    num_stages: int,
) -> tuple[int, str]:
    """Locate the legacy phase dimension without confusing v4 for v3."""
    if "phase_index" in state_names:
        return state_names.index("phase_index"), "metadata"

    if V4_STATE_MARKERS.intersection(state_names):
        markers = sorted(V4_STATE_MARKERS.intersection(state_names))
        raise RuntimeError(
            "The source uses the v4 28D observation schema, which is not "
            "dimension-compatible with the checkpoint's legacy 27D schema. "
            "v4 stores no phase_index inside observation.state and dimension "
            "18 is dig_target_from_tip_forward, not a stage label. Re-export "
            "or regenerate this episode with the legacy v3/27D checkpoint "
            f"observation contract. Detected v4 fields: {markers}"
        )

    if states.ndim != 2 or states.shape[1] != 28:
        raise RuntimeError(
            f"Cannot infer phase_index from state matrix shape {states.shape}; "
            "expected [N,28]"
        )

    # Some older exports contain the correct legacy v3 values but incomplete
    # or generic metadata names.  Dimension 18 is accepted only after checking
    # every value, so a continuous v4 geometry feature cannot be silently used
    # as a label.
    candidate_dimension = 18
    raw_stage = states[:, candidate_dimension]
    rounded = np.rint(raw_stage)
    finite = np.all(np.isfinite(raw_stage))
    integral = np.allclose(raw_stage, rounded, atol=1e-6)
    in_range = bool(
        raw_stage.size
        and np.min(rounded) >= 0
        and np.max(rounded) < num_stages
    )
    if finite and integral and in_range:
        return candidate_dimension, "validated_dimension_18_fallback"

    preview = state_names[:6]
    raise RuntimeError(
        "phase_index is absent from observation.state names and legacy "
        "dimension 18 did not contain valid integer stage ids. The source "
        "cannot be safely converted to the checkpoint's 27D schema. "
        f"names_count={len(state_names)}, names_preview={preview}, "
        f"dimension18_min={float(np.nanmin(raw_stage))}, "
        f"dimension18_max={float(np.nanmax(raw_stage))}"
    )


def numeric_vector(value: Any, size: int, label: str) -> list[float]:
    try:
        result = [float(item) for item in value]
    except Exception as exc:
        raise RuntimeError(f"{label} is not a numeric vector: {exc}") from exc
    if len(result) != size:
        raise RuntimeError(f"{label} must contain {size} values, got {len(result)}")
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError(f"{label} contains NaN or Inf")
    return result


def point_in_heading_frame(
    point_xyz: list[float],
    origin_xy: list[float],
    heading_rad: float,
) -> list[float]:
    dx = float(point_xyz[0]) - float(origin_xy[0])
    dy = float(point_xyz[1]) - float(origin_xy[1])
    c = math.cos(float(heading_rad))
    s = math.sin(float(heading_rad))
    return [
        c * dx + s * dy,
        -s * dx + c * dy,
        float(point_xyz[2]),
    ]


def heading_frame_delta_to_world(
    forward_left_up: list[float],
    heading_rad: float,
) -> list[float]:
    forward, left, up = forward_left_up
    c = math.cos(float(heading_rad))
    s = math.sin(float(heading_rad))
    return [
        c * forward - s * left,
        s * forward + c * left,
        up,
    ]


def checkpoint_stage_index(sample: dict[str, Any]) -> int:
    """Resolve a raw v4 phase using the canonical 10-stage contract."""
    raw_index = sample.get("phase.index")
    if raw_index is not None:
        try:
            value = float(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid canonical phase.index: {raw_index!r}"
            ) from exc
        index = int(round(value))
        if not math.isfinite(value) or abs(value - index) > 1.0e-6:
            raise RuntimeError(f"Non-integer canonical phase.index: {raw_index!r}")
        if not 0 <= index < len(DEFAULT_STAGE_NAMES):
            raise RuntimeError(f"Canonical phase.index is out of range: {index}")
        return index

    phase = str(sample.get("phase") or "").strip().lower()
    label = str(sample.get("label") or "").strip().lower()
    text = f"{phase} {label}"

    if (
        "loaded_transit" in text
        or "clearance_route_post" in text
        or "staged_unload" in text
        or "high_carry" in text
    ):
        return 8
    if "unload_to_bin" in text or "unload" in text or "dump" in text:
        return 9
    if "pre_dig" in text or "travel" in text or "align" in text:
        return 0
    if "approach_contact" in text or (
        "approach" in text and "contact" in text
    ):
        return 1
    if "insert" in text:
        return 2
    if "pull_mid" in text:
        return 3
    if "curl" in text:
        return 4
    if "pull_exit" in text:
        return 5
    if "secure" in text:
        return 6
    if "lift" in text or "carry" in text:
        return 7
    raise RuntimeError(
        "Cannot map raw v4 phase to the canonical stage order: "
        f"phase={phase!r}, label={label!r}"
    )


def reconstruct_legacy_state_27d(
    sample: dict[str, Any],
    v4_state: list[float],
    initial_origin_xy: list[float],
    initial_heading_rad: float,
    bucket_load_rate: float,
) -> list[float]:
    """Rebuild the exact legacy checkpoint state from retained raw fields."""
    legacy14 = numeric_vector(
        sample.get("obs.state_legacy_14d"),
        14,
        "obs.state_legacy_14d",
    )
    state_v4 = numeric_vector(v4_state, 28, "v4 observation.state")

    upper_heading = float(legacy14[2]) + float(legacy14[3])
    tip_world = legacy14[8:11]
    load_world = legacy14[11:14]

    dig_delta_world = heading_frame_delta_to_world(
        state_v4[18:21],
        upper_heading,
    )
    unload_delta_world = heading_frame_delta_to_world(
        state_v4[21:24],
        upper_heading,
    )
    dig_target_world = [
        tip_world[index] + dig_delta_world[index]
        for index in range(3)
    ]
    unload_landing_world = [
        load_world[index] + unload_delta_world[index]
        for index in range(3)
    ]

    dig_target_local = point_in_heading_frame(
        dig_target_world,
        initial_origin_xy,
        initial_heading_rad,
    )
    unload_landing_local = point_in_heading_frame(
        unload_landing_world,
        initial_origin_xy,
        initial_heading_rad,
    )

    # v4 stores the truck heading relative to the rotating upper structure.
    # Convert it back to the legacy heading relative to the episode's initial
    # body heading without depending on rounded scene metadata.
    truck_from_upper = math.atan2(state_v4[24], state_v4[25])
    truck_from_initial = (
        truck_from_upper + upper_heading - float(initial_heading_rad)
    )

    result = (
        legacy14
        + state_v4[4:8]
        + dig_target_local
        + unload_landing_local
        + [
            math.sin(truck_from_initial),
            math.cos(truck_from_initial),
            float(bucket_load_rate),
        ]
    )
    if len(result) != 27 or not all(math.isfinite(value) for value in result):
        raise RuntimeError("Reconstructed legacy state is not a finite 27D vector")
    return result


def reconstruct_v4_dataset_rows(
    raw_run: Path,
    raw_split: str,
    all_data: pd.DataFrame,
    old_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import excavator_dataset_tools as tools
    except Exception as exc:
        raise RuntimeError(
            "The v4 reconstruction requires excavator_dataset_tools.py "
            "beside this converter"
        ) from exc

    collected = tools.collect_lerobot_rows(str(raw_run), split=raw_split)
    exported_rows = list(collected.get("rows") or [])
    if len(exported_rows) != len(all_data):
        raise RuntimeError(
            "Raw/export alignment failed: "
            f"raw collector produced {len(exported_rows)} rows, "
            f"but parquet contains {len(all_data)} rows"
        )

    episode_rows = tools.load_index(str(raw_run), raw_split)
    sample_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    initial_pose_by_episode: dict[str, tuple[list[float], float]] = {}

    for episode in episode_rows:
        episode_id = str(episode.get("episode_id") or "")
        episode_dir = tools.episode_dir_from_row(episode, run_dir=str(raw_run))
        trajectory_path = tools.resolve_episode_file(
            episode_dir,
            tools.row_path_value(episode, "trajectory"),
        )
        trajectory = tools.read_jsonl(trajectory_path)
        if not trajectory:
            continue
        if not episode_id:
            episode_id = str(trajectory[0].get("id") or "")

        first_legacy = None
        for sample in trajectory:
            try:
                first_legacy = numeric_vector(
                    sample.get("obs.state_legacy_14d"),
                    14,
                    "obs.state_legacy_14d",
                )
                break
            except RuntimeError:
                continue
        if first_legacy is None:
            raise RuntimeError(
                f"Episode {episode_id!r} has no obs.state_legacy_14d"
            )
        initial_pose_by_episode[episode_id] = (
            first_legacy[:2],
            float(first_legacy[2]),
        )

        for sample in trajectory:
            sample_index = sample.get("i")
            if sample_index is None:
                continue
            sample_lookup[(episode_id, int(sample_index))] = sample

    reconstructed: list[list[float]] = []
    stages: list[int] = []
    previous_by_export_episode: dict[int, tuple[float, float]] = {}

    for row_index, exported_row in enumerate(exported_rows):
        export_episode = int(exported_row["episode_index"])
        frame_index = int(exported_row["frame_index"])
        raw_episode_id = str(exported_row.get("raw_episode_id") or "")
        raw_sample_index = int(exported_row["raw_sample_index"])
        key = (raw_episode_id, raw_sample_index)
        sample = sample_lookup.get(key)
        if sample is None:
            raise RuntimeError(f"Raw sample not found for {key}")

        parquet_episode = int(all_data.iloc[row_index]["episode_index"])
        parquet_frame = int(all_data.iloc[row_index]["frame_index"])
        if (export_episode, frame_index) != (parquet_episode, parquet_frame):
            raise RuntimeError(
                "Raw/parquet row identity mismatch at row "
                f"{row_index}: raw={(export_episode, frame_index)}, "
                f"parquet={(parquet_episode, parquet_frame)}"
            )

        source_state = numeric_vector(
            exported_row["observation.state"],
            28,
            "raw exported observation.state",
        )
        if not np.allclose(
            np.asarray(source_state, dtype=np.float32),
            old_states[row_index],
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Raw/parquet observation.state mismatch at row {row_index}"
            )

        legacy14 = numeric_vector(
            sample.get("obs.state_legacy_14d"),
            14,
            "obs.state_legacy_14d",
        )
        current_load = float(legacy14[7])
        current_time = float(sample.get("t", exported_row.get("timestamp", 0.0)))
        previous = previous_by_export_episode.get(export_episode)
        if previous is None:
            load_rate = 0.0
        else:
            previous_load, previous_time = previous
            dt = current_time - previous_time
            if not math.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(
                    f"Non-positive raw timestep in episode {export_episode}: {dt}"
                )
            load_rate = (current_load - previous_load) / dt
        previous_by_export_episode[export_episode] = (current_load, current_time)

        if raw_episode_id not in initial_pose_by_episode:
            raise RuntimeError(f"Initial pose missing for episode {raw_episode_id!r}")
        initial_origin_xy, initial_heading = initial_pose_by_episode[raw_episode_id]
        reconstructed.append(
            reconstruct_legacy_state_27d(
                sample,
                source_state,
                initial_origin_xy,
                initial_heading,
                load_rate,
            )
        )
        stages.append(checkpoint_stage_index(sample))

    states = np.asarray(reconstructed, dtype=np.float32)
    stage_ids = np.asarray(stages, dtype=np.int64)
    if states.shape != (len(all_data), 27):
        raise RuntimeError(f"Unexpected reconstructed state shape: {states.shape}")

    return states, stage_ids, {
        "mode": "v4_raw_trajectory_to_legacy_27d",
        "raw_run": str(raw_run),
        "raw_split": raw_split,
        "raw_frames": len(exported_rows),
        "stage_id_source": "raw phase.index with canonical text fallback",
        "canonical_stage_ids": {
            "curl_to_hold_material": 4,
            "pull_exit_cut": 5,
            "loaded_transit": 8,
            "unload_to_bin/dump": 9,
        },
    }


def vector_statistics(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": [int(array.shape[0])],
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def copy_tree_without_videos(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "videos":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(item, target)


def copy_videos(src: Path, dst: Path, mode: str) -> dict[str, int]:
    src_videos = src / "videos"
    dst_videos = dst / "videos"

    counts = {
        "hardlinked": 0,
        "copied": 0,
        "files": 0,
    }

    if not src_videos.exists():
        print("[WARN] Source has no videos directory")
        return counts

    for source in src_videos.rglob("*"):
        relative = source.relative_to(src_videos)
        target = dst_videos / relative

        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        counts["files"] += 1

        if mode == "copy":
            shutil.copy2(source, target)
            counts["copied"] += 1
            continue

        try:
            os.link(source, target)
            counts["hardlinked"] += 1
        except OSError as exc:
            print(
                f"[WARN] Hardlink failed for {relative}: {exc}; "
                "falling back to a full copy"
            )
            shutil.copy2(source, target)
            counts["copied"] += 1

    return counts


def dominant_future_stage_targets(
    stage_ids: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    horizon: int,
    minimum_purity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(stage_ids)
    target = np.empty(n, dtype=np.int64)
    purity = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)

    unique_episodes = np.unique(episode_ids)

    for episode_id in unique_episodes:
        row_indices = np.flatnonzero(episode_ids == episode_id)
        order = np.argsort(frame_ids[row_indices], kind="stable")
        ordered_rows = row_indices[order]
        ordered_stages = stage_ids[ordered_rows]

        for local_t, global_row in enumerate(ordered_rows):
            end = min(local_t + horizon, len(ordered_rows))
            window = ordered_stages[local_t:end]

            counts = np.bincount(window, minlength=10)
            dominant = int(np.argmax(counts))
            dominant_count = int(counts[dominant])

            target[global_row] = dominant
            purity[global_row] = dominant_count / float(len(window))

            complete_window = (end - local_t) == horizon
            valid[global_row] = (
                complete_window
                and purity[global_row] >= minimum_purity
            )

    return target, purity, valid


def remove_state_dimension_from_stats(
    obj: Any,
    dimension: int,
    old_width: int,
) -> Any:
    if isinstance(obj, dict):
        return {
            key: remove_state_dimension_from_stats(
                value,
                dimension,
                old_width,
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        if (
            len(obj) == old_width
            and all(not isinstance(x, (dict, list)) for x in obj)
        ):
            return obj[:dimension] + obj[dimension + 1 :]

        return [
            remove_state_dimension_from_stats(
                value,
                dimension,
                old_width,
            )
            for value in obj
        ]

    return obj


def scalar_statistics(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "count": int(x.size),
        "q01": float(np.quantile(x, 0.01)),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "q99": float(np.quantile(x, 0.99)),
    }


def fill_like(template: Any, value: float | int) -> Any:
    if isinstance(template, list):
        if not template:
            return []
        return [fill_like(item, value) for item in template]

    if isinstance(template, bool):
        return bool(value)

    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)

    if isinstance(template, float):
        return float(value)

    return value


def build_stats_like(
    template: dict[str, Any] | None,
    values: np.ndarray,
) -> dict[str, Any]:
    computed = scalar_statistics(values)

    if not isinstance(template, dict):
        return {
            key: [value]
            for key, value in computed.items()
        }

    result: dict[str, Any] = {}
    for key, template_value in template.items():
        if key in computed:
            result[key] = fill_like(
                template_value,
                computed[key],
            )
        else:
            result[key] = template_value

    for key, value in computed.items():
        if key not in result:
            result[key] = [value]

    return result


def append_readme(
    readme_path: Path,
    source: Path,
    horizon: int,
    minimum_purity: float,
    conversion_mode: str,
) -> None:
    if conversion_mode == "v4_raw_trajectory_to_legacy_27d":
        state_change = (
            "- Reconstructed the checkpoint's legacy 27D state from the v4 "
            "28D export and retained raw trajectory fields.\n"
            "- Preserved the v4 canonical 10-stage IDs used by the "
            "checkpoint stage head."
        )
    else:
        state_change = (
            "- Removed `phase_index` from `observation.state`.\n"
            "- Changed `observation.state` from 28 dimensions to 27 dimensions."
        )
    block = f"""

## Stage-label conversion

This dataset was derived from:

`{source}`

Changes:

{state_change}
- Added `observation.stage_current_id` as the checkpoint's scalar stage label
  in `[0, 9]`.
- Kept `stage_current_id` as a diagnostic compatibility alias.
- Added `stage_target_{horizon}` as the dominant stage in the next {horizon} frames.
- Added `stage_purity_{horizon}` and `stage_valid_{horizon}`.
- A future-stage label is valid only when the full {horizon}-frame window exists
  and the dominant-stage ratio is at least {minimum_purity:.2f}.
- Stage labels are supervision fields and must not be listed in the policy
  `input_features`.
"""
    with readme_path.open("a", encoding="utf-8") as f:
        f.write(block)


def validate_with_lerobot(dst: Path) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        print(f"[WARN] LeRobot import failed during validation: {exc}")
        return

    dataset = LeRobotDataset(str(dst))
    sample = dataset[0]

    assert tuple(sample["observation.state"].shape) == (27,)
    assert tuple(sample["action"].shape) == (4,)
    assert "observation.stage_current_id" in sample
    assert "stage_current_id" in sample
    assert "stage_target_30" in sample
    assert "stage_purity_30" in sample
    assert "stage_valid_30" in sample

    print("[OK] LeRobotDataset loaded the converted dataset")
    print(f"[OK] dataset[0] observation.state shape: {tuple(sample['observation.state'].shape)}")
    print(f"[OK] dataset[0] action shape: {tuple(sample['action'].shape)}")
    print(
        "[OK] stage fields:",
        {
            key: (
                tuple(sample[key].shape)
                if hasattr(sample[key], "shape")
                else type(sample[key]).__name__
            )
            for key in (
                "observation.stage_current_id",
                "stage_current_id",
                "stage_target_30",
                "stage_purity_30",
                "stage_valid_30",
            )
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a new LeRobot dataset by extracting phase_index from "
            "observation.state into independent stage supervision fields."
        )
    )
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument(
        "--raw-run",
        type=Path,
        default=None,
        help=(
            "Raw auto-collection run used to reconstruct a v4 28D export "
            "into the checkpoint's legacy 27D state"
        ),
    )
    parser.add_argument(
        "--raw-split",
        default="trainable",
        help="Raw-run episode split corresponding to the exported dataset",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(
            "/root/isaacsim/ExcavatorVLA/"
            "scripts/excavator_app/excavator_runtime.py"
        ),
    )
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--minimum-purity", type=float, default=0.70)
    parser.add_argument(
        "--video-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    raw_run = (
        args.raw_run.expanduser().resolve()
        if args.raw_run is not None
        else None
    )

    if not src.exists():
        raise FileNotFoundError(src)

    if src == dst:
        raise ValueError("Source and destination must be different")

    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")

    if not 0.0 <= args.minimum_purity <= 1.0:
        raise ValueError("--minimum-purity must be in [0, 1]")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}\n"
                "Use --overwrite only after confirming it is safe to delete."
            )
        shutil.rmtree(dst)

    info_path = src / "meta" / "info.json"
    stats_path = src / "meta" / "stats.json"
    data_files = sorted((src / "data").rglob("*.parquet"))

    if not info_path.exists():
        raise FileNotFoundError(info_path)

    if not data_files:
        raise FileNotFoundError(
            f"No parquet files found under {src / 'data'}"
        )

    info = load_json(info_path)
    features = info.get("features", {})
    state_spec = features.get("observation.state")

    if not isinstance(state_spec, dict):
        raise RuntimeError(
            "meta/info.json does not define observation.state"
        )

    state_names = list(state_spec.get("names") or [])
    old_shape = state_spec.get("shape")
    source_is_v4 = bool(V4_STATE_MARKERS.intersection(state_names))

    if old_shape != [28]:
        raise RuntimeError(
            f"Expected observation.state shape [28], got {old_shape}"
        )

    phase_dimension = (
        state_names.index("phase_index")
        if "phase_index" in state_names
        else None
    )

    stage_names = (
        DEFAULT_STAGE_NAMES.copy()
        if source_is_v4
        else read_stage_names(runtime)
    )

    if source_is_v4:
        if raw_run is None:
            raise RuntimeError(
                "The source is v4 28D. Pass --raw-run pointing to the raw "
                "auto-collection run so the legacy 27D state can be rebuilt."
            )
        if not raw_run.is_dir():
            raise FileNotFoundError(raw_run)

    print("=" * 88)
    print("SOURCE")
    print("=" * 88)
    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"source schema: {'v4 28D' if source_is_v4 else 'legacy v3 28D'}")
    if raw_run is not None:
        print(f"raw run: {raw_run}")
        print(f"raw split: {args.raw_split}")
    print(
        "phase dimension: "
        f"{phase_dimension if phase_dimension is not None else 'pending data validation'}"
    )
    print(f"stage names ({len(stage_names)}): {stage_names}")
    print(f"future-stage horizon: {args.horizon}")
    print(f"minimum purity: {args.minimum_purity}")
    print(f"video mode: {args.video_mode}")

    print("\nCopying non-video dataset files...")
    copy_tree_without_videos(src, dst)

    print("Copying/linking video files...")
    video_counts = copy_videos(
        src,
        dst,
        mode=args.video_mode,
    )
    print(f"video copy result: {video_counts}")

    print("\nReading data parquet files...")
    frames: list[pd.DataFrame] = []
    row_counts: list[int] = []

    for path in data_files:
        frame = pd.read_parquet(path)
        frames.append(frame)
        row_counts.append(len(frame))
        print(
            f"  {path.relative_to(src)}: "
            f"{len(frame)} rows"
        )

    all_data = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    required_columns = {
        "observation.state",
        "episode_index",
        "frame_index",
    }
    missing_columns = required_columns.difference(all_data.columns)
    if missing_columns:
        raise RuntimeError(
            f"Missing required data columns: {sorted(missing_columns)}"
        )

    old_states = np.stack(
        all_data["observation.state"].to_numpy()
    ).astype(np.float32)

    if old_states.shape[1] != 28:
        raise RuntimeError(
            f"Runtime state matrix is {old_states.shape}, expected [N,28]"
        )

    reconstruction_details: dict[str, Any] = {}
    if source_is_v4:
        assert raw_run is not None
        phase_dimension = None
        phase_dimension_source = "raw_trajectory_phase_remap"
        new_states, stage_current, reconstruction_details = (
            reconstruct_v4_dataset_rows(
                raw_run,
                args.raw_split,
                all_data,
                old_states,
            )
        )
        output_state_names = LEGACY_STATE_NAMES_27D.copy()
        print(
            "Reconstructed checkpoint state from raw trajectory: "
            f"{new_states.shape}"
        )
    else:
        phase_dimension, phase_dimension_source = resolve_phase_dimension(
            state_names,
            old_states,
            len(stage_names),
        )
        if phase_dimension != 18:
            print(
                f"[WARN] phase_index is at dimension {phase_dimension}, "
                "not dimension 18; using the name-based location"
            )
        elif phase_dimension_source != "metadata":
            print(
                "[WARN] observation.state metadata has no phase_index name; "
                "using legacy dimension 18 after validating every stage value"
            )
        print(
            f"Resolved phase dimension: {phase_dimension} "
            f"(source={phase_dimension_source})"
        )

        raw_stage = old_states[:, phase_dimension]
        stage_current = np.rint(raw_stage).astype(np.int64)

        if not np.allclose(raw_stage, stage_current, atol=1e-6):
            raise RuntimeError("phase_index contains non-integer values")

        if (
            np.min(stage_current) < 0
            or np.max(stage_current) >= len(stage_names)
        ):
            raise RuntimeError(
                "phase_index contains values outside the defined stage range: "
                f"min={np.min(stage_current)}, max={np.max(stage_current)}, "
                f"num_stages={len(stage_names)}"
            )

        new_states = np.delete(
            old_states,
            phase_dimension,
            axis=1,
        ).astype(np.float32)
        output_state_names = (
            state_names[:phase_dimension]
            + state_names[phase_dimension + 1 :]
        )

    if new_states.shape[1] != 27:
        raise AssertionError(new_states.shape)

    episode_ids = all_data["episode_index"].to_numpy(
        dtype=np.int64
    )
    frame_ids = all_data["frame_index"].to_numpy(
        dtype=np.int64
    )

    stage_target, stage_purity, stage_valid = (
        dominant_future_stage_targets(
            stage_ids=stage_current,
            episode_ids=episode_ids,
            frame_ids=frame_ids,
            horizon=args.horizon,
            minimum_purity=args.minimum_purity,
        )
    )

    horizon = args.horizon
    target_name = f"stage_target_{horizon}"
    purity_name = f"stage_purity_{horizon}"
    valid_name = f"stage_valid_{horizon}"

    all_data["observation.state"] = [
        row for row in new_states
    ]

    insert_at = list(all_data.columns).index(
        "observation.state"
    ) + 1

    all_data.insert(
        insert_at,
        "observation.stage_current_id",
        stage_current,
    )
    all_data.insert(
        insert_at + 1,
        "stage_current_id",
        stage_current,
    )
    all_data.insert(
        insert_at + 2,
        target_name,
        stage_target,
    )
    all_data.insert(
        insert_at + 3,
        purity_name,
        stage_purity.astype(np.float32),
    )
    all_data.insert(
        insert_at + 4,
        valid_name,
        stage_valid.astype(np.int64),
    )

    print("\nWriting transformed data parquet files...")
    offset = 0
    for source_path, row_count in zip(data_files, row_counts):
        relative = source_path.relative_to(src)
        target_path = dst / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)

        output_frame = all_data.iloc[
            offset : offset + row_count
        ].copy()

        output_frame.to_parquet(
            target_path,
            index=False,
            engine="pyarrow",
        )

        print(
            f"  {relative}: {len(output_frame)} rows"
        )
        offset += row_count

    if offset != len(all_data):
        raise AssertionError(
            f"Wrote {offset} rows, expected {len(all_data)}"
        )

    print("\nUpdating meta/info.json...")
    new_features: dict[str, Any] = {}

    for feature_name, feature_spec in features.items():
        if feature_name == "observation.state":
            new_state_spec = dict(feature_spec)
            new_state_spec["shape"] = [27]
            new_state_spec["names"] = output_state_names
            new_features[feature_name] = new_state_spec

            stage_feature_spec = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }
            new_features["observation.stage_current_id"] = dict(
                stage_feature_spec
            )
            new_features["stage_current_id"] = dict(stage_feature_spec)
            new_features[target_name] = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }
            new_features[purity_name] = {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            }
            new_features[valid_name] = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }
        else:
            new_features[feature_name] = feature_spec

    new_info = dict(info)
    new_info["features"] = new_features
    save_json(dst / "meta" / "info.json", new_info)

    print("Updating meta/stats.json...")
    if stats_path.exists():
        stats = load_json(stats_path)
        if "observation.state" in stats:
            if source_is_v4:
                stats["observation.state"] = vector_statistics(new_states)
            else:
                assert phase_dimension is not None
                stats["observation.state"] = (
                    remove_state_dimension_from_stats(
                        stats["observation.state"],
                        phase_dimension,
                        old_width=28,
                    )
                )

        template = (
            stats.get("task_index")
            or stats.get("frame_index")
            or stats.get("episode_index")
        )

        current_stage_stats = build_stats_like(
            template,
            stage_current,
        )
        stats["observation.stage_current_id"] = current_stage_stats
        stats["stage_current_id"] = current_stage_stats
        stats[target_name] = build_stats_like(
            template,
            stage_target,
        )
        stats[purity_name] = build_stats_like(
            template,
            stage_purity,
        )
        stats[valid_name] = build_stats_like(
            template,
            stage_valid.astype(np.int64),
        )

        save_json(dst / "meta" / "stats.json", stats)
    else:
        print("[WARN] Source has no meta/stats.json")

    stage_counts = Counter(stage_current.tolist())
    target_counts = Counter(
        stage_target[stage_valid].tolist()
    )

    stage_schema = {
        "num_stages": len(stage_names),
        "classes": {
            str(index): name
            for index, name in enumerate(stage_names)
        },
        "source_feature": (
            "raw trajectory: obs.state_legacy_14d + v4 observation.state"
            if source_is_v4
            else "observation.state"
        ),
        "source_dimension": phase_dimension,
        "source_name": (
            "raw phase/label resolved with the v4 canonical stage contract"
            if source_is_v4
            else "phase_index"
        ),
        "input_state_shape_before": [28],
        "input_state_shape_after": [27],
        "current_stage_field": "observation.stage_current_id",
        "current_stage_compatibility_alias": "stage_current_id",
        "future_stage_field": target_name,
        "future_stage_horizon_frames": horizon,
        "minimum_future_stage_purity": args.minimum_purity,
        "future_stage_purity_field": purity_name,
        "future_stage_valid_field": valid_name,
        "training_note": (
            "Stage fields are supervision labels. "
            "Do not include them in model input_features."
        ),
        "reconstruction": reconstruction_details,
    }
    save_json(
        dst / "meta" / "stage_schema.json",
        stage_schema,
    )

    conversion_manifest = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "source_total_frames": int(len(all_data)),
        "phase_dimension_removed": phase_dimension,
        "state_shape_before": [28],
        "state_shape_after": [27],
        "stage_names": stage_names,
        "stage_current_counts": {
            str(key): int(value)
            for key, value in sorted(stage_counts.items())
        },
        "stage_target_valid_counts": {
            str(key): int(value)
            for key, value in sorted(target_counts.items())
        },
        "future_stage_horizon_frames": horizon,
        "minimum_future_stage_purity": args.minimum_purity,
        "valid_future_stage_rows": int(
            np.sum(stage_valid)
        ),
        "valid_future_stage_ratio": float(
            np.mean(stage_valid)
        ),
        "video_copy": video_counts,
        "phase_source": phase_dimension_source,
        "reconstruction": reconstruction_details,
    }
    save_json(
        dst / "meta" / "stage_conversion_manifest.json",
        conversion_manifest,
    )

    readme_path = dst / "README.md"
    if readme_path.exists():
        append_readme(
            readme_path,
            source=src,
            horizon=horizon,
            minimum_purity=args.minimum_purity,
            conversion_mode=str(
                reconstruction_details.get("mode") or "legacy_phase_removal"
            ),
        )

    print("\n" + "=" * 88)
    print("VALIDATION")
    print("=" * 88)

    output_info = load_json(
        dst / "meta" / "info.json"
    )
    output_features = output_info["features"]

    assert output_features["observation.state"]["shape"] == [27]
    assert "phase_index" not in output_features["observation.state"]["names"]
    assert output_features["observation.stage_current_id"]["dtype"] == "int64"
    assert output_features["stage_current_id"]["dtype"] == "int64"
    assert output_features[target_name]["dtype"] == "int64"

    destination_data_files = sorted(
        (dst / "data").rglob("*.parquet")
    )
    destination_rows = sum(
        len(pd.read_parquet(path))
        for path in destination_data_files
    )
    assert destination_rows == len(all_data)

    source_video_count = sum(
        1 for path in (src / "videos").rglob("*")
        if path.is_file()
    )
    destination_video_count = sum(
        1 for path in (dst / "videos").rglob("*")
        if path.is_file()
    )
    assert source_video_count == destination_video_count

    print(f"[OK] source frames: {len(all_data)}")
    print(f"[OK] destination frames: {destination_rows}")
    print(f"[OK] observation.state: 28D -> 27D")
    if source_is_v4:
        print("[OK] v4 state reconstructed as checkpoint legacy 27D")
        print("[OK] v4 phases use canonical checkpoint stage ids")
    else:
        print("[OK] phase_index removed from model state")
    print(
        "[OK] observation.stage_current_id range: "
        f"{stage_current.min()}..{stage_current.max()}"
    )
    print(
        f"[OK] {target_name} range: "
        f"{stage_target.min()}..{stage_target.max()}"
    )
    print(
        f"[OK] valid future-stage labels: "
        f"{int(np.sum(stage_valid))}/{len(stage_valid)} "
        f"({100.0 * np.mean(stage_valid):.2f}%)"
    )
    print(f"[OK] videos: {destination_video_count}")

    validate_with_lerobot(dst)

    print("\nCurrent-stage counts:")
    for stage_id, name in enumerate(stage_names):
        print(
            f"  {stage_id:02d} {name:<28} "
            f"{stage_counts.get(stage_id, 0)}"
        )

    print("\nValid future-stage target counts:")
    for stage_id, name in enumerate(stage_names):
        print(
            f"  {stage_id:02d} {name:<28} "
            f"{target_counts.get(stage_id, 0)}"
        )

    print("\n" + "=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Converted dataset: {dst}")
    print(f"Stage schema: {dst / 'meta' / 'stage_schema.json'}")
    print(
        "The source dataset was not modified. "
        "Video hardlinks share file contents but the transformed "
        "Parquet and metadata files are independent copies."
    )


if __name__ == "__main__":
    main()
