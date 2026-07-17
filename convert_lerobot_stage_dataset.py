#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STAGE_NAMES = [
    "pre_dig_align",
    "approach_contact",
    "insert",
    "pull_mid",
    "pull_exit",
    "curl",
    "secure_load",
    "lift_carry",
    "unload_to_bin",
    "unload_dump",
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
) -> None:
    block = f"""

## Stage-label conversion

This dataset was derived from:

`{source}`

Changes:

- Removed `phase_index` from `observation.state`.
- Changed `observation.state` from 28 dimensions to 27 dimensions.
- Added `stage_current_id` as a scalar integer label in `[0, 9]`.
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

    if old_shape != [28]:
        raise RuntimeError(
            f"Expected observation.state shape [28], got {old_shape}"
        )

    if "phase_index" not in state_names:
        raise RuntimeError(
            "phase_index is not present in observation.state names"
        )

    phase_dimension = state_names.index("phase_index")

    if phase_dimension != 18:
        print(
            f"[WARN] phase_index is at dimension {phase_dimension}, "
            "not dimension 18; using the name-based location"
        )

    stage_names = read_stage_names(runtime)

    print("=" * 88)
    print("SOURCE")
    print("=" * 88)
    print(f"src: {src}")
    print(f"dst: {dst}")
    print(f"phase dimension: {phase_dimension}")
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

    raw_stage = old_states[:, phase_dimension]
    stage_current = np.rint(raw_stage).astype(np.int64)

    if not np.allclose(
        raw_stage,
        stage_current,
        atol=1e-6,
    ):
        raise RuntimeError(
            "phase_index contains non-integer values"
        )

    if np.min(stage_current) < 0 or np.max(stage_current) >= len(stage_names):
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
        "stage_current_id",
        stage_current,
    )
    all_data.insert(
        insert_at + 1,
        target_name,
        stage_target,
    )
    all_data.insert(
        insert_at + 2,
        purity_name,
        stage_purity.astype(np.float32),
    )
    all_data.insert(
        insert_at + 3,
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
            new_state_spec["names"] = (
                state_names[:phase_dimension]
                + state_names[phase_dimension + 1 :]
            )
            new_features[feature_name] = new_state_spec

            new_features["stage_current_id"] = {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            }
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

        stats["stage_current_id"] = build_stats_like(
            template,
            stage_current,
        )
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
        "source_feature": "observation.state",
        "source_dimension": phase_dimension,
        "source_name": "phase_index",
        "input_state_shape_before": [28],
        "input_state_shape_after": [27],
        "current_stage_field": "stage_current_id",
        "future_stage_field": target_name,
        "future_stage_horizon_frames": horizon,
        "minimum_future_stage_purity": args.minimum_purity,
        "future_stage_purity_field": purity_name,
        "future_stage_valid_field": valid_name,
        "training_note": (
            "Stage fields are supervision labels. "
            "Do not include them in model input_features."
        ),
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
    print(f"[OK] phase_index removed from model state")
    print(f"[OK] stage_current_id range: {stage_current.min()}..{stage_current.max()}")
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
