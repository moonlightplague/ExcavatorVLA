#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CURRENT_STAGE_KEY = "observation.stage_current_id"
OLD_STAGE_KEYS = (
    "observation.stage_target_30",
    "observation.stage_purity_30",
    "observation.stage_valid_30",
)
STAGE_SEQUENCE_KEY = "observation.stage_sequence_50"
STAGE_VALID_KEY = "observation.stage_sequence_valid_50"
CHUNK_SIZE = 50
NUM_STAGES = 10


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def copy_without_videos(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "videos":
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, copy_function=shutil.copy2)
        else:
            shutil.copy2(item, target)


def hardlink_or_copy_videos(src: Path, dst: Path) -> tuple[int, int]:
    src_root = src / "videos"
    dst_root = dst / "videos"
    hardlinked = 0
    copied = 0

    if not src_root.exists():
        return hardlinked, copied

    for source in src_root.rglob("*"):
        relative = source.relative_to(src_root)
        target = dst_root / relative

        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
            hardlinked += 1
        except OSError:
            shutil.copy2(source, target)
            copied += 1

    return hardlinked, copied


def make_stage_sequences(
    stage_ids: np.ndarray,
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(stage_ids)
    sequences = np.zeros((n, CHUNK_SIZE), dtype=np.int64)
    valid = np.zeros((n, CHUNK_SIZE), dtype=np.int64)

    for episode_id in np.unique(episode_ids):
        rows = np.flatnonzero(episode_ids == episode_id)
        order = np.argsort(frame_ids[rows], kind="stable")
        ordered_rows = rows[order]
        ordered_stages = stage_ids[ordered_rows]

        for local_index, global_row in enumerate(ordered_rows):
            available = min(CHUNK_SIZE, len(ordered_rows) - local_index)
            future = ordered_stages[local_index : local_index + available]

            sequences[global_row, :available] = future
            valid[global_row, :available] = 1

            if available < CHUNK_SIZE:
                sequences[global_row, available:] = int(future[-1])

    return sequences, valid


def scalar_stats(values: np.ndarray) -> dict[str, list[float | int]]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": [float(x.min())],
        "max": [float(x.max())],
        "mean": [float(x.mean())],
        "std": [float(x.std())],
        "count": [int(x.size)],
        "q01": [float(np.quantile(x, 0.01))],
        "q10": [float(np.quantile(x, 0.10))],
        "q50": [float(np.quantile(x, 0.50))],
        "q90": [float(np.quantile(x, 0.90))],
        "q99": [float(np.quantile(x, 0.99))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()

    if not src.exists():
        raise FileNotFoundError(src)
    if src == dst:
        raise ValueError("Source and destination must be different")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}\n"
                "Use --overwrite only after checking the path."
            )
        shutil.rmtree(dst)

    info_path = src / "meta" / "info.json"
    data_files = sorted((src / "data").rglob("*.parquet"))

    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not data_files:
        raise FileNotFoundError(f"No parquet files under {src / 'data'}")

    info = load_json(info_path)
    features = info.get("features", {})
    if CURRENT_STAGE_KEY not in features:
        raise RuntimeError(
            f"{CURRENT_STAGE_KEY} is missing from meta/info.json"
        )

    print("=" * 88)
    print("STAGE-SEQUENCE DATASET CONVERSION")
    print("=" * 88)
    print("source:", src)
    print("destination:", dst)
    print("stage sequence:", STAGE_SEQUENCE_KEY, "[50]")
    print("stage validity:", STAGE_VALID_KEY, "[50]")

    copy_without_videos(src, dst)
    hardlinked, copied = hardlink_or_copy_videos(src, dst)
    print(f"videos: hardlinked={hardlinked}, copied={copied}")

    frames: list[pd.DataFrame] = []
    row_counts: list[int] = []

    for path in data_files:
        frame = pd.read_parquet(path)
        required = {
            CURRENT_STAGE_KEY,
            "episode_index",
            "frame_index",
            "observation.state",
            "observation.effort",
            "action",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise RuntimeError(
                f"{path.relative_to(src)} missing columns: {sorted(missing)}"
            )
        frames.append(frame)
        row_counts.append(len(frame))

    all_data = pd.concat(frames, ignore_index=True)

    stage_ids = all_data[CURRENT_STAGE_KEY].to_numpy(dtype=np.int64)
    if stage_ids.min() < 0 or stage_ids.max() >= NUM_STAGES:
        raise RuntimeError(
            f"Stage IDs out of range: {stage_ids.min()}..{stage_ids.max()}"
        )

    episode_ids = all_data["episode_index"].to_numpy(dtype=np.int64)
    frame_ids = all_data["frame_index"].to_numpy(dtype=np.int64)

    sequences, valid = make_stage_sequences(
        stage_ids,
        episode_ids,
        frame_ids,
    )

    for old_key in OLD_STAGE_KEYS:
        if old_key in all_data.columns:
            all_data = all_data.drop(columns=[old_key])

    insert_at = list(all_data.columns).index(CURRENT_STAGE_KEY) + 1
    all_data.insert(
        insert_at,
        STAGE_SEQUENCE_KEY,
        [row for row in sequences],
    )
    all_data.insert(
        insert_at + 1,
        STAGE_VALID_KEY,
        [row for row in valid],
    )

    print("writing parquet...")
    offset = 0
    for source_path, row_count in zip(data_files, row_counts):
        relative = source_path.relative_to(src)
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        output = all_data.iloc[offset : offset + row_count].copy()
        output.to_parquet(target, index=False, engine="pyarrow")
        print(f"  {relative}: {len(output)} rows")
        offset += row_count

    if offset != len(all_data):
        raise AssertionError((offset, len(all_data)))

    new_features: dict[str, Any] = {}
    for name, spec in features.items():
        if name in OLD_STAGE_KEYS:
            continue
        new_features[name] = spec
        if name == CURRENT_STAGE_KEY:
            new_features[STAGE_SEQUENCE_KEY] = {
                "dtype": "int64",
                "shape": [CHUNK_SIZE],
                "names": [f"step_{i}" for i in range(CHUNK_SIZE)],
            }
            new_features[STAGE_VALID_KEY] = {
                "dtype": "int64",
                "shape": [CHUNK_SIZE],
                "names": [f"step_{i}" for i in range(CHUNK_SIZE)],
            }

    info["features"] = new_features
    save_json(dst / "meta" / "info.json", info)

    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        for old_key in OLD_STAGE_KEYS:
            stats.pop(old_key, None)
        stats[STAGE_SEQUENCE_KEY] = scalar_stats(sequences)
        stats[STAGE_VALID_KEY] = scalar_stats(valid)
        save_json(dst / "meta" / "stats.json", stats)

    stage_schema = {
        "num_stages": NUM_STAGES,
        "chunk_size": CHUNK_SIZE,
        "current_stage_key": CURRENT_STAGE_KEY,
        "stage_sequence_key": STAGE_SEQUENCE_KEY,
        "stage_sequence_valid_key": STAGE_VALID_KEY,
        "alignment": (
            "stage_sequence_50[t] corresponds to action chunk step t, "
            "for t=0..49"
        ),
        "padding": (
            "Stage IDs beyond the episode end repeat the final real stage; "
            "stage_sequence_valid_50 is 0 at those padded positions."
        ),
        "training_note": (
            "Stage sequence fields are supervision labels only and must not "
            "be added to policy.input_features."
        ),
    }
    save_json(dst / "meta" / "stage_sequence_schema.json", stage_schema)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "total_frames": int(len(all_data)),
        "total_episodes": int(len(np.unique(episode_ids))),
        "chunk_size": CHUNK_SIZE,
        "num_stages": NUM_STAGES,
        "valid_stage_tokens": int(valid.sum()),
        "total_stage_tokens": int(valid.size),
        "valid_stage_token_fraction": float(valid.mean()),
        "videos_hardlinked": hardlinked,
        "videos_copied": copied,
    }
    save_json(
        dst / "meta" / "stage_sequence_conversion_manifest.json",
        manifest,
    )

    readme = dst / "README.md"
    with readme.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## 50-step stage sequence supervision\n\n"
            "Each sample contains `observation.stage_sequence_50` and "
            "`observation.stage_sequence_valid_50`. These labels align "
            "one-to-one with the 50-step action chunk and are not policy "
            "input features.\n"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(str(dst))
    sample = dataset[0]

    assert tuple(sample["observation.state"].shape) == (27,)
    assert tuple(sample["observation.effort"].shape) == (4,)
    assert tuple(sample["action"].shape) == (4,)
    assert tuple(sample[STAGE_SEQUENCE_KEY].shape) == (CHUNK_SIZE,)
    assert tuple(sample[STAGE_VALID_KEY].shape) == (CHUNK_SIZE,)

    print()
    print("=" * 88)
    print("VALIDATION PASSED")
    print("=" * 88)
    print("frames:", dataset.num_frames)
    print("episodes:", dataset.num_episodes)
    print("state:", tuple(sample["observation.state"].shape))
    print("effort:", tuple(sample["observation.effort"].shape))
    print("action:", tuple(sample["action"].shape))
    print("stage sequence:", tuple(sample[STAGE_SEQUENCE_KEY].shape))
    print("stage valid:", tuple(sample[STAGE_VALID_KEY].shape))
    print("valid stage token fraction:", f"{valid.mean():.6f}")
    print("destination:", dst)


if __name__ == "__main__":
    main()
