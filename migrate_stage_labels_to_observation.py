#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


RENAME_MAP = {
    "stage_current_id": "observation.stage_current_id",
    "stage_target_30": "observation.stage_target_30",
    "stage_purity_30": "observation.stage_purity_30",
    "stage_valid_30": "observation.stage_valid_30",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
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


def link_or_copy_videos(src: Path, dst: Path) -> tuple[int, int]:
    source_root = src / "videos"
    destination_root = dst / "videos"
    hardlinked = 0
    copied = 0

    for source in source_root.rglob("*"):
        relative = source.relative_to(source_root)
        target = destination_root / relative

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


def recursively_replace_strings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            RENAME_MAP.get(str(key), str(key)): recursively_replace_strings(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [recursively_replace_strings(item) for item in value]
    if isinstance(value, str):
        return RENAME_MAP.get(value, value)
    return value


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
        raise ValueError("Source and destination must differ")

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}\n"
                "Use --overwrite only after checking the path."
            )
        shutil.rmtree(dst)

    info_path = src / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    info = load_json(info_path)
    features = info.get("features", {})

    missing_features = [key for key in RENAME_MAP if key not in features]
    if missing_features:
        raise RuntimeError(
            f"Missing source feature declarations: {missing_features}"
        )

    data_files = sorted((src / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files under {src / 'data'}")

    print("Source:", src)
    print("Destination:", dst)
    print("Field renames:")
    for old, new in RENAME_MAP.items():
        print(f"  {old} -> {new}")

    copy_without_videos(src, dst)
    hardlinked, copied = link_or_copy_videos(src, dst)
    print(f"Videos: hardlinked={hardlinked}, copied={copied}")

    total_rows = 0
    for source_path in data_files:
        relative = source_path.relative_to(src)
        target_path = dst / relative

        frame = pd.read_parquet(source_path)
        missing_columns = [key for key in RENAME_MAP if key not in frame.columns]
        if missing_columns:
            raise RuntimeError(
                f"{relative} is missing columns: {missing_columns}"
            )

        frame = frame.rename(columns=RENAME_MAP)
        frame.to_parquet(target_path, index=False, engine="pyarrow")
        total_rows += len(frame)
        print(f"Wrote {relative}: {len(frame)} rows")

    new_features: dict[str, Any] = {}
    for key, spec in features.items():
        new_features[RENAME_MAP.get(key, key)] = spec
    info["features"] = new_features
    save_json(dst / "meta" / "info.json", info)

    stats_path = src / "meta" / "stats.json"
    if stats_path.exists():
        stats = load_json(stats_path)
        stats = {
            RENAME_MAP.get(key, key): value
            for key, value in stats.items()
        }
        save_json(dst / "meta" / "stats.json", stats)

    for filename in (
        "stage_schema.json",
        "stage_conversion_manifest.json",
    ):
        source_path = src / "meta" / filename
        if source_path.exists():
            value = recursively_replace_strings(load_json(source_path))
            save_json(dst / "meta" / filename, value)

    migration_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(src),
        "destination_dataset": str(dst),
        "reason": (
            "LeRobot batch preprocessing preserves arbitrary auxiliary fields "
            "when they use the observation.* prefix."
        ),
        "renamed_fields": RENAME_MAP,
        "total_rows": total_rows,
        "videos_hardlinked": hardlinked,
        "videos_copied": copied,
        "training_note": (
            "The observation.stage_* fields are labels only. "
            "Do not add them to policy.input_features."
        ),
    }
    save_json(
        dst / "meta" / "observation_stage_label_migration.json",
        migration_manifest,
    )

    readme = dst / "README.md"
    with readme.open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## Auxiliary stage-label key migration\n\n"
            "The stage supervision fields use the `observation.` prefix so "
            "LeRobot's processor pipeline preserves them in training batches. "
            "They remain labels and are not policy input features.\n"
        )

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(str(dst))
    sample = dataset[0]

    assert tuple(sample["observation.state"].shape) == (27,)
    assert tuple(sample["observation.effort"].shape) == (4,)
    assert tuple(sample["action"].shape) == (4,)

    for new_key in RENAME_MAP.values():
        assert new_key in sample, (new_key, sample.keys())

    for old_key in RENAME_MAP:
        assert old_key not in sample

    print()
    print("[OK] LeRobotDataset validation passed")
    print("frames:", dataset.num_frames)
    print("episodes:", dataset.num_episodes)
    print("sample stage labels:")
    for key in RENAME_MAP.values():
        value = sample[key]
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} value={value}")
    print("Converted dataset:", dst)


if __name__ == "__main__":
    main()
