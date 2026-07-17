#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def heading(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact(value: Any, max_items: int = 8) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        shown = keys[:max_items]
        suffix = " ..." if len(keys) > max_items else ""
        return "{" + ", ".join(map(str, shown)) + suffix + "}"
    if isinstance(value, (list, tuple)):
        shown = list(value[:max_items])
        suffix = " ..." if len(value) > max_items else ""
        return f"{shown}{suffix}"
    return repr(value)


def infer_cell_shape(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if hasattr(value, "as_py"):
        value = value.as_py()
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    return tuple(arr.shape)


def print_tree(root: Path, max_depth: int = 3, max_entries: int = 120) -> None:
    root = root.resolve()
    count = 0
    for current, dirs, files in os.walk(root):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        if depth > max_depth:
            dirs[:] = []
            continue

        indent = "  " * depth
        if depth == 0:
            print(f"{root}/")
        else:
            print(f"{indent}{current_path.name}/")
        count += 1

        for name in sorted(files)[:30]:
            p = current_path / name
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            print(f"{indent}  {name}  ({size} bytes)")
            count += 1
            if count >= max_entries:
                print("... tree output truncated ...")
                return


def inspect_info(dataset_root: Path) -> dict[str, Any] | None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        print(f"[MISSING] {info_path}")
        return None

    info = safe_json_load(info_path)
    print(f"path: {info_path}")
    for key in (
        "codebase_version",
        "robot_type",
        "fps",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "total_videos",
        "chunks_size",
        "data_path",
        "video_path",
    ):
        if key in info:
            print(f"{key}: {info[key]}")

    features = info.get("features", {})
    print(f"\nfeatures ({len(features)}):")
    for name, spec in features.items():
        print(
            f"  {name}: dtype={spec.get('dtype')} "
            f"shape={spec.get('shape')} names={compact(spec.get('names'))}"
        )
    return info


def inspect_parquet(dataset_root: Path, info: dict[str, Any] | None) -> None:
    try:
        import pyarrow as pa
        import pyarrow.dataset as pads
        import pyarrow.parquet as pq
    except Exception as exc:
        print(f"[ERROR] pyarrow unavailable: {exc}")
        return

    parquet_files = sorted(dataset_root.rglob("*.parquet"))
    print(f"parquet files: {len(parquet_files)}")
    if not parquet_files:
        return

    by_top = Counter()
    total_rows_from_metadata = 0
    for p in parquet_files:
        rel = p.relative_to(dataset_root)
        by_top[rel.parts[0] if rel.parts else "."] += 1
        try:
            total_rows_from_metadata += pq.ParquetFile(p).metadata.num_rows
        except Exception:
            pass

    print(f"parquet files by top-level directory: {dict(by_top)}")
    print(f"sum of parquet metadata rows: {total_rows_from_metadata}")

    data_files = [p for p in parquet_files if "data" in p.relative_to(dataset_root).parts]
    if not data_files:
        data_files = parquet_files

    sample_file = data_files[0]
    pf = pq.ParquetFile(sample_file)
    print(f"\nsample data parquet: {sample_file}")
    print(f"rows={pf.metadata.num_rows}, row_groups={pf.metadata.num_row_groups}")
    print("schema:")
    print(pf.schema_arrow)

    table = pq.read_table(sample_file)
    print("\nfirst-row runtime values:")
    row = table.slice(0, 1).to_pylist()[0]
    feature_specs = (info or {}).get("features", {})
    for name in table.column_names:
        value = row.get(name)
        actual_shape = infer_cell_shape(value)
        expected_shape = feature_specs.get(name, {}).get("shape")
        print(
            f"  {name}: actual_shape={actual_shape} "
            f"expected_shape={expected_shape} value={compact(value)}"
        )

    scalar_candidates = [
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
    ]
    scalar_cols = [c for c in scalar_candidates if c in pf.schema_arrow.names]

    try:
        ds = pads.dataset([str(p) for p in data_files], format="parquet")
        available = set(ds.schema.names)
        scalar_cols = [c for c in scalar_candidates if c in available]
        if scalar_cols:
            meta_table = ds.to_table(columns=scalar_cols)
            meta = meta_table.to_pandas()
            print("\nscalar metadata summary:")
            print(meta.describe(include="all").to_string())

            if "episode_index" in meta and "frame_index" in meta:
                grouped = meta.groupby("episode_index", sort=True)
                lengths = grouped.size()
                print("\nepisode length summary:")
                print(lengths.describe().to_string())

                bad_start = 0
                bad_contiguous = 0
                bad_timestamp = 0
                fps = float((info or {}).get("fps", 0) or 0)

                for _, g in grouped:
                    frames = g["frame_index"].to_numpy()
                    if len(frames) and frames[0] != 0:
                        bad_start += 1
                    if len(frames) > 1 and not np.all(np.diff(frames) == 1):
                        bad_contiguous += 1
                    if fps > 0 and "timestamp" in g and len(g) > 1:
                        dt = np.diff(g["timestamp"].to_numpy(dtype=float))
                        expected = 1.0 / fps
                        if not np.all(np.isfinite(dt)) or np.max(np.abs(dt - expected)) > 5e-3:
                            bad_timestamp += 1

                print(f"episodes not starting at frame 0: {bad_start}")
                print(f"episodes with non-contiguous frame_index: {bad_contiguous}")
                if fps > 0 and "timestamp" in meta:
                    print(f"episodes with timestamp step inconsistent with fps={fps}: {bad_timestamp}")

            if "task_index" in meta:
                print(f"unique task_index count: {meta['task_index'].nunique()}")
        else:
            print("[WARN] no standard scalar metadata columns found")
    except Exception as exc:
        print(f"[WARN] aggregate parquet inspection failed: {type(exc).__name__}: {exc}")


def inspect_meta_files(dataset_root: Path) -> None:
    meta_root = dataset_root / "meta"
    if not meta_root.exists():
        print("[MISSING] meta directory")
        return

    for name in (
        "tasks.parquet",
        "episodes.parquet",
        "episodes_stats.parquet",
        "stats.json",
    ):
        path = meta_root / name
        if not path.exists():
            print(f"[MISSING] {path}")
            continue
        print(f"\n{name}: {path.stat().st_size} bytes")
        try:
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq
                table = pq.read_table(path)
                print(table.schema)
                print(table.slice(0, min(5, table.num_rows)).to_pandas().to_string(index=False))
            else:
                obj = safe_json_load(path)
                print(compact(obj, max_items=20))
        except Exception as exc:
            print(f"[ERROR] cannot read {path}: {type(exc).__name__}: {exc}")


def inspect_videos(dataset_root: Path, info: dict[str, Any] | None) -> None:
    video_files = sorted(
        [
            *dataset_root.rglob("*.mp4"),
            *dataset_root.rglob("*.avi"),
            *dataset_root.rglob("*.mkv"),
        ]
    )
    print(f"video files: {len(video_files)}")
    if video_files:
        for p in video_files[:10]:
            print(f"  {p.relative_to(dataset_root)} ({p.stat().st_size} bytes)")
        if len(video_files) > 10:
            print("  ...")

    image_features = []
    for name, spec in ((info or {}).get("features", {})).items():
        dtype = str(spec.get("dtype", "")).lower()
        if "image" in name or dtype in {"image", "video"}:
            image_features.append((name, spec))
    print(f"image/video feature declarations: {len(image_features)}")
    for name, spec in image_features:
        print(f"  {name}: {spec}")


def inspect_lerobot(dataset_root: Path) -> None:
    try:
        import lerobot
        print(f"lerobot module: {getattr(lerobot, '__file__', None)}")
        print(f"lerobot version: {getattr(lerobot, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"[ERROR] import lerobot failed: {type(exc).__name__}: {exc}")
        return

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print(f"LeRobotDataset.__init__ signature: {inspect.signature(LeRobotDataset.__init__)}")
    except Exception as exc:
        print(f"[WARN] cannot inspect LeRobotDataset: {type(exc).__name__}: {exc}")
        return

    attempts = [
        ("positional local path", lambda: LeRobotDataset(str(dataset_root))),
        (
            "repo_id=folder_name, root=parent",
            lambda: LeRobotDataset(
                repo_id=dataset_root.name,
                root=dataset_root.parent,
                local_files_only=True,
            ),
        ),
        (
            "repo_id=full path",
            lambda: LeRobotDataset(
                repo_id=str(dataset_root),
                local_files_only=True,
            ),
        ),
    ]

    dataset = None
    for label, fn in attempts:
        try:
            dataset = fn()
            print(f"[OK] loader construction succeeded using: {label}")
            break
        except Exception as exc:
            print(f"[FAIL] loader construction {label}: {type(exc).__name__}: {exc}")

    if dataset is None:
        print("[WARN] dataset could not be instantiated; raw format inspection above is still valid")
        return

    for attr in ("num_frames", "num_episodes", "fps", "features", "meta"):
        if hasattr(dataset, attr):
            try:
                print(f"{attr}: {compact(getattr(dataset, attr), max_items=20)}")
            except Exception as exc:
                print(f"{attr}: <error {exc}>")

    try:
        sample = dataset[0]
        print("\nLeRobotDataset[0] keys and tensor shapes:")
        for key, value in sample.items():
            shape = tuple(value.shape) if hasattr(value, "shape") else None
            dtype = getattr(value, "dtype", type(value).__name__)
            print(f"  {key}: shape={shape} dtype={dtype}")
    except Exception as exc:
        print(f"[WARN] dataset[0] failed: {type(exc).__name__}: {exc}")


def search_checkpoint_configs(checkpoint_root: Path | None) -> None:
    if checkpoint_root is None:
        print("checkpoint_root not provided")
        return
    if not checkpoint_root.exists():
        print(f"[MISSING] {checkpoint_root}")
        return

    keywords = {
        "input_features",
        "output_features",
        "chunk_size",
        "n_action_steps",
        "action",
        "observation.state",
        "observation.images",
    }

    json_files = sorted(checkpoint_root.rglob("*.json"))
    print(f"json config files under checkpoint: {len(json_files)}")
    for path in json_files:
        try:
            obj = safe_json_load(path)
            text = json.dumps(obj, ensure_ascii=False)
        except Exception:
            continue
        if any(k in text for k in keywords):
            print(f"\n--- {path} ---")
            if isinstance(obj, dict):
                for key in (
                    "type",
                    "chunk_size",
                    "n_action_steps",
                    "input_features",
                    "output_features",
                    "normalization_mapping",
                ):
                    if key in obj:
                        print(f"{key}: {json.dumps(obj[key], ensure_ascii=False, indent=2)[:8000]}")
            else:
                print(text[:8000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    checkpoint_root = (
        args.checkpoint_root.expanduser().resolve()
        if args.checkpoint_root is not None
        else None
    )

    heading("0. ENVIRONMENT")
    print(f"python: {sys.version}")
    print(f"platform: {platform.platform()}")
    print(f"dataset_root: {dataset_root}")
    print(f"checkpoint_root: {checkpoint_root}")
    print(f"dataset exists: {dataset_root.exists()}")

    if not dataset_root.exists():
        raise SystemExit(2)

    heading("1. DATASET TREE")
    print_tree(dataset_root)

    heading("2. META/INFO.JSON")
    info = inspect_info(dataset_root)

    heading("3. META TABLES")
    inspect_meta_files(dataset_root)

    heading("4. DATA PARQUET")
    inspect_parquet(dataset_root, info)

    heading("5. VIDEOS / IMAGE FEATURES")
    inspect_videos(dataset_root, info)

    heading("6. LEROBOT LOADER COMPATIBILITY")
    inspect_lerobot(dataset_root)

    heading("7. CHECKPOINT / POLICY EXPECTATIONS")
    search_checkpoint_configs(checkpoint_root)

    heading("8. INITIAL COMPATIBILITY CHECKLIST")
    features = (info or {}).get("features", {})
    checks = {
        "observation.state present": "observation.state" in features,
        "action present": "action" in features,
        "episode_index present": "episode_index" in features,
        "frame_index present": "frame_index" in features,
        "timestamp present": "timestamp" in features,
        "task_index present": "task_index" in features,
        "3 camera features present": sum(
            1 for k in features if k.startswith("observation.images.")
        ) == 3,
    }
    for name, ok in checks.items():
        print(f"[{'OK' if ok else 'CHECK'}] {name}")

    state_shape = features.get("observation.state", {}).get("shape")
    action_shape = features.get("action", {}).get("shape")
    print(f"declared observation.state shape: {state_shape}")
    print(f"declared action shape: {action_shape}")
    print("\nInspection only: this script does not modify the dataset.")


if __name__ == "__main__":
    main()
