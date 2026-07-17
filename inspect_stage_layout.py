#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_base_config(base_root: Path) -> Path | None:
    direct = base_root / "config.json"
    if direct.exists():
        return direct

    candidates = sorted(base_root.rglob("config.json"))
    for path in candidates:
        try:
            obj = load_json(path)
        except Exception:
            continue
        if isinstance(obj, dict) and (
            obj.get("type") == "smolvla"
            or "input_features" in obj
            or "output_features" in obj
        ):
            return path

    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--base-model-root", required=True, type=Path)
    parser.add_argument("--sample-rows", type=int, default=0)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    base_root = args.base_model_root.expanduser().resolve()

    info_path = root / "meta" / "info.json"
    data_files = sorted((root / "data").rglob("*.parquet"))

    if not info_path.exists():
        raise FileNotFoundError(info_path)
    if not data_files:
        raise FileNotFoundError(f"No data parquet found under {root / 'data'}")

    info = load_json(info_path)
    features = info.get("features", {})
    state_spec = features.get("observation.state", {})
    state_names = state_spec.get("names") or []

    print("=" * 100)
    print("A. DATASET FEATURE SCHEMA")
    print("=" * 100)
    print(f"dataset_root: {root}")
    print(f"fps: {info.get('fps')}")
    print(f"total_episodes: {info.get('total_episodes')}")
    print(f"total_frames: {info.get('total_frames')}")
    print(f"state declared shape: {state_spec.get('shape')}")
    print(f"action declared shape: {features.get('action', {}).get('shape')}")
    print(f"effort declared shape: {features.get('observation.effort', {}).get('shape')}")
    print("\nall top-level features:")
    for key, spec in features.items():
        print(f"  {key}: dtype={spec.get('dtype')} shape={spec.get('shape')}")

    print("\nfull observation.state names:")
    for i in range(int(state_spec.get("shape", [0])[0])):
        name = state_names[i] if i < len(state_names) else "<unnamed>"
        print(f"  {i:02d}: {name}")

    stage_like_feature_names = [
        name for name in features
        if any(token in name.lower() for token in ("stage", "phase", "mode"))
    ]
    print("\nseparate stage-like top-level features:")
    print(stage_like_feature_names if stage_like_feature_names else "  NONE")

    print("\nparquet columns:")
    parquet_schema = pq.ParquetFile(data_files[0]).schema_arrow
    for name in parquet_schema.names:
        print(f"  {name}")

    separate_stage_columns = [
        name for name in parquet_schema.names
        if any(token in name.lower() for token in ("stage", "phase", "mode"))
    ]
    print("\nseparate stage-like parquet columns:")
    print(separate_stage_columns if separate_stage_columns else "  NONE")

    print("\n" + "=" * 100)
    print("B. LOAD STATE MATRIX")
    print("=" * 100)

    arrays = []
    episode_arrays = []
    frame_arrays = []

    rows_remaining = args.sample_rows if args.sample_rows > 0 else None

    for path in data_files:
        cols = ["observation.state"]
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        if "episode_index" in schema_names:
            cols.append("episode_index")
        if "frame_index" in schema_names:
            cols.append("frame_index")

        table = pq.read_table(path, columns=cols)
        if rows_remaining is not None:
            if rows_remaining <= 0:
                break
            table = table.slice(0, min(rows_remaining, table.num_rows))
            rows_remaining -= table.num_rows

        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        arrays.append(states)

        if "episode_index" in table.column_names:
            episode_arrays.append(np.asarray(table["episode_index"], dtype=np.int64))
        if "frame_index" in table.column_names:
            frame_arrays.append(np.asarray(table["frame_index"], dtype=np.int64))

    X = np.concatenate(arrays, axis=0)
    episodes = np.concatenate(episode_arrays, axis=0) if episode_arrays else None
    frames = np.concatenate(frame_arrays, axis=0) if frame_arrays else None

    print(f"loaded rows: {X.shape[0]}")
    print(f"runtime state shape: {X.shape}")

    if X.ndim != 2:
        raise RuntimeError(f"Expected [N,D] state matrix, got {X.shape}")

    print("\nper-dimension diagnostics:")
    binary_mask = np.zeros(X.shape[1], dtype=bool)
    near_integer_mask = np.zeros(X.shape[1], dtype=bool)

    for i in range(X.shape[1]):
        col = X[:, i]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            print(f"  {i:02d}: all non-finite")
            continue

        near_zero_or_one = np.isclose(finite, 0.0, atol=1e-6) | np.isclose(
            finite, 1.0, atol=1e-6
        )
        binary_fraction = float(np.mean(near_zero_or_one))
        integer_fraction = float(np.mean(np.isclose(finite, np.round(finite), atol=1e-6)))
        binary_mask[i] = binary_fraction >= 0.9999
        near_integer_mask[i] = integer_fraction >= 0.9999

        unique = np.unique(finite)
        unique_preview = unique[:12].tolist()
        suffix = " ..." if unique.size > 12 else ""
        name = state_names[i] if i < len(state_names) else "<unnamed>"

        print(
            f"  {i:02d} {name}: "
            f"min={finite.min():+.6f} max={finite.max():+.6f} "
            f"mean={finite.mean():+.6f} std={finite.std():.6f} "
            f"unique_count={unique.size} "
            f"binary_fraction={binary_fraction:.6f} "
            f"integer_fraction={integer_fraction:.6f} "
            f"unique={unique_preview}{suffix}"
        )

    print("\n" + "=" * 100)
    print("C. SEARCH FOR ONE-HOT STAGE BLOCKS")
    print("=" * 100)

    candidates = []
    max_width = min(16, X.shape[1])

    for width in range(2, max_width + 1):
        for start in range(0, X.shape[1] - width + 1):
            end = start + width
            block = X[:, start:end]

            values_binary = np.isclose(block, 0.0, atol=1e-6) | np.isclose(
                block, 1.0, atol=1e-6
            )
            binary_fraction = float(np.mean(values_binary))
            row_sums = np.sum(block, axis=1)
            onehot_fraction = float(np.mean(np.isclose(row_sums, 1.0, atol=1e-6)))

            if binary_fraction >= 0.9999 and onehot_fraction >= 0.95:
                candidates.append(
                    (onehot_fraction, binary_fraction, start, end, width)
                )

    candidates.sort(reverse=True)

    if not candidates:
        print("No contiguous near-binary one-hot block found.")
    else:
        print("candidate blocks, best first:")
        for onehot_fraction, binary_fraction, start, end, width in candidates[:20]:
            names = [
                state_names[i] if i < len(state_names) else f"dim_{i}"
                for i in range(start, end)
            ]
            print(
                f"  dims [{start}:{end}] width={width} "
                f"binary_fraction={binary_fraction:.6f} "
                f"onehot_fraction={onehot_fraction:.6f}"
            )
            print(f"    names={names}")

        _, _, start, end, width = candidates[0]
        block = X[:, start:end]
        stage_ids = np.argmax(block, axis=1)
        row_sums = np.sum(block, axis=1)
        valid = np.isclose(row_sums, 1.0, atol=1e-6)

        print("\nbest candidate class counts:")
        counts = np.bincount(stage_ids[valid], minlength=width)
        for stage_id, count in enumerate(counts):
            name = (
                state_names[start + stage_id]
                if start + stage_id < len(state_names)
                else f"stage_{stage_id}"
            )
            print(f"  {stage_id:02d} {name}: {int(count)}")

        if episodes is not None and frames is not None:
            transition_counts: dict[tuple[int, int], int] = {}
            order = np.lexsort((frames, episodes))
            ep_sorted = episodes[order]
            st_sorted = stage_ids[order]
            valid_sorted = valid[order]

            for i in range(1, len(order)):
                if ep_sorted[i] != ep_sorted[i - 1]:
                    continue
                if not (valid_sorted[i] and valid_sorted[i - 1]):
                    continue
                a = int(st_sorted[i - 1])
                b = int(st_sorted[i])
                if a != b:
                    transition_counts[(a, b)] = transition_counts.get((a, b), 0) + 1

            print("\nobserved cross-stage transitions:")
            for (a, b), count in sorted(
                transition_counts.items(),
                key=lambda item: (-item[1], item[0]),
            ):
                print(f"  {a:02d} -> {b:02d}: {count}")

    print("\n" + "=" * 100)
    print("D. BASE MODEL CONFIG")
    print("=" * 100)

    config_path = find_base_config(base_root)
    if config_path is None:
        print(f"No config.json found under {base_root}")
    else:
        config = load_json(config_path)
        print(f"config_path: {config_path}")
        for key in ("type", "chunk_size", "n_action_steps"):
            print(f"{key}: {config.get(key)}")

        print("\ninput_features:")
        print(json.dumps(config.get("input_features"), indent=2, ensure_ascii=False))

        print("\noutput_features:")
        print(json.dumps(config.get("output_features"), indent=2, ensure_ascii=False))

        outputs = config.get("output_features") or {}
        stage_outputs = [
            key for key in outputs
            if any(token in key.lower() for token in ("stage", "phase", "mode"))
        ]
        print("\nstage-like output features in base model:")
        print(stage_outputs if stage_outputs else "  NONE")

    print("\n" + "=" * 100)
    print("E. DIRECT CONCLUSION")
    print("=" * 100)

    if separate_stage_columns or stage_like_feature_names:
        print("[FOUND] A separate stage-like dataset feature/column exists.")
    else:
        print("[NOT FOUND] No separate stage supervision field exists in the dataset.")

    if candidates:
        _, _, start, end, width = candidates[0]
        print(
            f"[FOUND] The strongest stage-like one-hot block is inside "
            f"observation.state at dims [{start}:{end}] with width={width}."
        )
        print(
            "[WARNING] If this block represents the target stage, feeding it into "
            "the model input leaks the answer."
        )
    else:
        print("[NOT FOUND] No convincing one-hot stage block was detected inside state.")

    if config_path is not None:
        outputs = config.get("output_features") or {}
        if any(
            any(token in key.lower() for token in ("stage", "phase", "mode"))
            for key in outputs
        ):
            print("[FOUND] Base model config already declares a stage output.")
        else:
            print("[NOT FOUND] Base model config has no stage output; it currently outputs action only.")


if __name__ == "__main__":
    main()
