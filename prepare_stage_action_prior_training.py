#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_NAMES = {
    0: "pre_dig_align",
    1: "approach_contact",
    2: "insert",
    3: "pull_mid",
    4: "pull_exit",
    5: "curl",
    6: "secure_load",
    7: "lift_carry",
    8: "unload_to_bin",
    9: "unload_dump",
}

ACTION_NAMES = {
    0: "swing",
    1: "boom",
    2: "arm",
    3: "bucket",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def action_vector(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (4,):
        raise ValueError(f"Expected 4D action, got {array.shape}")
    return array


def load_actions(dataset_root: Path, action_key: str) -> np.ndarray:
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {dataset_root / 'data'}"
        )

    chunks: list[np.ndarray] = []
    for index, path in enumerate(parquet_files, start=1):
        frame = pd.read_parquet(path, columns=[action_key])
        actions = np.stack(
            [action_vector(value) for value in frame[action_key]],
            axis=0,
        )
        chunks.append(actions)
        print(
            f"[{index:03d}/{len(parquet_files):03d}] "
            f"{path.relative_to(dataset_root)}: {len(actions)} rows"
        )

    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--min-prior-weight", type=float, default=0.8)
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    prior_path = args.prior.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    payload = load_json(prior_path)
    actions = load_actions(dataset_root, args.action_key)

    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)
    action_std = np.where(action_std > 1e-8, action_std, 1.0)

    direction = np.asarray(payload["direction"], dtype=np.int64)
    weight = np.asarray(payload["weight"], dtype=np.float64)
    reliable = np.asarray(payload["reliable"], dtype=np.int64).astype(bool)
    action_scale = np.asarray(payload["action_scale"], dtype=np.float64)

    if direction.shape != (10, 4):
        raise ValueError(f"Expected direction shape (10, 4), got {direction.shape}")
    if weight.shape != (10, 4):
        raise ValueError(f"Expected weight shape (10, 4), got {weight.shape}")
    if reliable.shape != (10, 4):
        raise ValueError(f"Expected reliable shape (10, 4), got {reliable.shape}")
    if action_scale.shape != (4,):
        raise ValueError(f"Expected action_scale shape (4,), got {action_scale.shape}")

    strong = reliable & (weight >= args.min_prior_weight) & (direction != 0)

    payload["action_mean"] = action_mean.tolist()
    payload["action_std"] = action_std.tolist()
    payload["min_prior_weight"] = float(args.min_prior_weight)
    payload["strong_mask"] = strong.astype(np.int64).tolist()
    payload["source_dataset"] = str(dataset_root)
    payload["direction_loss"] = {
        "prediction_space": "raw_action",
        "clean_action_reconstruction": "x_t - t * v_t",
        "formula": (
            "weighted_mean(relu(-direction * predicted_raw_action "
            "/ action_scale)^2)"
        ),
        "zero_behavior": "zero action receives zero directional penalty",
        "stage_supervision": "ground-truth stage sequence",
    }

    save_json(output_path, payload)

    print()
    print("=" * 88)
    print("ACTION NORMALIZATION")
    print("=" * 88)
    print("rows:", len(actions))
    print("action mean:", action_mean.tolist())
    print("action std:", action_std.tolist())
    print("action scale:", action_scale.tolist())

    print()
    print("=" * 88)
    print(f"STRONG RULES: weight >= {args.min_prior_weight}")
    print("=" * 88)
    for stage_id in range(10):
        for action_id in range(4):
            if not strong[stage_id, action_id]:
                continue
            sign = int(direction[stage_id, action_id])
            sign_text = "positive" if sign > 0 else "negative"
            print(
                f"{stage_id:2d} {STAGE_NAMES[stage_id]:18s} "
                f"{ACTION_NAMES[action_id]:7s} {sign_text:8s} "
                f"weight={weight[stage_id, action_id]:.4f}"
            )

    print()
    print("strong rules:", int(strong.sum()))
    print("output:", output_path)


if __name__ == "__main__":
    main()
