#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


STEP_PATTERN = re.compile(
    r"step:(?P<step>\d+)\s+"
    r"smpl:(?P<smpl>[0-9.]+[KMG]?)\s+"
    r"ep:(?P<ep>[0-9.]+[KMG]?)\s+"
    r"epch:(?P<epch>[0-9.]+)\s+"
    r"loss:(?P<train_loss>[0-9.eE+-]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.eE+-]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.eE+-]+)\s+"
    r"data_s:(?P<data_s>[0-9.eE+-]+)"
)


KEY_STD_FIELDS = {
    "train_loss",
    "grad_norm",
    "action_loss",
    "stage_ce_loss",
    "stage_raw_loss",
    "weighted_stage_loss",
    "stage_action_raw_loss",
    "weighted_stage_action_loss",
    "stage_action_violation_rate",
    "stage_accuracy",
    "loss",
}


def parse_scaled_number(text: str) -> float:
    suffixes = {"K": 1e3, "M": 1e6, "G": 1e9}
    if text and text[-1] in suffixes:
        return float(text[:-1]) * suffixes[text[-1]]
    return float(text)


def numeric_dict(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            result[key] = float(value)
        elif isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                result[key] = value
    return result


def parse_log(path: Path) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    pending: dict[str, float] | None = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ot_train.py:435" in line:
                match = STEP_PATTERN.search(line)
                if match:
                    pending = {
                        "step": float(match.group("step")),
                        "samples": parse_scaled_number(match.group("smpl")),
                        "episodes": parse_scaled_number(match.group("ep")),
                        "epoch": float(match.group("epch")),
                        "train_loss": float(match.group("train_loss")),
                        "grad_norm": float(match.group("grad_norm")),
                        "lr": float(match.group("lr")),
                        "update_s": float(match.group("update_s")),
                        "data_s": float(match.group("data_s")),
                    }
                continue

            if "policy_metrics:" not in line or pending is None:
                continue

            try:
                metrics = ast.literal_eval(
                    line.split("policy_metrics:", 1)[1].strip()
                )
            except (SyntaxError, ValueError):
                continue

            record = dict(pending)
            record.update(numeric_dict(metrics))
            records.append(record)
            pending = None

    return records


def summarize_blocks(
    records: list[dict[str, float]],
    block_size: int,
) -> pd.DataFrame:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)

    for record in records:
        step = int(record["step"])
        block_end = ((step - 1) // block_size + 1) * block_size
        grouped[block_end].append(record)

    rows: list[dict[str, float]] = []

    for block_end in sorted(grouped):
        block = grouped[block_end]
        row: dict[str, float] = {
            "block_start": float(block_end - block_size + 1),
            "block_end": float(block_end),
            "records": float(len(block)),
            "first_step": min(item["step"] for item in block),
            "last_step": max(item["step"] for item in block),
        }

        all_keys = sorted(
            {
                key
                for item in block
                for key in item.keys()
                if key != "step"
            }
        )

        for key in all_keys:
            values = [
                item[key]
                for item in block
                if key in item and math.isfinite(item[key])
            ]
            if not values:
                continue

            row[f"{key}_mean"] = sum(values) / len(values)
            row[f"{key}_last"] = values[-1]

            if key in KEY_STD_FIELDS and len(values) > 1:
                mean = row[f"{key}_mean"]
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                row[f"{key}_std"] = math.sqrt(variance)

        rows.append(row)

    return pd.DataFrame(rows)


def get(row: pd.Series, key: str) -> float:
    value = row.get(key, float("nan"))
    return float(value) if pd.notna(value) else float("nan")


def print_summary(frame: pd.DataFrame) -> None:
    print(
        "block      total    action   stage    stage_act "
        "raw_prior viol%  cover% stage_acc% grad   lr"
    )
    print("-" * 108)

    for _, row in frame.iterrows():
        start = int(get(row, "block_start"))
        end = int(get(row, "block_end"))

        total = get(row, "loss_mean")
        action = get(row, "action_loss_mean")
        stage = get(row, "weighted_stage_loss_mean")
        stage_action = get(row, "weighted_stage_action_loss_mean")
        raw_prior = get(row, "stage_action_raw_loss_mean")
        violation = 100.0 * get(row, "stage_action_violation_rate_mean")
        coverage = 100.0 * get(row, "stage_action_prior_coverage_mean")
        stage_acc = 100.0 * get(row, "stage_accuracy_mean")
        grad = get(row, "grad_norm_mean")
        lr = get(row, "lr_last")

        print(
            f"{start:4d}-{end:<4d} "
            f"{total:8.4f} "
            f"{action:8.4f} "
            f"{stage:8.4f} "
            f"{stage_action:9.4f} "
            f"{raw_prior:8.5f} "
            f"{violation:6.2f} "
            f"{coverage:7.2f} "
            f"{stage_acc:10.2f} "
            f"{grad:5.2f} "
            f"{lr:.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize all numeric SmolVLA training log metrics "
            "into non-overlapping N-step blocks."
        )
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path. Default: <log_stem>_50step_summary.csv",
    )
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    output_path = args.output
    if output_path is None:
        output_path = log_path.with_name(
            f"{log_path.stem}_{args.block_size}step_summary.csv"
        )
    output_path = output_path.expanduser().resolve()

    records = parse_log(log_path)
    if not records:
        raise RuntimeError(f"No complete training records found in {log_path}")

    summary = summarize_blocks(records, args.block_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print_summary(summary)
    print()
    print("complete records:", len(records))
    print("summary blocks:", len(summary))
    print("all numeric metrics CSV:", output_path)


if __name__ == "__main__":
    main()
