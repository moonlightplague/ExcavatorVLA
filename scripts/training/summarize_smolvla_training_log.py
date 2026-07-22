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


# Accept exact counters such as 950 and compact counters such as 1K / 1.5K.
SCALED = r"[0-9]+(?:\.[0-9]+)?[KMG]?"

STEP_PATTERN = re.compile(
    rf"step:(?P<step>{SCALED})\s+"
    rf"smpl:(?P<smpl>{SCALED})\s+"
    rf"ep:(?P<ep>{SCALED})\s+"
    r"epch:(?P<epch>[0-9.eE+-]+)\s+"
    r"loss:(?P<train_loss>[0-9.eE+-]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.eE+-]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.eE+-]+)\s+"
    r"data_s:(?P<data_s>[0-9.eE+-]+)"
)

# TQDM usually keeps the exact step even when the logger abbreviates it.
PROGRESS_PATTERN = re.compile(r"\|\s*(\d+)\s*/\s*(\d+)\s*\[")

NUMERIC_PAIR_PATTERN = re.compile(
    r"""['"](?P<key>[^'"]+)['"]\s*:\s*
        (?P<value>
            [-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?
            |nan|inf|-inf
        )
    """,
    re.VERBOSE | re.IGNORECASE,
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
    if text and text[-1].upper() in suffixes:
        return float(text[:-1]) * suffixes[text[-1].upper()]
    return float(text)


def numeric_dict(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            result[key] = float(value)
        elif isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                result[key] = number
    return result


def parse_metrics_payload(payload: str) -> dict[str, float]:
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, dict):
            return numeric_dict(parsed)
    except (SyntaxError, ValueError):
        pass

    # Fallback for payloads containing nan/inf or other values that
    # ast.literal_eval cannot parse.
    result: dict[str, float] = {}
    for match in NUMERIC_PAIR_PATTERN.finditer(payload):
        key = match.group("key")
        value_text = match.group("value").lower()
        value = float(value_text)
        if math.isfinite(value):
            result[key] = value
    return result


def parse_log(path: Path) -> tuple[list[dict[str, float]], dict[str, int]]:
    records: list[dict[str, float]] = []
    pending: dict[str, float] | None = None

    stats = {
        "step_lines": 0,
        "matched_step_lines": 0,
        "policy_metric_lines": 0,
        "complete_records": 0,
    }

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ot_train.py:435" in line:
                stats["step_lines"] += 1
                match = STEP_PATTERN.search(line)
                if match:
                    stats["matched_step_lines"] += 1

                    progress_match = PROGRESS_PATTERN.search(line)
                    if progress_match:
                        exact_step = float(progress_match.group(1))
                    else:
                        exact_step = parse_scaled_number(match.group("step"))

                    pending = {
                        "step": exact_step,
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

            if "policy_metrics:" not in line:
                continue

            stats["policy_metric_lines"] += 1
            if pending is None:
                continue

            metrics = parse_metrics_payload(
                line.split("policy_metrics:", 1)[1].strip()
            )
            if not metrics:
                continue

            record = dict(pending)
            record.update(metrics)
            records.append(record)
            stats["complete_records"] += 1
            pending = None

    return records, stats


def summarize_blocks(
    records: list[dict[str, float]],
    block_size: int,
) -> pd.DataFrame:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)

    for record in records:
        step = int(round(record["step"]))
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

            mean = sum(values) / len(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_last"] = values[-1]

            if key in KEY_STD_FIELDS and len(values) > 1:
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

        print(
            f"{start:4d}-{end:<4d} "
            f"{get(row, 'loss_mean'):8.4f} "
            f"{get(row, 'action_loss_mean'):8.4f} "
            f"{get(row, 'weighted_stage_loss_mean'):8.4f} "
            f"{get(row, 'weighted_stage_action_loss_mean'):9.4f} "
            f"{get(row, 'stage_action_raw_loss_mean'):8.5f} "
            f"{100.0 * get(row, 'stage_action_violation_rate_mean'):6.2f} "
            f"{100.0 * get(row, 'stage_action_prior_coverage_mean'):7.2f} "
            f"{100.0 * get(row, 'stage_accuracy_mean'):10.2f} "
            f"{get(row, 'grad_norm_mean'):5.2f} "
            f"{get(row, 'lr_last'):.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)

    output_path = args.output
    if output_path is None:
        output_path = log_path.with_name(
            f"{log_path.stem}_{args.block_size}step_summary.csv"
        )
    output_path = output_path.expanduser().resolve()

    records, stats = parse_log(log_path)
    if not records:
        raise RuntimeError(f"No complete training records found in {log_path}")

    summary = summarize_blocks(records, args.block_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print_summary(summary)
    print()
    print("step log lines:", stats["step_lines"])
    print("matched step lines:", stats["matched_step_lines"])
    print("policy metric lines:", stats["policy_metric_lines"])
    print("complete records:", stats["complete_records"])
    print("last parsed step:", int(max(record["step"] for record in records)))
    print("summary blocks:", len(summary))
    print("all numeric metrics CSV:", output_path)


if __name__ == "__main__":
    main()
