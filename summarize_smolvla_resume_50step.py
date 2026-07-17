#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCALED = r"[0-9]+(?:\.[0-9]+)?[KMG]?"

TRAIN_PATTERN = re.compile(
    rf"step:(?P<logged_step>{SCALED})\s+"
    rf"smpl:(?P<samples>{SCALED})\s+"
    rf"ep:(?P<episodes>{SCALED})\s+"
    r"epch:(?P<epoch>[0-9.eE+-]+)\s+"
    r"loss:(?P<train_loss>[0-9.eE+-]+)\s+"
    r"grdn:(?P<grad_norm>[0-9.eE+-]+)\s+"
    r"lr:(?P<lr>[0-9.eE+-]+)\s+"
    r"updt_s:(?P<update_s>[0-9.eE+-]+)\s+"
    r"data_s:(?P<data_s>[0-9.eE+-]+)"
)

# In resumed runs this is the exact number of optimizer steps completed after
# the resume point, e.g. "120/12810".
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


def parse_scaled(text: str) -> float:
    suffix = text[-1].upper()
    factors = {"K": 1e3, "M": 1e6, "G": 1e9}
    if suffix in factors:
        return float(text[:-1]) * factors[suffix]
    return float(text)


def numeric_dict(payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            value = float(value)
            if math.isfinite(value):
                out[key] = value
    return out


def parse_metrics(payload: str) -> dict[str, float]:
    try:
        parsed = ast.literal_eval(payload)
        if isinstance(parsed, dict):
            return numeric_dict(parsed)
    except (SyntaxError, ValueError):
        pass

    out: dict[str, float] = {}
    for match in NUMERIC_PAIR_PATTERN.finditer(payload):
        value = float(match.group("value"))
        if math.isfinite(value):
            out[match.group("key")] = value
    return out


def parse_log(log_path: Path, resume_step: int) -> list[dict[str, float]]:
    records: list[dict[str, float]] = []
    pending: dict[str, float] | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "ot_train.py:435" in line:
                match = TRAIN_PATTERN.search(line)
                if not match:
                    continue

                progress = PROGRESS_PATTERN.search(line)
                if progress:
                    local_step = int(progress.group(1))
                    global_step = resume_step + local_step
                else:
                    # Exact for un-abbreviated logs. For an abbreviated resumed
                    # log, provide --resume-step so the progress counter is used.
                    global_step = int(round(parse_scaled(match.group("logged_step"))))
                    local_step = global_step - resume_step

                pending = {
                    "local_step": float(local_step),
                    "global_step": float(global_step),
                    "samples": parse_scaled(match.group("samples")),
                    "episodes": parse_scaled(match.group("episodes")),
                    "epoch": float(match.group("epoch")),
                    "train_loss": float(match.group("train_loss")),
                    "grad_norm": float(match.group("grad_norm")),
                    "lr": float(match.group("lr")),
                    "update_s": float(match.group("update_s")),
                    "data_s": float(match.group("data_s")),
                }
                continue

            if "policy_metrics:" not in line or pending is None:
                continue

            metrics = parse_metrics(
                line.split("policy_metrics:", 1)[1].strip()
            )
            if not metrics:
                continue

            record = dict(pending)
            record.update(metrics)
            records.append(record)
            pending = None

    return records


def mean(block: list[dict[str, float]], key: str) -> float:
    values = [
        row[key]
        for row in block
        if key in row and math.isfinite(row[key])
    ]
    return sum(values) / len(values) if values else float("nan")


def last(block: list[dict[str, float]], key: str) -> float:
    values = [
        row[key]
        for row in block
        if key in row and math.isfinite(row[key])
    ]
    return values[-1] if values else float("nan")


def summarize(
    records: list[dict[str, float]],
    block_size: int,
    resume_step: int,
    include_partial: bool,
) -> list[dict[str, float]]:
    groups: dict[int, list[dict[str, float]]] = defaultdict(list)

    for row in records:
        local_step = int(round(row["local_step"]))
        if local_step <= 0:
            continue
        block_end_local = ((local_step - 1) // block_size + 1) * block_size
        groups[block_end_local].append(row)

    summaries: list[dict[str, float]] = []

    for block_end_local in sorted(groups):
        block = groups[block_end_local]
        observed_last = int(max(row["local_step"] for row in block))
        complete = observed_last >= block_end_local
        if not complete and not include_partial:
            continue

        block_start_local = block_end_local - block_size + 1
        block_start_global = resume_step + block_start_local
        block_end_global = resume_step + block_end_local

        all_keys = sorted(
            {
                key
                for row in block
                for key in row
                if key not in {"local_step", "global_step"}
            }
        )

        summary: dict[str, float] = {
            "local_start": float(block_start_local),
            "local_end": float(block_end_local),
            "global_start": float(block_start_global),
            "global_end": float(block_end_global),
            "records": float(len(block)),
            "complete": float(complete),
        }

        for key in all_keys:
            summary[f"{key}_mean"] = mean(block, key)
            summary[f"{key}_last"] = last(block, key)

        total = summary.get("loss_mean", float("nan"))
        for component, output_key in (
            ("action_loss_mean", "action_pct"),
            ("weighted_stage_loss_mean", "stage_pct"),
            ("weighted_stage_action_loss_mean", "stage_action_pct"),
        ):
            value = summary.get(component, float("nan"))
            summary[output_key] = (
                100.0 * value / total
                if math.isfinite(total) and total != 0 and math.isfinite(value)
                else float("nan")
            )

        summaries.append(summary)

    return summaries


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "   nan"
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "   nan"
    return f"{100.0 * value:6.2f}"


def print_tables(rows: list[dict[str, float]]) -> None:
    print("LOSS / OPTIMIZATION (mean within each 50-step block)")
    print(
        "global block   n  total   action   stage  stg_act "
        " act%  stg% aux%  grad    lr"
    )
    print("-" * 99)

    for row in rows:
        start = int(row["global_start"])
        end = int(row["global_end"])
        print(
            f"{start:5d}-{end:<5d} "
            f"{int(row['records']):2d} "
            f"{fmt(row.get('loss_mean', float('nan'))):>7} "
            f"{fmt(row.get('action_loss_mean', float('nan'))):>8} "
            f"{fmt(row.get('weighted_stage_loss_mean', float('nan'))):>7} "
            f"{fmt(row.get('weighted_stage_action_loss_mean', float('nan'))):>8} "
            f"{row.get('action_pct', float('nan')):5.1f} "
            f"{row.get('stage_pct', float('nan')):5.1f} "
            f"{row.get('stage_action_pct', float('nan')):5.1f} "
            f"{row.get('grad_norm_mean', float('nan')):5.2f} "
            f"{row.get('lr_last', float('nan')):.2e}"
        )

    print()
    print("STAGE / DIRECTION / LOW-MOTION (mean within each 50-step block)")
    print(
        "global block  stageAcc valid% dir_raw dirViol% prior% "
        "low_raw lowViol% lowCov% swingAbs excess"
    )
    print("-" * 111)

    for row in rows:
        start = int(row["global_start"])
        end = int(row["global_end"])
        print(
            f"{start:5d}-{end:<5d} "
            f"{pct(row.get('stage_accuracy_mean', float('nan'))):>8} "
            f"{pct(row.get('stage_valid_fraction_mean', float('nan'))):>6} "
            f"{fmt(row.get('stage_action_direction_raw_loss_mean', float('nan')), 5):>7} "
            f"{pct(row.get('stage_action_violation_rate_mean', float('nan'))):>8} "
            f"{pct(row.get('stage_action_prior_coverage_mean', float('nan'))):>6} "
            f"{fmt(row.get('stage_action_low_motion_raw_loss_mean', float('nan')), 6):>8} "
            f"{pct(row.get('stage_action_low_motion_violation_rate_mean', float('nan'))):>8} "
            f"{pct(row.get('stage_action_low_motion_coverage_mean', float('nan'))):>7} "
            f"{fmt(row.get('stage_action_low_motion_abs_swing_mean_mean', float('nan')), 5):>8} "
            f"{fmt(row.get('stage_action_low_motion_excess_mean_mean', float('nan')), 5):>6}"
        )


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--resume-step", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--include-partial", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    log_path = args.log.expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    records = parse_log(log_path, args.resume_step)
    if not records:
        raise RuntimeError(f"No complete log records found in {log_path}")

    rows = summarize(
        records,
        args.block_size,
        args.resume_step,
        args.include_partial,
    )
    if not rows:
        raise RuntimeError(
            "No complete block found yet. Add --include-partial to display "
            "the currently incomplete block."
        )

    output = args.output
    if output is None:
        output = log_path.with_name(
            f"{log_path.stem}_{args.block_size}step_important_summary.csv"
        )
    output = output.expanduser().resolve()

    print_tables(rows)
    write_csv(output, rows)

    print()
    print("parsed records:", len(records))
    print("last local step:", int(max(row["local_step"] for row in records)))
    print("last global step:", int(max(row["global_step"] for row in records)))
    print("CSV with all numeric means/last values:", output)


if __name__ == "__main__":
    main()
