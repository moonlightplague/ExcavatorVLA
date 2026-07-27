#!/usr/bin/env python3
"""Aggregate Isaac Sim excavation success-rate episode status files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


EPISODE_PATTERN = re.compile(r"episode_(\d+)$")
CSV_FIELDS = (
    "episode_index",
    "status",
    "success",
    "task_completed_success_and_bucket_open",
    "early_terminated",
    "termination_reason",
    "policy_steps_inferred",
    "policy_steps_executed",
    "final_newly_deposited_particle_count",
    "final_bucket_position_rad",
    "final_bucket_open",
    "success_time_from_policy_start_seconds",
    "success_time_from_simulation_launch_seconds",
    "task_completion_time_from_policy_start_seconds",
    "task_completion_time_from_simulation_launch_seconds",
    "rollout_time_from_policy_start_seconds",
    "rollout_time_from_simulation_launch_seconds",
    "error",
    "status_path",
)


def episode_index(path: Path) -> int:
    match = EPISODE_PATTERN.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(
            f"Status path is not under episode_NNNN: {path}"
        )
    return int(match.group(1))


def load_episode_rows(episodes_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for status_path in sorted(episodes_dir.glob("episode_*/episode_status.json")):
        index = episode_index(status_path)
        if index in seen:
            raise RuntimeError(f"Duplicate episode index: {index}")
        seen.add(index)
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "excavator_success_rate_episode_v1":
            raise RuntimeError(
                f"Unexpected status schema in {status_path}: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("status") == "completed":
            validation_path = status_path.parent / "rollout_validation.json"
            try:
                validation = json.loads(
                    validation_path.read_text(encoding="utf-8")
                )
                if validation.get("status") != "passed":
                    raise RuntimeError(
                        f"validation status={validation.get('status')!r}"
                    )
            except Exception as exc:
                payload["status"] = "validation_error"
                payload["error"] = (
                    "Missing or invalid rollout validation: "
                    f"{type(exc).__name__}: {exc}"
                )
        row = {
            field: payload.get(field)
            for field in CSV_FIELDS
            if field not in ("episode_index", "status_path")
        }
        row["episode_index"] = index
        row["status_path"] = str(status_path.resolve())
        rows.append(row)
    rows.sort(key=lambda item: int(item["episode_index"]))
    return rows


def numeric_summary(values: Iterable[Any]) -> dict[str, float | int | None]:
    finite = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not finite:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    p95_index = max(0, math.ceil(0.95 * len(finite)) - 1)
    return {
        "count": len(finite),
        "min": finite[0],
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "p95": finite[p95_index],
        "max": finite[-1],
    }


def wilson_interval(successes: int, total: int) -> list[float | None]:
    if total <= 0:
        return [None, None]
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def build_summary(
    rows: list[dict[str, Any]],
    expected_episodes: int,
) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    errors = [row for row in rows if row["status"] != "completed"]
    successes = [row for row in completed if bool(row["success"])]
    failures = [row for row in completed if not bool(row["success"])]
    task_completed = [
        row
        for row in successes
        if bool(row["task_completed_success_and_bucket_open"])
    ]
    completed_total = len(completed)
    success_total = len(successes)
    return {
        "schema_version": "excavator_success_rate_summary_v1",
        "expected_episodes": int(expected_episodes),
        "status_files_found": len(rows),
        "completed_episodes": completed_total,
        "infrastructure_error_episodes": len(errors),
        "missing_episodes": max(0, expected_episodes - len(rows)),
        "successes": success_total,
        "failures": len(failures),
        "success_rate_over_completed_episodes": (
            success_total / completed_total
            if completed_total
            else None
        ),
        "success_rate_95pct_wilson_interval": wilson_interval(
            success_total,
            completed_total,
        ),
        "task_completed_success_and_bucket_open": len(task_completed),
        "early_terminated_episodes": sum(
            bool(row["early_terminated"]) for row in completed
        ),
        "successful_episode_timing_seconds": {
            field: numeric_summary(row[field] for row in successes)
            for field in (
                "success_time_from_policy_start_seconds",
                "success_time_from_simulation_launch_seconds",
                "task_completion_time_from_policy_start_seconds",
                "task_completion_time_from_simulation_launch_seconds",
            )
        },
        "successful_episode_indices": [
            int(row["episode_index"]) for row in successes
        ],
        "failed_episode_indices": [
            int(row["episode_index"]) for row in failures
        ],
        "error_episode_indices": [
            int(row["episode_index"]) for row in errors
        ],
    }


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "episode_results.csv"
    temporary_csv = csv_path.with_name(csv_path.name + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})
    temporary_csv.replace(csv_path)

    summary_path = output_dir / "success_rate_summary.json"
    temporary_summary = summary_path.with_name(summary_path.name + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.expected_episodes <= 0:
        raise SystemExit("--expected-episodes must be positive")
    batch_dir = Path(args.batch_dir).expanduser().resolve()
    rows = load_episode_rows(batch_dir / "episodes")
    summary = build_summary(rows, args.expected_episodes)
    write_outputs(batch_dir, rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.strict and (
        summary["completed_episodes"] != args.expected_episodes
        or summary["infrastructure_error_episodes"] != 0
    ):
        raise SystemExit(
            "Batch is incomplete: "
            f"completed={summary['completed_episodes']}, "
            f"errors={summary['infrastructure_error_episodes']}, "
            f"expected={args.expected_episodes}"
        )


if __name__ == "__main__":
    main()
