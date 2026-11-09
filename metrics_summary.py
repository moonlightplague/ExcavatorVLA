"""Combine metrics for exactly one collect_repeat.sh batch, across retry rounds."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import fmean

try:
    from baseline.stream_export import read_jsonl, write_json
except ModuleNotFoundError:  # Direct script invocation from the shell launcher.
    from stream_export import read_jsonl, write_json


FIELDS = (
    "payload_mass_kg", "payload_particles", "runtime_seconds", "attempt_runtime_seconds",
    "simulated_runtime_seconds", "cycle_time_seconds", "cycle_time_wall_seconds",
    "spillage_mass_kg", "spillage_particles", "spillage_fraction", "scooped_mass_kg",
    "scooped_particles", "retained_mass_kg", "retained_particles",
    "payload_particles_per_sim_second", "scoops_completed", "returns_completed", "samples",
)
NESTED = {
    "score": ("score_summary", "score"),
    "bed_occupied_fraction": ("bed_fill", "occupied_fraction"),
    "bed_surface_std_m": ("bed_fill", "surface_std_m"),
    "bed_surface_range_m": ("bed_fill", "surface_range_m"),
    "bed_mean_depth_m": ("bed_fill", "mean_depth_m"),
}
TOTAL_FIELDS = tuple(k for k in FIELDS if k not in (
    "spillage_fraction", "payload_particles_per_sim_second", "cycle_time_seconds", "cycle_time_wall_seconds"))


def average_columns(rows, fields):
    values = {key: [row.get(key) for row in rows if row.get(key) is not None] for key in fields}
    for key, column in values.items():
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in column):
            raise ValueError(f"Invalid numeric metric: {key}")
    return ({key: fmean(v) if v else None for key, v in values.items()},
            {key: len(v) for key, v in values.items()})


def summarize(rows):
    flat = [{**row, **{key: row.get(path[0], {}).get(path[1]) for key, path in NESTED.items()}} for row in rows]
    averages, counts = average_columns(flat, (*FIELDS, *NESTED))
    totals = {}
    for key in TOTAL_FIELDS:
        values = [row[key] for row in rows if row.get(key) is not None]
        totals[key] = math.fsum(values) if values else None
    def pooled(numerator, denominator):
        paired = [r for r in rows if r.get(numerator) is not None and r.get(denominator) is not None]
        divisor = math.fsum(r[denominator] for r in paired)
        return math.fsum(r[numerator] for r in paired) / divisor if divisor else None
    cycles = [cycle for row in rows for cycle in row.get("scoop_cycles", [])]
    completed = [cycle for cycle in cycles if cycle.get("success")]
    scoop_averages, scoop_counts = average_columns(completed, (
        "payload_particles", "scooped_particles", "retained_particles", "spillage_particles",
        "spillage_fraction", "runtime_seconds", "simulated_runtime_seconds",
        "cycle_time_seconds", "cycle_time_wall_seconds"))
    return {
        "episode_count": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "available_metrics_episode_count": sum(bool(row.get("available")) for row in rows),
        "averages": averages, "non_null_episode_counts": counts, "totals": totals,
        "pooled_spillage_fraction": pooled("spillage_particles", "scooped_particles"),
        "pooled_payload_particles_per_sim_second": pooled("payload_particles", "simulated_runtime_seconds"),
        "scoop_cycles": {"attempted_count": len(cycles), "completed_count": len(completed),
                         "failed_count": len(cycles) - len(completed),
                         "completed_scoop_averages": scoop_averages,
                         "non_null_completed_scoop_counts": scoop_counts},
    }


def combine_batch(raw_root, batch_id, export_dir, requested, status):
    pattern = re.compile(r"run_.+_" + re.escape(batch_id) + r"_r\d+_w\d+")
    runs = sorted(path for path in Path(raw_root).glob(f"run_*_{batch_id}_r*_w*")
                  if path.is_dir() and pattern.fullmatch(path.name))
    records, sources, receipts = [], [], []
    for run in runs:
        metrics_path = run / "repeat_metrics.jsonl"
        rows = read_jsonl(metrics_path)
        if len({row["episode_id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate episode metric rows in {run}")
        if any(row.get("metrics_schema_version") != "repeat_episode_metrics_v1" for row in rows):
            raise ValueError(f"Unsupported metric schema in {run}")
        run_summary_path = run / "summary.json"
        run_summary = json.loads(run_summary_path.read_text()) if run_summary_path.exists() else {}
        exported = [json.loads(path.read_text()) for path in sorted(run.glob("episode_*/export.json"))]
        if any(not r.get("validated") or r.get("export_dir") != str(Path(export_dir).resolve()) for r in exported):
            raise ValueError(f"Invalid export receipt in {run}")
        receipts.extend(exported)
        sources.append({"run_id": run.name, "metrics_path": str(metrics_path.resolve()),
                        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest() if metrics_path.exists() else None,
                        "recorded_episode_count": len(rows), "exported_episode_count": len(exported),
                        "collection_attempt_count": run_summary.get("attempts", 0),
                        "planning_diagnostic_count": run_summary.get("planning_diagnostics", 0)})
        records.extend(rows)
    trainable = [row for row in records if row["status"] == "trainable"]
    attempts = sum(s["collection_attempt_count"] for s in sources)
    completed = (status == "complete" and len(trainable) == requested
                 and len(receipts) == requested and all(r.get("raw_pruned") for r in receipts))
    if status == "complete" and not completed:
        raise ValueError("Completed batch has missing metrics, exports, or cleanup receipts")
    combined = {"all_recorded_episodes": summarize(records), "trainable_episodes": summarize(trainable)}
    for name in ("rejected", "failed", "diagnostic"):
        subset = [row for row in records if row["status"] == name]
        if subset:
            combined[name + "_episodes"] = summarize(subset)
    return {
        "schema_version": "combined_repeat_averages_v1", "batch_id": batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "completed": completed, "requested_successful_episodes": requested,
        "export_dir": str(Path(export_dir).resolve()), "exported_episode_count": len(receipts),
        "collection_attempt_count": attempts,
        "acceptance_rate": len(trainable) / attempts if attempts else None,
        "export_acceptance_rate": len(receipts) / attempts if attempts else None,
        "planning_diagnostic_count": sum(s["planning_diagnostic_count"] for s in sources),
        "source_files": sources, "combined": combined,
        "source_metric_definitions": records[0].get("definitions", {}) if records else {},
        "aggregation_notes": {
            "scope": "This batch only, across every worker and retry round; pre-existing exported episodes are excluded.",
            "averages": "Arithmetic mean of individual non-null values, with denominators in non_null_episode_counts; missing values are never replaced by zero. Pooled ratios use rows with both numerator and denominator present.",
            "spillage": "Fractions use the 0-to-1 scale. Mean episode spillage_fraction and pooled total spillage / total scooped are separate metrics.",
            "cycle_times": "Episode cycle times are null on failures; runtime_seconds and simulated_runtime_seconds still include incomplete episodes.",
            "totals": "Totals sum episode measurements; summed wall runtimes across parallel workers are not elapsed batch time.",
            "coverage": "Planning-only attempts have no episode metrics and are excluded, including from average attempt_runtime_seconds.",
            "scoop_averages": "Pool individual completed scoop cycles, including completed scoops in rejected episodes; failed attempts are counted separately.",
            "extra_metric_field_paths": {key: '.'.join(path) for key, path in NESTED.items()},
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--status", choices=("complete", "failed", "interrupted"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine_batch(args.raw_root, args.batch_id, args.export_dir, args.requested, args.status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(f"Combined metrics and averages: {args.output}", flush=True)
    print(f"Acceptance: {result['exported_episode_count']}/{result['collection_attempt_count']} attempts "
          f"exported ({result['export_acceptance_rate']:.1%})" if result['collection_attempt_count'] else
          "Acceptance: no collection attempts completed", flush=True)


if __name__ == "__main__":
    main()
