#!/usr/bin/env python3
"""Salvage valid multi-scoop prefixes from episodes with a failed final scoop."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path


INDEX_FILES = (
    "episodes.jsonl",
    "successful_episodes.jsonl",
    "trainable_episodes.jsonl",
    "rejected_episodes.jsonl",
    "failed_episodes.jsonl",
    "diagnostic_episodes.jsonl",
    "segment_dig.jsonl",
    "segment_dig_secure.jsonl",
    "segment_lift_carry.jsonl",
    "segment_unload.jsonl",
)
CAMERA_KEYS = tuple(f"observation.images.{index}" for index in range(3))
SALVAGE_WARNING = "quality_warning/offline_successful_prefix_salvage"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".salvage_tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".salvage_tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def contiguous_success_prefix(results: list[dict]) -> list[dict]:
    prefix = []
    for row in results:
        if not isinstance(row, dict) or not bool(row.get("success", False)):
            break
        prefix.append(row)
    return prefix


def fixed_scoop_task_text(count: int) -> str:
    return (
        f"Excavate and dump {count} consecutive successful scoops of sand from the "
        "visible sand pile into the visible truck bed without resetting the scene."
    )


def row_scoop_index(row: dict) -> int:
    try:
        return int(row.get("observation.scoop_index", -1))
    except Exception:
        return -1


def image_references(rows: list[dict]) -> set[str]:
    references = set()
    for row in rows:
        for key in CAMERA_KEYS:
            value = str(row.get(key, "") or "").replace("\\", "/").strip()
            if value:
                references.add(value)
    return references


def validate_kept_rows(episode_dir: Path, rows: list[dict], scoop_count: int) -> dict:
    if not rows:
        raise ValueError("successful prefix has no trajectory rows")
    indices = [row_scoop_index(row) for row in rows]
    expected_indices = set(range(scoop_count))
    if not expected_indices.issubset(set(indices)):
        raise ValueError(
            f"trajectory misses successful scoop indices: expected={sorted(expected_indices)} "
            f"actual={sorted(set(indices))}"
        )
    final_phase = str(rows[-1].get("phase", "") or "")
    if "dump" not in final_phase and "unload" not in final_phase:
        raise ValueError(f"successful prefix does not end after unload: phase={final_phase}")

    times = [float(row.get("t", 0.0) or 0.0) for row in rows]
    deltas = [
        times[index] - times[index - 1]
        for index in range(1, len(times))
        if times[index] > times[index - 1]
    ]
    median_dt = statistics.median(deltas) if deltas else 0.0
    if median_dt <= 0.0:
        raise ValueError("trajectory has no positive timestamp delta")
    max_dt_error = max(
        (abs(delta - median_dt) for delta in deltas),
        default=0.0,
    )
    if max_dt_error > max(0.025, median_dt * 0.25):
        raise ValueError(
            f"trajectory sampling is not uniform: median_dt={median_dt:.6f} "
            f"max_error={max_dt_error:.6f}"
        )

    missing = []
    reused = 0
    previous_paths = None
    for row in rows:
        paths = []
        for key in CAMERA_KEYS:
            relative = str(row.get(key, "") or "").replace("\\", "/").strip()
            path = episode_dir / relative
            if not relative or not path.is_file() or path.stat().st_size <= 0:
                missing.append((row.get("i"), key, relative))
            paths.append(relative)
        if previous_paths is not None and paths == previous_paths:
            reused += 1
        previous_paths = paths
    if missing:
        raise ValueError(f"successful prefix has missing camera files: {missing[:3]}")
    if reused:
        raise ValueError(f"successful prefix reuses camera triplets: rows={reused}")

    return {
        "sample_count": len(rows),
        "median_dt": float(median_dt),
        "expected_hz": float(1.0 / median_dt),
        "max_dt_error": float(max_dt_error),
        "final_phase": final_phase,
    }


def aggregate_prefix(successful_results: list[dict], kept_rows: list[dict]) -> tuple[dict, dict]:
    last_result = successful_results[-1]
    max_bucket = max(
        int(row.get("max_bucket_from_pile_particles", 0) or 0)
        for row in successful_results
    )
    lift_bucket = sum(
        int(row.get("lift_bucket_from_pile_particles", 0) or 0)
        for row in successful_results
    )
    final_bucket = int(last_result.get("final_bucket_from_pile_particles", 0) or 0)
    final_bin = int(last_result.get("bin_from_pile_end", 0) or 0)
    final_spill = sum(
        int(row.get("spill_from_lift_particles", 0) or 0)
        for row in successful_results
    )
    raw_region_spill = max(
        int(row.get("raw_region_spill_end", 0) or 0)
        for row in successful_results
    )
    max_joint_error = max(
        float(row.get("max_joint_error_deg", 0.0) or 0.0)
        for row in successful_results
    )
    max_action_speed = max(
        float(row.get("max_action_speed", 0.0) or 0.0)
        for row in successful_results
    )
    score_weight = sum(max(1, int(row.get("samples", 0) or 0)) for row in successful_results)
    score = sum(
        float(row.get("score", 0.0) or 0.0)
        * max(1, int(row.get("samples", 0) or 0))
        for row in successful_results
    ) / max(1, score_weight)

    phase_metrics = {}
    for result in successful_results:
        scoop_number = int(result.get("scoop_index", 0) or 0) + 1
        for label, value in (result.get("phase_metrics", {}) or {}).items():
            phase_metrics[f"scoop_{scoop_number}:{label}"] = value

    counts = {
        "max_bucket_from_pile_particles": max_bucket,
        "lift_bucket_from_pile_particles": lift_bucket,
        "final_bucket_from_pile_particles": final_bucket,
        "final_bin_from_pile_particles": final_bin,
        "final_spill_from_pile_particles": final_spill,
        "raw_region_spill_from_pile_particles": raw_region_spill,
        "freeze_count": 0,
        "samples": len(kept_rows),
    }
    spill_ratio = float(final_spill) / max(1.0, float(lift_bucket))
    components = {
        "per_scoop_weighted_mean": float(score / 100.0),
        "successful_scoop_count": len(successful_results),
        "spill_ratio": spill_ratio,
        "smoothness_source": "validated_per_scoop_quality_scores",
    }
    return counts, {
        "score": float(max(0.0, min(100.0, score))),
        "components": components,
        "phase_metrics": phase_metrics,
        "max_joint_error_deg": max_joint_error,
        "max_action_speed": max_action_speed,
    }


def final_metrics_from_prefix(
    original_metrics: dict,
    final_row: dict,
    counts: dict,
    aggregate: dict,
) -> dict:
    metrics = {}
    q_real = list(final_row.get("obs.q", []) or [])
    q_cmd = list(final_row.get("obs.q_cmd", []) or [])
    metrics["q_real_rad"] = q_real
    metrics["q_cmd_rad"] = q_cmd
    metrics["q_real_deg"] = [math.degrees(float(value)) for value in q_real]
    metrics["q_cmd_deg"] = [math.degrees(float(value)) for value in q_cmd]
    metrics["target_xyz"] = final_row.get("target")
    metrics["bucket_tip_xyz"] = final_row.get("bucket.tip")
    metrics["bucket_load_xyz"] = final_row.get("bucket.load")
    metrics["bucket_pour_xyz"] = final_row.get("bucket.pour")
    metrics["bucket_load_estimate"] = float(
        (final_row.get("sand", {}) or {}).get("bucket_from_pile", 0) or 0
    )
    sand = copy.deepcopy(final_row.get("sand", {}) or {})
    sand.update(
        {
            "bucket_from_pile": counts["final_bucket_from_pile_particles"],
            "bin_from_pile": counts["final_bin_from_pile_particles"],
            "spill_from_pile": counts["final_spill_from_pile_particles"],
            "raw_region_spill_from_pile": counts[
                "raw_region_spill_from_pile_particles"
            ],
        }
    )
    metrics["sand"] = sand
    metrics.update(counts)
    metrics["max_joint_error_deg"] = aggregate["max_joint_error_deg"]
    metrics["max_action_speed"] = aggregate["max_action_speed"]
    effort_rows = [
        row
        for row in [final_row]
        if isinstance(row.get("observation.effort"), list)
    ]
    metrics["effort_available"] = bool(effort_rows)
    metrics["effort_sample_count"] = counts["samples"]
    metrics["effort_missing_samples"] = 0
    try:
        metrics["max_abs_effort"] = max(
            abs(float(value))
            for value in (final_row.get("observation.effort", []) or [])
        )
    except Exception:
        metrics["max_abs_effort"] = 0.0

    unload_drop = None
    for label, row in aggregate["phase_metrics"].items():
        if "after_dump_settle" in label and isinstance(row, dict):
            unload_drop = row.get("unload_drop")
    if unload_drop is not None:
        metrics["unload_drop"] = unload_drop
    metrics["salvaged_from_original_final_metrics"] = {
        "samples": int((original_metrics or {}).get("samples", 0) or 0),
        "final_bin_from_pile_particles": int(
            (original_metrics or {}).get("final_bin_from_pile_particles", 0) or 0
        ),
    }
    return metrics


def build_score_report(
    original_score: dict,
    counts: dict,
    aggregate: dict,
    final_metrics: dict,
) -> dict:
    thresholds = copy.deepcopy((original_score or {}).get("thresholds", {}) or {})
    return {
        "success": True,
        "execution_success": True,
        "raw_execution_success": True,
        "best_effort_bin_success": False,
        "score": aggregate["score"],
        "failure_reason": "",
        "warning_reason": SALVAGE_WARNING,
        "quality_warnings": [SALVAGE_WARNING],
        "raw_reason": "offline_successful_prefix_salvage",
        "components": aggregate["components"],
        "thresholds": thresholds,
        "counts": counts,
        "phase_metrics": aggregate["phase_metrics"],
        "final_metrics": final_metrics,
        "salvage": {
            "schema": "multi_scoop_successful_prefix_salvage_v1",
            "created_at": time.time(),
        },
    }


def update_meta(
    meta: dict,
    score: dict,
    successful_results: list[dict],
    kept_rows: list[dict],
    audit: dict,
) -> dict:
    result = copy.deepcopy(meta)
    count = len(successful_results)
    task = fixed_scoop_task_text(count)
    first_row = kept_rows[0]
    last_row = kept_rows[-1]
    result.update(
        {
            "status": "trainable",
            "success": True,
            "execution_success": True,
            "failure_reason": "",
            "warning_reason": SALVAGE_WARNING,
            "quality_warnings": [SALVAGE_WARNING],
            "task": task,
            "scoops_target": count,
            "scoops_min": count,
            "scoops_max": count,
            "scoops_completed": count,
            "adaptive_scoop_stop": False,
            "scoop_stop_reason": "completed_fixed_scoop_count_after_offline_prefix_salvage",
            "scoop_stop_natural": False,
            "scoop_stop_detail": {
                "offline_salvage": True,
                "successful_scoop_count": count,
            },
            "all_scoops_success": True,
            "scoop_results": successful_results,
            "score_summary": {
                "score": score["score"],
                "success": True,
                "failure_reason": "",
                "warning_reason": SALVAGE_WARNING,
                "quality_warnings": [SALVAGE_WARNING],
                "components": score["components"],
            },
            "final_counts": score["counts"],
            "final_metrics": score["final_metrics"],
            "duration_simulation_s": max(
                0.0,
                float(last_row.get("t", 0.0) or 0.0)
                - float(first_row.get("t", 0.0) or 0.0),
            ),
            "duration_wall_s": max(
                0.0,
                float(last_row.get("timestamp.wall", 0.0) or 0.0)
                - float(first_row.get("timestamp.wall", 0.0) or 0.0),
            ),
            "finished_at": float(last_row.get("timestamp.wall", time.time()) or time.time()),
            "finished_at_simulation": float(
                last_row.get("timestamp.simulation", 0.0) or 0.0
            ),
            "source_sampling_audit": {
                "ok": True,
                "expected_hz": audit["expected_hz"],
                "expected_dt_s": audit["median_dt"],
                "sample_count": audit["sample_count"],
                "min_dt_s": audit["median_dt"] - audit["max_dt_error"],
                "max_dt_s": audit["median_dt"] + audit["max_dt_error"],
                "median_dt_s": audit["median_dt"],
                "missing_camera_rows": 0,
                "reused_camera_rows": 0,
                "non_physical_timestamp_rows": 0,
                "non_monotonic_capture_rows": 0,
                "reason": "ok",
            },
            "salvage": {
                "schema": "multi_scoop_successful_prefix_salvage_v1",
                "created_at": time.time(),
                "original_status": meta.get("status"),
                "original_failure_reason": meta.get("failure_reason", ""),
                "original_scoop_result_count": len(meta.get("scoop_results", []) or []),
                "kept_successful_scoops": count,
                "kept_samples": len(kept_rows),
                "task_rewritten_as_fixed_scoop_count": True,
            },
        }
    )
    camera_summary = copy.deepcopy(result.get("camera_episode_summary", {}) or {})
    camera_summary["samples"] = len(kept_rows)
    camera_summary["image_counts"] = {
        str(index): len(kept_rows) for index in range(3)
    }
    camera_summary["counts_match_samples"] = True
    result["camera_episode_summary"] = camera_summary
    return result


def update_trajectory_rows(rows: list[dict], scoop_count: int) -> list[dict]:
    task = fixed_scoop_task_text(scoop_count)
    updated = []
    for new_index, source in enumerate(rows):
        row = copy.deepcopy(source)
        row["i"] = new_index
        row["task"] = task
        row["observation.scoops_target"] = scoop_count
        row["observation.scoops_min"] = scoop_count
        row["observation.scoops_max"] = scoop_count
        updated.append(row)
    return updated


def event_cutoff_rows(events: list[dict], last_scoop_index: int, meta: dict) -> list[dict]:
    cutoff = None
    token = f"scoop_index={last_scoop_index}; success=True"
    for index, row in enumerate(events):
        if str(row.get("event", "")) == "scoop_end" and token in str(
            row.get("detail", "")
        ):
            cutoff = index
    if cutoff is None:
        raise ValueError(f"missing successful scoop_end event for scoop {last_scoop_index}")
    kept = copy.deepcopy(events[: cutoff + 1])
    now = time.time()
    common = {
        "episode_index": meta.get("episode_index"),
        "episode_id": meta.get("episode_id"),
        "timestamp": now,
        "active_task": "offline_multi_scoop_prefix_salvage",
        "schema": meta.get("schema", "excavator_auto_multi_scoop_v2.0"),
        "dataset_release": meta.get("dataset_release", "v2.0"),
        "episode_mode": "multi_scoop",
        "recovery_mode": meta.get("recovery_mode", "expert_only"),
        "scoops_per_episode": last_scoop_index + 1,
        "scoops_min_per_episode": last_scoop_index + 1,
        "scoops_max_per_episode": last_scoop_index + 1,
        "adaptive_scoop_stop": False,
        "state_schema_version": meta.get("state_schema_version", ""),
    }
    kept.append(
        {
            **common,
            "event": "offline_prefix_salvage",
            "detail": (
                f"kept_successful_scoops={last_scoop_index + 1}; "
                "removed_failed_extension=True"
            ),
        }
    )
    kept.append(
        {
            **common,
            "event": "episode_end",
            "detail": (
                "status=trainable; success=True; execution_success=True; "
                "reason=offline_successful_prefix_salvage"
            ),
        }
    )
    return kept


def backup_episode_files(
    episode_dir: Path,
    run_dir: Path,
    source_files: list[Path],
) -> Path:
    backup_dir = (
        run_dir
        / ".multi_scoop_salvage_backup"
        / episode_dir.name
    )
    if backup_dir.exists():
        raise ValueError(f"backup already exists: {backup_dir}")
    originals = backup_dir / "original"
    originals.mkdir(parents=True)
    for path in source_files:
        if path.exists():
            shutil.copy2(path, originals / path.name)
    return backup_dir


def move_removed_images(
    episode_dir: Path,
    backup_dir: Path,
    kept_references: set[str],
    removed_references: set[str],
) -> int:
    moved = 0
    for relative in sorted(removed_references - kept_references):
        source = episode_dir / relative
        if not source.is_file():
            continue
        destination = backup_dir / "tail_images" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved += 1
    return moved


def updated_index_row(row: dict, meta: dict, score: dict) -> dict:
    result = copy.deepcopy(row)
    counts = score["counts"]
    result.update(
        {
            "success": True,
            "full_chain_success": True,
            "status": "trainable",
            "score": score["score"],
            "reason": "",
            "warning_reason": SALVAGE_WARNING,
            "quality_warnings": [SALVAGE_WARNING],
            "scoops_target": meta["scoops_target"],
            "scoops_min": meta["scoops_min"],
            "scoops_max": meta["scoops_max"],
            "scoops_completed": meta["scoops_completed"],
            "adaptive_scoop_stop": False,
            "scoop_stop_reason": meta["scoop_stop_reason"],
            "scoop_stop_natural": False,
            "all_scoops_success": True,
            "scoop_results": meta["scoop_results"],
            "duration_wall_s": meta["duration_wall_s"],
            "duration_simulation_s": meta["duration_simulation_s"],
            "samples": counts["samples"],
            "freeze_count": counts["freeze_count"],
            "max_bucket_from_pile_particles": counts[
                "max_bucket_from_pile_particles"
            ],
            "lift_bucket_from_pile_particles": counts[
                "lift_bucket_from_pile_particles"
            ],
            "final_bucket_from_pile_particles": counts[
                "final_bucket_from_pile_particles"
            ],
            "final_bin_from_pile_particles": counts[
                "final_bin_from_pile_particles"
            ],
            "final_spill_from_pile_particles": counts[
                "final_spill_from_pile_particles"
            ],
            "raw_region_spill_from_pile_particles": counts[
                "raw_region_spill_from_pile_particles"
            ],
            "salvage": meta["salvage"],
        }
    )
    return result


def rebuild_run_indexes(run_dir: Path, changed: dict[str, tuple[dict, dict]]) -> None:
    backup_root = run_dir / ".multi_scoop_salvage_backup" / "_run_indexes_original"
    backup_root.mkdir(parents=True, exist_ok=True)
    originals = {}
    for filename in INDEX_FILES:
        path = run_dir / filename
        rows = read_jsonl(path)
        originals[filename] = rows
        if path.exists() and not (backup_root / filename).exists():
            shutil.copy2(path, backup_root / filename)

    episode_rows = originals["episodes.jsonl"]
    new_episode_rows = []
    changed_rows = {}
    for row in episode_rows:
        episode_id = str(row.get("episode_id", "") or "")
        if episode_id in changed:
            meta, score = changed[episode_id]
            row = updated_index_row(row, meta, score)
            changed_rows[episode_id] = row
        new_episode_rows.append(row)
    missing = set(changed) - set(changed_rows)
    if missing:
        raise ValueError(f"episodes.jsonl misses repaired episodes: {sorted(missing)}")
    atomic_write_jsonl(run_dir / "episodes.jsonl", new_episode_rows)

    for filename in (
        "rejected_episodes.jsonl",
        "failed_episodes.jsonl",
        "diagnostic_episodes.jsonl",
    ):
        rows = [
            row
            for row in originals[filename]
            if str(row.get("episode_id", "") or "") not in changed
        ]
        atomic_write_jsonl(run_dir / filename, rows)

    for filename in ("successful_episodes.jsonl", "trainable_episodes.jsonl"):
        rows = [
            row
            for row in originals[filename]
            if str(row.get("episode_id", "") or "") not in changed
        ]
        rows.extend(changed_rows[episode_id] for episode_id in sorted(changed_rows))
        rows.sort(key=lambda row: int(row.get("episode_index", 0) or 0))
        atomic_write_jsonl(run_dir / filename, rows)

    for filename in (
        "segment_dig.jsonl",
        "segment_dig_secure.jsonl",
        "segment_lift_carry.jsonl",
        "segment_unload.jsonl",
    ):
        rows = []
        for row in originals[filename]:
            episode_id = str(row.get("episode_id", "") or "")
            if episode_id in changed_rows:
                row = copy.deepcopy(row)
                row["full_chain_success"] = True
                row["episode_status"] = "trainable"
                row["episode_reason"] = ""
                row["score"] = changed_rows[episode_id]["score"]
                row["salvage"] = changed_rows[episode_id]["salvage"]
            rows.append(row)
        atomic_write_jsonl(run_dir / filename, rows)

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        if not (backup_root / "summary.json").exists():
            shutil.copy2(summary_path, backup_root / "summary.json")
        summary = read_json(summary_path)
        statuses = {}
        for row in new_episode_rows:
            status = str(row.get("status", "") or "")
            statuses[status] = statuses.get(status, 0) + 1
        summary["updated_at"] = time.time()
        summary["episodes_index_rows"] = len(new_episode_rows)
        summary["successes"] = statuses.get("trainable", 0)
        summary["trainable"] = statuses.get("trainable", 0)
        summary["rejections"] = statuses.get("rejected", 0)
        summary["failures"] = statuses.get("failed", 0)
        summary["diagnostic"] = statuses.get("diagnostic", 0)
        summary["offline_prefix_salvage"] = {
            "schema": "multi_scoop_successful_prefix_salvage_v1",
            "salvaged_episodes": len(changed),
            "updated_at": time.time(),
        }
        atomic_write_json(summary_path, summary)


def salvage_episode(episode_dir: Path, minimum_scoops: int, apply: bool) -> dict:
    meta_path = episode_dir / "meta.json"
    score_path = episode_dir / "score.json"
    trajectory_path = episode_dir / "trajectory.jsonl"
    events_path = episode_dir / "events.jsonl"
    sand_path = episode_dir / "sand_metrics.jsonl"
    meta = read_json(meta_path)
    if str(meta.get("episode_mode", "")) != "multi_scoop":
        return {"eligible": False, "reason": "not_multi_scoop"}
    if str(meta.get("status", "")) == "trainable":
        return {"eligible": False, "reason": "already_trainable"}
    if isinstance(meta.get("salvage"), dict):
        return {"eligible": False, "reason": "already_salvaged"}

    results = list(meta.get("scoop_results", []) or [])
    prefix = contiguous_success_prefix(results)
    required = max(
        int(minimum_scoops),
        int(meta.get("scoops_min", meta.get("scoops_target", 1)) or 1),
    )
    if len(prefix) < required:
        return {
            "eligible": False,
            "reason": "insufficient_successful_prefix",
            "successful_prefix": len(prefix),
            "required": required,
        }
    if len(prefix) >= len(results):
        return {"eligible": False, "reason": "no_failed_extension"}
    if any(bool(row.get("success", False)) for row in results[len(prefix) :]):
        return {"eligible": False, "reason": "non_contiguous_success_after_failure"}

    rows = read_jsonl(trajectory_path)
    kept = [row for row in rows if row_scoop_index(row) < len(prefix)]
    removed = [row for row in rows if row_scoop_index(row) >= len(prefix)]
    if not removed:
        return {"eligible": False, "reason": "no_failed_trajectory_tail"}
    audit = validate_kept_rows(episode_dir, kept, len(prefix))
    successful_task_rows = update_trajectory_rows(kept, len(prefix))
    counts, aggregate = aggregate_prefix(prefix, kept)
    original_score = read_json(score_path)
    final_metrics = final_metrics_from_prefix(
        original_score.get("final_metrics", {}),
        successful_task_rows[-1],
        counts,
        aggregate,
    )
    score = build_score_report(original_score, counts, aggregate, final_metrics)
    new_meta = update_meta(meta, score, prefix, successful_task_rows, audit)
    events = read_jsonl(events_path)
    new_events = event_cutoff_rows(events, len(prefix) - 1, new_meta)
    sand_rows = read_jsonl(sand_path)
    kept_sand = [
        row
        for row in sand_rows
        if int(row.get("scoop_index", -1) or -1) < len(prefix)
    ]
    if len(kept_sand) != len(successful_task_rows):
        raise ValueError(
            f"sand/trajectory count mismatch after cutoff: "
            f"sand={len(kept_sand)} trajectory={len(successful_task_rows)}"
        )

    result = {
        "eligible": True,
        "episode": str(episode_dir),
        "episode_id": str(meta.get("episode_id", "") or ""),
        "original_status": meta.get("status"),
        "successful_scoops": len(prefix),
        "kept_samples": len(successful_task_rows),
        "removed_samples": len(removed),
        "score": score["score"],
        "final_bin": counts["final_bin_from_pile_particles"],
        "apply": bool(apply),
    }
    if not apply:
        return result

    run_dir = episode_dir.parent
    source_files = [
        meta_path,
        score_path,
        trajectory_path,
        events_path,
        sand_path,
    ]
    backup_dir = backup_episode_files(episode_dir, run_dir, source_files)
    atomic_write_jsonl(trajectory_path, successful_task_rows)
    atomic_write_jsonl(events_path, new_events)
    atomic_write_jsonl(sand_path, kept_sand)
    atomic_write_json(score_path, score)
    atomic_write_json(meta_path, new_meta)
    moved_images = move_removed_images(
        episode_dir,
        backup_dir,
        image_references(successful_task_rows),
        image_references(removed),
    )
    manifest = dict(result)
    manifest["moved_tail_images"] = moved_images
    manifest["backup_dir"] = str(backup_dir)
    atomic_write_json(backup_dir / "manifest.json", manifest)
    result["moved_tail_images"] = moved_images
    result["backup_dir"] = str(backup_dir)
    result["_meta"] = new_meta
    result["_score"] = score
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing run_* directories.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=[],
        help="Run directory glob. Repeatable; default: run_*.",
    )
    parser.add_argument(
        "--min-successful-scoops",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply repairs. Without this flag the command is a dry run.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    patterns = args.run_glob or ["run_*"]
    runs = sorted(
        {
            path.resolve()
            for pattern in patterns
            for path in root.glob(pattern)
            if path.is_dir()
        }
    )
    report = {
        "schema": "multi_scoop_successful_prefix_salvage_report_v1",
        "root": str(root),
        "apply": bool(args.apply),
        "runs_scanned": len(runs),
        "episodes": [],
        "errors": [],
    }
    for run_dir in runs:
        changed = {}
        for episode_dir in sorted(run_dir.glob("episode_*")):
            if not episode_dir.is_dir():
                continue
            try:
                row = salvage_episode(
                    episode_dir,
                    minimum_scoops=max(1, args.min_successful_scoops),
                    apply=bool(args.apply),
                )
            except Exception as exc:
                report["errors"].append(
                    {
                        "episode": str(episode_dir),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if row.get("eligible"):
                report["episodes"].append(
                    {
                        key: value
                        for key, value in row.items()
                        if not key.startswith("_")
                    }
                )
                if args.apply:
                    changed[row["episode_id"]] = (row["_meta"], row["_score"])
        if args.apply and changed:
            rebuild_run_indexes(run_dir, changed)

    report["eligible_count"] = len(report["episodes"])
    report["error_count"] = len(report["errors"])
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.report is not None:
        report_path = args.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(report_path, report)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
