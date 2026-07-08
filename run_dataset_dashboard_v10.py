import json
import io
import math
import os
import re
import shutil
import struct
import time
import threading
import hashlib
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qs, unquote, urlparse

try:
    import excavator_dataset_tools as shared_dataset_tools
except Exception:
    shared_dataset_tools = None


INDEX_FILES = {
    "all": "episodes.jsonl",
    "trainable": "trainable_episodes.jsonl",
    "success": "successful_episodes.jsonl",
    "rejected": "rejected_episodes.jsonl",
    "failed": "failed_episodes.jsonl",
    "diagnostic": "diagnostic_episodes.jsonl",
    "planning": "planning_diagnostics.jsonl",
}

SEGMENT_FILES = {
    "dig": "segment_dig.jsonl",
    "dig_secure": "segment_dig_secure.jsonl",
    "lift_carry": "segment_lift_carry.jsonl",
    "unload": "segment_unload.jsonl",
}

LEROBOT_EXPORT_SCHEMA = "excavator_lerobot_export_v3"
LEROBOT_CODEBASE_VERSION = "v3.0"
LEROBOT_DEFAULT_EXPORT_DIRNAME = "lerobot_v3"
EXPORT_TIME_POLICY_FILENAME = "export_time_policy.json"
LEROBOT_IMAGE_SHAPE = [256, 256, 3]
LEROBOT_IMAGE_KEYS = [
    "observation.images.0",
    "observation.images.1",
    "observation.images.2",
]
CAMERA_PREVIEW_MEANINGFUL_PHASE_PREFIXES = (
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "pull_exit_cut",
    "curl_to_hold_material",
    "secure_load",
    "lift_carry",
    "unload_to_bin",
)
CAMERA_PREVIEW_SKIP_INITIAL_PHASE_PREFIXES = (
    "clearance_route",
    "pre_dig_trace",
)
SUCCESS_POOL_DIRNAME = ".dashboard_success"
LEROBOT_IMAGE_KEY_ALIASES = {
    "observation.images.0": ["observation.images.0", "observation.images.camera", "observation.images.front"],
    "observation.images.1": ["observation.images.1", "observation.images.cameraleft", "observation.images.bucket"],
    "observation.images.2": ["observation.images.2", "observation.images.cameraright", "observation.images.side"],
}


def read_json(path: Union[str, os.PathLike], default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(path: Union[str, os.PathLike]) -> List[dict]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def read_jsonl_limited(path: Union[str, os.PathLike], limit: int = 10) -> List[dict]:
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if len(rows) >= int(limit):
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def latest_run(dataset_root: Union[str, os.PathLike]) -> str:
    if not os.path.isdir(dataset_root):
        return ""
    runs = [
        os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name)) and name.startswith("run_")
    ]
    if not runs:
        return ""
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return runs[0]


def index_path(run_dir: Union[str, os.PathLike], split: str = "trainable") -> str:
    return os.path.join(str(run_dir), INDEX_FILES.get(split, split))


def load_index(run_dir: Union[str, os.PathLike], split: str = "trainable") -> List[dict]:
    return read_jsonl(index_path(run_dir, split))


def iter_episodes(run_dir: Union[str, os.PathLike], split: str = "trainable") -> Iterable[dict]:
    for row in load_index(run_dir, split):
        yield row


def load_trajectory(row_or_path: Union[dict, str, os.PathLike]) -> List[dict]:
    if isinstance(row_or_path, dict):
        episode_dir = episode_dir_from_row(row_or_path)
        path = resolve_episode_file(episode_dir, row_or_path.get("trajectory", ""))
    else:
        path = str(row_or_path)
    return read_jsonl(path)


def load_episode_bundle(row: dict) -> Dict[str, object]:
    return {
        "index": row,
        "meta": read_json(row.get("meta", ""), default={}),
        "score": read_json(row.get("score_path", ""), default={}),
        "trajectory": load_trajectory(row),
    }


def summarize_run(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    run_dir = str(run_dir)
    raw_counts = {name: len(load_index(run_dir, name)) for name in INDEX_FILES}
    counts, catchup = success_index_catchup_counts(run_dir, raw_counts)
    segments = {
        name: len(read_jsonl(os.path.join(run_dir, filename)))
        for name, filename in SEGMENT_FILES.items()
    }
    attempts = counts["all"]
    trainable = counts["trainable"]
    rejected = counts["rejected"]
    failed = counts["failed"]
    success_rate = float(trainable) / float(attempts) if attempts > 0 else 0.0
    rejection_rate = float(rejected) / float(attempts) if attempts > 0 else 0.0
    failure_rate = float(failed) / float(attempts) if attempts > 0 else 0.0
    return {
        "run_dir": run_dir,
        "run_meta": read_json(os.path.join(run_dir, "run_meta.json"), default={}),
        "summary": read_json(os.path.join(run_dir, "summary.json"), default={}),
        "counts": counts,
        "segments": segments,
        "success_catchup": catchup,
        "success_rate": success_rate,
        "rejection_rate": rejection_rate,
        "failure_rate": failure_rate,
    }


def split_reason(reason: object) -> List[str]:
    text = str(reason or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";") if part.strip()]


def classify_reason(reason: object) -> str:
    text = str(reason or "").lower()
    if not text:
        return "ok"
    if "path_deviation:approach_contact" in text:
        return "execution/path_deviation/approach_contact"
    if "freeze_detected" in text:
        return "execution/freeze_detected"
    if "path_precheck_failed" in text:
        return "execution/path_precheck_failed"
    if "initial_pose_failed" in text:
        return "execution/initial_pose_failed"
    if "secure_not_retaining_material" in text:
        return "quality/secure_not_retaining_material"
    if "low_final_bin_particles" in text:
        return "quality/low_final_bin_particles"
    if "execution_failed/" in text:
        first = text.split(";", 1)[0].split(":", 1)[0].strip()
        return first.replace("execution_failed/", "execution/")[:96]
    if "quality_rejected/" in text:
        first = text.split(";", 1)[0].split(":", 1)[0].strip()
        return first.replace("quality_rejected/", "quality/")[:96]
    if "planning_failed" in text:
        return "planning/failed"
    if "preflight_failed" in text or "prepare_failed" in text:
        return "prepare/failed"
    if "score_low" in text:
        return "quality/score_low"
    return text.split(";", 1)[0].split(":", 1)[0][:96]


def normalized_status_name(status: object) -> str:
    text = str(status or "unknown").strip().lower()
    if text in {"fail", "failure", "failed"}:
        return "failed"
    if text in {"successful", "success"}:
        return "success"
    if not text:
        return "unknown"
    return text


def is_problem_status(status: object) -> bool:
    return normalized_status_name(status) in {"rejected", "failed", "diagnostic", "planning"}


def normalize_failure_reason_for_triage(reason: object, status: object = None) -> Optional[str]:
    """Return one stable blocker key for failure triage.

    Successful/trainable rows with an empty reason should not become the top failure.
    Empty reasons on rejected/failed rows are still counted as unclassified blockers.
    """
    text = str(reason or "").strip()
    status_key = normalized_status_name(status)
    if not text:
        if is_problem_status(status_key):
            return f"{status_key}/unclassified"
        return None
    key = classify_reason(text).strip()
    if not key or key == "ok":
        return f"{status_key}/unclassified" if is_problem_status(status_key) else None
    return key


def normalize_warning_fragment(fragment: object) -> Optional[str]:
    text = str(fragment or "").strip()
    if not text:
        return None
    head, sep, tail = text.partition(":")
    head = head.strip()
    tail = tail.strip()
    # Numeric suffixes such as score_low:35.0 or high_spill_ratio:0.99 are values,
    # not categories. Group them so they do not create one noisy warning per score.
    if sep and re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", tail):
        text = head
    else:
        text = head if head.startswith("quality_warning/") else text
    if text.startswith("quality_warning/"):
        text = "quality/" + text.split("/", 1)[1]
    return text[:120]


def is_quality_signal_key(key: object) -> bool:
    text = str(key or "").lower()
    return text.startswith("quality/") or "score_low" in text or "spill" in text or "low_final_bin" in text


def numeric_stats(values: Sequence[object]) -> Dict[str, object]:
    nums = []
    for value in values:
        try:
            if value is not None:
                nums.append(float(value))
        except Exception:
            pass
    if not nums:
        return {"count": 0}
    nums_sorted = sorted(nums)
    return {
        "count": len(nums_sorted),
        "min": nums_sorted[0],
        "median": median(nums_sorted),
        "mean": mean(nums_sorted),
        "max": nums_sorted[-1],
    }


def top_counter(counter: Counter, limit: int = 10) -> List[Dict[str, object]]:
    return [{"key": key, "count": count} for key, count in counter.most_common(limit)]


def compact_episode(row: dict) -> Dict[str, object]:
    return {
        "episode": row.get("episode_index"),
        "id": row.get("episode_id"),
        "status": row.get("status"),
        "score": row.get("score"),
        "initial_pose": row.get("initial_pose_id"),
        "reason": row.get("reason", ""),
        "warnings": row.get("warning_reason", ""),
        "max_bucket": row.get("max_bucket_from_pile_particles"),
        "lift_bucket": row.get("lift_bucket_from_pile_particles"),
        "bin": row.get("final_bin_from_pile_particles"),
        "spill": row.get("final_spill_from_pile_particles"),
        "freeze_count": row.get("freeze_count"),
    }


def analyze_events_for_rows(rows: Sequence[dict], limit_rows: Optional[int] = None) -> Dict[str, object]:
    event_counter = Counter()
    failure_counter = Counter()
    failed_stage_counter = Counter()
    freeze_details = Counter()
    stage_durations = defaultdict(list)
    scanned = 0
    for row in rows[:limit_rows] if limit_rows is not None else rows:
        path = row.get("events", "")
        if not path:
            continue
        scanned += 1
        for event in read_jsonl(path):
            name = str(event.get("event", ""))
            detail = str(event.get("detail", ""))
            event_counter[name] += 1
            if name == "execution_failure":
                failure_counter[detail.split(";", 1)[0]] += 1
            if name == "freeze":
                m = re.search(r"bucket_below_ground=([-0-9.]+)", detail)
                if m:
                    freeze_details[f"bucket_below_ground={m.group(1)}"] += 1
                blocked = re.search(r"blocked_joints=\[([^\]]*)\]", detail)
                if blocked:
                    freeze_details["blocked_joints=[" + blocked.group(1) + "]"] += 1
            if name == "stage_audit":
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                stage = str(data.get("stage_name") or detail.split(":", 1)[0])
                result = str(data.get("result", ""))
                if result == "failed":
                    failed_stage_counter[stage] += 1
                try:
                    duration_value = data.get("wall_duration_s")
                    if duration_value is None and isinstance(data.get("timing"), dict):
                        duration_value = data["timing"].get("wall_duration_s")
                    if duration_value is None:
                        duration_value = data.get("duration")
                    if result == "done" and duration_value is not None:
                        stage_durations[stage].append(float(duration_value))
                except Exception:
                    pass
    return {
        "episodes_scanned": scanned,
        "event_counts": top_counter(event_counter, 16),
        "execution_failures": top_counter(failure_counter, 12),
        "failed_stages": top_counter(failed_stage_counter, 12),
        "freeze_details": top_counter(freeze_details, 12),
        "stage_duration_s": {
            stage: numeric_stats(values)
            for stage, values in sorted(stage_durations.items())
        },
    }


def analyze_debug_timeline(run_dir: Union[str, os.PathLike], max_lines: Optional[int] = None) -> Dict[str, object]:
    path = os.path.join(str(run_dir), "debug_timeline.jsonl")
    tag_counter = Counter()
    result_counter = Counter()
    stage_fail_counter = Counter()
    lines = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_lines is not None and lines >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                lines += 1
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                tag = str(row.get("tag", ""))
                result = str(row.get("result", ""))
                reason = str(row.get("reason", ""))
                tag_counter[tag] += 1
                result_counter[f"{tag}:{result}"] += 1
                if "failed" in result.lower() or "execution_failed" in reason.lower():
                    data = row.get("data") if isinstance(row.get("data"), dict) else {}
                    stage = str(data.get("stage_name") or row.get("tag") or "unknown")
                    stage_fail_counter[stage] += 1
    except FileNotFoundError:
        pass
    return {
        "lines_scanned": lines,
        "tags": top_counter(tag_counter, 20),
        "results": top_counter(result_counter, 20),
        "failed_stage_or_tag": top_counter(stage_fail_counter, 20),
    }


def inspect_trajectory_schema(run_dir: Union[str, os.PathLike], max_episodes: int = 5, max_rows_per_episode: int = 8) -> Dict[str, object]:
    rows = load_index(run_dir, "trainable")
    if not rows:
        rows = load_index(run_dir, "all")
    rows = rows[: max(0, int(max_episodes))]
    field_counter = Counter()
    camera_counter = Counter()
    sample_count = 0
    examples = []
    for episode in rows:
        traj = episode.get("trajectory", "")
        for sample in read_jsonl_limited(traj, limit=max_rows_per_episode):
            sample_count += 1
            for key in sample.keys():
                field_counter[key] += 1
            for key in LEROBOT_IMAGE_KEYS:
                if sample_image_value(sample, key):
                    camera_counter[key] += 1
            cam = sample.get("observation.camera")
            if isinstance(cam, dict) and cam.get("available"):
                camera_counter["observation.camera.available"] += 1
            if len(examples) < 3:
                examples.append(
                    {
                        "episode": episode.get("episode_index"),
                        "sample_i": sample.get("i"),
                        "has_observation_state": "observation.state" in sample,
                        "has_obs_state": "obs.state" in sample,
                        "has_action": "action" in sample,
                        "has_task": "task" in sample,
                        "images": {
                            "0": sample_image_value(sample, "observation.images.0"),
                            "1": sample_image_value(sample, "observation.images.1"),
                            "2": sample_image_value(sample, "observation.images.2"),
                        },
                    }
                )
    required = [
        "task",
        "observation.state",
        "obs.state",
        "action",
        "sand",
        "env",
        "observation.images.0",
        "observation.images.1",
        "observation.images.2",
        "observation.camera",
    ]
    required_present = {key: field_counter.get(key, 0) > 0 for key in required}
    required_coverage = {key: field_counter.get(key, 0) for key in required}
    for key in LEROBOT_IMAGE_KEYS:
        required_present[key] = camera_counter.get(key, 0) > 0
        required_coverage[key] = camera_counter.get(key, 0)
    return {
        "episodes_scanned": len(rows),
        "samples_scanned": sample_count,
        "required_field_present": required_present,
        "field_coverage": required_coverage,
        "camera_coverage": dict(camera_counter),
        "top_fields": top_counter(field_counter, 40),
        "examples": examples,
    }


def analyze_run(run_dir: Union[str, os.PathLike], include_timeline: bool = True) -> Dict[str, object]:
    run_dir = str(run_dir)
    summary = summarize_run(run_dir)
    all_rows = load_index(run_dir, "all")
    trainable_rows = load_index(run_dir, "trainable")
    rejected_rows = load_index(run_dir, "rejected")
    failed_rows = load_index(run_dir, "failed")
    planning_rows = load_index(run_dir, "planning")

    reason_parts = Counter()
    # Failure triage is deliberately problem-only. Empty reason on a trainable/success
    # episode is not a failure category and must not appear as "ok" in Pareto charts.
    primary_reasons = Counter()
    failure_reasons = Counter()
    warning_parts = Counter()
    quality_alerts = Counter()
    initial_by_status = Counter()
    plan_by_status = Counter()
    for row in all_rows:
        status = normalized_status_name(row.get("status", "unknown"))
        initial_by_status[f"{row.get('initial_pose_id', 'unknown')}:{status}"] += 1
        plan_by_status[f"{row.get('chosen_plan_id', 'unknown')}:{status}"] += 1
        failure_key = normalize_failure_reason_for_triage(row.get("reason", ""), status)
        if failure_key and is_problem_status(status):
            primary_reasons[failure_key] += 1
            failure_reasons[failure_key] += 1
        for part in split_reason(row.get("reason", "")):
            reason_parts[part] += 1
        for part in split_reason(row.get("warning_reason", "")):
            grouped_warning = normalize_warning_fragment(part)
            if not grouped_warning:
                continue
            if is_quality_signal_key(grouped_warning):
                quality_alerts[grouped_warning] += 1
            else:
                warning_parts[grouped_warning] += 1

    material_fields = {
        "max_bucket": [row.get("max_bucket_from_pile_particles") for row in all_rows],
        "lift_bucket": [row.get("lift_bucket_from_pile_particles") for row in all_rows],
        "final_bin": [row.get("final_bin_from_pile_particles") for row in all_rows],
        "final_spill": [row.get("final_spill_from_pile_particles") for row in all_rows],
        "score": [row.get("score") for row in all_rows],
        "samples": [row.get("samples") for row in all_rows],
    }
    trainable_material_fields = {
        key: [row.get({
            "max_bucket": "max_bucket_from_pile_particles",
            "lift_bucket": "lift_bucket_from_pile_particles",
            "final_bin": "final_bin_from_pile_particles",
            "final_spill": "final_spill_from_pile_particles",
            "score": "score",
            "samples": "samples",
        }[key]) for row in trainable_rows]
        for key in material_fields
    }

    rejection_events = analyze_events_for_rows(rejected_rows)
    timeline = analyze_debug_timeline(run_dir) if include_timeline else {}

    return {
        "summary": summary,
        "segment_rates_vs_attempts": {
            name: (count / float(max(1, summary["counts"]["all"])))
            for name, count in summary.get("segments", {}).items()
        },
        "primary_reasons": top_counter(primary_reasons, 12),
        "failure_reasons": top_counter(failure_reasons, 12),
        "reason_fragments": top_counter(reason_parts, 16),
        "warning_fragments": top_counter(warning_parts, 16),
        "quality_alerts": top_counter(quality_alerts, 16),
        "initial_pose_by_status": top_counter(initial_by_status, 18),
        "chosen_plan_by_status": top_counter(plan_by_status, 12),
        "material_stats_all": {key: numeric_stats(values) for key, values in material_fields.items()},
        "material_stats_trainable": {key: numeric_stats(values) for key, values in trainable_material_fields.items()},
        "planning_diagnostics_count": len(planning_rows),
        "planning_examples": planning_rows[:5],
        "trajectory_schema": inspect_trajectory_schema(run_dir),
        "rejected_examples_tail": [compact_episode(row) for row in rejected_rows[-5:]],
        "trainable_examples_tail": [compact_episode(row) for row in trainable_rows[-5:]],
        "rejected_event_analysis": rejection_events,
        "debug_timeline_analysis": timeline,
    }


def compact_analysis(report: Dict[str, object]) -> Dict[str, object]:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    run_summary = summary.get("summary", {}) if isinstance(summary.get("summary"), dict) else {}
    return {
        "run_dir": summary.get("run_dir"),
        "requested": run_summary.get("requested"),
        "max_attempts_requested": run_summary.get("max_attempts_requested"),
        "counts": summary.get("counts", {}),
        "segments": summary.get("segments", {}),
        "wall_clock_breakdown": run_summary.get("wall_clock_breakdown", {}),
        "success_rate": summary.get("success_rate"),
        "rejection_rate": summary.get("rejection_rate"),
        "segment_rates_vs_attempts": report.get("segment_rates_vs_attempts", {}),
        "primary_reasons": report.get("primary_reasons", []),
        "failure_reasons": report.get("failure_reasons", report.get("primary_reasons", [])),
        "top_reason_fragments": report.get("reason_fragments", [])[:8],
        "top_warnings": report.get("warning_fragments", [])[:8],
        "top_quality_alerts": report.get("quality_alerts", [])[:8],
        "initial_pose_by_status": report.get("initial_pose_by_status", [])[:10],
        "chosen_plan_by_status": report.get("chosen_plan_by_status", [])[:8],
        "material_stats_all": report.get("material_stats_all", {}),
        "material_stats_trainable": report.get("material_stats_trainable", {}),
        "trajectory_schema": report.get("trajectory_schema", {}),
        "rejected_event_analysis": {
            "execution_failures": (report.get("rejected_event_analysis", {}) or {}).get("execution_failures", [])[:8],
            "failed_stages": (report.get("rejected_event_analysis", {}) or {}).get("failed_stages", [])[:8],
            "freeze_details": (report.get("rejected_event_analysis", {}) or {}).get("freeze_details", [])[:8],
        },
        "rejected_examples_tail": report.get("rejected_examples_tail", [])[-3:],
        "trainable_examples_tail": report.get("trainable_examples_tail", [])[-3:],
    }


def svg_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def ensure_dir(path: Union[str, os.PathLike]) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def write_text(path: Union[str, os.PathLike], text: str) -> str:
    path = str(path)
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def safe_float_value(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return default
    if not math.isfinite(number):
        return default
    return number


def counter_rows_to_pairs(rows: object, limit: int = 12) -> List[tuple]:
    pairs = []
    if not isinstance(rows, list):
        return pairs
    for row in rows[:limit]:
        if isinstance(row, dict):
            label = row.get("key", "")
            value = safe_float_value(row.get("count"), 0.0) or 0.0
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            label = row[0]
            value = safe_float_value(row[1], 0.0) or 0.0
        else:
            continue
        pairs.append((str(label), value))
    return pairs


def write_svg_bar_chart(
    path: Union[str, os.PathLike],
    title: str,
    rows: Sequence[tuple],
    width: int = 980,
    bar_height: int = 28,
    left: int = 300,
) -> str:
    rows = [(str(label), float(value)) for label, value in rows if safe_float_value(value) is not None]
    if not rows:
        rows = [("no data", 0.0)]
    top = 58
    right = 120
    bottom = 34
    gap = 8
    height = top + bottom + len(rows) * (bar_height + gap)
    plot_width = max(120, width - left - right)
    max_value = max(max(value for _, value in rows), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2937">{svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top - 12}" x2="{left + plot_width}" y2="{top - 12}" stroke="#d1d5db" stroke-width="1"/>',
    ]
    palette = ["#2563eb", "#059669", "#d97706", "#dc2626", "#7c3aed", "#0891b2", "#4b5563"]
    for index, (label, value) in enumerate(rows):
        y = top + index * (bar_height + gap)
        bar_w = max(1, int(plot_width * (value / max_value))) if max_value > 0 else 1
        label_text = label if len(label) <= 58 else label[:55] + "..."
        color = palette[index % len(palette)]
        parts.extend(
            [
                f'<text x="24" y="{y + 19}" font-family="Arial, sans-serif" font-size="14" fill="#374151">{svg_escape(label_text)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w}" height="{bar_height}" rx="3" fill="{color}"/>',
                f'<text x="{left + bar_w + 8}" y="{y + 19}" font-family="Arial, sans-serif" font-size="14" fill="#111827">{value:g}</text>',
            ]
        )
    parts.append("</svg>\n")
    return write_text(path, "\n".join(parts))


def write_svg_histogram(
    path: Union[str, os.PathLike],
    title: str,
    values: Sequence[object],
    bins: int = 12,
) -> str:
    nums = [safe_float_value(value) for value in values]
    nums = [value for value in nums if value is not None]
    if not nums:
        return write_svg_bar_chart(path, title, [("no data", 0)])
    lo = min(nums)
    hi = max(nums)
    if abs(hi - lo) < 1e-9:
        return write_svg_bar_chart(path, title, [(f"{lo:g}", len(nums))])
    bins = max(3, min(int(bins), 24))
    step = (hi - lo) / float(bins)
    counts = [0 for _ in range(bins)]
    for value in nums:
        index = int((value - lo) / step)
        if index >= bins:
            index = bins - 1
        counts[index] += 1
    rows = []
    for index, count in enumerate(counts):
        start = lo + step * index
        end = start + step
        rows.append((f"{start:.1f}-{end:.1f}", count))
    return write_svg_bar_chart(path, title, rows, left=180)


def write_svg_scatter(
    path: Union[str, os.PathLike],
    title: str,
    points: Sequence[dict],
    x_label: str,
    y_label: str,
    width: int = 880,
    height: int = 560,
) -> str:
    clean = []
    for point in points:
        x = safe_float_value(point.get("x") if isinstance(point, dict) else None)
        y = safe_float_value(point.get("y") if isinstance(point, dict) else None)
        if x is None or y is None:
            continue
        clean.append({"x": x, "y": y, "status": str(point.get("status", "unknown"))})
    if not clean:
        return write_svg_bar_chart(path, title, [("no data", 0)])
    left = 78
    top = 58
    right = 38
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point["x"] for point in clean]
    ys = [point["y"] for point in clean]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
        min_x -= 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y += 1.0
        min_y -= 1.0
    pad_x = (max_x - min_x) * 0.06
    pad_y = (max_y - min_y) * 0.06
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def sx(value: float) -> float:
        return left + ((value - min_x) / (max_x - min_x)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((value - min_y) / (max_y - min_y)) * plot_h

    colors = {
        "trainable": "#059669",
        "success": "#059669",
        "rejected": "#dc2626",
        "failed": "#7c2d12",
        "diagnostic": "#d97706",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#1f2937">{svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f9fafb" stroke="#d1d5db"/>',
    ]
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + plot_w * frac
        y = top + plot_h * frac
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
    for point in clean:
        color = colors.get(point["status"], "#4b5563")
        parts.append(
            f'<circle cx="{sx(point["x"]):.1f}" cy="{sy(point["y"]):.1f}" r="4" fill="{color}" fill-opacity="0.78"/>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">{svg_escape(x_label)}</text>',
            f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#374151">{svg_escape(y_label)}</text>',
            f'<text x="{left}" y="{top + plot_h + 22}" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{min_x:.1f}</text>',
            f'<text x="{left + plot_w}" y="{top + plot_h + 22}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{max_x:.1f}</text>',
            f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{min_y:.1f}</text>',
            f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#6b7280">{max_y:.1f}</text>',
            f'<circle cx="{width - 150}" cy="28" r="5" fill="#059669"/><text x="{width - 138}" y="33" font-family="Arial, sans-serif" font-size="12" fill="#374151">trainable</text>',
            f'<circle cx="{width - 72}" cy="28" r="5" fill="#dc2626"/><text x="{width - 60}" y="33" font-family="Arial, sans-serif" font-size="12" fill="#374151">rejected</text>',
        ]
    )
    parts.append("</svg>\n")
    return write_text(path, "\n".join(parts))



def report_status_color(status: object) -> str:
    value = str(status or "unknown").lower()
    colors = {
        "trainable": "#059669",
        "success": "#059669",
        "rejected": "#dc2626",
        "failed": "#7c2d12",
        "diagnostic": "#d97706",
        "planning": "#2563eb",
        "unknown": "#64748b",
        "other": "#64748b",
    }
    return colors.get(value, "#64748b")


def format_report_number(value: object, digits: int = 1, empty: str = "—") -> str:
    number = safe_float_value(value)
    if number is None:
        return empty
    if abs(number - round(number)) < 1.0e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.{max(0, int(digits))}f}"


def format_report_percent(value: object, digits: int = 1, empty: str = "—") -> str:
    number = safe_float_value(value)
    if number is None:
        return empty
    return f"{number * 100.0:.{max(0, int(digits))}f}%"


def short_report_text(value: object, limit: int = 96) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 1)] + "…"


def status_counter_from_rows(rows: Sequence[dict], counts: Optional[Dict[str, object]] = None) -> Counter:
    counter = Counter()
    for row in rows or []:
        counter[str(row.get("status") or "unknown")] += 1
    if counter:
        return counter
    counts = counts or {}
    for key in ["trainable", "rejected", "failed", "diagnostic"]:
        value = safe_float_value(counts.get(key), 0.0) or 0.0
        if value > 0:
            counter[key] = int(value)
    return counter


def write_svg_outcome_stack(
    path: Union[str, os.PathLike],
    title: str,
    status_counts: Counter,
    width: int = 980,
    height: int = 190,
) -> str:
    ordered = []
    for key in ["trainable", "success", "rejected", "failed", "diagnostic", "unknown"]:
        if status_counts.get(key, 0):
            ordered.append((key, float(status_counts.get(key, 0))))
    for key, value in status_counts.most_common():
        if key not in [item[0] for item in ordered] and value:
            ordered.append((key, float(value)))
    if not ordered:
        ordered = [("no data", 0.0)]
    total = max(sum(value for _, value in ordered), 1.0)
    left, top, plot_w, bar_h = 44, 78, width - 88, 38
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{svg_escape(title)}</title>',
        f'<desc>Episode outcome distribution as a stacked bar with counts and percentages.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="24" y="55" font-family="Arial, sans-serif" font-size="12" fill="#64748b">total outcomes={format_report_number(total, 0)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{bar_h}" rx="9" fill="#f1f5f9"/>',
    ]
    x = float(left)
    legend_x = left
    legend_y = top + bar_h + 34
    for status, value in ordered:
        pct = value / total if total else 0.0
        seg_w = plot_w * pct
        if seg_w < 1 and value > 0:
            seg_w = 1
        color = report_status_color(status)
        parts.append(
            f'<rect x="{x:.1f}" y="{top}" width="{seg_w:.1f}" height="{bar_h}" fill="{color}"><title>{svg_escape(status)} {value:g} ({pct * 100.0:.1f}%)</title></rect>'
        )
        if seg_w > 70:
            parts.append(
                f'<text x="{x + seg_w / 2:.1f}" y="{top + 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="700" fill="#ffffff">{svg_escape(status)} {pct * 100.0:.0f}%</text>'
            )
        parts.append(
            f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{color}"/><text x="{legend_x + 10}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">{svg_escape(status)} {format_report_number(value,0)} · {pct * 100.0:.1f}%</text>'
        )
        legend_x += 150
        if legend_x > width - 180:
            legend_x = left
            legend_y += 22
        x += seg_w
    parts.append('</svg>\n')
    return write_text(path, "\n".join(parts))


def write_svg_pareto_chart(
    path: Union[str, os.PathLike],
    title: str,
    rows: Sequence[tuple],
    width: int = 980,
    bar_height: int = 28,
    left: int = 360,
    limit: int = 12,
) -> str:
    clean = []
    for label, value in rows[: max(1, int(limit))]:
        number = safe_float_value(value)
        if number is not None:
            clean.append((str(label), float(number)))
    if not clean:
        clean = [("no data", 0.0)]
    total = max(sum(value for _, value in clean), 1.0)
    top, right, bottom, gap = 66, 156, 46, 8
    height = top + bottom + len(clean) * (bar_height + gap)
    plot_w = max(160, width - left - right)
    max_value = max(max(value for _, value in clean), 1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{svg_escape(title)}</title>',
        '<desc>Ranked Pareto-style bar chart showing count and cumulative share.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="24" y="54" font-family="Arial, sans-serif" font-size="12" fill="#64748b">ranked by count; cumulative share displayed at right</text>',
    ]
    cumulative = 0.0
    for index, (label, value) in enumerate(clean):
        cumulative += value
        pct = value / total if total else 0.0
        cum_pct = cumulative / total if total else 0.0
        y = top + index * (bar_height + gap)
        bar_w = max(1, plot_w * (value / max_value)) if max_value > 0 else 1
        color = "#1d4ed8" if index == 0 else "#64748b"
        label_text = short_report_text(label, 72)
        parts.extend(
            [
                f'<text x="24" y="{y + 19}" font-family="Arial, sans-serif" font-size="13" fill="#334155"><title>{svg_escape(label)}</title>{svg_escape(label_text)}</text>',
                f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="{bar_height}" rx="4" fill="{color}" opacity="0.88"/>',
                f'<text x="{left + bar_w + 8:.1f}" y="{y + 19}" font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#111827">{value:g}</text>',
                f'<text x="{width - 24}" y="{y + 19}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#64748b">{pct * 100.0:.1f}% / cum {cum_pct * 100.0:.1f}%</text>',
            ]
        )
    parts.append('</svg>\n')
    return write_text(path, "\n".join(parts))


def write_svg_status_by_category_chart(
    path: Union[str, os.PathLike],
    title: str,
    matrix: Dict[str, Counter],
    width: int = 980,
    bar_height: int = 26,
    left: int = 285,
    limit: int = 14,
) -> str:
    rows = []
    for category, counter in matrix.items():
        total = sum(counter.values())
        if total > 0:
            rows.append((str(category), counter, total))
    rows.sort(key=lambda item: (-item[2], item[0]))
    rows = rows[: max(1, int(limit))]
    if not rows:
        rows = [("no data", Counter({"unknown": 0}), 0)]
    top, right, bottom, gap = 62, 132, 52, 8
    height = top + bottom + len(rows) * (bar_height + gap)
    plot_w = max(160, width - left - right)
    max_total = max(max(total for _, _, total in rows), 1)
    status_order = ["trainable", "success", "rejected", "failed", "diagnostic", "unknown"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{svg_escape(title)}</title>',
        '<desc>Stacked horizontal bars by category and episode status.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
    ]
    for index, (category, counter, total) in enumerate(rows):
        y = top + index * (bar_height + gap)
        label = short_report_text(category, 48)
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#334155"><title>{svg_escape(category)}</title>{svg_escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{plot_w * total / max_total:.1f}" height="{bar_height}" rx="4" fill="#f1f5f9"/>')
        x = float(left)
        for status in status_order + [key for key in counter.keys() if key not in status_order]:
            value = counter.get(status, 0)
            if value <= 0:
                continue
            seg_w = (plot_w * total / max_total) * (value / max(1, total))
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(1.0, seg_w):.1f}" height="{bar_height}" fill="{report_status_color(status)}"><title>{svg_escape(category)} / {svg_escape(status)}: {value}</title></rect>'
            )
            x += max(1.0, seg_w)
        parts.append(f'<text x="{left + plot_w * total / max_total + 8:.1f}" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#111827">{total}</text>')
    legend_x, legend_y = left, height - 20
    seen = []
    for _, counter, _ in rows:
        for status in counter.keys():
            if status not in seen and counter.get(status, 0) > 0:
                seen.append(status)
    for status in seen[:8]:
        parts.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="5" fill="{report_status_color(status)}"/><text x="{legend_x + 10}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">{svg_escape(status)}</text>')
        legend_x += 108
    parts.append('</svg>\n')
    return write_text(path, "\n".join(parts))


def write_svg_metric_scatter(
    path: Union[str, os.PathLike],
    title: str,
    points: Sequence[dict],
    x_label: str,
    y_label: str,
    width: int = 980,
    height: int = 560,
    include_origin: bool = False,
) -> str:
    clean = []
    for point in points or []:
        if not isinstance(point, dict):
            continue
        x = safe_float_value(point.get("x"))
        y = safe_float_value(point.get("y"))
        if x is None or y is None:
            continue
        clean.append(
            {
                "x": x,
                "y": y,
                "status": str(point.get("status", "unknown")),
                "episode": point.get("episode") or point.get("label") or "",
                "title": str(point.get("title") or ""),
            }
        )
    if not clean:
        return write_svg_bar_chart(path, title, [("no data", 0)], left=180)
    left, top, right, bottom = 82, 64, 44, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [point["x"] for point in clean]
    ys = [point["y"] for point in clean]
    if include_origin:
        xs.append(0.0)
        ys.append(0.0)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if abs(max_x - min_x) < 1e-9:
        min_x -= 1.0
        max_x += 1.0
    if abs(max_y - min_y) < 1e-9:
        min_y -= 1.0
        max_y += 1.0
    pad_x = (max_x - min_x) * 0.07
    pad_y = (max_y - min_y) * 0.07
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    def sx(value: float) -> float:
        return left + ((float(value) - min_x) / (max_x - min_x)) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - ((float(value) - min_y) / (max_y - min_y)) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        f'<title>{svg_escape(title)}</title>',
        '<desc>Scatter plot colored by episode status.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{svg_escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for i in range(5):
        frac = i / 4.0
        x = left + plot_w * frac
        y = top + plot_h * frac
        x_value = min_x + (max_x - min_x) * frac
        y_value = max_y - (max_y - min_y) * frac
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#e2e8f0"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#64748b">{x_value:.1f}</text>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#64748b">{y_value:.1f}</text>')
    if include_origin and min_x <= 0.0 <= max_x:
        parts.append(f'<line x1="{sx(0):.1f}" y1="{top}" x2="{sx(0):.1f}" y2="{top + plot_h}" stroke="#0f172a" stroke-dasharray="4 4" opacity="0.45"/>')
    if include_origin and min_y <= 0.0 <= max_y:
        parts.append(f'<line x1="{left}" y1="{sy(0):.1f}" x2="{left + plot_w}" y2="{sy(0):.1f}" stroke="#0f172a" stroke-dasharray="4 4" opacity="0.45"/>')
    for point in clean:
        title_text = point["title"] or f'ep {point["episode"]} {point["status"]} ({point["x"]:.3g}, {point["y"]:.3g})'
        r = 5.2 if point["status"] in ["trainable", "success"] else 4.5
        parts.append(
            f'<circle cx="{sx(point["x"]):.1f}" cy="{sy(point["y"]):.1f}" r="{r}" fill="{report_status_color(point["status"])}" opacity="0.82" stroke="#ffffff" stroke-width="0.8"><title>{svg_escape(title_text)}</title></circle>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">{svg_escape(x_label)}</text>',
            f'<text transform="translate(24 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#334155">{svg_escape(y_label)}</text>',
        ]
    )
    legend_statuses = []
    for point in clean:
        if point["status"] not in legend_statuses:
            legend_statuses.append(point["status"])
    lx, ly = width - 230, 32
    for status in legend_statuses[:6]:
        parts.append(f'<circle cx="{lx}" cy="{ly}" r="5" fill="{report_status_color(status)}"/><text x="{lx + 10}" y="{ly + 4}" font-family="Arial, sans-serif" font-size="12" fill="#334155">{svg_escape(status)}</text>')
        lx += 95
        if lx > width - 58:
            lx = width - 230
            ly += 20
    parts.append('</svg>\n')
    return write_text(path, "\n".join(parts))


def report_material_row(stats: Dict[str, object], key: str) -> Dict[str, object]:
    row = stats.get(key, {}) if isinstance(stats.get(key), dict) else {}
    return {
        "metric": key,
        "count": row.get("count", 0),
        "min": row.get("min"),
        "median": row.get("median"),
        "mean": row.get("mean"),
        "max": row.get("max"),
    }


def build_report_context(
    compact: Dict[str, object],
    report: Optional[Dict[str, object]] = None,
    all_rows: Optional[Sequence[dict]] = None,
) -> Dict[str, object]:
    report = report or {}
    all_rows = list(all_rows or [])
    counts = compact.get("counts", {}) if isinstance(compact.get("counts"), dict) else {}
    attempts = int(safe_float_value(counts.get("all"), 0.0) or 0)
    trainable = int(safe_float_value(counts.get("trainable"), 0.0) or 0)
    rejected = int(safe_float_value(counts.get("rejected"), 0.0) or 0)
    failed = int(safe_float_value(counts.get("failed"), 0.0) or 0)
    success = int(safe_float_value(counts.get("success"), 0.0) or 0)
    diagnostic = int(safe_float_value(counts.get("diagnostic"), 0.0) or 0)
    denom = max(1, attempts)
    schema = compact.get("trajectory_schema", {}) if isinstance(compact.get("trajectory_schema"), dict) else {}
    field_coverage = schema.get("field_coverage", {}) if isinstance(schema.get("field_coverage"), dict) else {}
    camera_coverage = schema.get("camera_coverage", {}) if isinstance(schema.get("camera_coverage"), dict) else {}
    material_all = compact.get("material_stats_all", {}) if isinstance(compact.get("material_stats_all"), dict) else {}
    material_trainable = compact.get("material_stats_trainable", {}) if isinstance(compact.get("material_stats_trainable"), dict) else {}
    material_source = material_trainable if any((isinstance(v, dict) and v.get("count")) for v in material_trainable.values()) else material_all
    final_bin_median = safe_float_value((material_source.get("final_bin") or {}).get("median") if isinstance(material_source.get("final_bin"), dict) else None, 0.0) or 0.0
    final_spill_median = safe_float_value((material_source.get("final_spill") or {}).get("median") if isinstance(material_source.get("final_spill"), dict) else None, 0.0) or 0.0
    score_median = safe_float_value((material_source.get("score") or {}).get("median") if isinstance(material_source.get("score"), dict) else None, None)

    gates = []

    def gate(name: str, ok: bool, detail: str, weight: int, warn: bool = False):
        gates.append({"name": name, "status": "pass" if ok else ("warn" if warn else "fail"), "detail": detail, "weight": weight})

    gate("有 episode 采集记录", attempts > 0, f"attempts={attempts}", 10)
    gate("存在可训练样本", trainable > 0, f"trainable={trainable}, rate={trainable / denom * 100.0:.1f}%", 25)
    gate("拒绝率不过高", rejected / denom < 0.5, f"rejected={rejected}, rate={rejected / denom * 100.0:.1f}%", 10, warn=rejected > 0)
    state_ok = bool(field_coverage.get("observation.state") or field_coverage.get("obs.state"))
    gate("轨迹含状态向量", state_ok, f"observation.state={field_coverage.get('observation.state', 0)}, obs.state={field_coverage.get('obs.state', 0)}", 15)
    action_ok = bool(field_coverage.get("action"))
    gate("轨迹含 action", action_ok, f"action samples={field_coverage.get('action', 0)}", 15)
    camera_ok = any(camera_coverage.get(key, 0) for key in ["observation.images.0", "observation.images.1", "observation.images.2", "observation.camera.available"])
    gate("相机字段可见", camera_ok, "camera samples=" + str(camera_coverage or {}), 10)
    material_ok = final_bin_median > 0 or trainable > 0
    gate("物料进入目标容器", material_ok, f"final_bin median={format_report_number(final_bin_median, 1)}", 10, warn=final_spill_median > 0)
    score = sum(row["weight"] for row in gates if row["status"] == "pass")
    score += sum(max(0, row["weight"] // 2) for row in gates if row["status"] == "warn")
    score = min(100, int(score))

    top_reasons = compact.get("failure_reasons", compact.get("primary_reasons", []))
    if not isinstance(top_reasons, list):
        top_reasons = []
    top_reason = top_reasons[0] if top_reasons else {}
    top_reason_key = str(top_reason.get("key", "")) if isinstance(top_reason, dict) else ""
    top_reason_count = int(safe_float_value(top_reason.get("count") if isinstance(top_reason, dict) else 0, 0.0) or 0)
    problem_attempts = max(0, rejected + failed + diagnostic)
    warnings = compact.get("top_warnings", []) if isinstance(compact.get("top_warnings"), list) else []
    quality_alerts = compact.get("top_quality_alerts", []) if isinstance(compact.get("top_quality_alerts"), list) else []
    top_quality = quality_alerts[0] if quality_alerts else {}
    top_quality_key = str(top_quality.get("key", "")) if isinstance(top_quality, dict) else ""
    top_quality_count = int(safe_float_value(top_quality.get("count") if isinstance(top_quality, dict) else 0, 0.0) or 0)

    findings = []
    if attempts <= 0:
        findings.append({"severity": "critical", "title": "没有可分析的 attempts", "detail": "run 目录下 episodes.jsonl 为空或不存在。"})
    if attempts > 0 and trainable <= 0:
        findings.append({"severity": "critical", "title": "当前 run 无可训练样本", "detail": "不要直接导出训练；先修复 rejection / planning / execution 主因。"})
    if top_reason_key:
        denominator = problem_attempts or attempts or top_reason_count
        findings.append({
            "severity": "high" if top_reason_count >= max(1, denominator // 2) else "medium",
            "title": "主阻塞类目",
            "detail": f"{top_reason_key} · {top_reason_count}/{denominator} problem episodes",
        })
    elif problem_attempts > 0:
        findings.append({"severity": "medium", "title": "存在问题 episode 但未分类", "detail": f"problem episodes={problem_attempts}; reason 字段为空或格式未被归类。"})
    else:
        findings.append({"severity": "info", "title": "没有失败阻塞类目", "detail": "failure Pareto 只统计 rejected/failed/diagnostic，不再把成功 episode 的空 reason 记为 ok。"})
    if top_quality_key:
        findings.append({"severity": "medium", "title": "主要质量信号", "detail": f"{top_quality_key} · {top_quality_count} episode(s)"})
    if final_spill_median > 0 and final_bin_median <= 0:
        findings.append({"severity": "high", "title": "物料没有进入目标容器", "detail": f"median spill={format_report_number(final_spill_median, 1)}, median bin={format_report_number(final_bin_median, 1)}"})
    if not camera_ok:
        findings.append({"severity": "medium", "title": "相机字段覆盖不足", "detail": "LeRobot/VLA 训练前需要检查 observation.images.* 或 observation.camera。"})

    recommendations = []
    if trainable <= 0 and top_reason_key:
        recommendations.append("先修复主阻塞类目：按该 reason 筛选 episode，固定 seed 复现 3-5 次，确认是规划、执行还是质量阈值问题。")
    elif trainable > 0 and problem_attempts > 0 and top_reason_key:
        recommendations.append("保留可训练样本，同时针对主阻塞类目做小范围 ablation；不要把成功样本的 ok 当成失败类目。")
    elif trainable > 0 and problem_attempts == 0:
        recommendations.append("当前没有明显失败阻塞；可以先生成 --plots 静态报告，再做 LeRobot 导出 smoke test。")
    if "planning" in top_reason_key:
        recommendations.append("优先拆解 planning：按 initial_pose、truck body-frame XY/yaw、unload local XY 和 IK planar error 分组。")
    if "low_final_bin" in top_reason_key or final_bin_median <= 0:
        recommendations.append("检查 unload local XY：确认 unload point 是否落在 selected dump-bed mesh 的有效区域内。")
    if final_spill_median > max(1.0, final_bin_median * 2.0):
        recommendations.append("洒料显著高于入斗量：检查 lift/carry 阶段 bucket 姿态保持和 swing 速度/加速度峰值。")
    if top_quality_key and "score_low" in top_quality_key:
        recommendations.append("score_low 作为质量信号聚合展示，不作为 failure Pareto；需要结合 bin/spill 判断是否调阈值或调动作。")
    if not state_ok or not action_ok:
        recommendations.append("补齐 trajectory schema：训练前必须稳定记录 observation.state/obs.state 和 action。")
    if not camera_ok:
        recommendations.append("补齐三路相机样本覆盖；VLA 导出前确认 observation.images.0/1/2 都能解析到真实图片。")
    if not recommendations:
        recommendations.append("继续扩大采样量，并用同一套 status / blocker / quality 指标横向比较 run。")

    material_table = [report_material_row(material_source, key) for key in ["score", "max_bucket", "lift_bucket", "final_bin", "final_spill", "samples"]]
    status_counts = status_counter_from_rows(all_rows, counts)
    return {
        "run_dir": compact.get("run_dir"),
        "attempts": attempts,
        "trainable": trainable,
        "success": success,
        "rejected": rejected,
        "failed": failed,
        "diagnostic": diagnostic,
        "trainable_rate": trainable / denom,
        "success_file_rate": success / denom,
        "rejection_rate": rejected / denom,
        "failure_rate": failed / denom,
        "status_counts": dict(status_counts),
        "readiness_score": score,
        "readiness_gates": gates,
        "findings": findings,
        "recommendations": recommendations[:6],
        "problem_attempts": problem_attempts,
        "top_reason": top_reason,
        "top_problem_reason": top_reason,
        "top_reasons": top_reasons,
        "failure_reasons": top_reasons,
        "top_warnings": warnings,
        "general_warnings": warnings,
        "quality_alerts": quality_alerts,
        "top_quality_signal": top_quality,
        "material_table": material_table,
        "score_median": score_median,
        "final_bin_median": final_bin_median,
        "final_spill_median": final_spill_median,
        "camera_coverage": camera_coverage,
        "field_coverage": field_coverage,
    }


def report_plot_group(name: str) -> str:
    if name in {"outcome_mix.svg", "status_counts.svg", "segment_counts.svg", "segment_rates.svg"}:
        return "Outcome"
    if "reason" in name or "warning" in name or "initial_pose" in name or "plan" in name:
        return "Failure triage"
    if "scene" in name or "sand_xy" in name or "truck_xy" in name or "unload" in name:
        return "Scene coverage"
    if "score" in name or "bucket" in name or "spill" in name or "bin" in name:
        return "Material quality"
    return "Other"


def html_report_table(headers: Sequence[str], rows: Sequence[Sequence[object]], cls: str = "") -> str:
    parts = [f'<table class="{svg_escape(cls)}">', '<thead><tr>']
    for header in headers:
        parts.append(f'<th>{svg_escape(header)}</th>')
    parts.append('</tr></thead><tbody>')
    for row in rows:
        parts.append('<tr>')
        for cell in row:
            parts.append(f'<td>{svg_escape(cell)}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return "".join(parts)


def write_html_report(
    path: Union[str, os.PathLike],
    compact: Dict[str, object],
    plot_files: Sequence[Union[str, os.PathLike]],
    report: Optional[Dict[str, object]] = None,
    all_rows: Optional[Sequence[dict]] = None,
) -> str:
    context = build_report_context(compact, report=report, all_rows=all_rows)
    run_dir = context.get("run_dir", "") or compact.get("run_dir", "")
    output_dir = os.path.dirname(str(path)) or "."
    plot_titles = {
        "outcome_mix.svg": "Outcome Mix",
        "status_counts.svg": "Episode Status Counts",
        "segment_counts.svg": "Segment Counts",
        "primary_reasons.svg": "Primary Reasons",
        "reason_pareto.svg": "Reason Pareto",
        "initial_pose_status.svg": "Initial Pose Status",
        "initial_pose_status_stack.svg": "Initial Pose × Status",
        "warning_counts.svg": "Warnings",
        "score_histogram.svg": "Score Histogram",
        "bucket_material_medians.svg": "Material Medians",
        "bin_vs_spill_scatter.svg": "Final Bin vs Spill",
        "score_vs_spill_scatter.svg": "Score vs Spill",
        "sand_xy_scatter.svg": "Sand XY Coverage",
        "truck_xy_scatter.svg": "Truck XY Coverage",
        "unload_xy_scatter.svg": "Unload XY Coverage",
    }
    plot_groups: Dict[str, List[str]] = defaultdict(list)
    for plot_file in plot_files:
        name = os.path.basename(str(plot_file))
        if name.endswith(".svg"):
            plot_groups[report_plot_group(name)].append(name)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "info": 3}
    findings = sorted(context.get("findings", []), key=lambda item: severity_order.get(str(item.get("severity")), 9))
    gate_rows = []
    for gate in context.get("readiness_gates", []):
        gate_rows.append([str(gate.get("status", "")).upper(), gate.get("name", ""), gate.get("detail", ""), gate.get("weight", "")])
    material_rows = []
    for row in context.get("material_table", []):
        material_rows.append([
            row.get("metric", ""),
            format_report_number(row.get("count"), 0),
            format_report_number(row.get("min"), 1),
            format_report_number(row.get("median"), 1),
            format_report_number(row.get("mean"), 1),
            format_report_number(row.get("max"), 1),
        ])
    reason_rows = []
    for item in context.get("top_reasons", [])[:10]:
        if isinstance(item, dict):
            reason_rows.append([item.get("key", ""), item.get("count", "")])
    warning_rows = []
    for item in context.get("top_warnings", [])[:10]:
        if isinstance(item, dict):
            warning_rows.append([item.get("key", ""), item.get("count", "")])
    episode_rows = []
    for row in list(all_rows or [])[:120]:
        episode_rows.append([
            row.get("episode_index", ""),
            row.get("status", ""),
            format_report_number(row.get("score"), 1),
            format_report_number(row.get("max_bucket_from_pile_particles"), 0),
            format_report_number(row.get("lift_bucket_from_pile_particles"), 0),
            format_report_number(row.get("final_bin_from_pile_particles"), 0),
            format_report_number(row.get("final_spill_from_pile_particles"), 0),
            short_report_text(row.get("reason") or row.get("warning_reason"), 110),
        ])

    score = int(context.get("readiness_score", 0))
    score_class = "bad" if score < 35 else ("warn" if score < 70 else "good")
    status_counts = context.get("status_counts", {}) if isinstance(context.get("status_counts"), dict) else {}
    cards = [
        ("Attempts", format_report_number(context.get("attempts"), 0), "total rows in episodes.jsonl"),
        ("Trainable", format_report_number(context.get("trainable"), 0), format_report_percent(context.get("trainable_rate"))),
        ("Rejected", format_report_number(context.get("rejected"), 0), format_report_percent(context.get("rejection_rate"))),
        ("Failed", format_report_number(context.get("failed"), 0), format_report_percent(context.get("failure_rate"))),
        ("Median score", format_report_number(context.get("score_median"), 1), "material/quality score"),
        ("Median bin", format_report_number(context.get("final_bin_median"), 1), "particles into bin"),
        ("Median spill", format_report_number(context.get("final_spill_median"), 1), "particles spilled"),
        ("Readiness", f"{score}/100", "export/training gate"),
    ]
    html = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Excavator Dataset Report</title>",
        "<style>",
        ":root{font-family:Inter,Segoe UI,Arial,sans-serif;background:#eef2f7;color:#111827;--panel:#fff;--border:#dbe3ee;--muted:#64748b;--good:#059669;--bad:#dc2626;--warn:#d97706;--blue:#1d4ed8;}",
        "*{box-sizing:border-box} body{margin:0;padding:22px;background:linear-gradient(180deg,#e9eff8 0,#f8fafc 260px,#f8fafc 100%);} a{color:#1d4ed8}",
        ".wrap{max-width:1420px;margin:0 auto}.hero{background:#0f172a;color:white;border-radius:18px;padding:24px 26px;margin-bottom:16px;box-shadow:0 18px 45px rgba(15,23,42,.18)}",
        ".hero h1{margin:0 0 8px;font-size:30px;letter-spacing:-.02em}.run{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#cbd5e1;font-size:12px;word-break:break-all}",
        ".heroGrid{display:grid;grid-template-columns:minmax(280px,1.2fr) minmax(260px,.8fr);gap:18px;margin-top:18px}.scoreBox{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.14);border-radius:14px;padding:16px}.scoreNum{font-size:42px;font-weight:800}.bar{height:11px;background:rgba(255,255,255,.18);border-radius:999px;overflow:hidden}.bar span{display:block;height:100%;border-radius:999px}.bar .good{background:#10b981}.bar .warn{background:#f59e0b}.bar .bad{background:#ef4444}",
        ".cards{display:grid;grid-template-columns:repeat(8,minmax(120px,1fr));gap:10px;margin-bottom:16px}.card{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:13px 14px;box-shadow:0 1px 2px rgba(15,23,42,.04)}.label{font-size:11px;text-transform:uppercase;color:var(--muted);letter-spacing:.06em}.value{font-size:25px;font-weight:800;margin:4px 0 2px}.sub{font-size:12px;color:var(--muted)}",
        ".grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:15px;overflow:hidden;box-shadow:0 1px 2px rgba(15,23,42,.04)}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}",
        "h2{font-size:17px;margin:0 0 10px}h3{font-size:14px;margin:14px 0 8px;color:#334155}.muted{color:var(--muted);font-size:12px}.finding{border:1px solid #e2e8f0;border-radius:12px;padding:10px 12px;margin:8px 0;background:#f8fafc}.sev{display:inline-block;border-radius:999px;padding:2px 7px;font-size:11px;font-weight:800;text-transform:uppercase;margin-right:6px}.sev.critical,.sev.high{background:#fee2e2;color:#991b1b}.sev.medium{background:#fef3c7;color:#92400e}.sev.info{background:#dbeafe;color:#1e40af}",
        "ol{margin:8px 0 0 20px;padding:0}li{margin:7px 0}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:7px 8px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}th{color:#475569;background:#f8fafc;font-weight:700;position:sticky;top:0}.scroll{max-height:440px;overflow:auto;border:1px solid #e5e7eb;border-radius:10px}",
        ".plotGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:12px}.plot{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:10px}.plot img{display:block;width:100%;height:auto;border-radius:8px;background:white}.plot h3{margin:0 0 8px;font-size:14px}.groupTitle{font-size:15px;font-weight:800;margin:18px 0 10px;color:#0f172a}",
        "details{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:10px;margin-top:12px}summary{cursor:pointer;font-weight:700}.json{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:11px;color:#334155;max-height:280px;overflow:auto}",
        "@media(max-width:1100px){.cards{grid-template-columns:repeat(2,1fr)}.heroGrid,.grid{grid-template-columns:1fr}.span4,.span5,.span6,.span7,.span8,.span12{grid-column:span 1}.plotGrid{grid-template-columns:1fr}}",
        "</style>",
        "</head><body><div class=\"wrap\">",
        '<section class="hero">',
        '<h1>Excavator Dataset Report</h1>',
        f'<div class="run">{svg_escape(run_dir)}</div>',
        '<div class="heroGrid">',
        '<div><div class="muted" style="color:#cbd5e1">Purpose</div><div style="font-size:18px;line-height:1.45;margin-top:4px">判断本次自动采集是否已经适合进入 LeRobot/VLA 训练，并定位下一轮采集最该修复的失败链路。</div></div>',
        '<div class="scoreBox">',
        f'<div class="label" style="color:#cbd5e1">Training readiness</div><div class="scoreNum">{score}/100</div><div class="bar"><span class="{score_class}" style="width:{max(0,min(100,score))}%"></span></div>',
        f'<div class="sub" style="color:#cbd5e1;margin-top:7px">status rows: {svg_escape(dict(status_counts))}</div>',
        '</div></div></section>',
        '<section class="cards">',
    ]
    for label, value, sub in cards:
        html.append(f'<div class="card"><div class="label">{svg_escape(label)}</div><div class="value">{svg_escape(value)}</div><div class="sub">{svg_escape(sub)}</div></div>')
    html.append('</section><main class="grid">')
    html.append('<section class="panel span5"><h2>Key findings</h2>')
    for item in findings:
        sev = str(item.get("severity", "info"))
        html.append(f'<div class="finding"><span class="sev {svg_escape(sev)}">{svg_escape(sev)}</span><strong>{svg_escape(item.get("title", ""))}</strong><div class="muted" style="margin-top:4px">{svg_escape(item.get("detail", ""))}</div></div>')
    html.append('</section>')
    html.append('<section class="panel span7"><h2>Recommended next actions</h2><ol>')
    for rec in context.get("recommendations", []):
        html.append(f'<li>{svg_escape(rec)}</li>')
    html.append('</ol></section>')
    html.append('<section class="panel span6"><h2>Data readiness gates</h2>')
    html.append(html_report_table(["Status", "Gate", "Detail", "Weight"], gate_rows))
    html.append('</section>')
    html.append('<section class="panel span6"><h2>Material / score statistics</h2>')
    html.append(html_report_table(["Metric", "Count", "Min", "Median", "Mean", "Max"], material_rows))
    html.append('</section>')
    if reason_rows or warning_rows:
        html.append('<section class="panel span6"><h2>Top primary reasons</h2>')
        html.append(html_report_table(["Reason", "Count"], reason_rows or [["no data", 0]]))
        html.append('</section>')
        html.append('<section class="panel span6"><h2>Top warnings</h2>')
        html.append(html_report_table(["Warning", "Count"], warning_rows or [["no data", 0]]))
        html.append('</section>')
    html.append('<section class="panel span12"><h2>Charts</h2>')
    for group in ["Outcome", "Failure triage", "Scene coverage", "Material quality", "Other"]:
        names = plot_groups.get(group, [])
        if not names:
            continue
        html.append(f'<div class="groupTitle">{svg_escape(group)}</div><div class="plotGrid">')
        for name in names:
            title = plot_titles.get(name, os.path.splitext(name)[0].replace("_", " ").title())
            html.append(f'<div class="plot"><h3>{svg_escape(title)}</h3><img src="{svg_escape(name)}" alt="{svg_escape(title)}"></div>')
        html.append('</div>')
    html.append('</section>')
    if episode_rows:
        html.append('<section class="panel span12"><h2>Episode table</h2><div class="muted">前 120 条 episode；用于快速定位离群 score、极端 spill 或同类失败。</div><div class="scroll">')
        html.append(html_report_table(["Ep", "Status", "Score", "Max bucket", "Lift bucket", "Final bin", "Final spill", "Reason"], episode_rows))
        html.append('</div></section>')
    html.append('<section class="panel span12"><h2>Schema coverage</h2>')
    schema_rows = [[key, value] for key, value in sorted((context.get("field_coverage") or {}).items())]
    camera_rows = [[key, value] for key, value in sorted((context.get("camera_coverage") or {}).items())]
    html.append('<div class="grid" style="grid-template-columns:1fr 1fr"><div>')
    html.append('<h3>Trajectory fields</h3>' + html_report_table(["Field", "Samples"], schema_rows[:40] or [["no data", 0]]))
    html.append('</div><div><h3>Camera coverage</h3>' + html_report_table(["Camera key", "Samples"], camera_rows or [["no data", 0]]) + '</div></div>')
    html.append('<details><summary>Embedded diagnosis JSON</summary><pre class="json">')
    html.append(svg_escape(json.dumps(context, ensure_ascii=False, indent=2)))
    html.append('</pre></details></section>')
    html.extend([
        '</main>',
        '<footer class="muted" style="margin:18px 0 6px">Generated by excavator_dataset_tools.py. `analysis_compact.json` and `report_diagnosis.json` are written next to this HTML.</footer>',
        '</div></body></html>',
        '',
    ])
    return write_text(path, "\n".join(html))


def generate_plots(
    run_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
) -> Dict[str, object]:
    run_dir = str(run_dir)
    output_dir = ensure_dir(output_dir or os.path.join(run_dir, "analysis_plots"))
    report = analyze_run(run_dir, include_timeline=False)
    compact = compact_analysis(report)
    all_rows = load_index(run_dir, "all")
    trainable_rows = load_index(run_dir, "trainable")
    rejected_rows = load_index(run_dir, "rejected")

    counts = compact.get("counts", {}) if isinstance(compact.get("counts"), dict) else {}
    segments = compact.get("segments", {}) if isinstance(compact.get("segments"), dict) else {}
    attempts = safe_float_value(counts.get("all"), 0.0) or 0.0
    trainable_stats = compact.get("material_stats_trainable", {})
    if not isinstance(trainable_stats, dict) or not trainable_stats:
        trainable_stats = compact.get("material_stats_all", {})
    material_rows = []
    if isinstance(trainable_stats, dict):
        for key in ["max_bucket", "lift_bucket", "final_bin", "final_spill"]:
            stats = trainable_stats.get(key, {}) if isinstance(trainable_stats.get(key), dict) else {}
            if stats:
                material_rows.append((f"{key} median", stats.get("median", 0)))

    scatter_points = []
    score_vs_spill_points = []
    for row in trainable_rows + rejected_rows:
        status = str(row.get("status") or ("trainable" if row in trainable_rows else "rejected"))
        ep = row.get("episode_index")
        scatter_points.append(
            {
                "x": row.get("final_spill_from_pile_particles"),
                "y": row.get("final_bin_from_pile_particles"),
                "status": status,
                "episode": ep,
                "title": f"ep {ep} {status}: spill={row.get('final_spill_from_pile_particles')} bin={row.get('final_bin_from_pile_particles')}",
            }
        )
        score_vs_spill_points.append(
            {
                "x": row.get("final_spill_from_pile_particles"),
                "y": row.get("score"),
                "status": status,
                "episode": ep,
                "title": f"ep {ep} {status}: spill={row.get('final_spill_from_pile_particles')} score={row.get('score')}",
            }
        )

    status_counts = status_counter_from_rows(all_rows, counts)
    initial_pose_matrix: Dict[str, Counter] = defaultdict(Counter)
    for row in all_rows:
        pose = str(row.get("initial_pose_id") or "unknown")
        status = str(row.get("status") or "unknown")
        initial_pose_matrix[pose][status] += 1

    scene_sand_points = []
    scene_truck_points = []
    scene_unload_points = []
    for row in all_rows:
        try:
            scene = dashboard_scene_from_episode(row)
        except Exception:
            scene = {}
        status = str(row.get("status") or "unknown")
        ep = row.get("episode_index")
        sand_xy = (scene.get("sand_body_xy") or scene.get("sand_xy")) if isinstance(scene, dict) else None
        truck_xy = (scene.get("truck_body_xy") or scene.get("truck_xy")) if isinstance(scene, dict) else None
        unload_xy = (scene.get("unload_point_body_xy") or scene.get("unload_landing_body_xy") or scene.get("unload_body_xy") or scene.get("unload_point_xy") or scene.get("unload_landing_xy") or scene.get("unload_xy")) if isinstance(scene, dict) else None
        if isinstance(sand_xy, list) and len(sand_xy) >= 2:
            scene_sand_points.append({"x": sand_xy[0], "y": sand_xy[1], "status": status, "episode": ep, "title": f"ep {ep} {status}: sand_body=({sand_xy[0]:.3g},{sand_xy[1]:.3g})"})
        if isinstance(truck_xy, list) and len(truck_xy) >= 2:
            scene_truck_points.append({"x": truck_xy[0], "y": truck_xy[1], "status": status, "episode": ep, "title": f"ep {ep} {status}: truck_body=({truck_xy[0]:.3g},{truck_xy[1]:.3g})"})
        if isinstance(unload_xy, list) and len(unload_xy) >= 2:
            scene_unload_points.append({"x": unload_xy[0], "y": unload_xy[1], "status": status, "episode": ep, "title": f"ep {ep} {status}: unload_body=({unload_xy[0]:.3g},{unload_xy[1]:.3g})"})

    files = []
    files.append(
        write_svg_outcome_stack(
            os.path.join(output_dir, "outcome_mix.svg"),
            "Outcome Mix",
            status_counts,
        )
    )
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "status_counts.svg"),
            "Episode Status Counts",
            [(key, counts.get(key, 0)) for key in ["all", "trainable", "success", "rejected", "failed", "diagnostic", "planning"]],
            left=170,
        )
    )
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "segment_counts.svg"),
            "Segment Success Counts",
            [(key, value) for key, value in segments.items()],
            left=180,
        )
    )
    if attempts > 0:
        files.append(
            write_svg_bar_chart(
                os.path.join(output_dir, "segment_rates.svg"),
                "Segment Rates vs Attempts",
                [(key, float(value) / max(1.0, attempts) * 100.0) for key, value in segments.items()],
                left=210,
            )
        )
    reason_pairs = counter_rows_to_pairs(compact.get("primary_reasons"), limit=12)
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "primary_reasons.svg"),
            "Primary Episode Reasons",
            reason_pairs,
            left=330,
        )
    )
    files.append(
        write_svg_pareto_chart(
            os.path.join(output_dir, "reason_pareto.svg"),
            "Primary Reason Pareto",
            reason_pairs,
            left=380,
        )
    )
    files.append(
        write_svg_status_by_category_chart(
            os.path.join(output_dir, "initial_pose_status_stack.svg"),
            "Initial Pose × Status",
            initial_pose_matrix,
            left=300,
        )
    )
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "initial_pose_status.svg"),
            "Initial Pose By Status",
            counter_rows_to_pairs(compact.get("initial_pose_by_status"), limit=14),
            left=260,
        )
    )
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "warning_counts.svg"),
            "Top Warning Counts",
            counter_rows_to_pairs(compact.get("top_warnings"), limit=12),
            left=330,
        )
    )
    files.append(
        write_svg_histogram(
            os.path.join(output_dir, "score_histogram.svg"),
            "Episode Score Histogram",
            [row.get("score") for row in all_rows],
            bins=12,
        )
    )
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "bucket_material_medians.svg"),
            "Trainable Material Medians",
            material_rows,
            left=210,
        )
    )
    files.append(
        write_svg_metric_scatter(
            os.path.join(output_dir, "bin_vs_spill_scatter.svg"),
            "Final Bin vs Spill",
            scatter_points,
            "final spill particles",
            "final bin particles",
        )
    )
    files.append(
        write_svg_metric_scatter(
            os.path.join(output_dir, "score_vs_spill_scatter.svg"),
            "Score vs Spill",
            score_vs_spill_points,
            "final spill particles",
            "episode score",
        )
    )
    files.append(
        write_svg_metric_scatter(
            os.path.join(output_dir, "sand_xy_scatter.svg"),
            "Sand XY Coverage",
            scene_sand_points,
            "sand body-frame x / forward (m)",
            "sand body-frame y / left (m)",
            include_origin=True,
        )
    )
    files.append(
        write_svg_metric_scatter(
            os.path.join(output_dir, "truck_xy_scatter.svg"),
            "Truck XY Coverage",
            scene_truck_points,
            "truck body-frame x / forward (m)",
            "truck body-frame y / left (m)",
            include_origin=True,
        )
    )
    files.append(
        write_svg_metric_scatter(
            os.path.join(output_dir, "unload_xy_scatter.svg"),
            "Unload XY Coverage",
            scene_unload_points,
            "unload body-frame x / forward (m)",
            "unload body-frame y / left (m)",
            include_origin=True,
        )
    )
    diagnosis = build_report_context(compact, report=report, all_rows=all_rows)
    files.append(write_html_report(os.path.join(output_dir, "report.html"), compact, files, report=report, all_rows=all_rows))
    files.append(write_text(os.path.join(output_dir, "analysis_compact.json"), json.dumps(compact, ensure_ascii=True, indent=2)))
    files.append(write_text(os.path.join(output_dir, "report_diagnosis.json"), json.dumps(diagnosis, ensure_ascii=True, indent=2)))
    return {"run_dir": run_dir, "output_dir": output_dir, "files": files}

def print_plots(run_dir: Union[str, os.PathLike], output_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    result = generate_plots(run_dir, output_dir=output_dir)
    print("[PLOTS]", json.dumps(result, ensure_ascii=True, indent=2))


def trajectory_columns(trajectory: List[dict]) -> Dict[str, list]:
    cols = {
        "t": [],
        "phase": [],
        "task": [],
        "observation.state": [],
        "observation.images.0": [],
        "observation.images.1": [],
        "observation.images.2": [],
        "observation.camera": [],
        "obs.state": [],
        "obs.q": [],
        "obs.dq": [],
        "obs.q_cmd": [],
        "obs.q_err": [],
        "action": [],
        "goal.q": [],
        "target": [],
        "bucket.tip": [],
        "bucket.load": [],
        "sand": [],
    }
    for row in trajectory:
        for key in cols:
            if key in LEROBOT_IMAGE_KEYS:
                cols[key].append(sample_image_value(row, key))
            else:
                cols[key].append(row.get(key))
    return cols


def write_json(path: Union[str, os.PathLike], data: object) -> str:
    return write_text(path, json.dumps(data, ensure_ascii=True, indent=2))


def write_jsonl(path: Union[str, os.PathLike], rows: Sequence[dict]) -> str:
    path = str(path)
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")))
            f.write("\n")
    return path


def vector_or_none(value: object, length: Optional[int] = None) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)):
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            return None
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return None
    if length is not None and len(out) != int(length):
        return None
    return out


def row_path_value(row: dict, key: str) -> str:
    value = row.get(key, "")
    return str(value or "")


def _path_parts_any_platform(path: object) -> List[str]:
    text = str(path or "").strip()
    if not text:
        return []
    return [part for part in text.replace("\\", "/").split("/") if part]


def _basename_any_platform(path: object) -> str:
    parts = _path_parts_any_platform(path)
    return parts[-1] if parts else ""


def _dirname_any_platform(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    if "\\" not in text:
        return os.path.dirname(text)
    parts = _path_parts_any_platform(text)
    if len(parts) <= 1:
        return ""
    return "/".join(parts[:-1])


def _is_windows_absolute_path(path: object) -> bool:
    text = str(path or "").strip()
    return bool(re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"))


def _normalize_client_path_for_server(path: object) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        text = unquote(text)
    except Exception:
        pass
    if os.sep == "/" and "\\" in text:
        text = text.replace("\\", "/")
    return os.path.abspath(os.path.expanduser(text))


def sample_image_value(sample: dict, canonical_key: str) -> object:
    for key in LEROBOT_IMAGE_KEY_ALIASES.get(canonical_key, [canonical_key]):
        value = sample.get(key)
        if value:
            return value
    return None


def episode_dir_from_row(row: dict, run_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    for key in ["transferred_episode_dir", "dest_episode_dir", "source_episode_dir"]:
        candidate = _normalize_client_path_for_server(row.get(key))
        if candidate and os.path.isdir(candidate):
            return candidate
    if run_dir:
        episodes_root = os.path.join(str(run_dir), "episodes")
        for key in ["transferred_episode_dir", "dest_episode_dir"]:
            folder = _basename_any_platform(row.get(key))
            if folder:
                candidate = os.path.join(episodes_root, folder)
                if os.path.isdir(candidate):
                    return candidate
    for key in ["trajectory", "meta", "score_path", "events"]:
        path = row_path_value(row, key)
        if path:
            if os.path.isabs(path) and os.path.isdir(os.path.dirname(path)):
                return os.path.dirname(path)
            dirname = _dirname_any_platform(path)
            if dirname:
                if run_dir:
                    parent_name = _basename_any_platform(dirname)
                    if parent_name:
                        candidate = os.path.join(str(run_dir), parent_name)
                        if os.path.isdir(candidate):
                            return candidate
                        candidate = os.path.join(str(run_dir), "episodes", parent_name)
                        if os.path.isdir(candidate):
                            return candidate
                normalized = _normalize_client_path_for_server(dirname)
                if os.path.isdir(normalized):
                    return normalized
                return dirname
    return ""


def resolve_episode_file(episode_dir: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if os.path.isabs(text) and os.path.exists(text):
        return text
    normalized = _normalize_client_path_for_server(text)
    if os.path.isabs(text) and os.path.exists(normalized):
        return normalized
    basename = _basename_any_platform(text)
    if _is_windows_absolute_path(text) and episode_dir and basename:
        candidate = os.path.join(episode_dir, basename)
        if os.path.exists(candidate):
            return candidate
        return os.path.normpath(candidate)
    if episode_dir:
        rel_text = text.replace("\\", os.sep) if os.sep == "/" else text.replace("/", os.sep)
        candidate = os.path.normpath(os.path.join(episode_dir, rel_text))
        if os.path.exists(candidate):
            return candidate
        if basename:
            basename_candidate = os.path.join(episode_dir, basename)
            if os.path.exists(basename_candidate):
                return basename_candidate
        return candidate
    return normalized


def relpath_posix(path: Union[str, os.PathLike], base: Union[str, os.PathLike]) -> str:
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def safe_copy_file(src: str, dst: str) -> bool:
    if not src or not os.path.isfile(src):
        return False
    ensure_dir(os.path.dirname(dst) or ".")
    shutil.copy2(src, dst)
    return True


def _require_shared_dataset_tool(name: str):
    tool = getattr(shared_dataset_tools, name, None) if shared_dataset_tools is not None else None
    if tool is None:
        raise RuntimeError(f"shared VLA exporter unavailable: excavator_dataset_tools.{name}")
    return tool


def infer_export_fps(run_dir: Union[str, os.PathLike], explicit_fps: Optional[float] = None) -> float:
    return float(_require_shared_dataset_tool("infer_export_fps")(run_dir, explicit_fps))


def export_lerobot_dataset(
    run_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    split: str = "trainable",
    limit_episodes: Optional[int] = None,
    overwrite: bool = False,
    require_standard: bool = False,
    require_vla: bool = False,
    progress_callback=None,
    time_policy: object = None,
) -> Dict[str, object]:
    return _require_shared_dataset_tool("export_lerobot_dataset")(
        run_dir,
        output_dir=output_dir,
        split=split,
        limit_episodes=limit_episodes,
        overwrite=overwrite,
        require_standard=require_standard,
        require_vla=require_vla,
        progress_callback=progress_callback,
        time_policy=time_policy,
    )


def print_lerobot_export(
    run_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    split: str = "trainable",
    limit_episodes: Optional[int] = None,
    overwrite: bool = False,
    require_standard: bool = False,
    require_vla: bool = False,
    time_policy: object = None,
) -> None:
    result = export_lerobot_dataset(
        run_dir,
        output_dir=output_dir,
        split=split,
        limit_episodes=limit_episodes,
        overwrite=overwrite,
        require_standard=require_standard,
        require_vla=require_vla,
        time_policy=time_policy,
    )
    try:
        write_json(os.path.join(os.path.abspath(str(run_dir)), "lerobot_v3_export.json"), result)
    except Exception:
        pass
    print("[LEROBOT EXPORT]", json.dumps(result, ensure_ascii=True, indent=2))

def nested_dict_value(data: object, path: Sequence[str], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def vector_xy(value: object) -> Optional[List[float]]:
    vec = vector_or_none(value)
    if vec is None or len(vec) < 2:
        return None
    return [float(vec[0]), float(vec[1])]


def vector_xyz(value: object) -> Optional[List[float]]:
    vec = vector_or_none(value)
    if vec is None or len(vec) < 3:
        return None
    return [float(vec[0]), float(vec[1]), float(vec[2])]


def first_vector_xy(*values: object) -> Optional[List[float]]:
    for value in values:
        xy = vector_xy(value)
        if xy is not None:
            return xy
    return None


def polygon_xy_or_empty(value: object) -> List[List[float]]:
    if not isinstance(value, list):
        return []
    out: List[List[float]] = []
    for item in value:
        xy = vector_xy(item)
        if xy is not None:
            out.append(xy)
    return out


def normalize_degrees(value: object) -> Optional[float]:
    number = safe_float_value(value)
    if number is None:
        return None
    while number <= -180.0:
        number += 360.0
    while number > 180.0:
        number -= 360.0
    return number


def xy_to_body_frame(
    point_xy: object,
    robot_origin_xy: object = None,
    robot_body_yaw_deg: object = 0.0,
) -> Optional[List[float]]:
    point = vector_xy(point_xy)
    if point is None:
        return None
    origin = vector_xy(robot_origin_xy) or [0.0, 0.0]
    yaw = safe_float_value(robot_body_yaw_deg, 0.0) or 0.0
    angle = math.radians(yaw)
    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    c = math.cos(-angle)
    s = math.sin(-angle)
    return [dx * c - dy * s, dx * s + dy * c]


def polygon_to_body_frame(
    polygon_xy: object,
    robot_origin_xy: object = None,
    robot_body_yaw_deg: object = 0.0,
) -> List[List[float]]:
    polygon = polygon_xy_or_empty(polygon_xy)
    out: List[List[float]] = []
    for point in polygon:
        local = xy_to_body_frame(point, robot_origin_xy, robot_body_yaw_deg)
        if local is not None:
            out.append(local)
    return out




# v10: training-dataset status semantics.
# This dashboard is a dataset-generation control surface, not only a runtime log viewer.
# Rows that cannot become complete training episodes are surfaced as tag:skip instead
# of silently disappearing from the dataset-quality view.
def trajectory_runtime_s(trajectory: Sequence[dict]) -> float:
    if not trajectory:
        return 0.0
    t0 = safe_float_value(trajectory[0].get("t"), None)
    t1 = safe_float_value(trajectory[-1].get("t"), None)
    if t0 is not None and t1 is not None:
        return max(0.0, float(t1) - float(t0))
    return 0.0


def sample_has_state_action(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return False
    state = sample.get("observation.state") if sample.get("observation.state") is not None else sample.get("obs.state")
    action = sample.get("action")
    return vector_or_none(state) is not None and vector_or_none(action) is not None


def sample_has_required_cameras(sample: dict) -> bool:
    if not isinstance(sample, dict):
        return False
    # Dashboard triage checks whether camera fields are present; the shared exporter
    # performs the stricter file-exists validation.
    return all(bool(sample_image_value(sample, key)) for key in LEROBOT_IMAGE_KEYS)


def dataset_training_tag_for_row(row: dict, trajectory: Optional[Sequence[dict]] = None) -> Tuple[str, str]:
    raw_status = normalized_status_name(row.get("status", "unknown"))
    if trajectory is None:
        trajectory = load_trajectory(row)
    if not trajectory:
        return "skip", "trajectory_empty"
    # Scan a bounded prefix: enough to catch missing schema/camera without loading every
    # frame twice for large runs. The full export path remains the final authority.
    probe = list(trajectory[: min(len(trajectory), 32)])
    has_state_action = any(sample_has_state_action(sample) for sample in probe)
    has_camera_fields = any(sample_has_required_cameras(sample) for sample in probe)
    if raw_status in {"trainable", "success"} and not has_state_action:
        return "skip", "missing_state_or_action"
    if raw_status in {"trainable", "success"} and not has_camera_fields:
        return "skip", "missing_camera_fields"
    return raw_status, "ok"


def estimate_row_runtime_s(row: dict, trajectory: Optional[Sequence[dict]] = None) -> float:
    if trajectory is None:
        trajectory = load_trajectory(row)
    runtime = trajectory_runtime_s(trajectory or [])
    if runtime > 0:
        return runtime
    for key in ["duration_s", "wall_duration_s", "elapsed_s", "runtime_s", "episode_duration_s"]:
        value = safe_float_value(row.get(key), None)
        if value is not None and value > 0:
            return float(value)
    samples = safe_float_value(row.get("samples"), None)
    if samples is not None and samples > 0:
        # Camera/default collection frequency is commonly 10 Hz in this project.
        return float(samples) / 10.0
    return 0.0


def compute_dataset_generation_metrics(rows: Sequence[dict]) -> Dict[str, object]:
    tag_counts: Counter = Counter()
    raw_status_counts: Counter = Counter()
    skip_reasons: Counter = Counter()
    tag_runtime: Counter = Counter()
    total_frames = 0
    usable_frames = 0
    skip_examples: List[dict] = []
    for row in rows:
        trajectory = load_trajectory(row)
        tag, reason = dataset_training_tag_for_row(row, trajectory)
        raw_status = normalized_status_name(row.get("status", "unknown"))
        runtime = estimate_row_runtime_s(row, trajectory)
        frames = len(trajectory) if trajectory else int(safe_float_value(row.get("samples"), 0.0) or 0)
        tag_counts[tag] += 1
        raw_status_counts[raw_status] += 1
        tag_runtime[tag] += float(runtime)
        total_frames += int(max(0, frames))
        if tag in {"trainable", "success"}:
            usable_frames += int(max(0, frames))
        if tag == "skip":
            skip_reasons[reason] += 1
            if len(skip_examples) < 12:
                skip_examples.append({
                    "episode_index": row.get("episode_index"),
                    "episode_id": row.get("episode_id", ""),
                    "raw_status": raw_status,
                    "skip_reason": reason,
                    "trajectory": row.get("trajectory", ""),
                    "samples": row.get("samples"),
                })
    attempts = len(rows)
    usable_episodes = int(tag_counts.get("trainable", 0) + tag_counts.get("success", 0))
    skipped_episodes = int(tag_counts.get("skip", 0))
    total_runtime = sum(float(v) for v in tag_runtime.values())
    efficiency = float(usable_frames) / float(max(1, total_frames))
    stability = 1.0 - (float(skipped_episodes) / float(max(1, attempts)))
    return {
        "tag_counts": dict(tag_counts),
        "raw_status_counts": dict(raw_status_counts),
        "tag_runtime_seconds": {key: float(value) for key, value in tag_runtime.items()},
        "tag_runtime_total_seconds": float(total_runtime),
        "skip_reasons": top_counter(skip_reasons, 12),
        "skip_examples": skip_examples,
        "attempts": attempts,
        "usable_episodes": usable_episodes,
        "skipped_episodes": skipped_episodes,
        "total_frames": total_frames,
        "usable_frames": usable_frames,
        "data_efficiency_score": int(round(max(0.0, min(1.0, efficiency)) * 100.0)),
        "stability_score": int(round(max(0.0, min(1.0, stability)) * 100.0)),
    }

def dashboard_scene_from_episode(row: dict) -> Dict[str, object]:
    scene = row.get("scene_randomization") if isinstance(row.get("scene_randomization"), dict) else {}
    candidate = scene.get("candidate") if isinstance(scene.get("candidate"), dict) else {}
    applied = scene.get("applied") if isinstance(scene.get("applied"), dict) else {}
    scene_context = applied.get("scene_context") if isinstance(applied.get("scene_context"), dict) else {}

    # Raw XY values here are world-frame values from the simulator/randomizer.
    # The dashboard additionally exposes excavator body-frame XY values so plots remain
    # meaningful when robot_body_yaw_deg is randomized at init/reset time.
    sand_xy = (
        vector_xy(applied.get("sand_center"))
        or vector_xy(candidate.get("sand_xy"))
        or vector_xy(row.get("target_xyz"))
    )
    truck_xy = (
        vector_xy(applied.get("truck_translation_xyz"))
        or vector_xy(candidate.get("truck_center_xy"))
    )
    unload_xy = (
        vector_xy(scene_context.get("unload_bin_center"))
        or vector_xy(candidate.get("unload_xy"))
        or vector_xy(row.get("unload_landing_xyz"))
    )
    unload_landing_xy = vector_xy(row.get("unload_landing_xyz"))
    unload_point_xy = vector_xy(row.get("unload_point_xyz")) or vector_xy(row.get("unload_release_xyz"))

    robot_origin_xy = (
        first_vector_xy(
            applied.get("robot_origin_xy"),
            applied.get("robot_body_xy"),
            applied.get("robot_body_xyz"),
            applied.get("robot_translation_xyz"),
            applied.get("robot_base_xy"),
            applied.get("robot_base_xyz"),
            applied.get("excavator_origin_xy"),
            applied.get("excavator_translation_xyz"),
            scene_context.get("robot_origin_xy"),
            scene_context.get("excavator_origin_xy"),
            candidate.get("robot_origin_xy"),
            candidate.get("robot_xy"),
            candidate.get("robot_xyz"),
            row.get("robot_origin_xy"),
            row.get("robot_origin_xyz"),
            row.get("base_xy"),
            row.get("base_xyz"),
        )
        or [0.0, 0.0]
    )
    robot_body_yaw_deg = safe_float_value(
        applied.get("robot_body_yaw_deg", candidate.get("robot_body_yaw_deg")), 0.0
    )
    truck_yaw_deg = safe_float_value(applied.get("truck_yaw_deg", candidate.get("truck_yaw_deg")))
    truck_yaw_body_deg = normalize_degrees(
        (truck_yaw_deg if truck_yaw_deg is not None else 0.0)
        - (robot_body_yaw_deg if robot_body_yaw_deg is not None else 0.0)
    ) if truck_yaw_deg is not None else None

    unload_polygon_xy = polygon_xy_or_empty(scene_context.get("unload_polygon_xy"))
    unload_hull_xy = polygon_xy_or_empty(scene_context.get("unload_hull_xy"))

    return {
        "episode_index": row.get("episode_index"),
        "episode_id": row.get("episode_id"),
        "status": row.get("status"),
        "score": row.get("score"),
        "scene_xy_frame": "world_xy",
        "body_xy_frame": "excavator_body_xy",
        "body_x_axis": "excavator forward after init yaw",
        "body_y_axis": "excavator left after init yaw",
        "robot_origin_xy": robot_origin_xy,
        "robot_body_yaw_deg": robot_body_yaw_deg,
        "sand_xy": sand_xy,
        "truck_xy": truck_xy,
        "unload_xy": unload_xy,
        "unload_landing_xy": unload_landing_xy,
        "unload_point_xy": unload_point_xy,
        "sand_body_xy": xy_to_body_frame(sand_xy, robot_origin_xy, robot_body_yaw_deg),
        "truck_body_xy": xy_to_body_frame(truck_xy, robot_origin_xy, robot_body_yaw_deg),
        "unload_body_xy": xy_to_body_frame(unload_xy, robot_origin_xy, robot_body_yaw_deg),
        "unload_landing_body_xy": xy_to_body_frame(unload_landing_xy, robot_origin_xy, robot_body_yaw_deg),
        "unload_point_body_xy": xy_to_body_frame(unload_point_xy, robot_origin_xy, robot_body_yaw_deg),
        "sand_amount_multiplier": safe_float_value(
            applied.get("sand_amount_multiplier", candidate.get("sand_amount_multiplier"))
        ),
        "estimated_particle_count": safe_float_value(
            applied.get("estimated_particle_count", candidate.get("estimated_particle_count"))
        ),
        "truck_yaw_deg": truck_yaw_deg,
        "truck_yaw_body_deg": truck_yaw_body_deg,
        "truck_radius_m": safe_float_value(candidate.get("truck_radius_m")),
        "unload_radius_m": safe_float_value(candidate.get("unload_radius_m")),
        "sand_radius_m": safe_float_value(candidate.get("sand_radius_m")),
        "unload_polygon_xy": unload_polygon_xy,
        "unload_hull_xy": unload_hull_xy,
        "unload_polygon_body_xy": polygon_to_body_frame(unload_polygon_xy, robot_origin_xy, robot_body_yaw_deg),
        "unload_hull_body_xy": polygon_to_body_frame(unload_hull_xy, robot_origin_xy, robot_body_yaw_deg),
        "unload_shape": scene_context.get("manual_unload_range_shape"),
        "unload_mesh": scene_context.get("manual_unload_selected_path"),
    }

def dashboard_episode_summary(
    row: dict,
    dataset_tag: Optional[str] = None,
    dataset_skip_reason: str = "",
    runtime_s: Optional[float] = None,
) -> Dict[str, object]:
    scene = dashboard_scene_from_episode(row)
    tag = dataset_tag or normalized_status_name(row.get("status", "unknown"))
    scene["raw_status"] = row.get("status")
    scene["status"] = tag
    scene["dataset_skip_reason"] = dataset_skip_reason
    return {
        "episode_index": row.get("episode_index"),
        "episode_id": row.get("episode_id"),
        "status": tag,
        "raw_status": row.get("status"),
        "dataset_skip_reason": dataset_skip_reason,
        "score": row.get("score"),
        "reason": row.get("reason", ""),
        "warning_reason": row.get("warning_reason", ""),
        "task_prompt": dashboard_episode_task_prompt(row),
        "samples": row.get("samples"),
        "time_s": float(runtime_s if runtime_s is not None else estimate_row_runtime_s(row)),
        "freeze_count": row.get("freeze_count"),
        "max_bucket": row.get("max_bucket_from_pile_particles"),
        "lift_bucket": row.get("lift_bucket_from_pile_particles"),
        "final_bin": row.get("final_bin_from_pile_particles"),
        "final_spill": row.get("final_spill_from_pile_particles"),
        "initial_pose_id": row.get("initial_pose_id"),
        "q_initial_deg": row.get("q_initial_deg"),
        "target_xyz": row.get("target_xyz"),
        "unload_landing_xyz": row.get("unload_landing_xyz"),
        "scene": scene,
    }


def dashboard_episode_task_prompt(
    row: dict,
    trajectory: Optional[Sequence[dict]] = None,
    episode_meta: Optional[dict] = None,
) -> str:
    first_sample = trajectory[0] if trajectory else {}
    meta = episode_meta if isinstance(episode_meta, dict) else {}
    if not meta:
        try:
            episode_dir = episode_dir_from_row(row)
            meta = read_json(resolve_episode_file(episode_dir, row_path_value(row, "meta")), default={}) or {}
        except Exception:
            meta = {}
    shared_builder = getattr(shared_dataset_tools, "build_episode_task_text", None) if shared_dataset_tools is not None else None
    if shared_builder is not None:
        try:
            return str(shared_builder(first_sample, meta, row)).strip()
        except Exception:
            pass
    for source in [first_sample, meta, row]:
        if isinstance(source, dict):
            for key in ["task", "dataset_task_text"]:
                text = str(source.get(key) or "").strip()
                if text:
                    return text
    return "Dig soil from the marked area and dump it into the target container."


RUN_ACTIVITY_ACTIVE_SECONDS = 180.0
RUN_ACTIVITY_RECENT_SECONDS = 900.0
RUN_ACTIVITY_EPISODE_DIR_LIMIT = 32
RUN_ACTIVITY_ROOT_FILES = set(INDEX_FILES.values()) | set(SEGMENT_FILES.values()) | {
    "summary.json",
    "run_meta.json",
    "camera_config.json",
    "debug_timeline.jsonl",
    "lerobot_v3_export.json",
}


def _update_latest_mtime(latest: Tuple[float, str], path: str) -> Tuple[float, str]:
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        return latest
    if mtime > latest[0]:
        return float(mtime), path
    return latest


def run_activity_snapshot(
    run_dir: Union[str, os.PathLike],
    now: Optional[float] = None,
    active_seconds: float = RUN_ACTIVITY_ACTIVE_SECONDS,
    recent_seconds: float = RUN_ACTIVITY_RECENT_SECONDS,
) -> Dict[str, object]:
    """Return a cheap filesystem-based write monitor for a run folder.

    We intentionally do not try to inspect Isaac Sim internals. For this training
    data generator, the robust signal is whether run jsonl/summary/trajectory
    files are still being modified. The scan is bounded so a dashboard refresh
    does not walk image folders or huge exports.
    """
    run_dir = os.path.abspath(str(run_dir))
    now = float(time.time() if now is None else now)
    latest: Tuple[float, str] = (0.0, "")
    if not os.path.isdir(run_dir):
        return {
            "state": "missing",
            "active": False,
            "recent": False,
            "age_s": None,
            "latest_mtime": 0.0,
            "latest_file": "",
            "active_window_s": float(active_seconds),
        }
    latest = _update_latest_mtime(latest, run_dir)
    try:
        entries = list(os.scandir(run_dir))
    except Exception:
        entries = []

    episode_dirs = []
    for entry in entries:
        try:
            if entry.is_file():
                name = entry.name
                if name in RUN_ACTIVITY_ROOT_FILES or name.endswith((".json", ".jsonl", ".log")):
                    latest = _update_latest_mtime(latest, entry.path)
            elif entry.is_dir():
                dir_mtime = entry.stat().st_mtime
                latest = _update_latest_mtime(latest, entry.path)
                if entry.name.startswith(("episode_", "attempt_")):
                    episode_dirs.append((float(dir_mtime), entry.path))
        except Exception:
            continue

    # Only inspect the newest episode/attempt folders. Those are the ones that
    # can contain a trajectory jsonl currently being appended by Isaac Sim.
    episode_dirs.sort(reverse=True)
    for _, ep_dir in episode_dirs[:RUN_ACTIVITY_EPISODE_DIR_LIMIT]:
        try:
            for child in os.scandir(ep_dir):
                try:
                    if child.is_file() and child.name.endswith((".json", ".jsonl", ".log", ".txt")):
                        latest = _update_latest_mtime(latest, child.path)
                except Exception:
                    continue
        except Exception:
            continue

    latest_mtime, latest_path = latest
    if latest_mtime <= 0:
        age_s = None
        state = "unknown"
        active = False
        recent = False
    else:
        age_s = max(0.0, now - float(latest_mtime))
        active = age_s <= float(active_seconds)
        recent = age_s <= float(recent_seconds)
        state = "active" if active else ("recent" if recent else "idle")
    try:
        latest_file = relpath_posix(latest_path, run_dir) if latest_path else ""
    except Exception:
        latest_file = latest_path
    return {
        "state": state,
        "active": bool(active),
        "recent": bool(recent),
        "age_s": age_s,
        "latest_mtime": float(latest_mtime or 0.0),
        "latest_file": latest_file,
        "active_window_s": float(active_seconds),
    }


def summarize_run_activity(runs: Sequence[dict]) -> Dict[str, object]:
    active = [run for run in runs if isinstance(run.get("activity"), dict) and run["activity"].get("active")]
    recent = [run for run in runs if isinstance(run.get("activity"), dict) and run["activity"].get("state") == "recent"]
    return {
        "active_count": len(active),
        "recent_count": len(recent),
        "idle_count": max(0, len(runs) - len(active) - len(recent)),
        "active_window_s": RUN_ACTIVITY_ACTIVE_SECONDS,
        "recent_window_s": RUN_ACTIVITY_RECENT_SECONDS,
        "active_runs": [
            {
                "name": run.get("name"),
                "path": run.get("path"),
                "age_s": (run.get("activity") or {}).get("age_s"),
                "latest_file": (run.get("activity") or {}).get("latest_file"),
                "attempts": run.get("attempts"),
                "success": run.get("success"),
                "trainable": run.get("trainable"),
                "size_human": run.get("size_human"),
            }
            for run in active[:8]
        ],
    }


# Folder size accounting is intentionally cache-first.
# A full recursive size scan is O(number_of_files), which is too expensive for
# a training data factory where every run may contain thousands of frames.  The
# dashboard list endpoint must stay O(number_of_runs): it reads summary files,
# activity mtimes and cached sizes only.  Exact folder sizes are computed only
# when the user explicitly clicks "Refresh selected sizes".
RUN_SIZE_CACHE_TTL = 7 * 24 * 3600.0
RUN_SIZE_MEMORY_CACHE: Dict[str, Dict[str, object]] = {}
RUN_SIZE_CACHE_LOADED_ROOTS = set()
RUN_SIZE_CACHE_DIRNAME = ".dashboard_cache"
RUN_SIZE_CACHE_FILENAME = "folder_size_cache.json"
RUN_SIZE_CACHE_VERSION = 2
RUN_SIZE_EXCLUDE_DIRS = {".dashboard_cache", "__pycache__"}
RUN_DATA_SIZE_LIMIT = 8
RUN_PAYLOAD_CACHE_VERSION = 3
RUN_PAYLOAD_CACHE_DIRNAME = "run_payload_cache"
RUN_EPISODE_ANALYSIS_CACHE_DIRNAME = "run_episode_analysis_cache"
RUN_PAYLOAD_MEMORY_CACHE: Dict[str, Dict[str, object]] = {}
RUN_PAYLOAD_CACHE_LOCK = threading.Lock()
FRAME_CONTEXT_CACHE_TTL = 120.0
FRAME_CONTEXT_CACHE_LIMIT = 16
FRAME_CONTEXT_CACHE: Dict[Tuple[str, str], Dict[str, object]] = {}
FRAME_CONTEXT_CACHE_LOCK = threading.Lock()
FRAME_IMAGE_CACHE_TTL = 60.0
FRAME_IMAGE_CACHE_LIMIT = 512
FRAME_IMAGE_CACHE: Dict[str, Dict[str, object]] = {}
FRAME_IMAGE_CACHE_LOCK = threading.Lock()


def fast_jsonl_count(path: Union[str, os.PathLike]) -> int:
    count = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except FileNotFoundError:
        return 0
    except Exception:
        return 0
    return count




def success_index_catchup_counts(
    run_dir: Union[str, os.PathLike],
    counts: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, int], Dict[str, object]]:
    """Recover dashboard counts when disk-full shutdown left only successful_episodes.jsonl.

    Some interrupted runs can have empty/missing episodes.jsonl and
    trainable_episodes.jsonl while successful_episodes.jsonl was already flushed.
    For dashboard/export management, those rows are still valuable training data.
    This helper keeps raw counts visible while providing effective counts that do
    not display attempts/trainable as zero when success rows exist.
    """
    run_dir = os.path.abspath(str(run_dir))
    raw: Dict[str, int] = {}
    if counts is not None:
        for key in INDEX_FILES:
            try:
                raw[key] = int(counts.get(key, 0) or 0)
            except Exception:
                raw[key] = 0
    else:
        for key in INDEX_FILES:
            raw[key] = fast_jsonl_count(index_path(run_dir, key))
    effective = dict(raw)
    success = int(raw.get("success", 0) or 0)
    reasons: List[str] = []
    if success > 0 and int(raw.get("all", 0) or 0) <= 0:
        effective["all"] = success
        reasons.append("episodes_index_missing_success_recovered")
    if success > 0 and int(raw.get("trainable", 0) or 0) <= 0:
        effective["trainable"] = success
        reasons.append("trainable_index_missing_success_recovered")
    active = bool(reasons)
    catchup = {
        "active": active,
        "source": "successful_episodes.jsonl" if active else "normal_indexes",
        "reason": ";".join(reasons),
        "raw_counts": raw,
        "effective_counts": effective,
        "message": (
            "Recovered effective attempts/trainable from successful_episodes.jsonl. "
            "This usually means the run was interrupted after success rows were flushed "
            "but before episodes/trainable indexes or summary were completed."
        ) if active else "",
    }
    return effective, catchup


def load_dashboard_all_rows(run_dir: Union[str, os.PathLike]) -> Tuple[List[dict], Dict[str, object]]:
    """Load rows for dashboard inspection with success-index catch-up.

    Normal runs use episodes.jsonl.  If that index is empty but
    successful_episodes.jsonl contains rows, use the success index so the run can
    still be inspected, selected, pooled, and exported.
    """
    run_dir = os.path.abspath(str(run_dir))
    rows = load_index(run_dir, "all")
    raw_counts = {key: fast_jsonl_count(index_path(run_dir, key)) for key in INDEX_FILES}
    _, catchup = success_index_catchup_counts(run_dir, raw_counts)
    if rows:
        catchup = dict(catchup)
        catchup["rows_source"] = "episodes.jsonl"
        return rows, catchup
    success_rows = load_index(run_dir, "success")
    if success_rows:
        catchup = dict(catchup)
        catchup["active"] = True
        catchup["rows_source"] = "successful_episodes.jsonl"
        catchup["reason"] = catchup.get("reason") or "episodes_index_missing_success_rows_used"
        catchup["message"] = catchup.get("message") or "Using successful_episodes.jsonl because episodes.jsonl is empty."
        return success_rows, catchup
    catchup = dict(catchup)
    catchup["rows_source"] = "episodes.jsonl"
    return rows, catchup

def format_bytes(value: object) -> str:
    try:
        n = float(value)
    except Exception:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while n >= 1024.0 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(n)} {units[idx]}"
    return f"{n:.1f} {units[idx]}"


def folder_size_cache_path(dataset_root: Union[str, os.PathLike]) -> str:
    return os.path.join(os.path.abspath(str(dataset_root)), RUN_SIZE_CACHE_DIRNAME, RUN_SIZE_CACHE_FILENAME)


def folder_size_cache_key(path: Union[str, os.PathLike]) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def load_folder_size_cache(dataset_root: Union[str, os.PathLike]) -> None:
    root = os.path.abspath(str(dataset_root or ""))
    if not root or root in RUN_SIZE_CACHE_LOADED_ROOTS:
        return
    RUN_SIZE_CACHE_LOADED_ROOTS.add(root)
    payload = read_json(folder_size_cache_path(root), default={}) or {}
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) != RUN_SIZE_CACHE_VERSION:
        return
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return
    for key, value in entries.items():
        if isinstance(value, dict):
            RUN_SIZE_MEMORY_CACHE[str(key)] = dict(value)


def save_folder_size_cache(dataset_root: Union[str, os.PathLike]) -> None:
    root = os.path.abspath(str(dataset_root or ""))
    if not root:
        return
    try:
        ensure_dir(os.path.dirname(folder_size_cache_path(root)))
        # Keep only entries still under this dataset root to avoid the cache
        # growing unbounded when users switch roots.
        entries = {}
        for key, value in RUN_SIZE_MEMORY_CACHE.items():
            path = str(value.get("path") or key)
            try:
                if os.path.commonpath([root, os.path.abspath(path)]) != root:
                    continue
            except Exception:
                continue
            entries[key] = value
        write_json(folder_size_cache_path(root), {
            "version": RUN_SIZE_CACHE_VERSION,
            "created_at": time.time(),
            "root": root,
            "entries": entries,
        })
    except Exception:
        pass


def folder_signature(path: Union[str, os.PathLike]) -> Dict[str, object]:
    """Fast invalidation signature.

    Directory size is not available as a cheap OS metadata field on Windows or
    POSIX.  File managers get folder sizes by scanning or by using their own
    caches.  This signature uses only O(1) metadata plus the latest known writer
    mtime for run folders.  It is good enough to decide whether a cached exact
    size is stale without walking the full tree on every page load.
    """
    root = os.path.abspath(str(path))
    try:
        stat = os.stat(root)
        mtime = float(stat.st_mtime)
    except Exception:
        mtime = 0.0
    latest_mtime = mtime
    try:
        if os.path.basename(root).startswith("run_"):
            activity = run_activity_snapshot(root)
            latest_mtime = max(latest_mtime, float(activity.get("latest_mtime") or 0.0))
    except Exception:
        pass
    return {"path_mtime": mtime, "latest_mtime": latest_mtime}


def cached_folder_size_snapshot(path: Union[str, os.PathLike], dataset_root: Optional[Union[str, os.PathLike]] = None) -> Optional[Dict[str, object]]:
    if dataset_root is not None:
        load_folder_size_cache(dataset_root)
    key = folder_size_cache_key(path)
    cache = RUN_SIZE_MEMORY_CACHE.get(key)
    if not isinstance(cache, dict):
        return None
    sig = folder_signature(path)
    cached_latest = float(cache.get("latest_mtime", 0.0) or 0.0)
    # If files are still being written, keep serving the cache but mark it stale;
    # scanning active folders is exactly what makes the dashboard hang.
    stale = bool(float(sig.get("latest_mtime", 0.0) or 0.0) > cached_latest + 0.001)
    out = dict(cache)
    out["stale"] = stale
    out["source"] = "cache_stale" if stale else "cache"
    out["size_human"] = format_bytes(out.get("size_bytes", 0)) + (" (stale)" if stale else "")
    return out


def shallow_folder_size_snapshot(path: Union[str, os.PathLike]) -> Dict[str, object]:
    """Cheap fallback: direct child files only, no recursion."""
    root = os.path.abspath(str(path))
    total = 0
    files = 0
    dirs = 0
    try:
        for entry in os.scandir(root):
            try:
                if entry.is_file():
                    files += 1
                    total += int(entry.stat().st_size)
                elif entry.is_dir():
                    dirs += 1
            except Exception:
                pass
    except Exception:
        pass
    return {
        "path": root,
        "size_bytes": int(total),
        "size_human": f"metadata only · {format_bytes(total)}",
        "file_count": int(files),
        "dir_count": int(dirs),
        "complete": False,
        "truncated": True,
        "stale": True,
        "source": "metadata_only",
        "cached_at": 0.0,
        **folder_signature(root),
    }


def exact_directory_size_snapshot(path: Union[str, os.PathLike], dataset_root: Optional[Union[str, os.PathLike]] = None) -> Dict[str, object]:
    root = os.path.abspath(str(path))
    total = 0
    files = 0
    dirs = 0
    stack = [root]
    while stack:
        base = stack.pop()
        try:
            with os.scandir(base) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in RUN_SIZE_EXCLUDE_DIRS:
                                continue
                            dirs += 1
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            files += 1
                            total += int(entry.stat(follow_symlinks=False).st_size)
                    except Exception:
                        pass
        except Exception:
            pass
    now = time.time()
    sig = folder_signature(root)
    result = {
        "path": root,
        "size_bytes": int(total),
        "size_human": format_bytes(total),
        "file_count": int(files),
        "dir_count": int(dirs),
        "complete": True,
        "truncated": False,
        "stale": False,
        "source": "exact_scan",
        "cached_at": float(now),
        **sig,
    }
    RUN_SIZE_MEMORY_CACHE[folder_size_cache_key(root)] = dict(result)
    if dataset_root is not None:
        save_folder_size_cache(dataset_root)
    return result


def directory_size_snapshot(
    path: Union[str, os.PathLike],
    dataset_root: Optional[Union[str, os.PathLike]] = None,
    force_refresh: bool = False,
    exact: bool = False,
    max_files: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> Dict[str, object]:
    """Cache-first folder size snapshot.

    The old implementation walked directories during /api/runs.  This function
    keeps the name for compatibility, but default behavior is non-recursive:
    cached exact size if available, otherwise a cheap metadata-only fallback.
    Recursive exact size is performed only with force_refresh=True or exact=True.
    max_files/max_seconds are accepted for backward compatibility and ignored.
    """
    root = os.path.abspath(str(path))
    if force_refresh or exact:
        return exact_directory_size_snapshot(root, dataset_root=dataset_root)
    cached = cached_folder_size_snapshot(root, dataset_root=dataset_root)
    if cached is not None:
        return cached
    return shallow_folder_size_snapshot(root)


def run_data_folder_sizes(
    run_dir: Union[str, os.PathLike],
    limit: int = RUN_DATA_SIZE_LIMIT,
    dataset_root: Optional[Union[str, os.PathLike]] = None,
    refresh: bool = False,
    max_files: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> List[dict]:
    run_dir = os.path.abspath(str(run_dir))
    candidates = []
    try:
        for entry in os.scandir(run_dir):
            if entry.is_dir() and entry.name.lower().startswith("data"):
                candidates.append(entry.path)
    except Exception:
        pass
    lerobot_data = os.path.join(run_dir, LEROBOT_DEFAULT_EXPORT_DIRNAME, "data")
    if os.path.isdir(lerobot_data):
        candidates.append(lerobot_data)
    out = []
    seen = set()
    for path in candidates[: max(0, int(limit))]:
        norm = os.path.normcase(os.path.abspath(path))
        if norm in seen:
            continue
        seen.add(norm)
        snap = directory_size_snapshot(path, dataset_root=dataset_root, force_refresh=refresh, exact=refresh)
        out.append({
            "name": relpath_posix(path, run_dir),
            "path": path,
            "size_bytes": snap.get("size_bytes", 0),
            "size_human": snap.get("size_human", "-"),
            "file_count": snap.get("file_count", 0),
            "truncated": snap.get("truncated", False),
            "stale": snap.get("stale", False),
            "source": snap.get("source", "unknown"),
        })
    return out


def dashboard_refresh_folder_sizes(dataset_root: Union[str, os.PathLike], run_paths: Sequence[object]) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    refreshed = []
    skipped = []
    for raw_path in run_paths or []:
        run_dir = os.path.abspath(str(raw_path))
        if not run_is_under_root(root, run_dir) or not os.path.isdir(run_dir):
            skipped.append({"path": run_dir, "reason": "not_a_run_under_root"})
            continue
        activity = run_activity_snapshot(run_dir)
        if activity.get("active"):
            # Do not scan folders currently being written.  The cached value will
            # be marked stale until the run becomes idle.
            skipped.append({"path": run_dir, "reason": "active_writer"})
            continue
        run_snap = directory_size_snapshot(run_dir, dataset_root=root, force_refresh=True, exact=True)
        data_snaps = run_data_folder_sizes(run_dir, dataset_root=root, refresh=True)
        refreshed.append({
            "name": os.path.basename(run_dir),
            "path": run_dir,
            "size_human": run_snap.get("size_human"),
            "file_count": run_snap.get("file_count"),
            "data_folders": data_snaps,
        })
    save_folder_size_cache(root)
    return {"ok": True, "root": root, "refreshed": refreshed, "skipped": skipped, "cache_path": folder_size_cache_path(root)}


def normalize_dashboard_client_path(path: Union[str, os.PathLike, None]) -> str:
    """Normalize paths supplied by the browser before filesystem use.

    The dashboard HTML runs in a browser and cannot know the server OS reliably.
    Older UI code always used Windows backslashes for .dashboard_success.  On
    Linux/POSIX, backslash is a literal filename character, not a separator, so
    a path like ``excavator_auto_dataset\\.dashboard_success`` points to the
    wrong sibling directory.  For POSIX servers, treat client backslashes as
    separators.  Windows still accepts forward slashes via os.path.abspath().
    """
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        text = unquote(text)
    except Exception:
        pass
    if os.sep == "/" and "\\" in text:
        text = text.replace("\\", "/")
    return os.path.abspath(os.path.expanduser(text))


def run_is_under_root(dataset_root: Union[str, os.PathLike], run_dir: Union[str, os.PathLike]) -> bool:
    root = normalize_dashboard_client_path(dataset_root)
    path = normalize_dashboard_client_path(run_dir)
    try:
        return os.path.commonpath([root, path]) == root and os.path.basename(path).startswith("run_")
    except Exception:
        return False


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base = path
    for i in range(1, 10000):
        candidate = f"{base}_{i:03d}"
        if not os.path.exists(candidate):
            return candidate
    return f"{base}_{int(time.time())}"


SUCCESS_TRANSFER_PATH_FIELDS = {"trajectory", "meta", "score_path", "events"}
SUCCESS_TRANSFER_TEXT_FIELDS = {
    "status",
    "reason",
    "warning_reason",
    "episode_id",
    "initial_pose_id",
    "chosen_plan_id",
}
SUCCESS_POOL_STATUS_VALUES = {
    "trainable",
    "success",
    "successful",
    "rejected",
    "failed",
    "fail",
    "diagnostic",
    "planning",
    "skip",
    "unknown",
}
SUCCESS_POOL_TEXT_BASENAME_VALUES = SUCCESS_POOL_STATUS_VALUES | {"ok", "none", "null"}


def value_looks_like_path(value: object) -> bool:
    text = str(value or "")
    if not text:
        return False
    return bool(os.path.isabs(text) or "\\" in text or "/" in text)


def clean_path_polluted_text(value: object, field: str = "") -> object:
    if not isinstance(value, str) or not value:
        return value
    text = value.strip()
    if not value_looks_like_path(text):
        return value
    normalized = text.replace("\\", "/").rstrip("/")
    parts = [part for part in normalized.split("/") if part]
    tail = parts[-1] if parts else text
    tail_key = tail.strip().lower()
    field_key = str(field or "")
    if field_key == "status":
        if tail_key in {"successful", "success"}:
            return "success"
        if tail_key in {"fail", "failure", "failed"}:
            return "failed"
        if tail_key in SUCCESS_POOL_STATUS_VALUES:
            return tail_key
        if tail_key == "ok":
            return "trainable"
        return tail_key or "unknown"
    if "dashboard_success" in normalized.lower() or "/episodes/" in normalized.lower():
        if field_key in SUCCESS_TRANSFER_TEXT_FIELDS:
            return tail
        if tail_key in SUCCESS_POOL_TEXT_BASENAME_VALUES:
            return tail_key
    return value


def sanitize_success_pool_row(row: dict) -> dict:
    out = dict(row or {})
    for field in SUCCESS_TRANSFER_TEXT_FIELDS:
        if field in out:
            out[field] = clean_path_polluted_text(out.get(field), field)
    status = clean_path_polluted_text(out.get("status", "trainable"), "status")
    status_key = normalized_status_name(status)
    if status_key in {"successful"}:
        status_key = "success"
    if status_key in {"ok", "none", "null", ""}:
        status_key = "trainable"
    if status_key not in {"trainable", "success", "rejected", "failed", "diagnostic", "planning", "skip", "unknown"}:
        # .dashboard_success is a curated pool.  If a historical text-path bug
        # left a non-status string here, keep the episode visible as trainable
        # instead of creating one status-filter button per path.
        status_key = "trainable"
    out["status"] = status_key
    return out


def rewrite_row_paths_for_transfer(row: dict, src_dir: str, dst_dir: str) -> dict:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    out = sanitize_success_pool_row(dict(row))
    # Only rewrite fields that are known to hold filesystem paths.  The previous
    # implementation rewrote every string, which corrupted status="trainable"
    # into <dest_episode_dir>/trainable and exploded the dashboard status filter.
    for key in SUCCESS_TRANSFER_PATH_FIELDS:
        value = out.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            abs_value = os.path.abspath(value) if os.path.isabs(value) else os.path.abspath(os.path.join(src_dir, value))
            if os.path.commonpath([src_dir, abs_value]) == src_dir:
                rel = os.path.relpath(abs_value, src_dir)
                out[key] = os.path.join(dst_dir, rel)
        except Exception:
            pass
    out["source_episode_dir"] = src_dir
    out["transferred_episode_dir"] = dst_dir
    return out


def dashboard_delete_runs(dataset_root: Union[str, os.PathLike], run_paths: Sequence[object], zero_success_only: bool = True, allow_active: bool = False) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    trash_root = ensure_dir(os.path.join(root, ".dashboard_trash", time.strftime("deleted_%Y%m%d_%H%M%S")))
    deleted = []
    skipped = []
    for raw_path in run_paths or []:
        run_dir = os.path.abspath(str(raw_path))
        name = os.path.basename(run_dir)
        if not run_is_under_root(root, run_dir) or not os.path.isdir(run_dir):
            skipped.append({"path": run_dir, "reason": "not_a_run_under_root"})
            continue
        success_count = fast_jsonl_count(index_path(run_dir, "success"))
        if zero_success_only and success_count > 0:
            skipped.append({"path": run_dir, "reason": f"has_success:{success_count}"})
            continue
        activity = run_activity_snapshot(run_dir)
        if activity.get("active") and not allow_active:
            skipped.append({"path": run_dir, "reason": "active_writer"})
            continue
        dst = unique_path(os.path.join(trash_root, name))
        try:
            shutil.move(run_dir, dst)
            deleted.append({"name": name, "from": run_dir, "to": dst, "success": success_count})
        except Exception as exc:
            skipped.append({"path": run_dir, "reason": f"move_to_trash_failed:{type(exc).__name__}:{exc}"})
    return {"ok": True, "trash_dir": trash_root, "deleted": deleted, "skipped": skipped}



def dashboard_success_pool_dir(dataset_root: Union[str, os.PathLike], dest_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    """Return the canonical success-pool directory used by the dashboard.

    Browser paths are normalized first so Linux does not create or read a literal
    ``excavator_auto_dataset\\.dashboard_success`` sibling directory.
    """
    root = normalize_dashboard_client_path(dataset_root or "excavator_auto_dataset")
    if dest_dir:
        dest = normalize_dashboard_client_path(dest_dir)
        if os.path.basename(os.path.normpath(dest)).lower() == SUCCESS_POOL_DIRNAME.lower():
            return dest
        return os.path.join(dest, SUCCESS_POOL_DIRNAME)
    return os.path.join(root, SUCCESS_POOL_DIRNAME)


SUCCESS_TRANSFER_CACHE_FILENAME = "transfer_source_cache.json"
SUCCESS_TRANSFER_CACHE_VERSION = 1


def success_transfer_cache_path(pool_dir: Union[str, os.PathLike]) -> str:
    return os.path.join(os.path.abspath(str(pool_dir)), SUCCESS_TRANSFER_CACHE_FILENAME)


def file_update_signature(path: Union[str, os.PathLike]) -> Dict[str, object]:
    text = str(path or "")
    if not text:
        return {"path": "", "exists": False, "mtime_ns": 0, "size": 0}
    abs_path = os.path.abspath(text)
    try:
        st = os.stat(abs_path)
        return {
            "path": abs_path,
            "exists": True,
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            "size": int(st.st_size),
        }
    except Exception:
        return {"path": abs_path, "exists": False, "mtime_ns": 0, "size": 0}


def dashboard_transfer_source_key(run_dir: Union[str, os.PathLike], row: dict, src_dir: object = None) -> str:
    source_run = str(row.get("source_run_dir") or run_dir or "")
    source_episode_index = row.get("source_episode_index", row.get("episode_index", ""))
    source_dir = str(src_dir or row.get("source_episode_dir") or episode_dir_from_row(row) or "")
    try:
        source_run = os.path.normcase(os.path.abspath(source_run)) if source_run else ""
    except Exception:
        source_run = os.path.normcase(source_run)
    try:
        source_dir = os.path.normcase(os.path.abspath(source_dir)) if source_dir else ""
    except Exception:
        source_dir = os.path.normcase(source_dir)
    # Do not key on episode_id: several interrupted runs can reuse or corrupt it.
    # source_run + source episode index + source episode directory is the stable
    # identity; the signature hash below still detects real content updates.
    return "|".join([source_run, str(source_episode_index), source_dir])


def source_episode_update_signature(run_dir: Union[str, os.PathLike], row: dict, src_dir: str) -> Dict[str, object]:
    source_key = dashboard_transfer_source_key(run_dir, row, src_dir)
    watched = {"episode_dir": file_update_signature(src_dir)}
    for field in ["trajectory", "meta", "score_path", "events"]:
        value = row_path_value(row, field)
        watched[field] = file_update_signature(resolve_episode_file(src_dir, value)) if value else {"path": "", "exists": False, "mtime_ns": 0, "size": 0}
    row_hash = hashlib.sha1(json.dumps(row, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    latest_mtime_ns = max(int(item.get("mtime_ns", 0) or 0) for item in watched.values()) if watched else 0
    payload = {
        "source_key": source_key,
        "source_run_dir": os.path.abspath(str(run_dir or "")),
        "source_episode_index": row.get("episode_index"),
        "source_episode_id": row.get("episode_id", ""),
        "source_episode_dir": os.path.abspath(str(src_dir or "")),
        "row_hash": row_hash,
        "watched_files": watched,
        "latest_mtime_ns": int(latest_mtime_ns),
    }
    payload["source_signature_hash"] = hashlib.sha1(json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return payload


def load_success_transfer_cache(pool_dir: Union[str, os.PathLike]) -> Dict[str, dict]:
    payload = read_json(success_transfer_cache_path(pool_dir), default={}) or {}
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) != SUCCESS_TRANSFER_CACHE_VERSION:
        return {}
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        return {}
    return {str(key): dict(value) for key, value in sources.items() if isinstance(value, dict)}


def write_success_transfer_cache(pool_dir: Union[str, os.PathLike], sources: Dict[str, dict]) -> str:
    clean = {str(key): value for key, value in (sources or {}).items() if isinstance(value, dict)}
    return write_json(
        success_transfer_cache_path(pool_dir),
        {
            "version": SUCCESS_TRANSFER_CACHE_VERSION,
            "updated_at": time.time(),
            "note": "Maps original source run/episode signatures to .dashboard_success entries so repeat copy/cut skips unchanged success episodes.",
            "sources": clean,
        },
    )


def aggregate_row_source_key(row: dict) -> str:
    key = str(row.get("dashboard_transfer_source_key") or "").strip()
    if key:
        return key
    if row.get("source_run_dir") or row.get("source_episode_dir"):
        return dashboard_transfer_source_key(row.get("source_run_dir", ""), row, row.get("source_episode_dir", ""))
    return ""


def norm_episode_dir_key(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return os.path.normcase(os.path.abspath(text))
    except Exception:
        return os.path.normcase(text)


def success_transfer_cache_by_dest(cache: Dict[str, dict]) -> Dict[str, dict]:
    by_dest: Dict[str, dict] = {}

    def entry_score(entry: dict) -> int:
        source_run = str(entry.get("source_run_dir") or "")
        source_key = str(entry.get("source_key") or "")
        score = 0
        if entry.get("source_signature_hash"):
            score += 100
        if source_run:
            score += 10
        if SUCCESS_POOL_DIRNAME not in source_run and not source_key.startswith("|"):
            score += 5
        return score

    def keep_best(key: str, entry: dict) -> None:
        old = by_dest.get(key)
        if old is None or entry_score(entry) > entry_score(old):
            by_dest[key] = entry

    for entry in (cache or {}).values():
        if not isinstance(entry, dict):
            continue
        dest = norm_episode_dir_key(entry.get("transferred_episode_dir") or entry.get("dest_episode_dir"))
        if dest:
            keep_best(dest, entry)
        folder = str(entry.get("folder_name") or "").strip()
        if folder:
            keep_best(os.path.normcase(folder), entry)
    return by_dest


def hydrate_success_pool_row_from_transfer_cache(row: dict, by_dest: Dict[str, dict]) -> Tuple[dict, bool]:
    out = dict(row)
    dest = norm_episode_dir_key(out.get("transferred_episode_dir") or out.get("dest_episode_dir") or episode_dir_from_row(out))
    entry = by_dest.get(dest)
    if entry is None and dest:
        entry = by_dest.get(os.path.normcase(os.path.basename(dest)))
    if not isinstance(entry, dict):
        return out, False
    changed = False
    pool_row = entry.get("pool_row") if isinstance(entry.get("pool_row"), dict) else {}

    def fill(field: str, value: object) -> None:
        nonlocal changed
        if out.get(field) in (None, "") and value not in (None, ""):
            out[field] = value
            changed = True

    def replace_fake(field: str, value: object) -> None:
        nonlocal changed
        current = str(out.get(field) or "")
        new_value = str(value or "")
        if not new_value:
            return
        fake_current = current.startswith("|") or SUCCESS_POOL_DIRNAME in current
        if fake_current and current != new_value:
            out[field] = value
            changed = True

    fill("source_run_name", entry.get("source_run_name") or pool_row.get("source_run_name"))
    fill("source_run_dir", entry.get("source_run_dir") or pool_row.get("source_run_dir"))
    fill("source_episode_index", entry.get("source_episode_index") or pool_row.get("source_episode_index"))
    fill("source_episode_id", entry.get("source_episode_id") or pool_row.get("source_episode_id"))
    fill("source_episode_dir", entry.get("source_episode_dir") or pool_row.get("source_episode_dir"))
    fill("dashboard_transfer_source_key", entry.get("source_key") or pool_row.get("dashboard_transfer_source_key"))
    replace_fake("dashboard_transfer_source_key", entry.get("source_key") or pool_row.get("dashboard_transfer_source_key"))
    fill(
        "dashboard_transfer_source_signature_hash",
        entry.get("source_signature_hash") or pool_row.get("dashboard_transfer_source_signature_hash"),
    )
    fill(
        "dashboard_transfer_source_latest_mtime_ns",
        entry.get("latest_mtime_ns") or pool_row.get("dashboard_transfer_source_latest_mtime_ns"),
    )
    fill("dashboard_transfer_runtime_id", pool_row.get("dashboard_transfer_runtime_id"))
    fill("dashboard_transfer_mode", pool_row.get("dashboard_transfer_mode"))
    signature = pool_row.get("dashboard_transfer_source_signature")
    if isinstance(signature, dict) and not isinstance(out.get("dashboard_transfer_source_signature"), dict):
        out["dashboard_transfer_source_signature"] = signature
        changed = True
    return out, changed


def rewrite_row_paths_after_episode_relocation(row: dict, old_dir: str, new_dir: str) -> dict:
    old_dir = os.path.abspath(str(old_dir or ""))
    new_dir = os.path.abspath(str(new_dir or ""))
    out = dict(row)
    if not old_dir or not new_dir:
        return out
    for key, value in list(out.items()):
        if not isinstance(value, str) or not value:
            continue
        try:
            abs_value = os.path.abspath(value)
            if os.path.commonpath([old_dir, abs_value]) == old_dir:
                rel = os.path.relpath(abs_value, old_dir)
                out[key] = os.path.join(new_dir, rel)
        except Exception:
            pass
    out["transferred_episode_dir"] = new_dir
    return out


def run_timestamp_token_from_name(value: object) -> Optional[str]:
    text = str(value or "")
    m = re.search(r"run_(\d{4})(\d{2})(\d{2})_(\d{6})", text)
    if m:
        return f"{m.group(1)[2:]}{m.group(2)}{m.group(3)}_{m.group(4)}"
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{6})", text)
    if m:
        return f"{m.group(1)[2:]}{m.group(2)}{m.group(3)}_{m.group(4)}"
    m = re.search(r"(\d{2})(\d{2})(\d{2})_(\d{6})", text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}"
    return None


def source_run_timestamp_token(run_name: str, row: dict, runtime_id: str = "") -> str:
    for value in [
        run_name,
        row.get("source_run_name"),
        row.get("source_run_dir"),
        row.get("source_episode_dir"),
        row.get("trajectory"),
        runtime_id,
    ]:
        token = run_timestamp_token_from_name(value)
        if token:
            return token
    # Last resort: keep the requested shape even when a legacy row lacks a run timestamp.
    return time.strftime("%y%m%d_%H%M%S")


def success_pool_episode_folder_name(run_name: str, row: dict, runtime_id: str = "") -> str:
    token = source_run_timestamp_token(run_name, row, runtime_id)
    try:
        ep = int(row.get("source_episode_index", row.get("episode_index")))
    except Exception:
        ep = int(time.time()) % 1000000
    return f"{token}_ep{ep:06d}"


def normalize_existing_success_pool_rows(pool_dir: str, aggregate_rows: Sequence[dict]) -> Tuple[List[dict], Dict[str, int]]:
    episodes_root = ensure_dir(os.path.join(pool_dir, "episodes"))
    normalized: List[dict] = []
    seen_sources = set()
    stats = {"renamed": 0, "deduped": 0, "missing_dest": 0}
    for original in aggregate_rows or []:
        row = dict(original)
        source_key = aggregate_row_source_key(row)
        if source_key and source_key in seen_sources:
            stats["deduped"] += 1
            continue
        current_dir = str(row.get("transferred_episode_dir") or row.get("dest_episode_dir") or "")
        desired_name = success_pool_episode_folder_name(str(row.get("source_run_name") or os.path.basename(str(row.get("source_run_dir") or ""))), row, str(row.get("dashboard_transfer_runtime_id") or ""))
        desired_dir = os.path.join(episodes_root, desired_name)
        if current_dir:
            current_abs = os.path.abspath(current_dir)
            try:
                if os.path.isdir(current_abs) and os.path.basename(current_abs) != desired_name and not os.path.exists(desired_dir):
                    os.rename(current_abs, desired_dir)
                    row = rewrite_row_paths_after_episode_relocation(row, current_abs, desired_dir)
                    stats["renamed"] += 1
                elif os.path.isdir(desired_dir) and not os.path.isdir(current_abs):
                    row = rewrite_row_paths_after_episode_relocation(row, current_abs, desired_dir)
                elif not os.path.isdir(current_abs) and not os.path.isdir(desired_dir):
                    stats["missing_dest"] += 1
            except Exception:
                pass
        if source_key:
            seen_sources.add(source_key)
        normalized.append(row)
    return normalized, stats


def build_existing_success_source_cache(pool_dir: str, aggregate_rows: Sequence[dict]) -> Dict[str, dict]:
    cache = load_success_transfer_cache(pool_dir)
    for row in aggregate_rows or []:
        key = aggregate_row_source_key(row)
        if not key:
            continue
        source_hash = row.get("dashboard_transfer_source_signature_hash", "")
        if not source_hash and not (row.get("source_run_dir") or row.get("source_episode_dir")):
            continue
        cache[key] = {
            "source_key": key,
            "source_run_name": row.get("source_run_name", ""),
            "source_run_dir": row.get("source_run_dir", ""),
            "source_episode_index": row.get("source_episode_index"),
            "source_episode_id": row.get("source_episode_id", ""),
            "source_episode_dir": row.get("source_episode_dir", ""),
            "transferred_episode_dir": row.get("transferred_episode_dir", ""),
            "folder_name": os.path.basename(str(row.get("transferred_episode_dir") or "")),
            "pool_episode_index": row.get("episode_index"),
            "pool_episode_id": row.get("episode_id", ""),
            "pool_row": dict(row),
            "source_signature_hash": source_hash,
            "latest_mtime_ns": row.get("dashboard_transfer_source_latest_mtime_ns", 0),
            "created_from_existing_index": True,
        }

    # Also recover from transfer manifests. This matters for pools created before
    # transfer_source_cache.json existed, or when the process was restarted before
    # the cache file was flushed.
    manifest_dir = os.path.join(pool_dir, "transfer_manifests")
    try:
        manifest_files = [os.path.join(manifest_dir, name) for name in os.listdir(manifest_dir) if name.endswith(".json")]
    except Exception:
        manifest_files = []
    for manifest_path in manifest_files:
        manifest = read_json(manifest_path, default={}) or {}
        if not isinstance(manifest, dict):
            continue
        source_run_dir = str(manifest.get("source_run_dir") or "")
        source_run_name = os.path.basename(source_run_dir)
        for record in manifest.get("records", []) if isinstance(manifest.get("records"), list) else []:
            if not isinstance(record, dict):
                continue
            row_hint = {
                "episode_index": record.get("source_episode_index"),
                "source_episode_index": record.get("source_episode_index"),
                "source_episode_id": record.get("source_episode_id", ""),
                "source_run_dir": source_run_dir,
                "source_episode_dir": record.get("source_episode_dir", ""),
            }
            key = str(record.get("source_key") or dashboard_transfer_source_key(source_run_dir, row_hint, record.get("source_episode_dir", "")))
            if not key:
                continue
            cache.setdefault(
                key,
                {
                    "source_key": key,
                    "source_run_name": source_run_name,
                    "source_run_dir": source_run_dir,
                    "source_episode_index": record.get("source_episode_index"),
                    "source_episode_id": record.get("source_episode_id", ""),
                    "source_episode_dir": record.get("source_episode_dir", ""),
                    "transferred_episode_dir": record.get("dest_episode_dir", ""),
                    "folder_name": os.path.basename(str(record.get("dest_episode_dir") or "")),
                    "pool_episode_index": record.get("episode_index"),
                    "pool_episode_id": "",
                    "source_signature_hash": record.get("source_signature_hash", ""),
                    "latest_mtime_ns": 0,
                    "created_from_manifest": True,
                },
            )
    return cache

def success_pool_entry_content_complete(entry: Optional[dict], sample_limit: int = 16) -> Tuple[bool, str]:
    """Verify that a cached success-pool entry is usable before skipping it.

    A cache hit is not enough.  The destination episode directory must exist,
    the rewritten row must point to a non-empty trajectory, metadata files that
    are present in the row must still exist, and sampled camera files referenced
    from the trajectory must exist.  This prevents a half-copied episode from
    being marked as cached after a crash, disk-full event, or interrupted copy.
    """
    if not isinstance(entry, dict):
        return False, "cache_entry_missing"
    dest_dir = str(entry.get("transferred_episode_dir") or entry.get("dest_episode_dir") or "")
    if not dest_dir or not os.path.isdir(dest_dir):
        return False, "dest_episode_dir_missing"
    pool_row = entry.get("pool_row") if isinstance(entry.get("pool_row"), dict) else None
    if not pool_row:
        # Manifest-only legacy cache entries do not carry enough information to
        # prove that the copied episode is complete.  Force a reprocess instead
        # of silently skipping a possibly partial destination folder.
        return False, "cache_unverifiable_no_pool_row"
    trajectory_path = row_path_value(pool_row, "trajectory")
    if not trajectory_path or not os.path.isfile(trajectory_path):
        return False, "trajectory_missing"
    try:
        if os.path.getsize(trajectory_path) <= 0:
            return False, "trajectory_empty_file"
    except Exception:
        return False, "trajectory_unreadable"
    trajectory_probe = read_jsonl_limited(trajectory_path, limit=max(1, int(sample_limit)))
    if not trajectory_probe:
        return False, "trajectory_no_samples"
    for field in ["meta", "score_path"]:
        value = row_path_value(pool_row, field)
        if value and not os.path.isfile(value):
            return False, f"{field}_missing"
    # events can legitimately be absent in some historical rows, but if the row
    # points at an events file and it exists in the source-derived destination,
    # keep checking it.  Missing events should not block VLA export.
    checked_camera_samples = 0
    for sample in trajectory_probe:
        if not isinstance(sample, dict):
            continue
        # Keep the check aligned with LeRobot export requirements: state/action
        # plus the three camera keys must be resolvable for sampled frames.
        if not sample_has_state_action(sample):
            return False, "sample_missing_state_or_action"
        for key in LEROBOT_IMAGE_KEYS:
            image_value = sample_image_value(sample, key)
            if not image_value:
                return False, f"sample_missing_{key}"
            image_text = str(image_value or "")
            image_path = resolve_episode_file(dest_dir, image_text)
            source_dir = str(pool_row.get("source_episode_dir") or "")
            # Some historical trajectories store absolute image paths.  If such
            # a path points into the original source episode, validate the copied
            # destination-relative counterpart instead; otherwise cut would later
            # delete the source and leave the pool with broken absolute paths.
            try:
                if os.path.isabs(image_text) and source_dir:
                    image_abs = os.path.abspath(image_text)
                    source_abs = os.path.abspath(source_dir)
                    if os.path.commonpath([source_abs, image_abs]) == source_abs:
                        image_path = os.path.join(dest_dir, os.path.relpath(image_abs, source_abs))
            except Exception:
                pass
            if not image_path or not os.path.isfile(image_path):
                return False, f"image_missing:{key}"
        checked_camera_samples += 1
    if checked_camera_samples <= 0:
        return False, "no_camera_samples_checked"
    return True, "complete"


def existing_success_entry_is_current(entry: Optional[dict], signature: Dict[str, object]) -> Tuple[bool, str]:
    if not isinstance(entry, dict):
        return False, "not_seen"
    complete, complete_reason = success_pool_entry_content_complete(entry)
    cached_hash = str(entry.get("source_signature_hash") or entry.get("dashboard_transfer_source_signature_hash") or "")
    current_hash = str(signature.get("source_signature_hash") or "")
    if not complete:
        return False, f"cached_dest_incomplete:{complete_reason}"
    if cached_hash and current_hash and cached_hash == current_hash:
        return True, "already_transferred_unchanged"
    if not cached_hash:
        return True, "already_transferred_legacy_cache_complete"
    if cached_hash and current_hash and cached_hash != current_hash:
        return False, "source_updated"
    return False, "cache_incomplete"


def next_success_pool_episode_index(pool_dir: str) -> int:
    indexes = []
    for split in ("trainable", "success", "all"):
        for row in load_index(pool_dir, split):
            try:
                indexes.append(int(row.get("episode_index")))
            except Exception:
                pass
    return max(indexes, default=-1) + 1


def write_success_pool_indexes(pool_dir: str, rows: Sequence[dict]) -> None:
    ordered = sorted(rows, key=lambda row: int(row.get("episode_index", 0) or 0))
    write_jsonl(os.path.join(pool_dir, "trainable_episodes.jsonl"), ordered)
    write_jsonl(os.path.join(pool_dir, "successful_episodes.jsonl"), ordered)
    write_jsonl(os.path.join(pool_dir, "episodes.jsonl"), ordered)




def is_dashboard_success_pool_dir(run_dir: Union[str, os.PathLike]) -> bool:
    try:
        return os.path.basename(os.path.normpath(os.path.abspath(str(run_dir)))) == SUCCESS_POOL_DIRNAME
    except Exception:
        return False


def dashboard_success_pool_export_manifest_path(run_dir: Union[str, os.PathLike]) -> str:
    return os.path.join(os.path.abspath(str(run_dir)), LEROBOT_DEFAULT_EXPORT_DIRNAME, "manifest.json")


def dashboard_export_time_policy_path(run_dir: Union[str, os.PathLike]) -> str:
    return os.path.join(os.path.abspath(str(run_dir)), EXPORT_TIME_POLICY_FILENAME)


def normalize_export_time_policy(policy: object) -> Dict[str, object]:
    data = policy if isinstance(policy, dict) else {}
    try:
        speed_scale = float(data.get("speed_scale", 1.0))
    except Exception:
        speed_scale = 1.0
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        speed_scale = 1.0
    return {
        "version": 1,
        "speed_scale": float(speed_scale),
        "time_mode": "uniform_fps",
        "base_fps": None,
    }


def default_export_time_policy() -> Dict[str, object]:
    return normalize_export_time_policy({})


def export_time_policy_hash(policy: object) -> str:
    normalized = normalize_export_time_policy(policy)
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def export_time_policy_is_default(policy: object) -> bool:
    return normalize_export_time_policy(policy) == default_export_time_policy()


def load_dashboard_export_time_policy(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    path = dashboard_export_time_policy_path(run_dir)
    exists = os.path.isfile(path)
    raw = read_json(path, default={}) if exists else {}
    normalized = normalize_export_time_policy(raw)
    return {
        "path": path,
        "exists": exists,
        "policy": normalized,
        "hash": export_time_policy_hash(normalized),
        "is_default": export_time_policy_is_default(normalized),
    }


def save_dashboard_export_time_policy(run_dir: Union[str, os.PathLike], policy: object = None) -> Dict[str, object]:
    run_abs = os.path.abspath(str(run_dir))
    normalized = normalize_export_time_policy(policy)
    path = dashboard_export_time_policy_path(run_abs)
    existing = read_json(path, default={}) if os.path.isfile(path) else {}
    existing_normalized = normalize_export_time_policy(existing)
    if os.path.isfile(path) and existing_normalized == normalized:
        return {
            "ok": True,
            "path": path,
            "exists": True,
            "policy": normalized,
            "hash": export_time_policy_hash(normalized),
            "is_default": export_time_policy_is_default(normalized),
            "unchanged": True,
        }
    ensure_dir(os.path.dirname(path))
    write_json(path, normalized)
    with FRAME_CONTEXT_CACHE_LOCK:
        FRAME_CONTEXT_CACHE.clear()
    return {
        "ok": True,
        "path": path,
        "exists": True,
        "policy": normalized,
        "hash": export_time_policy_hash(normalized),
        "is_default": export_time_policy_is_default(normalized),
    }


def dashboard_effective_export_fps(run_dir: Union[str, os.PathLike], policy: object = None, explicit_fps: Optional[float] = None) -> float:
    normalized = normalize_export_time_policy(policy)
    try:
        base = infer_export_fps(run_dir, explicit_fps)
        return float(_require_shared_dataset_tool("effective_export_fps")(base, normalized))
    except Exception:
        return 10.0 * float(normalized.get("speed_scale") or 1.0)


def dashboard_time_policy_for_run(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    if not is_dashboard_success_pool_dir(run_dir):
        return {
            "path": dashboard_export_time_policy_path(run_dir),
            "exists": False,
            "policy": default_export_time_policy(),
            "hash": export_time_policy_hash(default_export_time_policy()),
            "is_default": True,
        }
    return load_dashboard_export_time_policy(run_dir)


def dashboard_apply_export_time_policy(
    run_dir: Union[str, os.PathLike],
    trajectory: Sequence[dict],
    row: Optional[dict] = None,
) -> Dict[str, object]:
    info = dashboard_time_policy_for_run(run_dir)
    policy = dict(info.get("policy") or default_export_time_policy())
    try:
        run_meta = read_json(os.path.join(os.path.abspath(str(run_dir)), "run_meta.json"), default={}) or {}
        if row is not None:
            try:
                episode_dir = episode_dir_from_row(row, run_dir=run_dir)
                episode_meta = read_json(resolve_episode_file(episode_dir, row_path_value(row, "meta")), default={}) or {}
                if isinstance(episode_meta.get("run_meta"), dict):
                    run_meta.update(episode_meta.get("run_meta") or {})
            except Exception:
                pass
        base_fps = infer_export_fps(run_dir, policy.get("base_fps"))
        result = _require_shared_dataset_tool("apply_export_time_policy_to_trajectory")(
            trajectory,
            policy=policy,
            base_fps=base_fps,
            state_names=run_meta.get("state_names"),
            action_names=run_meta.get("action_names"),
        )
        out = dict(result)
        out["time_policy"] = policy
        out["time_policy_hash"] = export_time_policy_hash(policy)
        out["base_fps"] = float(out.get("base_fps") or base_fps or 10.0)
        out["effective_fps"] = float(out.get("effective_fps") or dashboard_effective_export_fps(run_dir, policy, explicit_fps=base_fps))
        out["is_default"] = bool(info.get("is_default"))
        out["path"] = info.get("path", "")
        return out
    except Exception as exc:
        return {
            "samples": list(trajectory or []),
            "time_policy": policy,
            "time_policy_hash": export_time_policy_hash(policy),
            "base_fps": 10.0,
            "effective_fps": dashboard_effective_export_fps(run_dir, policy),
            "duration_s": trajectory_runtime_s(trajectory),
            "raw_duration_s": trajectory_runtime_s(trajectory),
            "error": f"{type(exc).__name__}: {exc}",
            "is_default": bool(info.get("is_default")),
            "path": info.get("path", ""),
        }


def dashboard_export_time_runtime(run_dir: Union[str, os.PathLike], trajectory: Sequence[dict]) -> Dict[str, object]:
    info = dashboard_time_policy_for_run(run_dir)
    policy = dict(info.get("policy") or default_export_time_policy())
    raw_duration = trajectory_runtime_s(trajectory)
    effective_fps_value = dashboard_effective_export_fps(run_dir, policy)
    if not trajectory:
        duration = 0.0
    else:
        duration = float(max(0, len(trajectory) - 1)) / float(max(0.001, effective_fps_value))
    return {
        "policy": policy,
        "time_policy_hash": export_time_policy_hash(policy),
        "effective_fps": effective_fps_value,
        "base_fps": policy.get("base_fps"),
        "duration_s": float(duration),
        "raw_duration_s": float(raw_duration),
        "is_default": bool(info.get("is_default")),
    }


def manifest_time_policy_hash(manifest: dict) -> str:
    for key in ["time_policy_hash", "export_time_policy_hash"]:
        text = str(manifest.get(key) or "").strip()
        if text:
            return text
    for key in ["time_policy", "export_time_policy"]:
        policy = manifest.get(key)
        if isinstance(policy, dict):
            return export_time_policy_hash(policy)
    return ""


def dashboard_success_pool_export_lookup(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    """Return manifest lookup tables used to mark .dashboard_success rows as export-ready."""
    manifest_path = dashboard_success_pool_export_manifest_path(run_dir)
    current_time_policy = load_dashboard_export_time_policy(run_dir)
    current_export_config_hash = ""
    try:
        config_builder = _require_shared_dataset_tool("lerobot_export_config_for_run")
        current_export_config_hash = str(config_builder(run_dir, time_policy=current_time_policy.get("policy", {})).get("export_config_hash") or "")
    except Exception:
        current_export_config_hash = ""
    manifest = read_json(manifest_path, default={}) or {}
    if not isinstance(manifest, dict) or not os.path.isfile(manifest_path):
        return {
            "has_manifest": False,
            "manifest_path": manifest_path,
            "current_time_policy": current_time_policy,
            "current_export_config_hash": current_export_config_hash,
            "time_policy_mismatch": False,
            "export_config_mismatch": False,
            "summary": {
                "has_manifest": False,
                "ready": 0,
                "not_ready": 0,
                "reason": "lerobot_v3_manifest_missing",
                "current_time_policy": current_time_policy.get("policy", {}),
                "current_time_policy_hash": current_time_policy.get("hash", ""),
            },
            "by_source_signature": {},
            "by_source_key": {},
            "by_raw_episode": {},
        }
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        episodes = []
    by_source_signature: Dict[str, dict] = {}
    by_source_key: Dict[str, dict] = {}
    by_raw_episode: Dict[str, dict] = {}
    ready = 0
    not_ready = 0
    manifest_policy_hash = manifest_time_policy_hash(manifest)
    current_policy_hash = str(current_time_policy.get("hash") or "")
    current_policy_is_default = bool(current_time_policy.get("is_default"))
    time_policy_mismatch = bool(
        (manifest_policy_hash and current_policy_hash and manifest_policy_hash != current_policy_hash)
        or (not manifest_policy_hash and not current_policy_is_default)
    )
    manifest_config_hash = str(manifest.get("export_config_hash") or "")
    export_config_mismatch = bool(
        (manifest_config_hash and current_export_config_hash and manifest_config_hash != current_export_config_hash)
        or (not manifest_config_hash and bool(current_export_config_hash))
    )
    for entry in episodes:
        if not isinstance(entry, dict):
            continue
        clean = dict(entry)
        if clean.get("export_ready") is True:
            ready += 1
        else:
            not_ready += 1
        for key in [
            clean.get("source_signature_hash"),
            clean.get("dashboard_transfer_source_signature_hash"),
        ]:
            text = str(key or "").strip()
            if text:
                by_source_signature[text] = clean
        source_key = str(clean.get("source_key") or clean.get("dashboard_transfer_source_key") or "").strip()
        if source_key:
            by_source_key[source_key] = clean
        for key in [clean.get("raw_episode_index"), clean.get("source_episode_index")]:
            text = str(key).strip() if key is not None else ""
            if text:
                by_raw_episode[text] = clean
    return {
        "has_manifest": True,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "current_time_policy": current_time_policy,
        "current_export_config_hash": current_export_config_hash,
        "manifest_time_policy_hash": manifest_policy_hash,
        "manifest_export_config_hash": manifest_config_hash,
        "time_policy_mismatch": time_policy_mismatch,
        "export_config_mismatch": export_config_mismatch,
        "by_source_signature": by_source_signature,
        "by_source_key": by_source_key,
        "by_raw_episode": by_raw_episode,
        "summary": {
            "has_manifest": True,
            "manifest_path": manifest_path,
            "video_layout": manifest.get("video_layout", ""),
            "export_config_hash": manifest.get("export_config_hash", ""),
            "ready": ready,
            "not_ready": not_ready,
            "total": len(episodes),
            "vla_training_ready": bool(manifest.get("vla_training_ready")),
            "reused_video_jobs": int(manifest.get("reused_video_jobs") or 0),
            "encoded_video_jobs": int(manifest.get("encoded_video_jobs") or 0),
            "total_video_jobs": int(manifest.get("total_video_jobs") or 0),
            "current_time_policy": current_time_policy.get("policy", {}),
            "current_time_policy_hash": current_policy_hash,
            "manifest_time_policy_hash": manifest_policy_hash,
            "current_export_config_hash": current_export_config_hash,
            "manifest_export_config_hash": manifest_config_hash,
            "time_policy_mismatch": time_policy_mismatch,
            "export_config_mismatch": export_config_mismatch,
        },
    }


def dashboard_row_export_status(row: dict, export_lookup: Dict[str, object]) -> Dict[str, object]:
    if not export_lookup.get("has_manifest"):
        return {
            "export_ready": False,
            "export_status": "missing",
            "export_reason": str((export_lookup.get("summary") or {}).get("reason") or "not_exported_or_missing_manifest"),
            "export_manifest_path": export_lookup.get("manifest_path", ""),
        }
    entry = None
    by_signature = export_lookup.get("by_source_signature") if isinstance(export_lookup.get("by_source_signature"), dict) else {}
    for key in [
        row.get("dashboard_transfer_source_signature_hash"),
        row.get("source_signature_hash"),
    ]:
        text = str(key or "").strip()
        if text and text in by_signature:
            entry = by_signature[text]
            break
    if entry is None:
        by_source_key = export_lookup.get("by_source_key") if isinstance(export_lookup.get("by_source_key"), dict) else {}
        source_key = str(row.get("dashboard_transfer_source_key") or row.get("source_key") or "").strip()
        if source_key and source_key in by_source_key:
            entry = by_source_key[source_key]
    if entry is None:
        by_raw_episode = export_lookup.get("by_raw_episode") if isinstance(export_lookup.get("by_raw_episode"), dict) else {}
        for key in [row.get("episode_index"), row.get("source_episode_index")]:
            text = str(key).strip() if key is not None else ""
            if text and text in by_raw_episode:
                entry = by_raw_episode[text]
                break
    if not isinstance(entry, dict):
        return {
            "export_ready": False,
            "export_status": "missing",
            "export_reason": "not_exported_or_missing_episode_manifest",
            "export_manifest_path": export_lookup.get("manifest_path", ""),
        }
    ready = entry.get("export_ready") is True
    entry_policy_hash = manifest_time_policy_hash(entry)
    current_hash = str((export_lookup.get("current_time_policy") or {}).get("hash") or "")
    time_policy_mismatch = bool(export_lookup.get("time_policy_mismatch"))
    if entry_policy_hash and current_hash:
        time_policy_mismatch = entry_policy_hash != current_hash
    reasons = entry.get("not_ready_reasons")
    if isinstance(reasons, list):
        reason_text = "; ".join(str(item) for item in reasons if str(item))
    else:
        reason_text = str(reasons or "")
    videos = entry.get("videos") if isinstance(entry.get("videos"), dict) else {}
    available_videos = sum(1 for value in videos.values() if isinstance(value, dict) and value.get("available") is True)
    entry_config_hash = str(entry.get("export_config_hash") or "")
    current_config_hash = str(export_lookup.get("current_export_config_hash") or "")
    export_config_mismatch = bool(export_lookup.get("export_config_mismatch"))
    if entry_config_hash and current_config_hash:
        export_config_mismatch = entry_config_hash != current_config_hash
    if ready and (time_policy_mismatch or export_config_mismatch):
        manifest_hash = entry_policy_hash or str(export_lookup.get("manifest_time_policy_hash") or "")
        manifest_config_hash = entry_config_hash or str(export_lookup.get("manifest_export_config_hash") or "")
        reason_kind = "time_policy_mismatch" if time_policy_mismatch else "export_config_mismatch"
        return {
            "export_ready": False,
            "export_status": reason_kind,
            "export_reason": (
                f"{reason_kind} current_time={current_hash[:8] or 'default'} "
                f"manifest_time={manifest_hash[:8] or 'none'} "
                f"current_config={current_config_hash[:8] or 'none'} "
                f"manifest_config={manifest_config_hash[:8] or 'none'}"
            ),
            "export_manifest_path": export_lookup.get("manifest_path", ""),
            "export_episode_index": entry.get("episode_index"),
            "export_video_layout": entry.get("video_layout", ""),
            "export_video_count": available_videos,
            "export_config_hash": entry.get("export_config_hash", ""),
        }
    return {
        "export_ready": bool(ready),
        "export_status": "ready" if ready else "not_ready",
        "export_reason": reason_text or ("ready" if ready else str(entry.get("export_status") or "not_ready")),
        "export_manifest_path": export_lookup.get("manifest_path", ""),
        "export_episode_index": entry.get("episode_index"),
        "export_video_layout": entry.get("video_layout", ""),
        "export_video_count": available_videos,
        "export_config_hash": entry.get("export_config_hash", ""),
    }


def episode_dir_key_from_row(row: dict) -> str:
    for value in [row.get("transferred_episode_dir"), row.get("dest_episode_dir")]:
        if value:
            try:
                return os.path.normcase(os.path.abspath(str(value)))
            except Exception:
                return os.path.normcase(str(value))
    try:
        ep_dir = episode_dir_from_row(row)
        return os.path.normcase(os.path.abspath(ep_dir)) if ep_dir else ""
    except Exception:
        return ""


def find_direct_episode_file(ep_dir: str, names: Sequence[str] = (), contains: Sequence[str] = (), suffixes: Sequence[str] = ()) -> str:
    try:
        entries = [entry for entry in os.scandir(ep_dir) if entry.is_file()]
    except Exception:
        return ""
    lowered_names = {str(name).lower() for name in names}
    lowered_contains = [str(part).lower() for part in contains]
    lowered_suffixes = [str(suffix).lower() for suffix in suffixes]
    for entry in entries:
        name = entry.name.lower()
        if name in lowered_names:
            return entry.path
    for entry in entries:
        name = entry.name.lower()
        if lowered_suffixes and not any(name.endswith(suffix) for suffix in lowered_suffixes):
            continue
        if lowered_contains and all(part in name for part in lowered_contains):
            return entry.path
    return ""


def find_success_pool_trajectory_file(ep_dir: str) -> str:
    explicit = find_direct_episode_file(
        ep_dir,
        names=["trajectory.jsonl", "trajectory.json", "samples.jsonl"],
        suffixes=[".jsonl", ".json"],
    )
    if explicit:
        return explicit
    by_name = find_direct_episode_file(ep_dir, contains=["trajectory"], suffixes=[".jsonl", ".json"])
    if by_name:
        return by_name
    try:
        candidates = []
        for entry in os.scandir(ep_dir):
            if not entry.is_file():
                continue
            name = entry.name.lower()
            if not name.endswith(".jsonl"):
                continue
            if any(skip in name for skip in ["event", "segment", "diagnostic", "planning"]):
                continue
            try:
                candidates.append((int(entry.stat().st_size), entry.path))
            except Exception:
                candidates.append((0, entry.path))
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else ""
    except Exception:
        return ""


def parse_source_episode_index_from_pool_folder(folder_name: object) -> Optional[int]:
    m = re.search(r"_ep(\d{1,9})$", str(folder_name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def recover_success_pool_row_from_episode_folder(pool_dir: str, ep_dir: str, next_ep_index: int) -> Optional[dict]:
    ep_dir = os.path.abspath(str(ep_dir))
    folder_name = os.path.basename(ep_dir)
    trajectory_path = find_success_pool_trajectory_file(ep_dir)
    if not trajectory_path or not os.path.isfile(trajectory_path):
        return None
    try:
        if os.path.getsize(trajectory_path) <= 0:
            return None
    except Exception:
        return None
    meta_path = find_direct_episode_file(ep_dir, names=["meta.json", "episode_meta.json"], contains=["meta"], suffixes=[".json"])
    score_path = find_direct_episode_file(ep_dir, names=["score.json", "episode_score.json"], contains=["score"], suffixes=[".json"])
    events_path = find_direct_episode_file(ep_dir, names=["events.jsonl", "event_log.jsonl"], contains=["event"], suffixes=[".jsonl"])
    meta = read_json(meta_path, default={}) or {}
    score_data = read_json(score_path, default={}) or {}
    sample_probe = read_jsonl_limited(trajectory_path, limit=1)
    source_ep = parse_source_episode_index_from_pool_folder(folder_name)
    row = {
        "episode_index": int(next_ep_index),
        "episode_id": folder_name,
        "status": "trainable",
        "trajectory": trajectory_path,
        "transferred_episode_dir": ep_dir,
        "dashboard_recovered_from_episode_folder": True,
    }
    if meta_path:
        row["meta"] = meta_path
    if score_path:
        row["score_path"] = score_path
    if events_path:
        row["events"] = events_path
    if source_ep is not None:
        row["source_episode_index"] = source_ep
    for source in [meta, score_data]:
        if not isinstance(source, dict):
            continue
        for key in [
            "status",
            "score",
            "reason",
            "warning_reason",
            "samples",
            "freeze_count",
            "max_bucket_from_pile_particles",
            "lift_bucket_from_pile_particles",
            "final_bin_from_pile_particles",
            "final_spill_from_pile_particles",
            "initial_pose_id",
            "chosen_plan_id",
            "q_initial_deg",
            "target_xyz",
            "unload_landing_xyz",
            "unload_point_xyz",
            "scene_randomization",
        ]:
            if key not in row and key in source:
                row[key] = source.get(key)
    if "samples" not in row:
        try:
            row["samples"] = fast_jsonl_count(trajectory_path)
        except Exception:
            row["samples"] = len(sample_probe)
    # Do not trust arbitrary recovered status values from meta/score.  The pool
    # itself is curated for trainable/success data; dataset_training_tag_for_row()
    # will still demote incomplete rows to skip after schema/camera checks.
    row = sanitize_success_pool_row(row)
    if row.get("status") not in {"trainable", "success"}:
        row["status"] = "trainable"
    return row


def reconcile_success_pool_indexes(pool_dir: Union[str, os.PathLike], recover_orphan_folders: bool = True) -> Dict[str, object]:
    pool_dir = os.path.abspath(str(pool_dir))
    if not is_dashboard_success_pool_dir(pool_dir):
        return {"ok": True, "is_success_pool": False, "changed": False}
    episodes_root = os.path.join(pool_dir, "episodes")
    aggregate: List[dict] = []
    seen_dirs = set()
    seen_ids = set()
    sanitized = 0
    duplicate_rows = 0
    raw_index_rows = {split: load_index(pool_dir, split) for split in ["trainable", "success", "all"]}
    initial_trainable_rows = len(raw_index_rows.get("trainable") or [])
    # Read all three pool indexes.  Some interrupted transfers wrote only one of
    # them; this prevents Load .dashboard_success from depending on a single file.
    for split in ["trainable", "success", "all"]:
        for original in raw_index_rows.get(split, []):
            row = sanitize_success_pool_row(original)
            if row != original:
                sanitized += 1
            dir_key = episode_dir_key_from_row(row)
            row_id = str(row.get("episode_id") or "")
            unique_key = dir_key or row_id or json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)
            if unique_key in seen_dirs or (row_id and row_id in seen_ids and not dir_key):
                duplicate_rows += 1
                continue
            if dir_key:
                seen_dirs.add(unique_key)
            if row_id:
                seen_ids.add(row_id)
            aggregate.append(row)
    next_index = max([int(row.get("episode_index", -1) or -1) for row in aggregate], default=-1) + 1
    recovered = 0
    incomplete_folders = 0
    scanned_folders = 0
    if recover_orphan_folders and os.path.isdir(episodes_root):
        try:
            entries = sorted([entry for entry in os.scandir(episodes_root) if entry.is_dir()], key=lambda entry: entry.name)
        except Exception:
            entries = []
        for entry in entries:
            scanned_folders += 1
            dir_key = os.path.normcase(os.path.abspath(entry.path))
            if dir_key in seen_dirs:
                continue
            row = recover_success_pool_row_from_episode_folder(pool_dir, entry.path, next_index)
            if row is None:
                incomplete_folders += 1
                continue
            aggregate.append(row)
            seen_dirs.add(dir_key)
            seen_ids.add(str(row.get("episode_id") or ""))
            next_index += 1
            recovered += 1
    hydrated_from_cache = 0
    transfer_cache = load_success_transfer_cache(pool_dir)
    if transfer_cache:
        by_dest = success_transfer_cache_by_dest(transfer_cache)
        hydrated: List[dict] = []
        for row in aggregate:
            fixed, row_changed = hydrate_success_pool_row_from_transfer_cache(row, by_dest)
            hydrated.append(fixed)
            if row_changed:
                hydrated_from_cache += 1
        aggregate = hydrated
    changed = bool(
        sanitized
        or recovered
        or hydrated_from_cache
        or len(aggregate) != initial_trainable_rows
        or (aggregate and not os.path.exists(index_path(pool_dir, "trainable")))
    )
    if changed:
        write_success_pool_indexes(pool_dir, aggregate)
    return {
        "ok": True,
        "is_success_pool": True,
        "changed": changed,
        "indexed_rows": len(aggregate),
        "scanned_episode_folders": scanned_folders,
        "recovered_orphan_folders": recovered,
        "incomplete_or_unreadable_folders": incomplete_folders,
        "sanitized_rows": sanitized,
        "hydrated_rows_from_transfer_cache": hydrated_from_cache,
        "duplicate_index_rows_dropped": duplicate_rows,
        "episodes_root": episodes_root,
    }


def maybe_reconcile_success_pool_for_dashboard(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    if is_dashboard_success_pool_dir(run_dir):
        return reconcile_success_pool_indexes(run_dir, recover_orphan_folders=True)
    return {"ok": True, "is_success_pool": False, "changed": False}

def short_success_episode_folder_name(run_name: str, row: dict, runtime_id: str = "") -> str:
    """Canonical success-pool folder name: YYMMDD_HHMMSS_epXXXXXX."""
    return success_pool_episode_folder_name(run_name, row, runtime_id)

def dashboard_cut_trash_root(dataset_root: Union[str, os.PathLike], runtime_id: str) -> str:
    return ensure_dir(os.path.join(os.path.abspath(str(dataset_root or "")), ".dashboard_trash", f"cut_success_{runtime_id}"))


def safe_move_source_episode_dir_to_trash_for_cut(
    root: str,
    run_dir: str,
    src_dir: str,
    trash_root: str,
) -> Tuple[bool, str]:
    root_abs = os.path.abspath(str(root or ""))
    run_abs = os.path.abspath(str(run_dir or ""))
    src_abs = os.path.abspath(str(src_dir or ""))
    trash_root = ensure_dir(str(trash_root or dashboard_cut_trash_root(root_abs, time.strftime("%Y%m%d_%H%M%S"))))
    if not src_abs or not os.path.exists(src_abs):
        return True, "already_missing"
    try:
        if os.path.commonpath([root_abs, run_abs]) != root_abs:
            return False, "run_not_under_dataset_root"
        if os.path.commonpath([run_abs, src_abs]) != run_abs:
            return False, "source_not_under_run"
        if os.path.commonpath([root_abs, os.path.abspath(trash_root)]) != root_abs:
            return False, "trash_not_under_dataset_root"
    except Exception:
        return False, "source_path_invalid"
    if os.path.normcase(run_abs) == os.path.normcase(src_abs):
        return False, "refuse_move_run_as_episode"
    try:
        dst_dir = unique_path(os.path.join(trash_root, "episodes", os.path.basename(run_abs), os.path.basename(src_abs)))
        ensure_dir(os.path.dirname(dst_dir))
        shutil.move(src_abs, dst_dir)
        return True, f"moved_to_trash:{dst_dir}"
    except Exception as exc:
        return False, f"move_to_trash_failed:{type(exc).__name__}:{exc}"


def remaining_success_episode_dirs(run_dir: str, rows: Optional[Sequence[dict]] = None) -> List[str]:
    remaining = []
    for row in list(rows if rows is not None else load_index(run_dir, "success")):
        src_dir = episode_dir_from_row(row)
        if src_dir and os.path.isdir(src_dir):
            remaining.append(os.path.abspath(src_dir))
    return remaining


def move_source_run_to_trash_if_cut_complete(root: str, run_dir: str, rows: Sequence[dict], trash_root: str) -> Dict[str, object]:
    result = {"trashed": False, "reason": "not_checked", "remaining_success_dirs": 0, "trash_dir": ""}
    if not run_is_under_root(root, run_dir) or not os.path.isdir(run_dir):
        result["reason"] = "run_missing_or_not_under_root"
        return result
    remaining = remaining_success_episode_dirs(run_dir, rows)
    result["remaining_success_dirs"] = len(remaining)
    if remaining:
        result["reason"] = "success_episode_dirs_remaining"
        return result
    try:
        trash_root = ensure_dir(str(trash_root or dashboard_cut_trash_root(root, time.strftime("%Y%m%d_%H%M%S"))))
        dst_dir = unique_path(os.path.join(trash_root, "runs", os.path.basename(os.path.abspath(run_dir))))
        ensure_dir(os.path.dirname(dst_dir))
        shutil.move(os.path.abspath(run_dir), dst_dir)
        result.update({"trashed": True, "reason": "zero_success_left_moved_run_to_trash", "trash_dir": dst_dir})
    except Exception as exc:
        result["reason"] = f"run_move_to_trash_failed:{type(exc).__name__}:{exc}"
    return result


def dashboard_transfer_success_records(
    dataset_root: Union[str, os.PathLike],
    run_paths: Sequence[object],
    dest_dir: Union[str, os.PathLike],
    mode: str = "copy",
    progress_callback=None,
) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    dest_root = ensure_dir(dashboard_success_pool_dir(root, dest_dir))
    episodes_root = ensure_dir(os.path.join(dest_root, "episodes"))
    mode = "move" if str(mode).lower() in {"move", "cut"} else "copy"
    runtime_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
    cut_trash_root = dashboard_cut_trash_root(root, runtime_id) if mode == "move" else ""
    transferred = []
    skipped = []
    run_rows: List[Tuple[str, str, List[dict]]] = []
    total_records = 0
    for raw_path in run_paths or []:
        run_dir = os.path.abspath(str(raw_path))
        run_name = os.path.basename(run_dir)
        if not run_is_under_root(root, run_dir) or not os.path.isdir(run_dir):
            skipped.append({"path": run_dir, "reason": "not_a_run_under_root"})
            continue
        rows = load_index(run_dir, "success")
        if not rows:
            skipped.append({"path": run_dir, "reason": "no_success_records"})
            continue
        run_rows.append((run_dir, run_name, rows))
        total_records += len(rows)
    done_records = 0

    def progress(message: str = ""):
        if progress_callback:
            try:
                progress_callback(done_records, max(1, total_records), message)
            except Exception:
                pass

    progress("preparing success pool")
    aggregate_rows = load_index(dest_root, "trainable")
    aggregate_rows, existing_pool_stats = normalize_existing_success_pool_rows(dest_root, aggregate_rows)
    existing_sources = build_existing_success_source_cache(dest_root, aggregate_rows)
    next_ep_index = max([int(row.get("episode_index", -1) or -1) for row in aggregate_rows], default=-1) + 1
    cached_skipped = 0
    updated_reprocessed = 0
    cut_source_episode_trashed = 0
    cut_source_episode_trash_failed = 0
    cut_source_runs_trashed = []
    cut_source_runs_trash_failed = []
    if not os.path.exists(os.path.join(dest_root, "camera_config.json")):
        for run_dir, _, _ in run_rows:
            if safe_copy_file(os.path.join(run_dir, "camera_config.json"), os.path.join(dest_root, "camera_config.json")):
                break
    if not os.path.exists(os.path.join(dest_root, "run_meta.json")):
        for run_dir, _, _ in run_rows:
            if safe_copy_file(os.path.join(run_dir, "run_meta.json"), os.path.join(dest_root, "run_meta.json")):
                break

    for run_dir, run_name, rows in run_rows:
        rewritten_rows = []
        run_manifest = {"source_run_dir": run_dir, "mode": mode, "runtime_id": runtime_id, "records": [], "skipped": [], "cached_skipped": 0, "updated_reprocessed": 0, "cut_source_episode_trashed": 0, "cut_source_episode_trash_failed": 0}
        for row in rows:
            src_dir = episode_dir_from_row(row)
            if not src_dir or not os.path.isdir(src_dir):
                item = {"episode_index": row.get("episode_index"), "reason": "episode_dir_missing", "source_episode_dir": src_dir}
                skipped.append({"path": run_dir, **item})
                run_manifest["skipped"].append(item)
                done_records += 1
                progress(f"skipped missing episode dir in {run_name}")
                continue
            source_signature = source_episode_update_signature(run_dir, row, src_dir)
            source_key = str(source_signature.get("source_key") or "")
            is_current, cache_reason = existing_success_entry_is_current(existing_sources.get(source_key), source_signature)
            if is_current:
                entry = existing_sources.get(source_key, {})
                item = {
                    "episode_index": row.get("episode_index"),
                    "reason": cache_reason,
                    "source_key": source_key,
                    "source_episode_dir": src_dir,
                    "dest_episode_dir": entry.get("transferred_episode_dir") or entry.get("dest_episode_dir", ""),
                    "source_signature_hash": source_signature.get("source_signature_hash"),
                }
                if mode == "move":
                    removed, remove_reason = safe_move_source_episode_dir_to_trash_for_cut(root, run_dir, src_dir, cut_trash_root)
                    item["cut_source_trash_reason"] = remove_reason
                    if removed:
                        cut_source_episode_trashed += 1
                        run_manifest["cut_source_episode_trashed"] += 1
                    else:
                        cut_source_episode_trash_failed += 1
                        run_manifest["cut_source_episode_trash_failed"] += 1
                skipped.append({"path": run_dir, **item})
                run_manifest["skipped"].append(item)
                cached_skipped += 1
                run_manifest["cached_skipped"] += 1
                done_records += 1
                progress(f"cached {run_name}: {done_records}/{max(1, total_records)}")
                continue
            if cache_reason == "source_updated":
                # Replace the older aggregate row for this source key so the VLA export
                # uses only the newest copy, while the old physical folder is left in
                # place for audit/recovery.
                aggregate_rows = [existing_row for existing_row in aggregate_rows if aggregate_row_source_key(existing_row) != source_key]
                updated_reprocessed += 1
                run_manifest["updated_reprocessed"] += 1
            dst_name = short_success_episode_folder_name(run_name, row, runtime_id)
            existing_entry = existing_sources.get(source_key) if isinstance(existing_sources.get(source_key), dict) else {}
            existing_dest = str(existing_entry.get("transferred_episode_dir") or "")
            dst_dir = os.path.join(episodes_root, dst_name)
            try:
                if cache_reason == "source_updated" and existing_dest:
                    existing_abs = os.path.abspath(existing_dest)
                    try:
                        if os.path.commonpath([episodes_root, existing_abs]) == episodes_root:
                            dst_dir = existing_abs
                            if os.path.isdir(dst_dir):
                                shutil.rmtree(dst_dir)
                    except Exception:
                        dst_dir = os.path.join(episodes_root, dst_name)
                elif os.path.exists(dst_dir):
                    # A canonical folder alone is not proof that the transfer is
                    # complete. Verify the destination content using a rewritten
                    # row; only then allow cache skip. If incomplete, remove the
                    # partial destination and reprocess from the source episode.
                    legacy_row = rewrite_row_paths_for_transfer(row, src_dir, dst_dir)
                    legacy_entry = {"transferred_episode_dir": dst_dir, "pool_row": legacy_row}
                    complete, complete_reason = success_pool_entry_content_complete(legacy_entry)
                    if complete:
                        item = {"episode_index": row.get("episode_index"), "reason": "canonical_folder_exists_complete", "source_key": source_key, "source_episode_dir": src_dir, "dest_episode_dir": dst_dir}
                        if mode == "move":
                            removed, remove_reason = safe_move_source_episode_dir_to_trash_for_cut(root, run_dir, src_dir, cut_trash_root)
                            item["cut_source_trash_reason"] = remove_reason
                            if removed:
                                cut_source_episode_trashed += 1
                                run_manifest["cut_source_episode_trashed"] += 1
                            else:
                                cut_source_episode_trash_failed += 1
                                run_manifest["cut_source_episode_trash_failed"] += 1
                        skipped.append({"path": run_dir, **item})
                        run_manifest["skipped"].append(item)
                        cached_skipped += 1
                        run_manifest["cached_skipped"] += 1
                        done_records += 1
                        progress(f"cached {run_name}: {done_records}/{max(1, total_records)}")
                        continue
                    try:
                        existing_abs = os.path.abspath(dst_dir)
                        if os.path.commonpath([episodes_root, existing_abs]) == episodes_root:
                            shutil.rmtree(existing_abs)
                        else:
                            raise RuntimeError("canonical_dest_not_under_episodes_root")
                    except Exception as exc:
                        item = {"episode_index": row.get("episode_index"), "reason": f"partial_dest_remove_failed:{complete_reason}:{type(exc).__name__}:{exc}", "source_key": source_key, "source_episode_dir": src_dir, "dest_episode_dir": dst_dir}
                        skipped.append({"path": run_dir, **item})
                        run_manifest["skipped"].append(item)
                        done_records += 1
                        progress(f"skipped partial dest {run_name}: {done_records}/{max(1, total_records)}")
                        continue
                if mode == "move":
                    shutil.move(src_dir, dst_dir)
                else:
                    shutil.copytree(src_dir, dst_dir)
                rewritten = rewrite_row_paths_for_transfer(row, src_dir, dst_dir)
                rewritten["source_run_name"] = run_name
                rewritten["source_run_dir"] = run_dir
                rewritten["source_episode_index"] = row.get("episode_index")
                rewritten["source_episode_id"] = row.get("episode_id", "")
                rewritten["dashboard_transfer_runtime_id"] = runtime_id
                rewritten["dashboard_transfer_mode"] = mode
                rewritten["dashboard_transfer_source_key"] = source_key
                rewritten["dashboard_transfer_source_signature_hash"] = source_signature.get("source_signature_hash")
                rewritten["dashboard_transfer_source_latest_mtime_ns"] = source_signature.get("latest_mtime_ns", 0)
                rewritten["dashboard_transfer_source_signature"] = source_signature
                rewritten["episode_index"] = int(next_ep_index)
                rewritten["episode_id"] = f"dashboard_success_{next_ep_index:06d}_{runtime_id}"
                rewritten_rows.append(rewritten)
                aggregate_rows.append(rewritten)
                record = {
                    "episode_index": int(next_ep_index),
                    "source_episode_index": row.get("episode_index"),
                    "source_episode_id": row.get("episode_id", ""),
                    "source_episode_dir": src_dir,
                    "dest_episode_dir": dst_dir,
                    "folder_name": os.path.basename(dst_dir),
                    "source_key": source_key,
                    "source_signature_hash": source_signature.get("source_signature_hash"),
                    "cache_reason": cache_reason,
                }
                run_manifest["records"].append(record)
                existing_sources[source_key] = {
                    "source_key": source_key,
                    "source_run_name": run_name,
                    "source_run_dir": run_dir,
                    "source_episode_index": row.get("episode_index"),
                    "source_episode_id": row.get("episode_id", ""),
                    "source_episode_dir": src_dir,
                    "transferred_episode_dir": dst_dir,
                    "folder_name": os.path.basename(dst_dir),
                    "pool_episode_index": int(next_ep_index),
                    "pool_episode_id": rewritten["episode_id"],
                    "pool_row": dict(rewritten),
                    "source_signature_hash": source_signature.get("source_signature_hash"),
                    "latest_mtime_ns": source_signature.get("latest_mtime_ns", 0),
                    "updated_at": time.time(),
                    "mode": mode,
                }
                next_ep_index += 1
            except Exception as exc:
                item = {"episode_index": row.get("episode_index"), "reason": f"{mode}_failed:{type(exc).__name__}:{exc}", "source_episode_dir": src_dir, "source_key": source_key}
                skipped.append({"path": run_dir, **item})
                run_manifest["skipped"].append(item)
            finally:
                done_records += 1
                progress(f"{mode} {run_name}: {done_records}/{max(1, total_records)}")
        manifest_dir = ensure_dir(os.path.join(dest_root, "transfer_manifests"))
        run_manifest.update({"created_at": time.time(), "record_count": len(rewritten_rows), "pool_dir": dest_root, "cache_path": success_transfer_cache_path(dest_root), "existing_pool_normalization": existing_pool_stats})
        run_delete_result = {"deleted": False, "reason": "not_cut"}
        if mode == "move":
            run_delete_result = move_source_run_to_trash_if_cut_complete(root, run_dir, rows, cut_trash_root)
            if run_delete_result.get("trashed"):
                cut_source_runs_trashed.append({"run": run_name, "path": run_dir})
            elif run_delete_result.get("remaining_success_dirs", 0) == 0:
                cut_source_runs_trash_failed.append({"run": run_name, "path": run_dir, "reason": run_delete_result.get("reason")})
        run_manifest["source_run_delete"] = run_delete_result
        write_json(os.path.join(manifest_dir, f"{run_name}_{runtime_id}.json"), run_manifest)
        transferred.append({"run": run_name, "source_run_dir": run_dir, "dest_dir": dest_root, "records": len(rewritten_rows), "cached_skipped": run_manifest["cached_skipped"], "updated_reprocessed": run_manifest["updated_reprocessed"], "cut_source_episode_trashed": run_manifest["cut_source_episode_trashed"], "cut_source_episode_trash_failed": run_manifest["cut_source_episode_trash_failed"], "source_run_delete": run_delete_result, "mode": mode})
    if aggregate_rows:
        write_success_pool_indexes(dest_root, aggregate_rows)
    cache_path = write_success_transfer_cache(dest_root, existing_sources)
    summary = {"ok": True, "mode": mode, "dest_root": dest_root, "episodes_dir": episodes_root, "runtime_id": runtime_id, "transferred": transferred, "skipped": skipped, "cached_skipped": cached_skipped, "updated_reprocessed": updated_reprocessed, "cut_source_episode_trashed": cut_source_episode_trashed, "cut_source_episode_trash_failed": cut_source_episode_trash_failed, "cut_source_runs_trashed": cut_source_runs_trashed, "cut_source_runs_trash_failed": cut_source_runs_trash_failed, "existing_pool_normalization": existing_pool_stats, "cache_path": cache_path, "total_pool_records": len(aggregate_rows)}
    write_json(os.path.join(dest_root, f"transfer_summary_{runtime_id}.json"), summary)
    progress("complete")
    return summary


DASHBOARD_JOBS: Dict[str, Dict[str, object]] = {}
DASHBOARD_JOBS_LOCK = threading.Lock()
DASHBOARD_JOB_COUNTER = 0


def dashboard_job_snapshot(job_id: str) -> Dict[str, object]:
    with DASHBOARD_JOBS_LOCK:
        job = DASHBOARD_JOBS.get(str(job_id))
        if not job:
            return {"ok": False, "error": f"job_not_found:{job_id}"}
        return dict(job)


def dashboard_job_update(job_id: str, **updates) -> None:
    with DASHBOARD_JOBS_LOCK:
        job = DASHBOARD_JOBS.get(str(job_id))
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()


def dashboard_start_job(kind: str, title: str, worker) -> Dict[str, object]:
    global DASHBOARD_JOB_COUNTER
    with DASHBOARD_JOBS_LOCK:
        DASHBOARD_JOB_COUNTER += 1
        job_id = f"{int(time.time() * 1000)}_{DASHBOARD_JOB_COUNTER:04d}"
        DASHBOARD_JOBS[job_id] = {
            "ok": True,
            "job_id": job_id,
            "kind": kind,
            "title": title,
            "status": "queued",
            "percent": 0.0,
            "current": 0,
            "total": 1,
            "message": "queued",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    def run_worker():
        dashboard_job_update(job_id, status="running", message="started", percent=0.0)
        try:
            result = worker(job_id)
            dashboard_job_update(job_id, status="done", message="done", percent=100.0, result=result)
        except Exception as exc:
            dashboard_job_update(job_id, status="error", message=str(exc), error=f"{type(exc).__name__}: {exc}", percent=100.0)

    thread = threading.Thread(target=run_worker, name=f"dashboard-job-{job_id}", daemon=True)
    thread.start()
    return dashboard_job_snapshot(job_id)


def dashboard_start_success_transfer_job(dataset_root: Union[str, os.PathLike], run_paths: Sequence[object], dest_dir: Union[str, os.PathLike], mode: str = "copy") -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    paths = list(run_paths or [])
    mode = "move" if str(mode).lower() in {"move", "cut"} else "copy"

    def worker(job_id: str):
        def progress(done: int, total: int, message: str):
            percent = 0.0 if total <= 0 else max(0.0, min(100.0, float(done) * 100.0 / float(total)))
            dashboard_job_update(job_id, current=int(done), total=int(max(1, total)), percent=percent, message=message or "working")
        return dashboard_transfer_success_records(root, paths, dest_dir, mode=mode, progress_callback=progress)

    return dashboard_start_job("success_records", f"{mode} success records", worker)


def dashboard_export_success_pool(
    dataset_root: Union[str, os.PathLike],
    overwrite: bool = True,
    require_vla: bool = False,
    time_policy: object = None,
    progress_callback=None,
) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    pool_dir = dashboard_success_pool_dir(root)

    def progress(percent: float, message: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(float(percent), str(message or ""), current, total)
        except Exception:
            pass

    if not os.path.isdir(pool_dir):
        raise FileNotFoundError(pool_dir)
    progress(2.0, "reconciling .dashboard_success indexes")
    reconcile_result = reconcile_success_pool_indexes(pool_dir, recover_orphan_folders=True)
    rows = load_index(pool_dir, "trainable")
    if not rows:
        raise ValueError(f"no success pool trainable rows found: {pool_dir}")
    progress(4.0, f"success pool rows={len(rows)}; recovered={reconcile_result.get('recovered_orphan_folders', 0)}")
    policy_info = save_dashboard_export_time_policy(pool_dir, time_policy if time_policy is not None else load_dashboard_export_time_policy(pool_dir).get("policy", {}))
    final_export_dir = os.path.join(pool_dir, LEROBOT_DEFAULT_EXPORT_DIRNAME)
    staging_export_dir = unique_path(os.path.join(pool_dir, f".{LEROBOT_DEFAULT_EXPORT_DIRNAME}_staging_{time.strftime('%Y%m%d_%H%M%S')}"))
    try:
        shared_exporter = getattr(shared_dataset_tools, "export_lerobot_dataset", None) if shared_dataset_tools is not None else None
        if shared_exporter is None:
            raise RuntimeError("shared VLA exporter unavailable: excavator_dataset_tools.export_lerobot_dataset")
        progress(5.0, "using shared segmented VLA exporter")
        result = shared_exporter(
            pool_dir,
            output_dir=staging_export_dir,
            split="trainable",
            overwrite=True,
            require_standard=False,
            require_vla=require_vla,
            reuse_from_dir=final_export_dir if os.path.isdir(final_export_dir) else None,
            time_policy=policy_info.get("policy"),
            progress_callback=progress,
        )
        progress(99.0, "publishing final lerobot_v3 folder")
        if overwrite and os.path.exists(final_export_dir):
            shutil.rmtree(final_export_dir)
        if os.path.exists(final_export_dir):
            raise FileExistsError(f"{final_export_dir} already exists; pass overwrite=True")
        shutil.move(staging_export_dir, final_export_dir)
        result = dict(result)
        result["export_dir"] = final_export_dir
        result["staging_export_dir"] = staging_export_dir
        result["success_pool_reconcile"] = reconcile_result
        result["time_policy"] = policy_info.get("policy")
        result["time_policy_hash"] = policy_info.get("hash")
        write_json(os.path.join(final_export_dir, "manifest.json"), result)
        progress(100.0, "VLA export complete", 1, 1)
        return result
    except Exception:
        # Keep the staging folder for post-mortem inspection, but never publish a
        # half-built export as lerobot_v3.  This prevents stale two-camera exports
        # from looking like a valid VLA dataset after a failed third-camera encode.
        raise


def dashboard_start_success_pool_export_job(
    dataset_root: Union[str, os.PathLike],
    overwrite: bool = True,
    require_vla: bool = True,
    time_policy: object = None,
) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))

    def worker(job_id: str):
        def progress(percent: float, message: str, current: Optional[int] = None, total: Optional[int] = None):
            dashboard_job_update(
                job_id,
                current=int(current if current is not None else round(percent)),
                total=int(max(1, total if total is not None else 100)),
                percent=max(0.0, min(100.0, float(percent))),
                message=message or "working",
            )

        progress(0.0, "starting .dashboard_success VLA export", 0, 100)
        result = dashboard_export_success_pool(root, overwrite=overwrite, require_vla=require_vla, time_policy=time_policy, progress_callback=progress)
        if require_vla and not result.get("vla_training_ready"):
            raise RuntimeError(f"VLA export incomplete: {json.dumps(result, ensure_ascii=True)}")
        return result

    return dashboard_start_job("export_success_vla", "Export .dashboard_success VLA", worker)

def list_dashboard_runs(dataset_root: Union[str, os.PathLike], limit: int = 80) -> List[dict]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    if not os.path.isdir(root):
        return []
    runs = []
    now = time.time()
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("run_"):
            continue
        summary = read_json(os.path.join(path, "summary.json"), default={}) or {}
        activity = run_activity_snapshot(path, now=now)
        latest_mtime = float(activity.get("latest_mtime") or 0.0)
        raw_index_counts = {key: fast_jsonl_count(index_path(path, key)) for key in INDEX_FILES}
        effective_counts, catchup = success_index_catchup_counts(path, raw_index_counts)
        success_count = int(effective_counts.get("success", 0) or 0)
        # Bounded scans keep "Loading runs..." fast even when many episode
        # image/video/data folders contain tens of thousands of files.
        size_snapshot = directory_size_snapshot(
            path,
            dataset_root=root,
        )
        data_sizes = run_data_folder_sizes(
            path,
            dataset_root=root,
        )
        runs.append(
            {
                "name": name,
                "path": path,
                "mtime": max(float(os.path.getmtime(path)), latest_mtime),
                "attempts": int(effective_counts.get("all", 0) or 0),
                "success": success_count,
                "trainable": int(effective_counts.get("trainable", 0) or 0),
                "rejected": int(effective_counts.get("rejected", 0) or 0),
                "failed": int(effective_counts.get("failed", 0) or 0),
                "requested": summary.get("requested"),
                "raw_index_counts": raw_index_counts,
                "success_catchup": catchup,
                "size_bytes": size_snapshot.get("size_bytes", 0),
                "size_human": size_snapshot.get("size_human", "-"),
                "size_truncated": size_snapshot.get("truncated", False),
                "data_folders": data_sizes,
                "data_size_human": ", ".join(f"{item.get('name')}={item.get('size_human')}" for item in data_sizes) if data_sizes else "-",
                "activity": activity,
            }
        )
    # Active writers stay visible at the top; within each state, sort by latest write time.
    runs.sort(
        key=lambda item: (
            1 if ((item.get("activity") or {}).get("active")) else 0,
            1 if ((item.get("activity") or {}).get("state") == "recent") else 0,
            float(item.get("mtime", 0.0)),
        ),
        reverse=True,
    )
    return runs[: max(1, int(limit))]




def stat_signature(path: Union[str, os.PathLike]) -> Dict[str, object]:
    text = str(path or "")
    if not text:
        return {"exists": False, "mtime_ns": 0, "size": 0}
    abs_path = os.path.abspath(text)
    try:
        st = os.stat(abs_path)
        return {
            "exists": True,
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))),
            "size": int(st.st_size),
        }
    except Exception:
        return {"exists": False, "mtime_ns": 0, "size": 0}


def direct_child_dirs_signature(path: Union[str, os.PathLike]) -> Dict[str, object]:
    root = os.path.abspath(str(path or ""))
    names: List[str] = []
    latest_mtime_ns = 0
    try:
        for entry in os.scandir(root):
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                names.append(entry.name)
                st = entry.stat(follow_symlinks=False)
                latest_mtime_ns = max(latest_mtime_ns, int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
            except Exception:
                continue
    except Exception:
        pass
    names.sort()
    names_hash = hashlib.sha1("\n".join(names).encode("utf-8", errors="replace")).hexdigest()
    return {
        "exists": os.path.isdir(root),
        "count": len(names),
        "latest_mtime_ns": int(latest_mtime_ns),
        "names_hash": names_hash,
    }


def dashboard_run_cache_dataset_root(run_dir: Union[str, os.PathLike]) -> str:
    run_abs = os.path.abspath(str(run_dir or ""))
    if is_dashboard_success_pool_dir(run_abs):
        return os.path.dirname(run_abs)
    return os.path.dirname(run_abs)


def dashboard_run_payload_cache_path(run_dir: Union[str, os.PathLike]) -> str:
    run_abs = os.path.abspath(str(run_dir or ""))
    dataset_root = dashboard_run_cache_dataset_root(run_abs)
    digest = hashlib.sha1(os.path.normcase(run_abs).encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(dataset_root, RUN_SIZE_CACHE_DIRNAME, RUN_PAYLOAD_CACHE_DIRNAME, f"{digest}.json")


def dashboard_run_payload_signature(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    run_abs = os.path.abspath(str(run_dir or ""))
    is_success_pool = is_dashboard_success_pool_dir(run_abs)
    files: Dict[str, object] = {}
    for key, filename in INDEX_FILES.items():
        files[f"index/{key}"] = stat_signature(os.path.join(run_abs, filename))
    for key, filename in SEGMENT_FILES.items():
        files[f"segment/{key}"] = stat_signature(os.path.join(run_abs, filename))
    for filename in [
        "summary.json",
        "run_meta.json",
        "camera_config.json",
        "lerobot_v3_export.json",
        "transfer_source_cache.json",
    ]:
        files[f"root/{filename}"] = stat_signature(os.path.join(run_abs, filename))
    files["export/manifest"] = stat_signature(dashboard_success_pool_export_manifest_path(run_abs))
    files["export/time_policy"] = stat_signature(dashboard_export_time_policy_path(run_abs))
    signature: Dict[str, object] = {
        "cache_version": RUN_PAYLOAD_CACHE_VERSION,
        "run_dir": run_abs,
        "run_stat": {"exists": os.path.isdir(run_abs)} if is_success_pool else stat_signature(run_abs),
        "folder_signature": {"path_mtime": 0.0, "latest_mtime": 0.0} if is_success_pool else folder_signature(run_abs),
        "files": files,
        "is_success_pool": is_success_pool,
    }
    if is_success_pool:
        # Direct directory listing only.  This detects new/orphan/removed pool
        # episode folders without walking camera/image trees or reading trajectories.
        signature["success_pool_episode_dirs"] = direct_child_dirs_signature(os.path.join(run_abs, "episodes"))
    payload = json.dumps(signature, ensure_ascii=True, sort_keys=True, default=str)
    signature["signature_hash"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return signature


def clone_jsonable(data: object) -> object:
    return json.loads(json.dumps(data, ensure_ascii=True, default=str))


def cached_dashboard_run_payload(run_dir: str, signature: Dict[str, object], timing_ms: Optional[Dict[str, float]] = None) -> Optional[Dict[str, object]]:
    signature_hash = str(signature.get("signature_hash") or "")
    cache_key = os.path.normcase(os.path.abspath(str(run_dir or "")))
    t0 = time.perf_counter()
    with RUN_PAYLOAD_CACHE_LOCK:
        cached = RUN_PAYLOAD_MEMORY_CACHE.get(cache_key)
    if timing_ms is not None:
        timing_ms["memory_cache_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    if isinstance(cached, dict) and cached.get("signature_hash") == signature_hash and isinstance(cached.get("payload"), dict):
        t_clone = time.perf_counter()
        payload = clone_jsonable(cached.get("payload"))  # type: ignore[assignment]
        if timing_ms is not None:
            timing_ms["memory_clone_ms"] = round((time.perf_counter() - t_clone) * 1000.0, 3)
        if isinstance(payload, dict):
            payload["analysis_cache"] = {
                "hit": True,
                "source": "memory",
                "signature_hash": signature_hash,
                "cache_path": cached.get("cache_path", ""),
                "created_at": cached.get("created_at", 0.0),
                "timing_ms": dict(timing_ms or {}),
            }
            return payload
    cache_path = dashboard_run_payload_cache_path(run_dir)
    t_disk = time.perf_counter()
    disk = read_json(cache_path, default={}) or {}
    if timing_ms is not None:
        timing_ms["disk_cache_read_ms"] = round((time.perf_counter() - t_disk) * 1000.0, 3)
    if (
        isinstance(disk, dict)
        and int(disk.get("version", 0) or 0) == RUN_PAYLOAD_CACHE_VERSION
        and disk.get("signature_hash") == signature_hash
        and isinstance(disk.get("payload"), dict)
    ):
        with RUN_PAYLOAD_CACHE_LOCK:
            RUN_PAYLOAD_MEMORY_CACHE[cache_key] = dict(disk)
        t_clone = time.perf_counter()
        payload = clone_jsonable(disk.get("payload"))  # type: ignore[assignment]
        if timing_ms is not None:
            timing_ms["disk_clone_ms"] = round((time.perf_counter() - t_clone) * 1000.0, 3)
        if isinstance(payload, dict):
            payload["analysis_cache"] = {
                "hit": True,
                "source": "disk",
                "signature_hash": signature_hash,
                "cache_path": cache_path,
                "created_at": disk.get("created_at", 0.0),
                "timing_ms": dict(timing_ms or {}),
            }
            return payload
    return None


def save_dashboard_run_payload_cache(run_dir: str, signature: Dict[str, object], payload: Dict[str, object]) -> str:
    cache_path = dashboard_run_payload_cache_path(run_dir)
    clean_payload = dict(payload)
    clean_payload.pop("analysis_cache", None)
    entry = {
        "version": RUN_PAYLOAD_CACHE_VERSION,
        "created_at": time.time(),
        "run_dir": os.path.abspath(str(run_dir or "")),
        "signature_hash": signature.get("signature_hash", ""),
        "signature": signature,
        "payload": clone_jsonable(clean_payload),
        "cache_path": cache_path,
    }
    ensure_dir(os.path.dirname(cache_path))
    write_json(cache_path, entry)
    cache_key = os.path.normcase(os.path.abspath(str(run_dir or "")))
    with RUN_PAYLOAD_CACHE_LOCK:
        RUN_PAYLOAD_MEMORY_CACHE[cache_key] = dict(entry)
    return cache_path


def dashboard_episode_analysis_cache_path(run_dir: Union[str, os.PathLike]) -> str:
    run_abs = os.path.abspath(str(run_dir or ""))
    dataset_root = dashboard_run_cache_dataset_root(run_abs)
    digest = hashlib.sha1(os.path.normcase(run_abs).encode("utf-8", errors="replace")).hexdigest()
    return os.path.join(dataset_root, RUN_SIZE_CACHE_DIRNAME, RUN_EPISODE_ANALYSIS_CACHE_DIRNAME, f"{digest}.json")


def load_dashboard_episode_analysis_cache(run_dir: Union[str, os.PathLike]) -> Dict[str, dict]:
    payload = read_json(dashboard_episode_analysis_cache_path(run_dir), default={}) or {}
    if not isinstance(payload, dict) or int(payload.get("version", 0) or 0) != RUN_PAYLOAD_CACHE_VERSION:
        return {}
    entries = payload.get("entries") if isinstance(payload.get("entries"), dict) else {}
    return {str(key): value for key, value in entries.items() if isinstance(value, dict)}


def save_dashboard_episode_analysis_cache(run_dir: Union[str, os.PathLike], entries: Dict[str, dict]) -> str:
    cache_path = dashboard_episode_analysis_cache_path(run_dir)
    ensure_dir(os.path.dirname(cache_path))
    write_json(
        cache_path,
        {
            "version": RUN_PAYLOAD_CACHE_VERSION,
            "created_at": time.time(),
            "run_dir": os.path.abspath(str(run_dir or "")),
            "entries": entries,
        },
    )
    return cache_path


def dashboard_episode_analysis_signature(
    run_dir: Union[str, os.PathLike],
    row: dict,
    time_policy_hash: str = "",
) -> Dict[str, object]:
    run_abs = os.path.abspath(str(run_dir or ""))
    episode_dir = episode_dir_from_row(row, run_dir=run_abs)
    files: Dict[str, object] = {"episode_dir": stat_signature(episode_dir)}
    for key in ["trajectory", "meta", "score"]:
        value = row_path_value(row, key)
        files[key] = stat_signature(resolve_episode_file(episode_dir, value)) if value else {"exists": False, "mtime_ns": 0, "size": 0}
    row_payload = json.dumps(row, ensure_ascii=True, sort_keys=True, default=str)
    signature = {
        "cache_version": RUN_PAYLOAD_CACHE_VERSION,
        "run_dir": run_abs,
        "episode_index": row.get("episode_index"),
        "episode_id": row.get("episode_id", ""),
        "row_hash": hashlib.sha1(row_payload.encode("utf-8", errors="replace")).hexdigest(),
        "files": files,
        "time_policy_hash": str(time_policy_hash or ""),
        "is_success_pool": is_dashboard_success_pool_dir(run_abs),
    }
    payload = json.dumps(signature, ensure_ascii=True, sort_keys=True, default=str)
    signature["signature_hash"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return signature


def dashboard_compute_episode_analysis(
    run_dir: Union[str, os.PathLike],
    row: dict,
    success_pool_mode: bool,
) -> Dict[str, object]:
    trajectory = load_trajectory(row)
    tag, reason = dataset_training_tag_for_row(row, trajectory)
    if success_pool_mode and trajectory:
        transformed = dashboard_export_time_runtime(run_dir, trajectory)
        runtime_s = float(transformed.get("duration_s") or 0.0)
        time_policy = {
            "policy": transformed.get("policy", default_export_time_policy()),
            "time_policy_hash": transformed.get("time_policy_hash", ""),
            "effective_fps": transformed.get("effective_fps"),
            "base_fps": transformed.get("base_fps"),
            "duration_s": transformed.get("duration_s"),
            "raw_duration_s": transformed.get("raw_duration_s"),
            "error": transformed.get("error", ""),
        }
    else:
        runtime_s = estimate_row_runtime_s(row, trajectory)
        time_policy = None
    frames = len(trajectory) if trajectory else int(safe_float_value(row.get("samples"), 0.0) or 0)
    episode = dashboard_episode_summary(
        row,
        dataset_tag=tag,
        dataset_skip_reason=reason if tag == "skip" else "",
        runtime_s=runtime_s,
    )
    if time_policy is not None:
        episode["export_time_policy"] = time_policy
    return {
        "tag": tag,
        "skip_reason": reason if tag == "skip" else "",
        "raw_status": normalized_status_name(row.get("status", "unknown")),
        "runtime_s": float(runtime_s),
        "frames": int(max(0, frames)),
        "episode": episode,
        "time_policy": time_policy,
    }


def dashboard_dataset_metrics_from_episode_analyses(rows: Sequence[dict], analyses: Sequence[dict]) -> Dict[str, object]:
    tag_counts: Counter = Counter()
    raw_status_counts: Counter = Counter()
    skip_reasons: Counter = Counter()
    tag_runtime: Counter = Counter()
    total_frames = 0
    usable_frames = 0
    skip_examples: List[dict] = []
    for row, analysis in zip(rows, analyses):
        tag = str(analysis.get("tag") or normalized_status_name(row.get("status", "unknown")))
        raw_status = str(analysis.get("raw_status") or normalized_status_name(row.get("status", "unknown")))
        reason = str(analysis.get("skip_reason") or "")
        runtime = safe_float_value(analysis.get("runtime_s"), 0.0) or 0.0
        frames = int(safe_float_value(analysis.get("frames"), row.get("samples", 0)) or 0)
        tag_counts[tag] += 1
        raw_status_counts[raw_status] += 1
        tag_runtime[tag] += float(runtime)
        total_frames += int(max(0, frames))
        if tag in {"trainable", "success"}:
            usable_frames += int(max(0, frames))
        if tag == "skip":
            skip_reasons[reason] += 1
            if len(skip_examples) < 12:
                skip_examples.append({
                    "episode_index": row.get("episode_index"),
                    "episode_id": row.get("episode_id", ""),
                    "raw_status": raw_status,
                    "skip_reason": reason,
                    "samples": row.get("samples"),
                })
    total_runtime = sum(float(value) for value in tag_runtime.values())
    return {
        "tag_counts": dict(tag_counts),
        "raw_status_counts": dict(raw_status_counts),
        "skip_reasons": top_counter(skip_reasons, 16),
        "skip_examples": skip_examples,
        "skipped_episodes": int(tag_counts.get("skip", 0)),
        "total_frames": int(total_frames),
        "usable_frames": int(usable_frames),
        "usable_frame_ratio": float(usable_frames) / float(max(1, total_frames)),
        "tag_runtime_seconds": {key: round(float(value), 3) for key, value in tag_runtime.items()},
        "total_runtime_seconds": round(float(total_runtime), 3),
        "data_efficiency_score": round(100.0 * float(usable_frames) / float(max(1, total_frames)), 2),
    }


def dashboard_episode_analyses_for_rows(
    run_dir: Union[str, os.PathLike],
    rows: Sequence[dict],
    success_pool_mode: bool,
    time_policy_hash: str = "",
) -> Tuple[List[dict], Dict[str, object]]:
    cache = load_dashboard_episode_analysis_cache(run_dir)
    next_cache: Dict[str, dict] = {}
    analyses: List[dict] = []
    hits = 0
    misses = 0
    for row in rows:
        signature = dashboard_episode_analysis_signature(run_dir, row, time_policy_hash=time_policy_hash)
        signature_hash = str(signature.get("signature_hash") or "")
        entry = cache.get(signature_hash)
        if isinstance(entry, dict) and isinstance(entry.get("analysis"), dict):
            analysis = clone_jsonable(entry.get("analysis"))  # type: ignore[assignment]
            hits += 1
        else:
            analysis = dashboard_compute_episode_analysis(run_dir, row, success_pool_mode)
            misses += 1
        next_cache[signature_hash] = {
            "signature_hash": signature_hash,
            "signature": signature,
            "analysis": analysis,
            "updated_at": time.time(),
        }
        analyses.append(analysis)
    cache_path = save_dashboard_episode_analysis_cache(run_dir, next_cache)
    return analyses, {
        "cache_path": cache_path,
        "hits": hits,
        "misses": misses,
        "entries": len(next_cache),
    }


def dashboard_run_payload_uncached(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    success_pool_reconcile = maybe_reconcile_success_pool_for_dashboard(run_dir)
    report = analyze_run(run_dir, include_timeline=False)
    compact = compact_analysis(report)
    rows, success_catchup = load_dashboard_all_rows(run_dir)
    success_export_lookup = (
        dashboard_success_pool_export_lookup(run_dir)
        if is_dashboard_success_pool_dir(run_dir)
        else {
            "has_manifest": False,
            "manifest_path": "",
            "summary": {"has_manifest": False, "reason": "not_success_pool"},
        }
    )
    success_pool_mode = is_dashboard_success_pool_dir(run_dir)
    current_policy_info = success_export_lookup.get("current_time_policy") if isinstance(success_export_lookup, dict) else {}
    current_policy_hash = str(current_policy_info.get("hash", "") if isinstance(current_policy_info, dict) else "")
    analyses, episode_cache = dashboard_episode_analyses_for_rows(
        run_dir,
        rows,
        success_pool_mode=success_pool_mode,
        time_policy_hash=current_policy_hash,
    )
    dataset_metrics = dashboard_dataset_metrics_from_episode_analyses(rows, analyses)
    episodes = []
    for row, analysis in zip(rows, analyses):
        episode = clone_jsonable(analysis.get("episode") or dashboard_episode_summary(row))  # type: ignore[assignment]
        if is_dashboard_success_pool_dir(run_dir):
            episode.update(dashboard_row_export_status(row, success_export_lookup))
        episodes.append(episode)
    scene_points = [episode["scene"] for episode in episodes]
    status_counts = Counter(str(episode.get("status", "unknown")) for episode in episodes)
    diagnosis = build_report_context(compact, report=report, all_rows=rows)
    diagnosis["dataset_metrics"] = dataset_metrics
    diagnosis["skip_reasons"] = dataset_metrics.get("skip_reasons", [])
    diagnosis["skipped_episodes"] = dataset_metrics.get("skipped_episodes", 0)
    diagnosis["usable_frames"] = dataset_metrics.get("usable_frames", 0)
    diagnosis["total_frames"] = dataset_metrics.get("total_frames", 0)
    diagnosis["data_efficiency_score"] = dataset_metrics.get("data_efficiency_score", 0)
    diagnosis["success_catchup"] = success_catchup
    diagnosis["success_pool_reconcile"] = success_pool_reconcile
    diagnosis["success_export"] = success_export_lookup.get("summary", {})
    diagnosis["episode_analysis_cache"] = episode_cache
    compact = dict(compact)
    compact_counts = dict(compact.get("counts", {}) if isinstance(compact.get("counts"), dict) else {})
    compact_counts["skip"] = int(status_counts.get("skip", 0))
    compact["counts"] = compact_counts
    return {
        "run_dir": run_dir,
        "compact": compact,
        "diagnosis": diagnosis,
        "dataset_metrics": dataset_metrics,
        "tag_runtime_seconds": dataset_metrics.get("tag_runtime_seconds", {}),
        "run_activity": run_activity_snapshot(run_dir),
        "success_catchup": success_catchup,
        "success_pool_reconcile": success_pool_reconcile,
        "status_counts": dict(status_counts),
        "episodes": episodes,
        "scene_points": scene_points,
        "generated_at": time.time(),
    }



def dashboard_run_payload(run_dir: Union[str, os.PathLike], force_refresh: bool = False) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    timing_ms: Dict[str, float] = {}
    total_t0 = time.perf_counter()
    if not force_refresh:
        signature_t0 = time.perf_counter()
        signature = dashboard_run_payload_signature(run_dir)
        timing_ms["signature_ms"] = round((time.perf_counter() - signature_t0) * 1000.0, 3)
        cached = cached_dashboard_run_payload(run_dir, signature, timing_ms=timing_ms)
        if cached is not None:
            cache = cached.get("analysis_cache") if isinstance(cached, dict) else None
            if isinstance(cache, dict):
                timing_ms["total_ms"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
                cache["timing_ms"] = dict(timing_ms)
            return cached
    rebuild_t0 = time.perf_counter()
    payload = dashboard_run_payload_uncached(run_dir)
    timing_ms["rebuild_ms"] = round((time.perf_counter() - rebuild_t0) * 1000.0, 3)
    signature_t0 = time.perf_counter()
    signature = dashboard_run_payload_signature(run_dir)
    timing_ms["post_rebuild_signature_ms"] = round((time.perf_counter() - signature_t0) * 1000.0, 3)
    save_t0 = time.perf_counter()
    cache_path = save_dashboard_run_payload_cache(run_dir, signature, payload)
    timing_ms["cache_save_ms"] = round((time.perf_counter() - save_t0) * 1000.0, 3)
    timing_ms["total_ms"] = round((time.perf_counter() - total_t0) * 1000.0, 3)
    payload["analysis_cache"] = {
        "hit": False,
        "source": "rebuilt",
        "signature_hash": signature.get("signature_hash", ""),
        "cache_path": cache_path,
        "created_at": time.time(),
        "force_refresh": bool(force_refresh),
        "timing_ms": dict(timing_ms),
    }
    return payload


def downsample_indices(count: int, max_points: int) -> List[int]:
    count = int(count)
    max_points = max(8, int(max_points))
    if count <= max_points:
        return list(range(count))
    step = float(count - 1) / float(max_points - 1)
    out = []
    last = -1
    for i in range(max_points):
        idx = int(round(i * step))
        if idx != last:
            out.append(idx)
            last = idx
    if out[-1] != count - 1:
        out.append(count - 1)
    return out


def contiguous_stage_spans(samples: Sequence[dict], t0: float) -> List[dict]:
    spans = []
    current = None
    start_t = None
    last_t = None
    for sample in samples:
        t = safe_float_value(sample.get("t"))
        if t is None:
            continue
        phase = str(sample.get("phase") or sample.get("label") or "unknown")
        rel_t = float(t) - float(t0)
        if current is None:
            current = phase
            start_t = rel_t
        elif phase != current:
            spans.append({"stage": current, "start": start_t, "end": last_t if last_t is not None else rel_t})
            current = phase
            start_t = rel_t
        last_t = rel_t
    if current is not None:
        spans.append({"stage": current, "start": start_t, "end": last_t if last_t is not None else start_t})
    return spans


def radians_vector_to_degrees(value: object, length: int = 4) -> List[Optional[float]]:
    vec = vector_or_none(value)
    if vec is None:
        return [None for _ in range(length)]
    out = []
    for index in range(length):
        if index < len(vec):
            out.append(float(vec[index]) * 180.0 / math.pi)
        else:
            out.append(None)
    return out


def dashboard_find_episode_row(run_dir: Union[str, os.PathLike], episode_index: Union[int, str]) -> Tuple[Optional[dict], List[dict]]:
    run_dir = os.path.abspath(str(run_dir))
    maybe_reconcile_success_pool_for_dashboard(run_dir)
    rows, _success_catchup = load_dashboard_all_rows(run_dir)
    wanted = str(episode_index)
    for row in rows:
        if str(row.get("episode_index")) == wanted or str(row.get("episode_id")) == wanted:
            return row, rows
    return None, rows


def dashboard_frame_context(
    run_dir: Union[str, os.PathLike],
    episode_index: Union[int, str],
    refresh: bool = False,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    wanted = str(episode_index)
    key = (run_dir, wanted)
    now = time.time()
    if not refresh:
        with FRAME_CONTEXT_CACHE_LOCK:
            cached = FRAME_CONTEXT_CACHE.get(key)
            if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= FRAME_CONTEXT_CACHE_TTL:
                return cached

    selected, _rows = dashboard_find_episode_row(run_dir, wanted)
    if selected is None:
        context = {"ok": False, "status": 404, "error": f"episode_not_found:{episode_index}", "created_at": now}
    else:
        trajectory = load_trajectory(selected)
        context = dashboard_build_frame_context(run_dir, wanted, selected, trajectory, created_at=now)
    with FRAME_CONTEXT_CACHE_LOCK:
        FRAME_CONTEXT_CACHE[key] = context
        if len(FRAME_CONTEXT_CACHE) > FRAME_CONTEXT_CACHE_LIMIT:
            oldest_key = min(FRAME_CONTEXT_CACHE, key=lambda item: float(FRAME_CONTEXT_CACHE[item].get("created_at", 0.0) or 0.0))
            FRAME_CONTEXT_CACHE.pop(oldest_key, None)
    return context


def dashboard_build_frame_context(
    run_dir: Union[str, os.PathLike],
    episode_index: Union[int, str],
    selected: dict,
    trajectory: Sequence[dict],
    created_at: Optional[float] = None,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    wanted = str(episode_index)
    image_paths: Dict[str, List[Tuple[str, object]]] = {}
    for camera_key in LEROBOT_IMAGE_KEYS:
        paths = []
        for sample in trajectory:
            if isinstance(sample, dict):
                paths.append(dashboard_resolve_sample_image_path(selected, sample, camera_key))
            else:
                paths.append(("", None))
        image_paths[camera_key] = paths
    return {
        "ok": True,
        "created_at": float(created_at if created_at is not None else time.time()),
        "run_dir": run_dir,
        "episode_index": wanted,
        "row": selected,
        "trajectory": list(trajectory),
        "image_paths": image_paths,
        "allowed_bases": [
            episode_dir_from_row(selected),
            selected.get("transferred_episode_dir"),
            selected.get("dest_episode_dir"),
            selected.get("source_episode_dir"),
        ],
    }


def dashboard_store_frame_context(context: Dict[str, object]) -> Dict[str, object]:
    if not context.get("ok"):
        return context
    run_dir = os.path.abspath(str(context.get("run_dir") or ""))
    episode_index = str(context.get("episode_index") or "")
    if not run_dir or not episode_index:
        return context
    with FRAME_CONTEXT_CACHE_LOCK:
        FRAME_CONTEXT_CACHE[(run_dir, episode_index)] = context
        if len(FRAME_CONTEXT_CACHE) > FRAME_CONTEXT_CACHE_LIMIT:
            oldest_key = min(FRAME_CONTEXT_CACHE, key=lambda item: float(FRAME_CONTEXT_CACHE[item].get("created_at", 0.0) or 0.0))
            FRAME_CONTEXT_CACHE.pop(oldest_key, None)
    return context


def dashboard_resolve_sample_image_path(row: dict, sample: dict, camera_key: str) -> Tuple[str, object]:
    image_value = sample_image_value(sample, camera_key)
    if not image_value:
        return "", image_value
    ep_dir = episode_dir_from_row(row)
    image_text = str(image_value or "")
    image_path = resolve_episode_file(ep_dir, image_text)
    source_dir = str(row.get("source_episode_dir") or "")
    dest_dir = str(row.get("transferred_episode_dir") or row.get("dest_episode_dir") or ep_dir or "")
    try:
        if (os.path.isabs(image_text) or _is_windows_absolute_path(image_text)) and source_dir and dest_dir:
            if _is_windows_absolute_path(image_text) or _is_windows_absolute_path(source_dir):
                image_norm = image_text.replace("\\", "/")
                source_norm = source_dir.replace("\\", "/").rstrip("/")
                if image_norm.lower().startswith(source_norm.lower() + "/"):
                    rel_parts = [part for part in image_norm[len(source_norm):].lstrip("/").split("/") if part]
                    mapped = os.path.join(os.path.abspath(dest_dir), *rel_parts)
                    if os.path.isfile(mapped):
                        image_path = mapped
            else:
                image_abs = os.path.abspath(image_text)
                source_abs = os.path.abspath(source_dir)
                if os.path.commonpath([source_abs, image_abs]) == source_abs:
                    mapped = os.path.join(os.path.abspath(dest_dir), os.path.relpath(image_abs, source_abs))
                    if os.path.isfile(mapped):
                        image_path = mapped
    except Exception:
        pass
    return image_path, image_value


def dashboard_path_under_any(path: str, bases: Sequence[object]) -> bool:
    try:
        path_abs = os.path.abspath(str(path or ""))
    except Exception:
        return False
    for base in bases:
        text = str(base or "")
        if not text:
            continue
        try:
            base_abs = os.path.abspath(text)
            if os.path.exists(base_abs) and os.path.commonpath([base_abs, path_abs]) == base_abs:
                return True
        except Exception:
            continue
    return False


def dashboard_image_content_type(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    if ext == ".ppm":
        return "image/x-portable-pixmap"
    return "application/octet-stream"


def dashboard_ppm_to_bmp_bytes(data: bytes) -> Optional[bytes]:
    # Minimal P6 PPM fallback for browser preview when Pillow is unavailable.
    # The collector writes 8-bit RGB PPM frames, so BMP can be generated without
    # external dependencies.
    try:
        cursor = 0
        tokens: List[bytes] = []
        length = len(data)
        while len(tokens) < 4 and cursor < length:
            while cursor < length and data[cursor] in b" \t\r\n":
                cursor += 1
            if cursor < length and data[cursor] == ord("#"):
                while cursor < length and data[cursor] not in b"\r\n":
                    cursor += 1
                continue
            start = cursor
            while cursor < length and data[cursor] not in b" \t\r\n":
                cursor += 1
            if start < cursor:
                tokens.append(data[start:cursor])
        if len(tokens) < 4 or tokens[0] != b"P6":
            return None
        while cursor < length and data[cursor] in b" \t\r\n":
            cursor += 1
        width = int(tokens[1])
        height = int(tokens[2])
        maxval = int(tokens[3])
        if width <= 0 or height <= 0 or maxval <= 0 or maxval > 255:
            return None
        expected = width * height * 3
        rgb = data[cursor:cursor + expected]
        if len(rgb) < expected:
            return None
        row_stride = width * 3
        bmp_row_stride = (row_stride + 3) & ~3
        padding = b"\x00" * (bmp_row_stride - row_stride)
        pixel_rows = []
        for y in range(height - 1, -1, -1):
            row = rgb[y * row_stride:(y + 1) * row_stride]
            bgr = bytearray(row_stride)
            bgr[0::3] = row[2::3]
            bgr[1::3] = row[1::3]
            bgr[2::3] = row[0::3]
            pixel_rows.append(bytes(bgr) + padding)
        pixel_data = b"".join(pixel_rows)
        file_size = 14 + 40 + len(pixel_data)
        file_header = b"BM" + struct.pack("<IHHI", file_size, 0, 0, 54)
        dib_header = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0, len(pixel_data), 2835, 2835, 0, 0)
        return file_header + dib_header + pixel_data
    except Exception:
        return None


def dashboard_read_preview_image(path: str) -> Tuple[bytes, str]:
    try:
        stat = os.stat(path)
        cache_key = f"{os.path.abspath(path)}|{int(stat.st_mtime_ns)}|{int(stat.st_size)}"
        now = time.time()
        with FRAME_IMAGE_CACHE_LOCK:
            cached = FRAME_IMAGE_CACHE.get(cache_key)
            if cached and now - float(cached.get("created_at", 0.0) or 0.0) <= FRAME_IMAGE_CACHE_TTL:
                return cached.get("data", b""), str(cached.get("content_type") or "application/octet-stream")
    except Exception:
        cache_key = ""
        now = time.time()

    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in {".ppm", ".pnm", ".pgm", ".pbm"}:
        with open(path, "rb") as f:
            raw_data = f.read()
        bmp_data = dashboard_ppm_to_bmp_bytes(raw_data)
        if bmp_data:
            data, content_type = bmp_data, "image/bmp"
            if cache_key:
                with FRAME_IMAGE_CACHE_LOCK:
                    FRAME_IMAGE_CACHE[cache_key] = {"created_at": now, "data": data, "content_type": content_type}
                    if len(FRAME_IMAGE_CACHE) > FRAME_IMAGE_CACHE_LIMIT:
                        oldest_key = min(FRAME_IMAGE_CACHE, key=lambda item: float(FRAME_IMAGE_CACHE[item].get("created_at", 0.0) or 0.0))
                        FRAME_IMAGE_CACHE.pop(oldest_key, None)
            return data, content_type
        try:
            from PIL import Image  # type: ignore
            with Image.open(io.BytesIO(raw_data)) as image:
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data, content_type = buffer.getvalue(), "image/png"
        except Exception:
            data, content_type = raw_data, dashboard_image_content_type(path)
        if cache_key:
            with FRAME_IMAGE_CACHE_LOCK:
                FRAME_IMAGE_CACHE[cache_key] = {"created_at": now, "data": data, "content_type": content_type}
                if len(FRAME_IMAGE_CACHE) > FRAME_IMAGE_CACHE_LIMIT:
                    oldest_key = min(FRAME_IMAGE_CACHE, key=lambda item: float(FRAME_IMAGE_CACHE[item].get("created_at", 0.0) or 0.0))
                    FRAME_IMAGE_CACHE.pop(oldest_key, None)
        return data, content_type
    with open(path, "rb") as f:
        data, content_type = f.read(), dashboard_image_content_type(path)
    if cache_key:
        with FRAME_IMAGE_CACHE_LOCK:
            FRAME_IMAGE_CACHE[cache_key] = {"created_at": now, "data": data, "content_type": content_type}
            if len(FRAME_IMAGE_CACHE) > FRAME_IMAGE_CACHE_LIMIT:
                oldest_key = min(FRAME_IMAGE_CACHE, key=lambda item: float(FRAME_IMAGE_CACHE[item].get("created_at", 0.0) or 0.0))
                FRAME_IMAGE_CACHE.pop(oldest_key, None)
    return data, content_type


def dashboard_camera_key(value: object) -> Optional[str]:
    text = str(value or "").strip()
    if text in LEROBOT_IMAGE_KEYS:
        return text
    aliases = {
        "0": "observation.images.0",
        "cam0": "observation.images.0",
        "camera0": "observation.images.0",
        "1": "observation.images.1",
        "cam1": "observation.images.1",
        "camera1": "observation.images.1",
        "2": "observation.images.2",
        "cam2": "observation.images.2",
        "camera2": "observation.images.2",
    }
    return aliases.get(text.lower())


def dashboard_ppm_quick_mean(path: str, sample_pixels: int = 1024) -> Optional[float]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        cursor = 0
        tokens: List[bytes] = []
        length = len(data)
        while len(tokens) < 4 and cursor < length:
            while cursor < length and data[cursor] in b" \t\r\n":
                cursor += 1
            if cursor < length and data[cursor] == ord("#"):
                while cursor < length and data[cursor] not in b"\r\n":
                    cursor += 1
                continue
            start = cursor
            while cursor < length and data[cursor] not in b" \t\r\n":
                cursor += 1
            if start < cursor:
                tokens.append(data[start:cursor])
        if len(tokens) < 4 or tokens[0] != b"P6":
            return None
        while cursor < length and data[cursor] in b" \t\r\n":
            cursor += 1
        width = int(tokens[1])
        height = int(tokens[2])
        maxval = int(tokens[3])
        if width <= 0 or height <= 0 or maxval <= 0:
            return None
        rgb = data[cursor:cursor + width * height * 3]
        if not rgb:
            return None
        stride = max(3, (len(rgb) // max(1, int(sample_pixels))) // 3 * 3)
        total = 0
        count = 0
        for offset in range(0, len(rgb) - 2, stride):
            total += int(rgb[offset]) + int(rgb[offset + 1]) + int(rgb[offset + 2])
            count += 3
        if count <= 0:
            return None
        return float(total) / float(count)
    except Exception:
        return None


def camera_preview_phase(sample: object) -> str:
    if not isinstance(sample, dict):
        return ""
    return str(sample.get("phase") or sample.get("label") or "").strip()


def camera_preview_phase_startswith(phase: str, prefixes: Sequence[str]) -> bool:
    text = str(phase or "").strip().lower()
    return any(text.startswith(str(prefix).lower()) for prefix in prefixes)


def camera_preview_frame_has_image(
    frame_index: int,
    image_paths: Dict[str, List[Tuple[str, object]]],
) -> bool:
    idx = int(frame_index)
    for key in LEROBOT_IMAGE_KEYS:
        paths = image_paths.get(key) if isinstance(image_paths, dict) else None
        if isinstance(paths, list) and 0 <= idx < len(paths):
            path, _value = paths[idx]
            if path and os.path.isfile(path):
                return True
    return False


def build_episode_camera_preview(
    row: dict,
    trajectory: Sequence[dict],
    first_t: float = 0.0,
    image_paths: Optional[Dict[str, List[Tuple[str, object]]]] = None,
) -> Dict[str, object]:
    cameras = []
    for index, key in enumerate(LEROBOT_IMAGE_KEYS):
        present = 0
        existing = 0
        first_valid_frame = None
        first_nonblack_frame = None
        sampled_dark_frames = 0
        sampled_frames = 0
        missing_examples = []
        for frame_index, sample in enumerate(trajectory or []):
            if not isinstance(sample, dict):
                continue
            cached_paths = image_paths.get(key) if isinstance(image_paths, dict) else None
            if isinstance(cached_paths, list) and frame_index < len(cached_paths):
                path, value = cached_paths[frame_index]
            else:
                path, value = dashboard_resolve_sample_image_path(row, sample, key)
            if value:
                present += 1
            if path and os.path.isfile(path):
                existing += 1
                if first_valid_frame is None:
                    first_valid_frame = frame_index
                if sampled_frames < 12 or frame_index in {0, len(trajectory or []) // 2, max(0, len(trajectory or []) - 1)}:
                    sampled_frames += 1
                    mean_value = dashboard_ppm_quick_mean(path)
                    if mean_value is not None:
                        if mean_value <= 2.0:
                            sampled_dark_frames += 1
                        elif first_nonblack_frame is None:
                            first_nonblack_frame = frame_index
            elif value and len(missing_examples) < 3:
                missing_examples.append({"frame_index": frame_index, "value": value, "resolved": path})
        cameras.append({
            "key": key,
            "index": index,
            "label": f"Cam {index}",
            "present_frames": int(present),
            "existing_frames": int(existing),
            "available": bool(existing > 0),
            "first_valid_frame": first_valid_frame,
            "first_nonblack_frame": first_nonblack_frame,
            "sampled_dark_frames": int(sampled_dark_frames),
            "sampled_frames": int(sampled_frames),
            "looks_all_black": bool(existing > 0 and sampled_frames > 0 and sampled_dark_frames == sampled_frames and first_nonblack_frame is None),
            "missing_examples": missing_examples,
        })
    frame_meta = []
    for frame_index, sample in enumerate(trajectory or []):
        t = safe_float_value(sample.get("t") if isinstance(sample, dict) else None, first_t)
        phase = camera_preview_phase(sample)
        frame_meta.append({
            "index": int(frame_index),
            "t": float(t - first_t) if t is not None else None,
            "raw_t": float(t) if t is not None else None,
            "phase": phase or "unknown",
        })
    first_available = 0
    for cam in cameras:
        preferred_frame = cam.get("first_nonblack_frame")
        if preferred_frame is None:
            preferred_frame = cam.get("first_valid_frame")
        if preferred_frame is not None:
            first_available = int(preferred_frame or 0)
            break
    for frame_index, sample in enumerate(trajectory or []):
        phase = camera_preview_phase(sample)
        if (
            camera_preview_phase_startswith(phase, CAMERA_PREVIEW_MEANINGFUL_PHASE_PREFIXES)
            and not camera_preview_phase_startswith(phase, CAMERA_PREVIEW_SKIP_INITIAL_PHASE_PREFIXES)
            and camera_preview_frame_has_image(frame_index, image_paths)
        ):
            first_available = int(frame_index)
            break
    return {
        "frame_count": len(trajectory or []),
        "initial_frame_index": int(first_available),
        "cameras": cameras,
        "frames": frame_meta,
    }


def dashboard_episode_frame_image(
    run_dir: Union[str, os.PathLike],
    episode_index: Union[int, str],
    camera: object,
    frame_index: Union[int, str],
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    context = dashboard_frame_context(run_dir, episode_index)
    if not context.get("ok"):
        return context
    key = dashboard_camera_key(camera)
    if not key:
        return {"ok": False, "status": 400, "error": f"camera_not_found:{camera}"}
    trajectory = context.get("trajectory") if isinstance(context.get("trajectory"), list) else []
    if not trajectory:
        return {"ok": False, "status": 404, "error": "trajectory_empty"}
    try:
        idx = int(frame_index)
    except Exception:
        idx = 0
    idx = max(0, min(len(trajectory) - 1, idx))

    candidate_indices = [idx]
    for radius in range(1, min(64, len(trajectory))):
        if idx - radius >= 0:
            candidate_indices.append(idx - radius)
        if idx + radius < len(trajectory):
            candidate_indices.append(idx + radius)
    image_path = ""
    image_value = None
    resolved_index = idx
    image_paths = context.get("image_paths") if isinstance(context.get("image_paths"), dict) else {}
    camera_paths = image_paths.get(key) if isinstance(image_paths.get(key), list) else []
    for candidate in candidate_indices:
        if candidate < len(camera_paths):
            path, value = camera_paths[candidate]
        else:
            sample = trajectory[candidate]
            path, value = dashboard_resolve_sample_image_path(context.get("row") or {}, sample, key)
        if path and os.path.isfile(path):
            image_path = path
            image_value = value
            resolved_index = candidate
            break
    if not image_path:
        return {"ok": False, "status": 404, "error": f"image_not_found:{key}:frame={idx}"}

    allowed_bases = context.get("allowed_bases") if isinstance(context.get("allowed_bases"), list) else []
    if not dashboard_path_under_any(image_path, allowed_bases):
        return {"ok": False, "status": 403, "error": "image_path_not_under_episode_dir"}
    try:
        data, content_type = dashboard_read_preview_image(image_path)
    except Exception as exc:
        return {"ok": False, "status": 500, "error": f"image_read_failed:{type(exc).__name__}:{exc}"}
    return {
        "ok": True,
        "data": data,
        "content_type": content_type,
        "path": image_path,
        "camera": key,
        "requested_frame_index": idx,
        "resolved_frame_index": resolved_index,
        "image_value": image_value,
    }


def dashboard_episode_payload(
    run_dir: Union[str, os.PathLike],
    episode_index: Union[int, str],
    max_points: int = 1800,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    selected, _rows = dashboard_find_episode_row(run_dir, episode_index)
    if selected is None:
        return {"ok": False, "reason": f"episode_not_found:{episode_index}", "run_dir": run_dir}
    raw_trajectory = load_trajectory(selected)
    time_transform = dashboard_apply_export_time_policy(run_dir, raw_trajectory, selected)
    trajectory = list(time_transform.get("samples") or raw_trajectory)
    if not raw_trajectory:
        episode = dashboard_episode_summary(selected, dataset_tag="skip", dataset_skip_reason="trajectory_empty", runtime_s=0.0)
        empty_series = {
            "t": [], "phase": [], "bucket_from_pile": [], "bucket_total": [], "bucket_mass": [],
            "q_deg": [], "dq_deg_s": [], "ddq_deg_s2": [], "cmd_q_deg": [], "q_err_deg": [],
            "action_deg_s": [], "action_accel_deg_s2": [], "effort": [],
        }
        return {
            "ok": True,
            "reason": "trajectory_empty",
            "run_dir": run_dir,
            "episode": episode,
            "sample_count": 0,
            "returned_points": 0,
            "stage_spans": [],
            "joint_names": ["swing", "boom", "arm", "bucket"],
            "series": empty_series,
            "camera_preview": {"frame_count": 0, "initial_frame_index": 0, "cameras": [], "frames": []},
            "time_policy": time_transform,
        }
    episode_meta = {}
    try:
        episode_dir = episode_dir_from_row(selected)
        episode_meta = read_json(resolve_episode_file(episode_dir, row_path_value(selected, "meta")), default={}) or {}
    except Exception:
        episode_meta = {}
    episode_summary = dashboard_episode_summary(selected, runtime_s=trajectory_runtime_s(trajectory))
    episode_summary["task_prompt"] = dashboard_episode_task_prompt(selected, trajectory=trajectory, episode_meta=episode_meta)
    episode_summary["export_time_policy"] = {
        "policy": time_transform.get("time_policy", default_export_time_policy()),
        "time_policy_hash": time_transform.get("time_policy_hash", ""),
        "base_fps": time_transform.get("base_fps"),
        "effective_fps": time_transform.get("effective_fps"),
        "duration_s": time_transform.get("duration_s"),
        "raw_duration_s": time_transform.get("raw_duration_s"),
        "error": time_transform.get("error", ""),
    }
    first_t = safe_float_value(trajectory[0].get("t"), 0.0) or 0.0
    frame_context = dashboard_store_frame_context(dashboard_build_frame_context(run_dir, episode_index, selected, trajectory))
    indices = downsample_indices(len(trajectory), max_points)
    series = {
        "t": [],
        "phase": [],
        "bucket_from_pile": [],
        "bucket_total": [],
        "bucket_mass": [],
        "q_deg": [],
        "dq_deg_s": [],
        "ddq_deg_s2": [],
        "cmd_q_deg": [],
        "q_err_deg": [],
        "action_deg_s": [],
        "action_accel_deg_s2": [],
        "effort": [],
    }
    for index in indices:
        sample = trajectory[index]
        t = safe_float_value(sample.get("t"), first_t) or first_t
        sand = sample.get("sand") if isinstance(sample.get("sand"), dict) else {}
        effort = vector_or_none(sample.get("observation.effort"))
        series["t"].append(float(t) - float(first_t))
        series["phase"].append(str(sample.get("phase") or sample.get("label") or "unknown"))
        series["bucket_from_pile"].append(safe_float_value(sand.get("bucket_from_pile"), 0.0))
        series["bucket_total"].append(safe_float_value(sand.get("bucket"), 0.0))
        series["bucket_mass"].append(safe_float_value(sand.get("bucket_from_pile_mass"), 0.0))
        series["q_deg"].append(radians_vector_to_degrees(sample.get("obs.q")))
        series["dq_deg_s"].append(radians_vector_to_degrees(sample.get("obs.dq")))
        series["ddq_deg_s2"].append(radians_vector_to_degrees(sample.get("obs.ddq")))
        series["cmd_q_deg"].append(radians_vector_to_degrees(sample.get("obs.q_cmd")))
        series["q_err_deg"].append(radians_vector_to_degrees(sample.get("obs.q_err")))
        series["action_deg_s"].append(radians_vector_to_degrees(sample.get("action")))
        series["action_accel_deg_s2"].append(radians_vector_to_degrees(sample.get("action.ddq")))
        series["effort"].append(effort[:4] if effort else [None, None, None, None])
    return {
        "ok": True,
        "run_dir": run_dir,
        "episode": episode_summary,
        "sample_count": len(trajectory),
        "returned_points": len(indices),
        "stage_spans": contiguous_stage_spans(trajectory, first_t),
        "joint_names": ["swing", "boom", "arm", "bucket"],
        "series": series,
        "time_policy": episode_summary["export_time_policy"],
        "camera_preview": build_episode_camera_preview(
            selected,
            trajectory,
            first_t,
            image_paths=frame_context.get("image_paths") if isinstance(frame_context.get("image_paths"), dict) else None,
        ),
    }


def dashboard_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Excavator Training Dataset Generation Dashboard</title>
<style>
:root{
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, "Microsoft YaHei", sans-serif;
  color:#111827;background:#f5f7fb;--border:#d8e0eb;--muted:#667085;--panel:#ffffff;
  --good:#067647;--bad:#b42318;--warn:#b54708;--info:#175cd3;--ink:#101828;
}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#eef3fb 0,#f6f8fb 220px,#f6f8fb 100%);}
.topbarTitleRow{display:flex;align-items:center;justify-content:space-between;gap:12px}
.darkModeToggle{min-width:36px;width:36px;height:34px;padding:0;border-radius:999px;background:#fff;color:#111827;border:1px solid #cbd5e1;font-size:17px;line-height:1;display:inline-flex;align-items:center;justify-content:center;box-shadow:0 1px 2px rgba(16,24,40,.05)}
.darkModeToggle:hover{background:#f8fafc}
body.dark{background:linear-gradient(180deg,#0b1220 0,#111827 240px,#111827 100%);color:#e5e7eb;--panel:#111827;--border:#334155;--muted:#94a3b8;--ink:#f8fafc}
body.dark .topbar{background:rgba(15,23,42,.96);border-bottom-color:#334155;box-shadow:0 6px 18px rgba(0,0,0,.30)}
body.dark h1,body.dark h2,body.dark h3,body.dark .cameraPreviewTitle,body.dark .cameraCardTitle,body.dark .diagCardValue,body.dark .runMonitorTitle,body.dark .managerTable .nameCell{color:#f8fafc}
body.dark .panel,body.dark .kpi,body.dark .diagCard,body.dark .subPanel,body.dark .managerPanel,body.dark .episodeSide,body.dark .timelinePaneHeader,body.dark .cameraPreview,body.dark .cameraCard,body.dark .miniChart,body.dark .finding,body.dark .action{background:#0f172a;border-color:#334155;color:#e5e7eb}
body.dark input,body.dark select,body.dark button.secondary,body.dark .cameraPlayerBtn,body.dark .filterBtn,body.dark .runBadge,body.dark .unitLegend,body.dark .meshFrameBadge,body.dark .darkModeToggle{background:#111827;color:#e5e7eb;border-color:#475569}
body.dark button{border-color:#334155}
body.dark .muted,body.dark .panelHint,body.dark .filterCount,body.dark .diagCardDetail,body.dark .diagCardTitle,body.dark .sideFooter,body.dark .sideSortHint,body.dark .chartLegend,body.dark .plotNote{color:#94a3b8}
body.dark .tableWrap,body.dark .managerTableWrap,body.dark .episodeTabsList{background:#0b1220;border-color:#334155}
body.dark th,body.dark .episodeDataSheet th,body.dark .managerTable th{background:#111827;color:#cbd5e1;border-bottom-color:#334155}
body.dark td,body.dark .episodeDataSheet td,body.dark .managerTable td{color:#e5e7eb;border-bottom-color:#1e293b}
body.dark tbody tr:hover,body.dark .episodeDataSheet tbody tr:hover td{background:#172033}
body.dark tbody tr.selected,body.dark .episodeDataSheet tbody tr.selected td,body.dark .episodeDataSheet tbody tr.selected .epCol{background:#1e3a5f}
body.dark .episodeDataSheet .epCol{background:#0f172a;color:#f8fafc}
body.dark .chart{background:#0f172a;border-color:#334155}body.dark svg text{fill:#cbd5e1!important}body.dark svg rect[fill="#fcfcfd"],body.dark svg rect[fill="#ffffff"]{fill:#0f172a!important}body.dark svg line[stroke="#eef2f6"]{stroke:#334155!important}body.dark svg rect[stroke="#d0d5dd"]{stroke:#475569!important}
body.dark .empty,body.dark .reportHint,body.dark .taskPrompt,body.dark .diagStat,body.dark .transferProgress{background:#111827;border-color:#334155;color:#94a3b8}
body.dark .barTrack,body.dark .scoreBar,body.dark .transferProgressTrack{background:#334155}
body.dark .codeBox{background:#020617;color:#cbd5e1}
.shell{max-width:1720px;margin:0 auto;padding:16px 18px 28px}
.topbar{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);box-shadow:0 6px 18px rgba(16,24,40,.06)}
.topbar .shell{padding-top:12px;padding-bottom:12px}
h1{font-size:21px;line-height:1.2;margin:0 0 10px;color:#101828}
h2{font-size:15px;line-height:1.25;margin:0;color:#101828}
h3{font-size:13px;margin:0 0 8px;color:#344054;text-transform:uppercase;letter-spacing:.04em}
.controls{display:grid;gap:8px;align-items:center}.rootControls{grid-template-columns:auto minmax(260px,1fr) auto auto auto minmax(320px,560px) auto}.runControls{grid-template-columns:auto minmax(320px,1fr) auto auto minmax(120px,auto)}
label{font-size:12px;color:#475467;font-weight:600;white-space:nowrap}
input,select,button{min-height:34px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;padding:0 10px;font-size:13px;min-width:0}
input.path{width:100%}select{width:100%}
button{background:#1f2937;color:#fff;border-color:#1f2937;cursor:pointer;font-weight:600}
button.secondary{background:#fff;color:#111827;border-color:#cbd5e1}.linkBtn{border:0;background:transparent;color:#175cd3;padding:0;min-height:0;font-weight:800;text-align:left;cursor:pointer}.linkBtn:hover{text-decoration:underline}
.statusLine{margin-top:8px;display:flex;gap:10px;align-items:center;min-height:18px}.runMonitor{margin-top:9px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:#475467}.runMonitorTitle{font-weight:800;color:#101828}.runBadge{display:inline-flex;align-items:center;gap:6px;border:1px solid #d0d5dd;border-radius:999px;background:#fff;color:#344054;padding:4px 9px;min-height:26px;font-size:12px;cursor:pointer}.runBadge.active{border-color:#12b76a;background:#ecfdf3;color:#027a48}.runBadge.recent{border-color:#fdb022;background:#fffaeb;color:#b54708}.activityDot{width:9px;height:9px;border-radius:999px;display:inline-block;background:#98a2b3;box-shadow:0 0 0 2px rgba(152,162,179,.14)}.activityDot.active{background:#12b76a;box-shadow:0 0 0 3px rgba(18,183,106,.18)}.activityDot.recent{background:#fdb022;box-shadow:0 0 0 3px rgba(253,176,34,.18)}.activityDot.idle{background:#f04438;box-shadow:0 0 0 3px rgba(240,68,56,.14)}.activityDot.missing,.activityDot.unknown{background:#98a2b3}.managerPanel{margin-bottom:14px;padding:0}.managerPanel>summary{cursor:pointer;list-style:none;padding:14px 16px;display:flex;align-items:flex-start;justify-content:space-between;gap:14px;border-bottom:1px solid #eaecf0;position:relative}.managerPanel>summary::-webkit-details-marker{display:none}.managerPanel>summary:after{content:"";width:10px;height:10px;border-right:2px solid #667085;border-bottom:2px solid #667085;transform:rotate(-45deg);transition:transform .2s ease;flex:0 0 auto;margin-top:4px}.managerPanel[open]>summary:after{transform:rotate(45deg);margin-top:7px}.managerPanel:not([open])>summary{border-bottom:0}.managerBody{padding:12px 14px 14px}.managerToolbar{display:flex;flex-wrap:nowrap;overflow-x:auto;gap:8px;align-items:center;margin-bottom:10px;padding-bottom:2px}.managerToolbar button{flex:0 0 auto}.managerToolbar input.path{flex:1 0 360px;min-width:260px}.managerToolbar .danger{background:#b42318;border-color:#b42318;color:#fff}.managerToolbar .warn{background:#b54708;border-color:#b54708;color:#fff}.managerSummary{font-size:12px;color:#475467;margin-bottom:8px;min-height:18px}.managerTableWrap{max-height:260px;overflow:auto;border:1px solid #eaecf0;border-radius:10px;background:#fff;scrollbar-width:none;-ms-overflow-style:none}.managerTableWrap::-webkit-scrollbar{display:none}.managerTable{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}.managerTable th,.managerTable td{padding:7px 8px;border-bottom:1px solid #eef2f6;white-space:nowrap;vertical-align:middle}.managerTable th{position:sticky;top:0;background:#f8fafc;z-index:2;text-transform:uppercase;letter-spacing:.04em;font-size:10.5px;color:#475467}.managerTable .nameCell{font-weight:800;color:#101828}.managerTable .num{text-align:right;font-variant-numeric:tabular-nums}.managerTable .zeroSuccess{color:#b42318;font-weight:850}.managerTable .successRun{color:#067647;font-weight:850}.managerTable .dataSizeCell{max-width:280px;overflow:hidden;text-overflow:ellipsis}.ok{color:#047857}.error{color:#b91c1c}.muted{color:var(--muted);font-size:12px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.small{font-size:12px}.nowrap{white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;box-shadow:0 1px 2px rgba(16,24,40,.04);padding:14px;min-width:0;overflow:hidden}
.panelHeader{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin-bottom:10px}.panelHint{font-size:12px;color:#667085;line-height:1.35}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span5{grid-column:span 5}.span6{grid-column:span 6}.span7{grid-column:span 7}.span8{grid-column:span 8}.span12{grid-column:span 12}
.kpis{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:10px;margin-bottom:14px}.kpi{background:#fff;border:1px solid var(--border);border-radius:14px;padding:12px;min-width:0;box-shadow:0 1px 2px rgba(16,24,40,.04)}.kpiLabel{font-size:11px;color:#667085;text-transform:uppercase;letter-spacing:.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.kpiValue{margin-top:5px;font-size:24px;line-height:1.05;font-weight:800;color:#101828;overflow-wrap:anywhere}.kpiSub{margin-top:5px;font-size:12px;color:#667085;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.kpi.good .kpiValue{color:var(--good)}.kpi.bad .kpiValue{color:var(--bad)}.kpi.warn .kpiValue{color:var(--warn)}
.hero{margin-bottom:14px}.heroGrid{display:grid;grid-template-columns:280px minmax(0,1fr) minmax(0,1fr);gap:14px}.scoreBox{border-radius:14px;border:1px solid #e5e7eb;background:#f8fafc;padding:14px;min-height:160px}.scoreNumber{font-size:54px;line-height:1;font-weight:850;color:#101828}.scoreLabel{font-size:12px;color:#667085;text-transform:uppercase;letter-spacing:.05em}.scoreBar{height:10px;background:#e5e7eb;border-radius:999px;margin-top:14px;overflow:hidden}.scoreFill{height:100%;border-radius:999px;background:#175cd3}.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.chip{font-size:11px;border:1px solid #d0d5dd;border-radius:999px;padding:3px 8px;color:#344054;background:#fff}
.finding,.action{border:1px solid #eaecf0;border-radius:10px;padding:9px 10px;margin-top:8px;background:#fff}.findingTitle{font-size:13px;font-weight:700;color:#101828}.findingDetail{font-size:12px;color:#667085;line-height:1.35;margin-top:3px}.sev-critical{border-left:4px solid #b42318}.sev-high{border-left:4px solid #f04438}.sev-medium{border-left:4px solid #f79009}.sev-info{border-left:4px solid #2e90fa}.action{font-size:13px;line-height:1.45;color:#344054;background:#fcfcfd}
.barRows{display:grid;gap:8px}.barRow{display:grid;grid-template-columns:minmax(130px,260px) minmax(120px,1fr) 52px;gap:10px;align-items:center}.barLabel{font-size:12px;color:#344054;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.barTrack{height:14px;border-radius:999px;background:#eef2f6;overflow:hidden}.barFill{height:100%;border-radius:999px}.barVal{text-align:right;font-size:12px;color:#475467;font-variant-numeric:tabular-nums}.empty{font-size:12px;color:#98a2b3;padding:12px;border:1px dashed #d0d5dd;border-radius:10px;background:#fcfcfd}
.chartBox{min-height:280px}.chart{display:block;width:100%;height:auto;border:1px solid #edf2f7;border-radius:10px;background:#fff;overflow:visible}.chartLegend{display:flex;flex-wrap:wrap;gap:10px;margin-top:7px;font-size:12px;color:#475467}.legendItem{display:inline-flex;align-items:center;gap:5px}.legendDot{width:9px;height:9px;border-radius:999px;display:inline-block}.timelineChart{margin-top:10px}.timelineChart .chart{min-height:210px}.timelineGrid{display:grid;grid-template-columns:1fr;gap:10px}.detailGrid{display:grid;grid-template-columns:minmax(420px,5fr) minmax(560px,7fr);gap:14px;align-items:start}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}.tableWrap{max-height:560px;overflow:auto;border:1px solid #eaecf0;border-radius:10px;background:#fff}th,td{padding:8px 9px;border-bottom:1px solid #eaecf0;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#f8fafc;color:#475467;font-size:11px;text-transform:uppercase;letter-spacing:.04em;z-index:1}td{color:#344054}.sortable{cursor:pointer;user-select:none;color:#175cd3}tbody tr{cursor:pointer}tbody tr:hover{background:#f8fafc}tbody tr.selected{background:#e0f2fe}.reasonCell{max-width:360px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.pill{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700}.pill.trainable,.pill.success,.gate-pass{background:#ecfdf3;color:#067647}.pill.rejected,.gate-fail{background:#fef3f2;color:#b42318}.pill.failed{background:#fff1f3;color:#912018}.pill.diagnostic,.gate-warn{background:#fffaeb;color:#b54708}.gate-pass,.gate-fail,.gate-warn{border-radius:999px;padding:3px 7px;font-weight:700;font-size:11px}.metricsTable td:nth-child(n+2),.metricsTable th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}.schemaGrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.codeBox{white-space:pre-wrap;font-size:12px;max-height:320px;overflow:auto;background:#0f172a;color:#e2e8f0;padding:12px;border-radius:10px}.reportHint{padding:8px 10px;background:#f8fafc;border:1px solid #eaecf0;border-radius:10px;color:#667085;font-size:12px;margin-top:8px;word-break:break-all}
@media(max-width:1280px){.kpis{grid-template-columns:repeat(4,minmax(0,1fr))}.heroGrid,.detailGrid{grid-template-columns:1fr}.span3,.span4,.span5,.span6,.span7,.span8{grid-column:span 12}.controls{grid-template-columns:1fr}.controls label{display:none}.barRow{grid-template-columns:minmax(110px,210px) 1fr 44px}}
@media(max-width:720px){.shell{padding:12px}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.kpiValue{font-size:20px}.panel{padding:12px}.schemaGrid{grid-template-columns:1fr}}

/* v4 layout fixes */
.statusFilterBar{margin-top:10px;border-top:1px solid #eaecf0;padding-top:10px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap}
.statusFilterTitle{font-size:12px;color:#475467;font-weight:700;margin-right:4px}.statusFilter{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.filterBtn{min-height:28px;border:1px solid #d0d5dd;border-radius:999px;background:#fff;color:#344054;padding:0 10px;font-size:12px;font-weight:700;cursor:pointer}.filterBtn.active{color:#fff;border-color:transparent}.filterBtn.all.active{background:#344054}.filterBtn.trainable.active,.filterBtn.success.active{background:#067647}.filterBtn.rejected.active{background:#d92d20}.filterBtn.failed.active,.filterBtn.fail.active{background:#f79009;color:#111827}.filterBtn.diagnostic.active{background:#6941c6}.filterBtn.planning.active{background:#175cd3}.filterBtn.skip.active{background:#475467}.filterBtn.unknown.active{background:#667085}.filterCount{font-size:12px;color:#667085;white-space:nowrap}.chartMeta{font-size:11px;color:#667085;line-height:1.35;margin-top:6px}.miniChartGrid{display:grid;grid-template-columns:1fr;gap:10px}.miniChart{border:1px solid #eaecf0;border-radius:12px;background:#fcfcfd;padding:10px;min-width:0}.miniChart h3{margin:0 0 6px;font-size:12px;color:#344054;text-transform:none;letter-spacing:0}.miniChart .chart{max-height:165px}.smallChartBox{min-height:0}.pill.failed,.pill.fail{background:#fff7ed;color:#c2410c}.pill.rejected{background:#fef3f2;color:#b42318}.pill.diagnostic{background:#f4f3ff;color:#5925dc}.pill.skip{background:#f2f4f7;color:#344054}.episodeInspector{display:grid;grid-template-columns:minmax(560px,42%) minmax(0,1fr);gap:14px;align-items:start;min-height:0}.episodeSide{background:#f8fafc;border:1px solid #eaecf0;border-radius:12px;padding:10px;display:flex;flex-direction:column;min-height:0;height:var(--episodeAsideHeight,640px);max-height:var(--episodeAsideHeight,640px);overflow:hidden;align-self:start}.sideTabsToolbar{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px}.sideSortHint{font-size:11px;color:#667085;line-height:1.35;text-align:right;max-width:190px}.episodeTabsList{flex:1;min-height:0;overflow:auto;scrollbar-width:none;-ms-overflow-style:none;border:1px solid #eaecf0;border-radius:10px;background:#fff}.episodeTabsList::-webkit-scrollbar{display:none;width:0;height:0}.episodeDataSheet{min-width:1080px;width:100%;border-collapse:separate;border-spacing:0;font-size:11.5px;line-height:1.25}.episodeDataSheet th,.episodeDataSheet td{padding:7px 8px;border-bottom:1px solid #eef2f6;white-space:nowrap;vertical-align:middle}.episodeDataSheet th{position:sticky;top:0;z-index:4;background:#f8fafc;color:#475467;text-transform:uppercase;letter-spacing:.04em;font-size:10.5px}.episodeDataSheet th.sortable{cursor:pointer;color:#175cd3;user-select:none}.episodeDataSheet th.sortable:hover{background:#eff8ff}.episodeDataSheet tbody tr{cursor:pointer}.episodeDataSheet tbody tr:hover td{background:#f8fafc}.episodeDataSheet tbody tr.selected td{background:#e0f2fe}.episodeDataSheet .epCol{position:sticky;left:0;z-index:3;background:#fff;font-weight:800;color:#101828}.episodeDataSheet th.epCol{z-index:5;background:#f8fafc}.episodeDataSheet tbody tr:hover .epCol{background:#f8fafc}.episodeDataSheet tbody tr.selected .epCol{background:#e0f2fe}.episodeDataSheet .num{text-align:right;font-variant-numeric:tabular-nums}.episodeDataSheet .reasonCell{max-width:360px;overflow:hidden;text-overflow:ellipsis}.sideFooter{font-size:11px;color:#98a2b3;margin-top:7px;line-height:1.35}.timelinePane{min-width:0;display:flex;flex-direction:column;height:auto;align-self:start}.timelinePaneHeader{position:relative;top:auto;z-index:2;background:#fff;border:1px solid #eaecf0;border-radius:12px;padding:10px 12px;margin-bottom:16px}.timelinePaneHeader + .timelineGrid{margin-top:0}#bucketChart{margin-top:0}.timelineChart svg{display:block}.timelineSvgClickable{cursor:crosshair}.timelineCursor,.cameraTimelineCursor{pointer-events:none}.timelineGrid{gap:12px;min-width:0}.timelineChart .chart{min-height:230px}.unitLegend{border:1px solid #d0d5dd;border-radius:999px;padding:2px 7px;background:#fff;color:#475467;font-weight:700}.plotNote{font-size:11px;color:#667085;margin-top:6px;line-height:1.35}.densityBadge{display:inline-block;margin-left:6px;border:1px solid #d0d5dd;border-radius:999px;padding:1px 6px;font-size:10px;color:#475467;background:#fff}.meshFrameBadge{display:inline-block;border:1px solid #d0d5dd;border-radius:999px;padding:2px 7px;background:#fff;color:#475467;font-size:11px;margin-top:6px}
@media(max-width:1280px){.episodeInspector{grid-template-columns:1fr;min-height:0}.episodeSide{height:min(560px,var(--episodeAsideHeight,560px));max-height:min(560px,var(--episodeAsideHeight,560px))}.timelinePaneHeader{position:static}.miniChartGrid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:760px){.miniChartGrid{grid-template-columns:1fr}.statusFilterBar{align-items:flex-start}.episodeTabMetrics{grid-template-columns:repeat(2,1fr)}}

.taskPrompt{margin-top:8px;border-left:3px solid #175cd3;padding:7px 9px;background:#f8fafc;border-radius:8px;color:#344054;font-size:12px;line-height:1.4}
.taskPromptLabel{font-weight:800;color:#175cd3;margin-right:6px}
.cameraPreview{border:1px solid #eaecf0;border-radius:12px;background:#f8fafc;padding:10px;margin-bottom:14px}
.cameraPreviewHeader{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.cameraTimeline{margin:8px 0 10px}.cameraTimeline svg{display:block;width:100%;height:auto;border:1px solid #edf2f7;border-radius:10px;background:#fff;cursor:crosshair}body.dark .cameraTimeline svg{background:#0f172a;border-color:#334155}
.cameraPreviewTitle{font-size:13px;font-weight:850;color:#101828}
.cameraFrameControls{display:flex;align-items:center;gap:8px;min-width:320px;flex:1}
.cameraFrameControls input[type=range]{flex:1;min-width:160px}
.cameraPlayerControls{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.cameraPlayerBtn{min-height:28px;border:1px solid #d0d5dd;border-radius:8px;background:#fff;color:#344054;padding:0 9px;font-size:12px;font-weight:800;cursor:pointer}
.cameraPlayerBtn:hover{background:#f8fafc}
.cameraPlayerBtn.primary{background:#175cd3;border-color:#175cd3;color:#fff;min-width:58px}
.cameraPlayerBtn.primary.playing{background:#b42318;border-color:#b42318}
.cameraGrid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.cameraCard{border:1px solid #dbe3ee;border-radius:10px;background:#fff;overflow:hidden;min-width:0}
.cameraCard.loading img{opacity:.74}
.cameraCardHeader{display:flex;justify-content:space-between;gap:8px;padding:7px 8px;border-bottom:1px solid #eef2f6;font-size:11px;color:#475467}
.cameraCardTitle{font-weight:850;color:#101828}
.cameraCard img{display:block;width:min(100%,256px);max-width:256px;aspect-ratio:1/1;object-fit:contain;background:#0f172a;margin:0 auto}
.cameraCard.missing img{display:none}
.cameraMissing{display:none;min-height:160px;align-items:center;justify-content:center;padding:16px;color:#98a2b3;font-size:12px;text-align:center}
.cameraCard.missing .cameraMissing{display:flex}
.cameraOpenLink{font-size:11px;color:#175cd3;text-decoration:none}
@media(max-width:900px){.cameraGrid{grid-template-columns:1fr}}


/* v8 diagnosis refactor */
.triageGrid{display:grid;grid-template-columns:minmax(380px,1.35fr) minmax(0,1fr) minmax(0,1fr);gap:12px;align-items:stretch}
.compactScore{min-height:132px}.runtimePieCard{min-height:132px}.runtimePieCard .diagStats{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:4px}.runtimePieCard .diagStatValue{font-size:15px}.pieWrap{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:center}.pieLegend{display:grid;gap:5px;font-size:11px;color:#475467}.pieLegendRow{display:flex;align-items:center;justify-content:space-between;gap:8px}.pieSwatch{width:9px;height:9px;border-radius:99px;display:inline-block;margin-right:5px}.pieSvg{width:130px;height:130px;display:block}.pieCenter{fill:#fff;opacity:.96}.body.dark .pieSvg .pieCenter,.dark .pieSvg .pieCenter,body.dark .pieSvg .pieCenter{fill:#0f172a}.datasetScore{font-size:12px;color:#667085;margin-top:6px}.compactScore{min-height:132px}.compactScore .scoreNumber{font-size:46px}.diagCard{border:1px solid #e5e7eb;border-radius:14px;background:#fff;padding:13px;min-width:0;display:flex;flex-direction:column;gap:8px}.diagCardTitle{font-size:11px;color:#667085;text-transform:uppercase;letter-spacing:.05em;font-weight:800}.diagCardValue{font-size:20px;line-height:1.15;font-weight:850;color:#101828;overflow-wrap:anywhere}.diagCardDetail{font-size:12px;line-height:1.4;color:#667085}.diagCard.bad{border-left:4px solid #d92d20}.diagCard.warn{border-left:4px solid #f79009}.diagCard.good{border-left:4px solid #067647}.diagCard.info{border-left:4px solid #175cd3}.diagStats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:auto}.diagStat{border:1px solid #eef2f6;border-radius:10px;background:#f8fafc;padding:7px}.diagStatLabel{font-size:10px;color:#667085;text-transform:uppercase;letter-spacing:.04em}.diagStatValue{font-size:17px;font-weight:800;color:#101828}.actionsStrip{margin-top:12px;border-top:1px solid #eaecf0;padding-top:10px;display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:start}.actionsStrip h3{margin-top:4px}.actionsInline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.actionsInline .action{margin:0;min-height:52px;background:#fcfcfd}.detailsPanel{padding:0}.detailsPanel>summary{cursor:pointer;list-style:none;padding:14px 16px;font-weight:850;color:#101828;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #eaecf0}.detailsPanel>summary::-webkit-details-marker{display:none}.detailsPanel>summary:after{content:"展开";font-size:12px;color:#667085;border:1px solid #d0d5dd;border-radius:999px;padding:3px 9px;background:#fff}.detailsPanel[open]>summary:after{content:"收起"}.diagDetailsGrid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px;padding:14px}.diagDetailsGrid>.span3{grid-column:span 3}.diagDetailsGrid>.span4{grid-column:span 4}.diagDetailsGrid>.span6{grid-column:span 6}.diagDetailsGrid>.span12{grid-column:span 12}.subPanel{border:1px solid #eaecf0;border-radius:12px;background:#fff;padding:12px;min-width:0;overflow:hidden}.subPanel h2{font-size:14px;margin:0}.qualityNote{font-size:11px;color:#667085;line-height:1.35;margin-top:6px}
@media(max-width:1280px){.triageGrid{grid-template-columns:1fr 1fr}.actionsInline{grid-template-columns:1fr}.actionsStrip{grid-template-columns:1fr}.diagDetailsGrid>.span3,.diagDetailsGrid>.span4,.diagDetailsGrid>.span6{grid-column:span 12}}
@media(max-width:720px){.triageGrid{grid-template-columns:1fr}.diagStats{grid-template-columns:1fr}}
.transferProgress{margin:8px 0 10px;border:1px solid #dbe3ee;border-radius:10px;background:#f8fafc;padding:8px 10px}.transferProgressMeta{display:flex;justify-content:space-between;gap:10px;font-size:12px;color:#475467;margin-bottom:6px}.transferProgressTrack{height:8px;border-radius:999px;background:#e5e7eb;overflow:hidden}.transferProgressFill{height:100%;border-radius:999px;background:#12b76a;transition:width .22s ease}.transferProgressFill.busy{background:linear-gradient(90deg,#12b76a,#60a5fa,#12b76a);background-size:180% 100%;animation:progressSlide 1.1s linear infinite}@keyframes progressSlide{from{background-position:0 0}to{background-position:180% 0}}

.terminalPanel{margin-bottom:14px}.terminalPanel .panelHeader,.rawAttemptPanel .panelHeader{margin-bottom:8px}.terminalActions,.rawAttemptActions{display:flex;align-items:center;gap:8px}.terminalBox{max-height:180px;min-height:72px}.rawAttemptActions:after{content:"debug only";font-size:12px;color:#667085}
.epCellInner{display:inline-flex;align-items:center;gap:6px}.exportReadyDot{width:8px;height:8px;border-radius:999px;display:inline-block;box-shadow:0 0 0 2px #fff,0 0 0 3px #d0d5dd;flex:0 0 auto}.exportReadyDot.ready{background:#12b76a;box-shadow:0 0 0 2px #fff,0 0 0 3px rgba(18,183,106,.35)}.exportReadyDot.notReady{background:#f04438;box-shadow:0 0 0 2px #fff,0 0 0 3px rgba(240,68,56,.32)}.exportReadyDot.stale{background:#f79009;box-shadow:0 0 0 2px #fff,0 0 0 3px rgba(247,144,9,.34)}.exportReadyDot.unknown{background:#98a2b3}
.modalOverlay{position:fixed;inset:0;z-index:1000;background:rgba(15,23,42,.48);display:none;align-items:center;justify-content:center;padding:18px}.modalOverlay.open{display:flex}.modalOverlay.loading{cursor:wait}.modalCard{width:min(560px,calc(100vw - 36px));background:#fff;border:1px solid #d0d5dd;border-radius:14px;box-shadow:0 24px 72px rgba(16,24,40,.32);padding:16px;color:#101828}.modalOverlay.loading .modalCard{box-shadow:0 24px 72px rgba(16,24,40,.36),0 0 0 3px rgba(23,92,211,.14)}.modalCard h2{margin:0 0 4px}.modalIntro{font-size:12px;color:#667085;line-height:1.45;margin:0 0 12px}.modalGrid{display:grid;grid-template-columns:150px minmax(0,1fr);gap:9px 10px;align-items:center}.modalGrid input,.modalGrid select{width:100%}.modalNote{margin-top:10px;border:1px solid #eaecf0;border-radius:10px;background:#f8fafc;padding:9px 10px;font-size:12px;color:#475467;line-height:1.45}.modalActions{display:flex;justify-content:flex-end;gap:8px;flex-wrap:wrap;margin-top:14px}.modalActions .danger{background:#b42318;border-color:#b42318;color:#fff}body.dark .modalCard{background:#0f172a;border-color:#334155;color:#e5e7eb}body.dark .modalNote{background:#111827;border-color:#334155;color:#94a3b8}
body main details>summary:after,.managerPanel>summary:after,.detailsPanel>summary:after{content:""!important;width:10px!important;height:10px!important;border:0!important;border-right:2px solid #667085!important;border-bottom:2px solid #667085!important;border-radius:0!important;padding:0!important;background:transparent!important;transform:rotate(-45deg);transition:transform .2s ease;flex:0 0 auto;margin-top:4px}body main details[open]>summary:after,.managerPanel[open]>summary:after,.detailsPanel[open]>summary:after{content:""!important;transform:rotate(45deg);margin-top:7px}

</style>
</head>
<body>
<header class="topbar">
  <div class="shell">
    <div class="topbarTitleRow"><h1>Excavator Training Dataset Generation Dashboard</h1><button id="darkModeToggle" class="darkModeToggle" type="button" title="Toggle dark mode" aria-label="Toggle dark mode">🌙</button></div>
    <div class="controls rootControls">
      <label for="rootInput">Dataset root</label>
      <input id="rootInput" class="path" value="excavator_auto_dataset">
      <button id="loadRunsBtn">Load runs</button>
      <button class="secondary" id="loadSuccessPoolBtn">Load .dashboard_success</button>
      <button class="secondary" id="exportSuccessPoolBtn">Export success VLA</button>
      <select id="runSelect"></select>
      <button id="loadRunBtn">Analyze run</button>
    </div>
    <div class="controls runControls" style="margin-top:8px">
      <label for="runInput">Run folder</label>
      <input id="runInput" class="path" placeholder="D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_...">
      <button class="secondary" id="reloadBtn">Reload</button>
      <button class="secondary" id="copyPathBtn">Copy path</button>
      <span id="status" class="muted"></span>
    </div>
    <div id="runMonitor" class="runMonitor"><span class="runMonitorTitle">Active writers:</span><span class="muted">loading...</span></div>
    <div class="statusFilterBar">
      <div><span class="statusFilterTitle">Status filter</span><span id="statusFilter" class="statusFilter"></span></div>
      <span id="filterCount" class="filterCount">All statuses</span>
    </div>
  </div>
</header>
<main class="shell">
  <section class="panel terminalPanel">
    <div class="panelHeader">
      <h2>Terminal output</h2>
      <div class="terminalActions">
        <button type="button" class="secondary" id="clearTerminalBtn">Clear</button>
      </div>
    </div>
    <pre id="terminalBox" class="codeBox terminalBox">Dashboard ready.</pre>
  </section>
  <details class="panel managerPanel" open>
    <summary>
      <div><h2>Run folder manager</h2></div>
    </summary>
    <div class="managerBody">
      <div class="managerToolbar">
        <button type="button" class="secondary" id="selectAllRunsBtn">Select all</button>
        <button type="button" class="secondary" id="selectZeroSuccessBtn">Select 0 success</button>
        <button type="button" class="secondary" id="selectSuccessRunsBtn">Select success&gt;0</button>
        <input id="successDestInput" class="path" readonly placeholder="dataset_root\.dashboard_success">
        <button type="button" id="copySuccessBtn">Copy selected success</button>
        <button type="button" class="warn" id="moveSuccessBtn">Cut selected success</button>
        <button type="button" class="secondary" id="clearRunSelectionBtn">Clear</button>
        <button type="button" class="secondary" id="refreshSelectedSizesBtn">Refresh selected sizes</button>
        <button type="button" class="danger" id="deleteSelectedRunsBtn">Trash selected 0-success</button>
      </div>
      <div id="successTransferProgress" class="transferProgress" style="display:none">
        <div class="transferProgressMeta"><span id="successTransferProgressText">idle</span><span id="successTransferProgressPct">0%</span></div>
        <div class="transferProgressTrack"><div id="successTransferProgressFill" class="transferProgressFill" style="width:0%"></div></div>
      </div>
      <div id="managerSummary" class="managerSummary">loading run folders...</div>
      <div class="managerTableWrap"><table id="runManagerTable" class="managerTable"></table></div>
    </div>
  </details>
  <section class="panel hero">
    <div class="panelHeader">
      <div><h2>Run diagnosis</h2></div>
      <div id="reportHint" class="reportHint"></div>
    </div>
    <div class="triageGrid">
      <div id="runtimePieCard" class="diagCard runtimePieCard"></div>
      <div id="topBlockerCard" class="diagCard"></div>
      <div id="qualitySignalCard" class="diagCard"></div>
    </div>
  </section>

  <section class="grid">
    <section class="panel span3"><div class="panelHeader"><h2>Outcome mix</h2><span class="panelHint">status counts</span></div><div id="outcomeBars" class="barRows"></div></section>
    <section class="panel span5"><div class="panelHeader"><h2>Blocker reason Pareto</h2><span class="panelHint">only rejected / failed / diagnostic; excludes ok</span></div><div id="reasonBars" class="barRows"></div></section>
    <section class="panel span4"><div class="panelHeader"><h2>Quality signals</h2><span class="panelHint">score_low / spill / low bin grouped</span></div><div id="qualityBars" class="barRows"></div><div class="qualityNote">带数值的 warning 会按类别聚合，例如 score_low:35.0 → quality/score_low。</div></section>
    <details class="panel span12 detailsPanel">
      <summary><span>Training/export diagnostics</span><span class="panelHint">readiness gates, material stats, schema, non-quality warnings</span></summary>
      <div class="diagDetailsGrid">
        <section class="subPanel span6"><div class="panelHeader"><h2>Readiness gates</h2><span class="panelHint">training export blockers</span></div><div class="tableWrap"><table id="gatesTable"></table></div></section>
        <section class="subPanel span6"><div class="panelHeader"><h2>Material / score statistics</h2><span class="panelHint">median and range</span></div><div class="tableWrap"><table id="materialTable" class="metricsTable"></table></div></section>
        <section class="subPanel span4"><div class="panelHeader"><h2>Initial pose × status</h2><span class="panelHint">pose triage matrix</span></div><div class="tableWrap"><table id="initialPoseTable" class="metricsTable"></table></div></section>
        <section class="subPanel span4"><div class="panelHeader"><h2>Segment completion</h2><span class="panelHint">jsonl counts</span></div><div id="segmentBars" class="barRows"></div></section>
        <section class="subPanel span4"><div class="panelHeader"><h2>General warnings</h2><span class="panelHint">non-quality warnings only</span></div><div id="warningBars" class="barRows"></div></section>
        <section class="subPanel span12"><div class="panelHeader"><h2>Trajectory schema</h2><span class="panelHint">field coverage</span></div><div id="schemaTables" class="schemaGrid"></div></section>
      </div>
    </details>

    <section class="panel span4"><div class="panelHeader"><h2>Sand XY coverage</h2><span class="panelHint">excavator body-frame XY after init yaw; 1:1 X/Y scale</span></div><div id="sandScatter" class="chartBox"></div></section>
    <section class="panel span8"><div class="panelHeader"><h2>Truck XY coverage</h2><span class="panelHint">truck center + selected dump-bed mesh transformed into excavator body frame; 1:1 X/Y scale</span></div><div id="truckScatter" class="chartBox"></div></section>
    <section class="panel span8"><div class="panelHeader"><h2>Unload local XY</h2><span class="panelHint">selected unload mesh local frame; mesh is edge-aligned to X/Y</span></div><div id="unloadScatter" class="chartBox"></div></section>
    <section class="panel span4"><div class="panelHeader"><h2>Scene distributions</h2><span class="panelHint">secondary variables</span></div><div class="miniChartGrid">
      <div class="miniChart"><h3>Sand amount multiplier</h3><div id="sandAmountHist" class="smallChartBox"></div></div>
      <div class="miniChart"><h3>Truck yaw relative to body</h3><div id="truckYawHist" class="smallChartBox"></div></div>
      <div class="miniChart"><h3>Robot initial yaw</h3><div id="robotYawHist" class="smallChartBox"></div></div>
    </div></section>
    <section class="panel span6"><div class="panelHeader"><h2>Score vs spill</h2><span class="panelHint">quality trade-off</span></div><div id="scoreSpillScatter" class="chartBox"></div></section>
    <section class="panel span6"><div class="panelHeader"><h2>Bin vs spill</h2><span class="panelHint">dumping effectiveness</span></div><div id="binSpillScatter" class="chartBox"></div></section>
  </section>

  <section class="panel" style="margin-top:14px">
    <div class="panelHeader"><div><h2>Episode inspection</h2></div></div>
    <div class="episodeInspector">
      <aside class="episodeSide">
        <div class="sideTabsToolbar">
          <div><h3>Attempts datasheet</h3><div id="episodeSideMeta" class="muted"></div></div>
          <div class="sideSortHint">Click a column header to sort. Wheel/trackpad scroll is enabled; scrollbars are hidden.</div>
        </div>
        <div id="episodeTabs" class="episodeTabsList" role="region" aria-label="Attempt datasheet"></div>
        <div class="sideFooter">Database-style view: sticky header, sortable columns, hidden side-panel scrollbars.</div>
      </aside>
      <section class="timelinePane">
        <div class="timelinePaneHeader">
          <h2 id="episodeTitle">Attempt timeline</h2>
          <div id="episodeMeta" class="muted" style="margin-top:6px"></div>
          <div id="episodeTaskPrompt" class="taskPrompt" style="display:none"></div>
        </div>
        <div id="cameraPreview" class="cameraPreview"><div class="empty">Select an attempt to preview Cam 0/1/2.</div></div>
        <div class="timelineGrid">
          <div id="bucketChart" class="timelineChart"></div>
          <div id="qChart" class="timelineChart"></div>
          <div id="dqChart" class="timelineChart"></div>
          <div id="ddqChart" class="timelineChart"></div>
          <div id="effortChart" class="timelineChart"></div>
        </div>
      </section>
    </div>
  </section>
  <section class="panel rawAttemptPanel" style="margin-top:14px">
    <div class="panelHeader">
      <h2>Raw selected attempt</h2>
      <div class="rawAttemptActions">
        <button type="button" class="secondary" id="copyRawAttemptBtn">Copy raw</button>
      </div>
    </div>
    <pre id="rawBox" class="codeBox">{}</pre>
  </section>
</main>
<div id="exportTimeModal" class="modalOverlay" role="dialog" aria-modal="true" aria-labelledby="exportTimeTitle">
  <div class="modalCard">
    <h2 id="exportTimeTitle">VLA export time scale</h2>
    <p class="modalIntro">Set the training-time playback speed before exporting. Preview applies the same timestamp/action/dq/ddq recomputation used by the exporter.</p>
    <div class="modalGrid">
      <label for="exportSpeedScaleInput">Speed scale</label>
      <input id="exportSpeedScaleInput" type="number" min="0.01" step="0.1" value="1">
    </div>
    <div id="exportTimePolicyPreview" class="modalNote">1x export uses the original timeline.</div>
    <div class="modalActions">
      <button type="button" class="secondary" id="cancelExportTimeBtn">Cancel</button>
      <button type="button" class="secondary" id="resetTimePolicyBtn">Reset 1x</button>
      <button type="button" class="secondary" id="applyTimePreviewBtn">Apply preview</button>
      <button type="button" id="exportTimeNowBtn">Export now</button>
    </div>
  </div>
</div>
<script>

const jointNames = ["swing","boom","arm","bucket"];
const statusPalette = {
  trainable:"#067647", success:"#067647", rejected:"#d92d20",
  failed:"#f79009", fail:"#f79009", diagnostic:"#6941c6",
  planning:"#175cd3", skip:"#475467", unknown:"#667085", all:"#344054"
};
const lineColors = ["#175cd3","#067647","#b54708","#d92d20","#6941c6","#0086c9"];
const stagePalette = ["#dbeafe","#dcfce7","#fef3c7","#fee2e2","#ede9fe","#cffafe","#fce7f3","#e2e8f0"];
let currentRun = null;
let currentEpisodeIndex = null;
let episodeSort = {key:"episode_index", dir:1};
let selectedStatuses = new Set();
let runMonitorTimer = null;
let availableStatuses = [];
let runRecords = [];
let selectedRunPaths = new Set();
let cameraFrameIndex = 0;
let cameraPreviewTimer = null;
let cameraPlayerTimer = null;
let cameraPlayerPlaying = false;
const CAMERA_PLAYER_FIXED_FPS = 10;
let cameraPreviewLoadSeq = 0;
const cameraFrameImageCache = new Map();
const cameraFrameImageCacheOrder = [];
const cameraFrameImageCacheLimit = 360;
let darkModeEnabled = false;
let timelineDragState = {active:false, svg:null, moved:false, suppressClick:false};
let exportTimeModalOpen = false;
let currentExportTimePolicy = {version:1, speed_scale:1.0, time_mode:"uniform_fps", base_fps:null};

function $(id){return document.getElementById(id)}
function esc(s){return String(s ?? "").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function terminalTimestamp(){const d=new Date(); return d.toLocaleTimeString("en-GB",{hour12:false})}
function terminalWrite(text, cls="muted"){
  const box=$("terminalBox"); if(!box) return;
  const line=`[${terminalTimestamp()}] ${String(text||"")}`;
  const current=box.textContent&&box.textContent!=="Dashboard ready."?box.textContent:"";
  const lines=(current?current+"\n":"")+line;
  box.textContent=lines.split("\n").slice(-240).join("\n");
  box.scrollTop=box.scrollHeight;
  box.dataset.level=cls||"muted";
}
function setStatus(text, cls="muted"){const el=$("status"); if(el){el.className=cls; el.textContent=text} terminalWrite(text,cls)}
async function api(path, params){const qs=new URLSearchParams(params||{}); const r=await fetch(path+"?"+qs.toString()); if(!r.ok) throw new Error(await r.text()); return await r.json()}
async function postJSON(path, payload){let r; try{r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload||{})});}catch(err){throw new Error(`Failed to fetch ${path}: ${err.message||err}`);} if(!r.ok) throw new Error(await r.text()); return await r.json()}
function normalizeExportTimePolicyJS(policy){
  const p=policy&&typeof policy==="object"?policy:{};
  let speed=Number(p.speed_scale);
  if(!Number.isFinite(speed)||speed<=0) speed=1;
  return {version:1, speed_scale:speed, time_mode:"uniform_fps", base_fps:null};
}
function currentRunTimePolicyInfo(){
  const info=(((currentRun||{}).diagnosis||{}).success_export||{}).current_time_policy;
  if(info&&typeof info==="object") return info;
  return {policy:{version:1,speed_scale:1,time_mode:"uniform_fps",base_fps:null},hash:"",is_default:true};
}
function fillExportTimePolicyInputs(policy){
  const p=normalizeExportTimePolicyJS(policy);
  currentExportTimePolicy=p;
  const speed=$("exportSpeedScaleInput");
  if(speed) speed.value=String(p.speed_scale);
  updateExportTimePolicyPreview();
}
function readExportTimePolicyInputs(){
  return normalizeExportTimePolicyJS({
    speed_scale:$("exportSpeedScaleInput")?.value,
    base_fps:null
  });
}
function exportTimePolicyText(policy){
  const p=normalizeExportTimePolicyJS(policy);
  return `${fmt(p.speed_scale,3)}x, uniform export time`;
}
function updateExportTimePolicyPreview(extra){
  const p=readExportTimePolicyInputs();
  currentExportTimePolicy=p;
  const el=$("exportTimePolicyPreview");
  if(el){
    el.textContent=(extra?`${extra} `:"")+`Preview/export policy: ${exportTimePolicyText(p)}. Timestamps use the exporter uniform timeline; joint velocity, acceleration and action are recomputed from that timeline.`;
  }
}
function showExportTimeModal(){
  syncSuccessPoolPath();
  const info=currentRunTimePolicyInfo();
  fillExportTimePolicyInputs((info&&info.policy)||currentExportTimePolicy);
  const modal=$("exportTimeModal");
  if(modal){modal.classList.add("open"); exportTimeModalOpen=true;}
}
function hideExportTimeModal(){
  const modal=$("exportTimeModal");
  if(modal){modal.classList.remove("open"); exportTimeModalOpen=false;}
}
function setExportModalBusy(busy, message){
  const modal=$("exportTimeModal");
  if(modal) modal.classList.toggle("loading",!!busy);
  ["cancelExportTimeBtn","resetTimePolicyBtn","applyTimePreviewBtn","exportTimeNowBtn","exportSpeedScaleInput"].forEach(id=>{
    const el=$(id); if(el) el.disabled=!!busy;
  });
  const apply=$("applyTimePreviewBtn");
  if(apply) apply.textContent=busy?"Applying...":"Apply preview";
  const exportNow=$("exportTimeNowBtn");
  if(exportNow) exportNow.textContent=busy?"Working...":"Export now";
  if(message) updateExportTimePolicyPreview(message);
}
async function saveExportTimePolicyForPreview(reset=false, reload=true){
  syncSuccessPoolPath();
  const policy=reset?{speed_scale:1,time_mode:"uniform_fps",base_fps:null}:readExportTimePolicyInputs();
  const result=await postJSON("/api/manage/export_time_policy", {root:$("rootInput").value, reset:!!reset, time_policy:policy});
  currentExportTimePolicy=normalizeExportTimePolicyJS(result.policy||policy);
  fillExportTimePolicyInputs(currentExportTimePolicy);
  updateExportTimePolicyPreview(`Saved.`);
  if(!reload) return result;
  $("runInput").value=result.pool_dir||successPoolPath();
  await loadRun(false);
  if(currentEpisodeIndex!==null && currentEpisodeIndex!==undefined){
    await loadEpisode(currentEpisodeIndex);
  }
  return result;
}
async function applyTimePreviewAndClose(){
  setExportModalBusy(true,"Applying preview...");
  try{
    await saveExportTimePolicyForPreview(false, true);
    setExportModalBusy(false,"Preview applied.");
    hideExportTimeModal();
  }catch(err){
    setExportModalBusy(false,"Preview failed.");
    throw err;
  }
}
function finite(v){return typeof v==="number" && Number.isFinite(v)}
function number(v){const n=Number(v); return Number.isFinite(n)?n:null}
function numeric(values){return (values||[]).map(v=>Number(v)).filter(v=>Number.isFinite(v))}
function statusKey(s){
  let t=String(s||"unknown").trim().toLowerCase();
  if(t.includes("\\") || t.includes("/")){
    const parts=t.split(/[\\/]+/).filter(Boolean);
    t=parts.length?parts[parts.length-1]:t;
  }
  if(t==="fail"||t==="failed"||t==="failure") return "failed";
  if(t==="successful") return "success";
  if(t==="ok") return "trainable";
  return t||"unknown"
}
function statusColor(s){return statusPalette[statusKey(s)] || statusPalette.unknown}
function cssClassStatus(s){return statusKey(s).replace(/[^a-zA-Z0-9_-]/g,"_")}
function fmt(v,d=1){const n=Number(v); if(!Number.isFinite(n)) return ""; if(Math.abs(n)>=10000) return n.toFixed(0); if(Math.abs(n)>=1000) return n.toFixed(1); return n.toFixed(d)}
function fmtAxis(v){const n=Number(v); if(!Number.isFinite(n)) return ""; if(Math.abs(n)>=10000) return n.toFixed(0); if(Math.abs(n)>=1000) return n.toFixed(0); if(Math.abs(n)>=10) return n.toFixed(1); return n.toFixed(2)}
function pct(v){const n=Number(v); return Number.isFinite(n)?`${(n*100).toFixed(1)}%`:""}
function shortText(s,n){s=String(s||""); return s.length>n?s.slice(0,n-1)+"…":s}
function basename(path){return String(path||"").split(/[\/]/).pop()}
function applyDarkMode(enabled){
  darkModeEnabled=!!enabled;
  document.body.classList.toggle("dark", darkModeEnabled);
  const btn=$("darkModeToggle");
  if(btn){btn.textContent=darkModeEnabled?"☀️":"🌙"; btn.title=darkModeEnabled?"Switch to light mode":"Switch to dark mode";}
  try{localStorage.setItem("excavatorDashboardDarkMode", darkModeEnabled?"1":"0");}catch(_e){}
}
function toggleDarkMode(){applyDarkMode(!darkModeEnabled)}
function initDarkMode(){
  let stored="";
  try{stored=localStorage.getItem("excavatorDashboardDarkMode")||"";}catch(_e){}
  const preferDark=window.matchMedia&&window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyDarkMode(stored?stored==="1":preferDark);
}

function pathSepForBase(base){
  const text=String(base||'');
  return (/^[A-Za-z]:[\\/]/.test(text) || text.includes('\\')) ? '\\' : '/';
}
function joinOne(base, child){
  const root=String(base||'').replace(/[\\/]+$/,'');
  return `${root}${pathSepForBase(root)}${child}`;
}
function successPoolPath(){
  const root=String($('rootInput').value||'excavator_auto_dataset').replace(/[\\/]+$/,'');
  return joinOne(root, '.dashboard_success');
}
function syncSuccessPoolPath(){const input=$('successDestInput'); if(input) input.value=successPoolPath();}
function setTransferProgress(text, percent=0, busy=false){
  const box=$('successTransferProgress'), fill=$('successTransferProgressFill'), label=$('successTransferProgressText'), pctEl=$('successTransferProgressPct');
  if(!box||!fill) return;
  const p=Math.max(0, Math.min(100, Number(percent)||0));
  box.style.display='block';
  if(label) label.textContent=text||'working';
  if(pctEl) pctEl.textContent=`${Math.round(p)}%`;
  fill.style.width=`${p}%`;
  fill.classList.toggle('busy', !!busy);
}
function hideTransferProgressSoon(){setTimeout(()=>{const box=$('successTransferProgress'); if(box) box.style.display='none';}, 3500);}
async function pollDashboardJob(jobId, onDone){
  let last = null;
  for(;;){
    const job = await api('/api/manage/job', {job_id:jobId, _:Date.now()});
    const percent = Number(job.percent || 0);
    const msg = job.message || job.title || 'working';
    setTransferProgress(`${msg} (${job.current||0}/${job.total||1})`, percent, job.status==='running' || job.status==='queued');
    if(msg !== last?.message || Math.round(percent) !== Math.round(Number(last?.percent || -1))){
      terminalWrite(`${job.type||"job"}: ${msg} (${Math.round(percent)}%)`, job.status==='error'?'error':'muted');
    }
    last = job;
    if(job.status === 'done'){
      setTransferProgress('complete', 100, false);
      terminalWrite(`${job.type||"job"}: complete`, "ok");
      if(onDone) await onDone(job.result || job);
      hideTransferProgressSoon();
      return job;
    }
    if(job.status === 'error'){
      setTransferProgress(job.error || 'job failed', 100, false);
      terminalWrite(`${job.type||"job"}: ${job.error || 'job failed'}`, "error");
      throw new Error(job.error || 'job failed');
    }
    await new Promise(resolve=>setTimeout(resolve, 500));
  }
}

async function loadRuns(opts={}){
  const silent = !!opts.silent;
  syncSuccessPoolPath();
  if(!silent) setStatus("Loading runs...");
  const previous = $("runSelect").value || $("runInput").value || "";
  const data = await api("/api/runs", {root:$("rootInput").value});
  const runs = data.runs || [];
  runRecords = runs;
  const validPaths = new Set(runs.map(r=>r.path));
  selectedRunPaths = new Set([...selectedRunPaths].filter(path=>validPaths.has(path)));
  const sel = $("runSelect"); sel.innerHTML = "";
  let selectedPath = "";
  for(const run of runs){
    const opt=document.createElement("option"); opt.value=run.path;
    const activity=run.activity || {};
    const state=activity.state || "unknown";
    const dot=state==="active" ? "🟢" : (state==="recent" ? "🟡" : "🔴");
    const age=activity.age_s==null ? "unknown" : ageText(activity.age_s);
    const label=state==="active" ? `active ${age}` : (state==="recent" ? `recent ${age}` : `idle ${age}`);
    const catchup=(run.success_catchup&&run.success_catchup.active)?"  CATCHUP":"";
    opt.textContent=`${dot} ${run.name}${catchup}  attempts=${run.attempts ?? "-"} success=${run.success ?? 0} trainable=${run.trainable ?? "-"} size=${run.size_human || "-"}  ${label}`;
    if(previous && run.path === previous){opt.selected=true; selectedPath=run.path;}
    sel.appendChild(opt);
  }
  renderRunManager(runs);
  if(!selectedPath && runs.length){selectedPath=runs[0].path; sel.value=selectedPath;}
  if(selectedPath) $("runInput").value=selectedPath;
  renderRunMonitor(data);
  if(runs.length){ if(!silent) setStatus(`Loaded ${runs.length} runs; active writers=${(data.activity_summary||{}).active_count||0}`, "ok"); }
  else if(!silent) setStatus("No run_* folders found", "error");
}
function ageText(seconds){
  const s=Number(seconds);
  if(!Number.isFinite(s)) return "unknown";
  if(s<60) return `${Math.round(s)}s`;
  if(s<3600) return `${Math.round(s/60)}m`;
  return `${(s/3600).toFixed(1)}h`;
}

function renderRunManager(runs){
  const table=$("runManagerTable"); if(!table) return;
  const totalSize=(runs||[]).reduce((a,r)=>a+Number(r.size_bytes||0),0);
  const zero=(runs||[]).filter(r=>Number(r.success||0)===0);
  const successRuns=(runs||[]).filter(r=>Number(r.success||0)>0);
  const selected=[...selectedRunPaths].length;
  const catchupRuns=(runs||[]).filter(r=>r.success_catchup&&r.success_catchup.active);
  const summary=$("managerSummary");
  if(summary) summary.textContent=`folders=${runs.length}; selected=${selected}; 0-success=${zero.length}; success>0=${successRuns.length}; catch-up=${catchupRuns.length}; cached/displayed size=${humanBytes(totalSize)} · sizes are cache-first; refresh selected for exact scan`;
  if(!runs.length){table.innerHTML='<tbody><tr><td class="muted">no run folders</td></tr></tbody>';return;}
  const rows=[`<thead><tr><th><input type="checkbox" data-action="toggle-all-runs"></th><th>State</th><th>Run</th><th>Catch-up</th><th class="num">Attempts</th><th class="num">Success</th><th class="num">Trainable</th><th class="num">Rejected</th><th class="num">Failed</th><th>Run size</th><th>data* size</th><th>Latest write</th></tr></thead><tbody>`];
  for(const run of runs){
    const activity=run.activity||{};
    const state=activity.state||"unknown";
    const checked=selectedRunPaths.has(run.path) ? "checked" : "";
    const success=Number(run.success||0);
    const successCls=success>0?"successRun":"zeroSuccess";
    const activeTitle=activity.latest_file ? `${state}: ${activity.latest_file}` : state;
    const catchup=run.success_catchup || {};
    const catchupText=catchup.active ? `success-index · raw all=${((catchup.raw_counts||{}).all ?? 0)} raw trainable=${((catchup.raw_counts||{}).trainable ?? 0)}` : "normal";
    rows.push(`<tr>
      <td><input type="checkbox" ${checked} data-action="toggle-run-selection" data-run-path="${esc(run.path)}"></td>
      <td title="${esc(activeTitle)}"><span class="activityDot ${esc(state)}"></span> ${esc(state)}</td>
      <td class="nameCell" title="${esc(run.path)}"><button type="button" class="linkBtn" data-action="choose-run" data-run-path="${esc(run.path)}">${esc(run.name)}</button></td>
      <td title="${esc(catchup.message||catchup.reason||'')}">${catchup.active?'<span class="pill diagnostic">catch-up</span>':'<span class="muted">normal</span>'}<div class="muted">${esc(catchupText)}</div></td>
      <td class="num">${esc(run.attempts ?? "-")}</td>
      <td class="num ${successCls}">${esc(success)}</td>
      <td class="num">${esc(run.trainable ?? "-")}</td>
      <td class="num">${esc(run.rejected ?? "-")}</td>
      <td class="num">${esc(run.failed ?? "-")}</td>
      <td>${esc(run.size_human || "-")}${run.size_truncated?" *":""}</td>
      <td class="dataSizeCell" title="${esc(run.data_size_human || "-")}">${esc(run.data_size_human || "-")}</td>
      <td>${activity.age_s==null?"-":esc(ageText(activity.age_s)+" ago")}</td>
    </tr>`);
  }
  rows.push('</tbody>');
  table.innerHTML=rows.join('');
}
function humanBytes(value){
  let n=Number(value)||0; const units=["B","KB","MB","GB","TB","PB"]; let i=0;
  while(n>=1024 && i<units.length-1){n/=1024;i++;}
  return i===0?`${Math.round(n)} ${units[i]}`:`${n.toFixed(1)} ${units[i]}`;
}
function toggleRunSelection(path, checked){if(checked) selectedRunPaths.add(path); else selectedRunPaths.delete(path); renderRunManager(runRecords);}
function toggleAllRuns(checked){selectedRunPaths = checked ? new Set(runRecords.map(r=>r.path)) : new Set(); renderRunManager(runRecords);}
function selectRuns(predicate){selectedRunPaths = new Set(runRecords.filter(predicate).map(r=>r.path)); renderRunManager(runRecords);}
function selectedRunList(){return [...selectedRunPaths];}
async function refreshSelectedSizes(){
  let paths=selectedRunList();
  if(!paths.length){
    const current=$('runInput').value || $('runSelect').value;
    if(current) paths=[current];
  }
  if(!paths.length){setStatus('Select run folders first, or choose one run', 'error'); return;}
  if(!confirm(`Exact-size scan ${paths.length} selected run folder(s)? This is intentionally manual because recursive size is O(number of files). Active writers will be skipped.`)) return;
  setStatus('Refreshing exact folder-size cache for selected runs...');
  const result=await postJSON('/api/manage/refresh_sizes', {root:$('rootInput').value, paths});
  const refreshed=(result.refreshed||[]).length, skipped=(result.skipped||[]).length;
  await loadRuns({silent:true});
  setStatus(`Size cache refreshed: refreshed=${refreshed}, skipped=${skipped}. cache=${result.cache_path||''}`, skipped?'error':'ok');
}
async function deleteSelectedZeroSuccessRuns(){
  const paths=selectedRunList();
  if(!paths.length){setStatus("No run folders selected", "error"); return;}
  const count=paths.length;
  if(!confirm(`Move ${count} selected run folder(s) with 0 success to .dashboard_trash? Active writers and success>0 runs will be skipped.`)) return;
  setStatus("Moving selected 0-success runs to trash...");
  const result=await postJSON("/api/manage/delete_runs", {root:$("rootInput").value, paths, zero_success_only:true, allow_active:false});
  const deleted=(result.deleted||[]).length, skipped=(result.skipped||[]).length;
  setStatus(`Trash complete: deleted=${deleted}, skipped=${skipped}. trash=${result.trash_dir||""}`, skipped?"error":"ok");
  selectedRunPaths.clear();
  await loadRuns({silent:true});
}
async function transferSuccessRecords(mode, allSuccess=false){
  syncSuccessPoolPath();
  const dest=$('successDestInput').value.trim() || successPoolPath();
  const paths=allSuccess ? runRecords.filter(r=>Number(r.success||0)>0).map(r=>r.path) : selectedRunList();
  if(!paths.length){setStatus(allSuccess?'No success>0 runs found':'No run folders selected', 'error'); return;}
  const verb=mode==='move'?'Cut/move':'Copy';
  if(!confirm(`${verb} success episodes from ${paths.length} run folder(s) to:\n${joinOne(dest,'episodes')}\n\nFolder names are canonical: YYMMDD_HHMMSS_epXXXXXX. The operation runs as a background job with percentage progress.`)) return;
  setStatus(`${verb} success records job starting...`);
  setTransferProgress('starting success transfer', 0, true);
  const job=await postJSON('/api/manage/success_records', {root:$('rootInput').value, paths, dest_dir:dest, mode});
  if(!job.job_id){throw new Error(job.error || 'success_records job did not start');}
  await pollDashboardJob(job.job_id, async (result)=>{
    const done=(result.transferred||[]).reduce((a,r)=>a+Number(r.records||0),0);
    const cached=Number(result.cached_skipped||0);
    const updated=Number(result.updated_reprocessed||0);
    const skipped=(result.skipped||[]).length;
    const cutEpTrashed=Number(result.cut_source_episode_trashed||0);
    const cutRunTrashed=(result.cut_source_runs_trashed||[]).length;
    const cutFailed=Number(result.cut_source_episode_trash_failed||0)+(result.cut_source_runs_trash_failed||[]).length;
    setStatus(`${verb} complete: new=${done}, cached=${cached}, updated=${updated}, skipped=${skipped}, cut_ep_trashed=${cutEpTrashed}, cut_run_trashed=${cutRunTrashed}, dest=${result.dest_root||dest}`, (skipped>cached||cutFailed)?'error':'ok');
    await loadRuns({silent:true});
  });
}

async function loadSuccessPool(){
  syncSuccessPoolPath();
  const pool=successPoolPath();
  $('runInput').value=pool;
  setStatus('Loading .dashboard_success...');
  await loadRun();
}
async function exportSuccessPool(){
  showExportTimeModal();
}
async function exportSuccessPoolWithPolicy(){
  syncSuccessPoolPath();
  const root=$('rootInput').value;
  const pool=successPoolPath();
  const policy=readExportTimePolicyInputs();
  await saveExportTimePolicyForPreview(false, false);
  hideExportTimeModal();
  setStatus('Starting .dashboard_success VLA export...');
  setTransferProgress('starting VLA export', 0, true);
  const job=await postJSON('/api/manage/export_success_vla', {root, overwrite:true, require_vla:true, time_policy:policy});
  if(!job.job_id){throw new Error(job.error || 'export_success_vla job did not start');}
  await pollDashboardJob(job.job_id, async (result)=>{
    const videoSummary=Object.entries(result.videos||{}).map(([k,v])=>`${k}:${v&&v.available?'ok':'fail'}`).join(', ');
    const encoded=Number(result.encoded_video_jobs||0);
    const reused=Number(result.reused_video_jobs||0);
    const totalJobs=Number(result.total_video_jobs||0);
    const policyText=exportTimePolicyText(result.time_policy||policy);
    const jobSummary=totalJobs?`encoded=${encoded}, reused=${reused}/${totalJobs}`:`encoded=${encoded}, reused=${reused}`;
    setTransferProgress(`Export complete: ${result.total_episodes||0} episodes, ${result.total_frames||0} frames, ${jobSummary}, ${policyText}`, 100, false);
    setStatus(`Export complete: ready=${!!result.vla_training_ready}; ${policyText}; ${jobSummary}; videos=${videoSummary}; dir=${result.export_dir||''}`, result.vla_training_ready?'ok':'error');
    $('runInput').value=pool;
    await loadRun(false);
  });
}

function renderRunMonitor(data){
  const el=$("runMonitor"); if(!el) return;
  const summary=data.activity_summary || {};
  const activeRuns=summary.active_runs || [];
  const activeCount=Number(summary.active_count || 0);
  const recentCount=Number(summary.recent_count || 0);
  const idleCount=Number(summary.idle_count || 0);
  const windowS=Number(summary.active_window_s || 180);
  const title=`<span class="runMonitorTitle"><span class="activityDot ${activeCount?"active":"idle"}"></span> Active writers: ${activeCount}</span>`;
  const subtitle=`<span class="muted">green = updated within ${ageText(windowS)}; recent=${recentCount}; idle=${idleCount}</span>`;
  const badges=activeRuns.map(run=>{
    const latest=run.latest_file ? ` · ${shortText(run.latest_file,42)}` : "";
    return `<button type="button" class="runBadge active" data-action="choose-run" data-run-path="${esc(run.path||"")}" title="${esc(run.path||"")}"><span class="activityDot active"></span>${esc(run.name||"")} <span class="muted">${esc(ageText(run.age_s))} · attempts=${esc(run.attempts ?? "-")} success=${esc(run.success ?? 0)} trainable=${esc(run.trainable ?? "-")} size=${esc(run.size_human||"-")}${esc(latest)}</span></button>`;
  }).join("");
  el.innerHTML=title + subtitle + (badges || `<span class="runBadge"><span class="activityDot idle"></span>no run folder updated recently</span>`);
}
function chooseRun(path){
  if(!path) return;
  $("runInput").value=path;
  const sel=$("runSelect");
  for(const opt of Array.from(sel.options)){ if(opt.value===path){sel.value=path; break;} }
  loadRun().catch(e=>setStatus(e.message,"error"));
}
async function loadRun(force=false){
  const runDir=$("runInput").value || $("runSelect").value;
  if(!runDir){setStatus("Choose a run folder first", "error"); return}
  const started=performance.now();
  setStatus(force ? "Refreshing run analysis..." : "Checking run analysis cache...");
  const slowTimer=setTimeout(()=>setStatus(force ? "Still refreshing run analysis..." : "Still checking run analysis cache..."),1500);
  const rebuildTimer=setTimeout(()=>setStatus(force ? "Large run: still refreshing analysis..." : "Large run: still loading analysis cache..."),4200);
  let data;
  try{
    data=await api("/api/run", {run_dir:runDir, force:force ? "1" : "0", _:Date.now()});
  }finally{
    clearTimeout(slowTimer);
    clearTimeout(rebuildTimer);
  }
  const wallMs=performance.now()-started;
  currentRun=data; currentEpisodeIndex=null;
  renderRun(data);
  const cache=data.analysis_cache || {};
  const timing=cache.timing_ms||{};
  const timingBits=[];
  if(Number.isFinite(Number(timing.signature_ms))) timingBits.push(`signature=${fmt(Number(timing.signature_ms),0)}ms`);
  if(Number.isFinite(Number(timing.disk_cache_read_ms))) timingBits.push(`disk=${fmt(Number(timing.disk_cache_read_ms),0)}ms`);
  if(Number.isFinite(Number(timing.memory_clone_ms))) timingBits.push(`clone=${fmt(Number(timing.memory_clone_ms),0)}ms`);
  if(Number.isFinite(Number(timing.rebuild_ms))) timingBits.push(`rebuild=${fmt(Number(timing.rebuild_ms),0)}ms`);
  const timingText=timingBits.length?` (${timingBits.join(", ")})`:` (${fmt(wallMs,0)}ms)`;
  if(cache.hit){
    setStatus(`Run loaded from ${cache.source || "analysis"} cache${timingText}`, "ok");
  }else{
    setStatus(`${force ? "Run refreshed and cached" : "Run analyzed and cached"}${timingText}`, "ok");
  }
  const eps=filteredEpisodes();
  if(eps.length) loadEpisode(eps[0].episode_index);
}
async function loadEpisode(index){
  if(!currentRun) return;
  currentEpisodeIndex=index; markSelectedTab(index); setStatus(`Loading attempt ${index}...`);
  const data=await api("/api/episode", {run_dir:currentRun.run_dir, episode_index:index, max_points:2400});
  if(!data.ok){setStatus(data.reason||"episode load failed", "error"); return}
  renderEpisode(data); setStatus(`Attempt ${index} loaded`, "ok");
}

function renderRun(data){
  const compact=data.compact||{}; const diagnosis=data.diagnosis||{}; const counts=compact.counts||{}; const segments=compact.segments||{};
  const attempts=Number(counts.all||0), trainable=Number(counts.trainable||0), rejected=Number(counts.rejected||0), failed=Number(counts.failed||counts.fail||0), diagnostic=Number(counts.diagnostic||0), skip=Number(counts.skip||0);
  const datasetMetrics=data.dataset_metrics || diagnosis.dataset_metrics || {};
  const blockerRows=diagnosis.failure_reasons || compact.failure_reasons || compact.primary_reasons || [];
  const qualityRows=diagnosis.quality_alerts || compact.top_quality_alerts || [];
  const warningRows=diagnosis.general_warnings || compact.top_warnings || [];
  const topBlocker=(diagnosis.top_problem_reason&&diagnosis.top_problem_reason.key)||((blockerRows||[])[0]||{}).key||"";
  // KPI strip removed: counts/readiness/efficiency are consolidated into Runtime distribution.
  renderTopBlocker(diagnosis, blockerRows);
  renderQualitySignal(diagnosis, qualityRows);
  renderRuntimePie("runtimePieCard", data.tag_runtime_seconds || datasetMetrics.tag_runtime_seconds || {}, datasetMetrics, diagnosis, {attempts, trainable, rejected, failed, diagnostic, skip}, compact.wall_clock_breakdown || diagnosis.wall_clock_breakdown || {});
  renderGates(diagnosis.readiness_gates||[]);
  renderMaterialTable(diagnosis.material_table||[]);
  renderSchemaTables(diagnosis.field_coverage||{}, diagnosis.camera_coverage||{});
  renderInitialPoseTable(compact.initial_pose_by_status||[]);
  renderBarList("outcomeBars", ["trainable","success","rejected","failed","diagnostic","planning","skip"].map(k=>({key:k,count:counts[k]||0,status:k})).filter(r=>r.count||r.key==="trainable"||r.key==="rejected"||r.key==="skip"), {colorByStatus:true});
  renderBarList("reasonBars", blockerRows, {info:true});
  renderBarList("qualityBars", qualityRows, {quality:true});
  renderBarList("warningBars", warningRows, {warn:true});
  renderBarList("segmentBars", Object.entries(segments).map(([key,count])=>({key,count})), {info:true});
  const act=data.run_activity || {};
  const actState=act.state || "unknown";
  const actText=act.age_s==null ? "write state unknown" : `${actState} · last write ${ageText(act.age_s)} ago`;
  const catchup=data.success_catchup || (diagnosis&&diagnosis.success_catchup) || {};
  const catchupText=catchup.active ? ` &nbsp;|&nbsp; <span class="pill diagnostic" title="${esc(catchup.message||catchup.reason||'')}">success catch-up</span> rows=${esc(catchup.rows_source||catchup.source||'successful_episodes.jsonl')}` : "";
  const reconcile=data.success_pool_reconcile || (diagnosis&&diagnosis.success_pool_reconcile) || {};
  const reconcileText=reconcile.is_success_pool ? ` &nbsp;|&nbsp; <span class="pill diagnostic" title="folders=${esc(reconcile.scanned_episode_folders||0)}, incomplete=${esc(reconcile.incomplete_or_unreadable_folders||0)}, sanitized=${esc(reconcile.sanitized_rows||0)}">pool index ${esc(reconcile.indexed_rows||0)}/${esc(reconcile.scanned_episode_folders||0)} folders; recovered=${esc(reconcile.recovered_orphan_folders||0)}</span>` : "";
  const analysisCache=data.analysis_cache || {};
  const cacheText=analysisCache.source ? ` &nbsp;|&nbsp; <span class="pill ${analysisCache.hit?'success':'diagnostic'}" title="${esc(analysisCache.cache_path||'')}">analysis cache: ${analysisCache.hit?'hit':'rebuilt'} (${esc(analysisCache.source)})</span>` : "";
  $("reportHint").innerHTML = data.run_dir ? `<span class="activityDot ${esc(actState)}"></span> ${esc(actText)}${catchupText}${reconcileText}${cacheText} &nbsp;|&nbsp; Static report: analysis_plots/report.html  |  generate with --plots` : "";
  setupStatusFilter(data.episodes||[]);
  refreshFilteredViews();
}

function renderRuntimePie(id, data, metrics, diagnosis={}, counts={}, wallClock={}){
  const el=$(id); if(!el) return;
  el.className="diagCard runtimePieCard info";
  const entries=Object.entries(data||{}).map(([key,value])=>({key:statusKey(key), value:Number(value)||0})).filter(r=>r.value>0);
  const total=entries.reduce((a,r)=>a+r.value,0);
  const wall=wallClock&&wallClock.available?wallClock:null;
  const wallTotal=wall?Number(wall.total_wall_s||0):0;
  const overhead=wall?Math.max(0,wallTotal-total):0;
  const overheadPct=wallTotal>0?pct(overhead/wallTotal):"";
  const topStages=(wall&&Array.isArray(wall.top_stages)?wall.top_stages:[]).slice(0,4).map(s=>`${esc(s.stage)} ${fmt(Number(s.total_s||0),1)}s`).join(" · ");
  const wallHtml=wall?`<div class="datasetScore">wall-clock: ${fmt(wallTotal,1)}s · recorded: ${fmt(total,1)}s · overhead: ${fmt(overhead,1)}s ${overheadPct}${topStages?`<br>top wall stages: ${topStages}`:""}</div>`:"";
  const attempts=Number(counts.attempts||0);
  const trainable=Number(counts.trainable||0);
  const rejected=Number(counts.rejected||0);
  const failed=Number(counts.failed||0);
  const skip=Number(counts.skip||0);
  const readiness=Number(diagnosis.readiness_score||0);
  const efficiency=Number((metrics&&metrics.data_efficiency_score)||0);
  const medianScore=diagnosis.score_median;
  const statRows=[
    {label:"Readiness", value:`${fmt(readiness,0)}/100`},
    {label:"Data eff.", value:`${fmt(efficiency,0)}/100`},
    {label:"Attempts", value:attempts},
    {label:"Trainable", value:`${trainable} ${pct(trainable/Math.max(1,attempts))}`},
    {label:"Rejected", value:`${rejected} ${pct(rejected/Math.max(1,attempts))}`},
    {label:"Failed", value:`${failed} ${pct(failed/Math.max(1,attempts))}`},
    {label:"Skip", value:`${skip} ${skip?"unexportable":"0"}`},
    {label:"Median score", value:fmt(medianScore,1)||"—"},
  ];
  const statHtml=`<div class="diagStats">${statRows.map(s=>`<div class="diagStat"><div class="diagStatLabel">${esc(s.label)}</div><div class="diagStatValue">${esc(s.value)}</div></div>`).join("")}</div>`;
  const framesHtml=`<div class="datasetScore">usable frames: ${esc((metrics&&metrics.usable_frames)||0)} / ${esc((metrics&&metrics.total_frames)||0)}</div>`;
  if(total<=0){
    const fallbackCounts=(metrics&&metrics.tag_counts)||{};
    const countEntries=Object.entries(fallbackCounts).map(([key,value])=>({key:statusKey(key), value:Number(value)||0})).filter(r=>r.value>0);
    if(!countEntries.length){
      el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">no runtime data</div><div class="diagCardDetail">Counts, readiness and data efficiency are consolidated here; no duplicate KPI strip is shown.</div>${statHtml}${framesHtml}${wallHtml}`;
      return;
    }
    el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">count fallback</div><div class="diagCardDetail">No runtime seconds were available. Counts/readiness/efficiency are consolidated here instead of repeated in KPI cards.</div>${pieMarkup(countEntries,"episodes")}${statHtml}${framesHtml}${wallHtml}`;
    return;
  }
  el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">${fmt(total,1)}s recorded</div><div class="diagCardDetail">Tag runtime share plus dataset balance. Wall-clock overhead is included when the run summary has phase timing.</div>${pieMarkup(entries,"s")}${statHtml}${framesHtml}${wallHtml}`;
}
function pieMarkup(entries, unit){
  const total=entries.reduce((a,r)=>a+r.value,0)||1;
  let angle=-Math.PI/2;
  const cx=64, cy=64, r=54;
  let paths="";
  for(const item of entries){
    const slice=(item.value/total)*Math.PI*2;
    const x1=cx+r*Math.cos(angle), y1=cy+r*Math.sin(angle);
    const x2=cx+r*Math.cos(angle+slice), y2=cy+r*Math.sin(angle+slice);
    const large=slice>Math.PI?1:0;
    const color=statusColor(item.key);
    if(slice>=Math.PI*2-1e-6){
      paths+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${color}"><title>${esc(item.key)} ${fmt(item.value,1)}${unit}</title></circle>`;
    }else{
      paths+=`<path d="M ${cx} ${cy} L ${x1.toFixed(1)} ${y1.toFixed(1)} A ${r} ${r} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} Z" fill="${color}"><title>${esc(item.key)} ${fmt(item.value,1)}${unit}</title></path>`;
    }
    angle+=slice;
  }
  const legend=entries.sort((a,b)=>b.value-a.value).map(item=>`<div class="pieLegendRow"><span><span class="pieSwatch" style="background:${statusColor(item.key)}"></span>${esc(item.key)}</span><b>${fmt(item.value,1)}${esc(unit)}</b></div>`).join("");
  return `<div class="pieWrap"><svg class="pieSvg" viewBox="0 0 128 128" role="img">${paths}<circle class="pieCenter" cx="${cx}" cy="${cy}" r="25"></circle></svg><div class="pieLegend">${legend}</div></div>`;
}
function renderKPIs(items){const el=$("kpiCards"); if(!el) return; el.innerHTML=(items||[]).map(it=>`<div class="kpi ${esc(it.cls||"")}"><div class="kpiLabel">${esc(it.label)}</div><div class="kpiValue">${esc(it.value)}</div><div class="kpiSub">${esc(it.sub||"")}</div></div>`).join("")}
function renderReadiness(diagnosis, topBlocker){const score=Number(diagnosis.readiness_score||0); const blocker=topBlocker?`blocker: ${shortText(topBlocker,26)}`:"no blocker"; $("readinessCard").innerHTML=`<div class="scoreLabel">Training readiness</div><div class="scoreNumber">${fmt(score,0)}</div><div class="scoreBar"><div class="scoreFill" style="width:${Math.max(0,Math.min(100,score))}%"></div></div><div class="chips"><span class="chip">${esc(blocker)}</span><span class="chip">${(diagnosis.readiness_gates||[]).length} gates</span></div>`}
function renderDiagCard(id, cls, title, value, detail, stats){const statHtml=(stats||[]).length?`<div class="diagStats">${stats.map(s=>`<div class="diagStat"><div class="diagStatLabel">${esc(s.label)}</div><div class="diagStatValue">${esc(s.value)}</div></div>`).join("")}</div>`:""; $(id).className=`diagCard ${esc(cls||"info")}`; $(id).innerHTML=`<div class="diagCardTitle">${esc(title)}</div><div class="diagCardValue">${esc(value)}</div><div class="diagCardDetail">${esc(detail||"")}</div>${statHtml}`}
function renderTriageSummary(diagnosis, c){const problem=Number(diagnosis.problem_attempts ?? (c.rejected+c.failed+c.diagnostic)); const rate=c.attempts?pct(c.trainable/Math.max(1,c.attempts)):""; const cls=c.trainable>0?"good":"bad"; renderDiagCard("triageSummaryCard", cls, "Dataset balance", `${c.trainable}/${c.attempts} trainable`, `problem episodes: ${problem}; rejected=${c.rejected}, failed=${c.failed}`, [{label:"trainable",value:rate||"0%"},{label:"problem",value:pct(problem/Math.max(1,c.attempts))},{label:"diagnostic",value:c.diagnostic}])}
function renderTopBlocker(diagnosis, rows){const top=(diagnosis.top_problem_reason&&diagnosis.top_problem_reason.key)?diagnosis.top_problem_reason:(rows||[])[0]; if(!top||!top.key){renderDiagCard("topBlockerCard","good","Top blocker","None","Failure Pareto excludes trainable/success ok rows. No rejected/failed blocker is currently dominant.",[]);return} const count=Number(top.count||0); const denom=Number(diagnosis.problem_attempts||diagnosis.attempts||count||1); const cls=count>=Math.max(1,denom/2)?"bad":"warn"; renderDiagCard("topBlockerCard",cls,"Top blocker",shortText(top.key,58),`${count}/${denom} problem episode(s). This is the first item to debug; it is not an \"ok\" status.`,[{label:"count",value:count},{label:"share",value:pct(count/Math.max(1,denom))},{label:"scope",value:"problem-only"}])}
function renderQualitySignal(diagnosis, rows){const top=(diagnosis.top_quality_signal&&diagnosis.top_quality_signal.key)?diagnosis.top_quality_signal:(rows||[])[0]; if(!top||!top.key){renderDiagCard("qualitySignalCard","good","Quality signal","No dominant signal","score_low / spill / low bin are grouped separately from failure blockers.",[]);return} const count=Number(top.count||0); const cls=String(top.key).includes("spill")||String(top.key).includes("low")?"warn":"info"; renderDiagCard("qualitySignalCard",cls,"Quality signal",shortText(top.key,58),`${count} episode(s). Median bin/spill live here with the quality signal instead of in the KPI strip.`,[{label:"median score",value:fmt(diagnosis.score_median,1)||"—"},{label:"median bin",value:`${fmt(diagnosis.final_bin_median,1)||"0"} particles`},{label:"median spill",value:`${fmt(diagnosis.final_spill_median,0)||"0"} particles`}])}
function renderFindings(id, rows){const el=$(id); if(!el) return; el.innerHTML=(rows||[]).length?(rows||[]).map(r=>`<div class="finding sev-${esc(r.severity||"info")}"><div class="findingTitle">${esc(r.title||r.key||"finding")}</div><div class="findingDetail">${esc(r.detail||r.message||"")}</div></div>`).join(""):'<div class="empty">no findings</div>'}
function renderActions(id, rows){const el=$(id); if(!el) return; el.innerHTML=(rows||[]).length?(rows||[]).map((r,i)=>`<div class="action"><b>${i+1}.</b> ${esc(r)}</div>`).join(""):'<div class="empty">no actions</div>'}
function renderBarList(id, rows, opts={}){const el=$(id); if(!el) return; const data=(rows||[]).map(r=>({key:String(r.key??r[0]??""), count:Number(r.count??r[1]??0), status:r.status})).filter(r=>Number.isFinite(r.count)&&r.count>=0); if(!data.length){el.innerHTML='<div class="empty">no data</div>';return} const max=Math.max(...data.map(r=>r.count),1); el.innerHTML=data.map(r=>{const color=opts.colorByStatus?statusColor(r.status||r.key):(opts.quality?"#7a5af8":opts.warn?"#b54708":opts.info?"#175cd3":"#344054"); return `<div class="barRow"><div class="barLabel" title="${esc(r.key)}">${esc(shortText(r.key,48))}</div><div class="barTrack"><div class="barFill" style="width:${Math.max(1,100*r.count/max)}%;background:${color}"></div></div><div class="barVal">${esc(r.count)}</div></div>`}).join("")}
function renderGates(rows){const html=['<thead><tr><th>Gate</th><th>Status</th><th>Detail</th></tr></thead><tbody>']; for(const g of rows||[]){const st=g.status||g.state||""; const cls=st==="pass"?"gate-pass":st==="warn"?"gate-warn":"gate-fail"; html.push(`<tr><td>${esc(g.name||g.key||"")}</td><td><span class="${cls}">${esc(st)}</span></td><td>${esc(g.detail||g.reason||"")}</td></tr>`)} html.push('</tbody>'); $("gatesTable").innerHTML=rows&&rows.length?html.join(""):'<tbody><tr><td class="muted">no gates</td></tr></tbody>'}
function renderMaterialTable(rows){const html=['<thead><tr><th>Metric</th><th>Count</th><th>Min</th><th>Median</th><th>Mean</th><th>Max</th></tr></thead><tbody>']; for(const r of rows||[]){html.push(`<tr><td>${esc(r.key||r.metric||"")}</td><td>${esc(r.count??"")}</td><td>${fmt(r.min,1)}</td><td>${fmt(r.median,1)}</td><td>${fmt(r.mean,1)}</td><td>${fmt(r.max,1)}</td></tr>`)} html.push('</tbody>'); $("materialTable").innerHTML=rows&&rows.length?html.join(""):'<tbody><tr><td class="muted">no stats</td></tr></tbody>'}
function renderInitialPoseTable(rows){const parts=['<thead><tr><th>Initial pose</th><th>Status</th><th>Count</th></tr></thead><tbody>']; for(const r of rows||[]){const key=String(r.key||""); const split=key.lastIndexOf(":"); const pose=split>=0?key.slice(0,split):key; const status=split>=0?key.slice(split+1):""; parts.push(`<tr><td>${esc(shortText(pose,46))}</td><td><span class="pill ${esc(cssClassStatus(status))}">${esc(status)}</span></td><td>${esc(r.count??"")}</td></tr>`)} parts.push('</tbody>'); $("initialPoseTable").innerHTML=rows&&rows.length?parts.join(""):'<tbody><tr><td class="muted">no data</td></tr></tbody>'}
function renderSchemaTables(fieldCoverage, cameraCoverage){function table(obj,title){const rows=Object.entries(obj||{}).map(([k,v])=>`<tr><td>${esc(shortText(k,34))}</td><td>${esc(v)}</td></tr>`).join(""); return `<div><h3>${esc(title)}</h3><div class="tableWrap"><table class="metricsTable"><tbody>${rows||'<tr><td class="muted">no data</td></tr>'}</tbody></table></div></div>`} $("schemaTables").innerHTML=table(fieldCoverage,"Fields")+table(cameraCoverage,"Cameras")}

function setupStatusFilter(episodes){
  const counts=new Map();
  for(const ep of episodes||[]){const k=statusKey(ep.status); counts.set(k,(counts.get(k)||0)+1)}
  availableStatuses=[...counts.keys()].sort((a,b)=>statusOrder(a)-statusOrder(b));
  selectedStatuses=new Set(availableStatuses);
  const el=$("statusFilter");
  const buttons=[`<button class="filterBtn all active" data-status="__all__">All</button>`];
  for(const st of availableStatuses){const label=shortText(st,28); buttons.push(`<button class="filterBtn ${esc(st)} active" data-status="${esc(st)}" title="${esc(st)}"><span style="display:inline-block;width:8px;height:8px;border-radius:9px;background:${statusColor(st)};margin-right:5px"></span>${esc(label)} ${counts.get(st)||0}</button>`)}
  el.innerHTML=buttons.join("");
  for(const btn of el.querySelectorAll(".filterBtn")){btn.addEventListener("click",()=>toggleStatusFilter(btn.dataset.status))}
  updateFilterUI();
}
function statusOrder(s){return {trainable:1,success:2,rejected:3,failed:4,diagnostic:5,planning:6,skip:7,unknown:99}[statusKey(s)]||50}
function toggleStatusFilter(status){
  if(status==="__all__"){
    const allActive=selectedStatuses.size===availableStatuses.length;
    selectedStatuses=allActive?new Set():new Set(availableStatuses);
  }else{
    if(selectedStatuses.has(status)) selectedStatuses.delete(status); else selectedStatuses.add(status);
  }
  if(!selectedStatuses.size) selectedStatuses=new Set(availableStatuses);
  updateFilterUI();
  refreshFilteredViews();
  const eps=filteredEpisodes();
  if(eps.length && !eps.some(ep=>String(ep.episode_index)===String(currentEpisodeIndex))) loadEpisode(eps[0].episode_index);
}
function updateFilterUI(){
  const el=$("statusFilter"); if(!el) return;
  for(const btn of el.querySelectorAll(".filterBtn")){
    const st=btn.dataset.status;
    btn.classList.toggle("active", st==="__all__" ? selectedStatuses.size===availableStatuses.length : selectedStatuses.has(st));
  }
}
function statusAllowed(s){return selectedStatuses.has(statusKey(s))}
function filteredEpisodes(){return ((currentRun&&currentRun.episodes)||[]).filter(ep=>statusAllowed(ep.status))}
function filteredScenePoints(){return ((currentRun&&currentRun.scene_points)||[]).filter(p=>statusAllowed(p.status))}
function refreshFilteredViews(){
  if(!currentRun) return;
  const pts=filteredScenePoints(); const eps=filteredEpisodes(); const total=(currentRun.episodes||[]).length;
  $("filterCount").textContent=`Showing ${eps.length}/${total} attempts`;
  renderScenePlots(pts, eps);
  renderEpisodes(eps);
}
function validXY(v){return Array.isArray(v)&&finite(Number(v[0]))&&finite(Number(v[1]))}
function normalizeDegJS(v){let n=Number(v); if(!Number.isFinite(n))return null; while(n<=-180)n+=360; while(n>180)n-=360; return n}
function transformWorldToBodyXY(pt, scene){
  if(!validXY(pt)) return null;
  const origin=validXY(scene.robot_origin_xy)?scene.robot_origin_xy:[0,0];
  const yaw=Number(scene.robot_body_yaw_deg||0)*Math.PI/180;
  const dx=Number(pt[0])-Number(origin[0]), dy=Number(pt[1])-Number(origin[1]);
  const c=Math.cos(-yaw), s=Math.sin(-yaw);
  return [dx*c-dy*s, dx*s+dy*c];
}
function sceneBodyXY(scene, bodyKey, worldKey){
  if(validXY(scene[bodyKey])) return scene[bodyKey];
  return transformWorldToBodyXY(scene[worldKey], scene);
}
function sceneBodyPoly(scene){
  const body=(Array.isArray(scene.unload_polygon_body_xy)&&scene.unload_polygon_body_xy.length)?scene.unload_polygon_body_xy:scene.unload_hull_body_xy;
  if(Array.isArray(body)&&body.length) return body;
  const world=(Array.isArray(scene.unload_polygon_xy)&&scene.unload_polygon_xy.length)?scene.unload_polygon_xy:scene.unload_hull_xy;
  if(!Array.isArray(world)) return [];
  return world.map(pt=>transformWorldToBodyXY(pt, scene)).filter(validXY);
}
function sceneTruckYawBody(scene){
  if(Number.isFinite(Number(scene.truck_yaw_body_deg))) return Number(scene.truck_yaw_body_deg);
  const truck=Number(scene.truck_yaw_deg), robot=Number(scene.robot_body_yaw_deg||0);
  return Number.isFinite(truck)?normalizeDegJS(truck-robot):null;
}
function renderScenePlots(pts, eps){
  drawScatter(
    "sandScatter",
    pts.map(p=>{const xy=sceneBodyXY(p,"sand_body_xy","sand_xy"); return {x:xy&&xy[0], y:xy&&xy[1], status:p.status, label:p.episode_index, yaw:p.robot_body_yaw_deg};}),
    "body x / forward from excavator", "body y / left from excavator",
    {origin:true, unit:"m", width:640, height:420, equalAspect:true, bodyFrame:true}
  );
  drawScatter(
    "truckScatter",
    pts.map(p=>{const xy=sceneBodyXY(p,"truck_body_xy","truck_xy"); return {x:xy&&xy[0], y:xy&&xy[1], status:p.status, label:p.episode_index, yaw:sceneTruckYawBody(p), polygon:sceneBodyPoly(p), robotYaw:p.robot_body_yaw_deg};}),
    "body x / forward from excavator", "body y / left from excavator",
    {origin:true, unit:"m", polygons:true, heading:true, aggregate:true, densityThreshold:70, maxPolygons:24, width:900, height:520, equalAspect:true, truckMeshOverlay:true, bodyFrame:true}
  );
  drawScatter("unloadScatter", canonicalUnloadPoints(pts), "selected mesh local x", "selected mesh local y", {origin:true, unit:"m", polygons:true, meshLocal:true, aggregate:true, densityThreshold:180, maxPolygons:140, width:900, height:520, equalAspect:true});
  drawHistogram("sandAmountHist", pts.map(p=>p.sand_amount_multiplier), "x", {mini:true, bins:10});
  drawHistogram("truckYawHist", pts.map(p=>p.truck_yaw_body_deg ?? p.truck_yaw_deg), "deg", {mini:true, bins:12});
  drawHistogram("robotYawHist", pts.map(p=>p.robot_body_yaw_deg), "deg", {mini:true, bins:12});
  drawScatter("scoreSpillScatter", eps.map(ep=>({x:ep.final_spill, y:ep.score, status:ep.status, label:ep.episode_index})), "final spill particles", "score", {unit:"", aggregate:true, densityThreshold:120, width:720, height:420});
  drawScatter("binSpillScatter", eps.map(ep=>({x:ep.final_spill, y:ep.final_bin, status:ep.status, label:ep.episode_index})), "final spill particles", "final bin particles", {unit:"particles", aggregate:true, densityThreshold:120, width:720, height:420});
}

function renderEpisodes(episodes){
  const sorted=[...(episodes||[])].sort((a,b)=>{const av=sortValue(a,episodeSort.key), bv=sortValue(b,episodeSort.key); if(av===bv) return Number(a.episode_index||0)-Number(b.episode_index||0); if(av===null)return 1; if(bv===null)return -1; return (av<bv?-1:1)*episodeSort.dir});
  const label=episodeSortLabel(episodeSort.key); const arrow=episodeSort.dir>0?"↑":"↓";
  const hasExportFields=sorted.some(ep=>Object.prototype.hasOwnProperty.call(ep,"export_ready"));
  const staleExportStatuses=new Set(["time_policy_mismatch","export_config_mismatch"]);
  const exportMeta=hasExportFields?` · export ready ${sorted.filter(ep=>ep.export_ready===true).length}, stale ${sorted.filter(ep=>staleExportStatuses.has(ep.export_status)).length}, missing ${sorted.filter(ep=>ep.export_ready===false&&!staleExportStatuses.has(ep.export_status)).length}, unknown ${sorted.filter(ep=>ep.export_ready!==true&&ep.export_ready!==false).length}`:"";
  $("episodeSideMeta").textContent=`${sorted.length} shown · sorted by ${label} ${arrow}${exportMeta}`;
  if(!sorted.length){$("episodeTabs").innerHTML='<div class="empty">No attempts match the current status filter.</div>'; syncEpisodeInspectorHeight(); return}
  $("episodeTabs").innerHTML=episodeTableHtml(sorted);
  markSelectedTab(currentEpisodeIndex);
  syncEpisodeInspectorHeight();
}
function episodeSortLabel(key){return ({episode_index:"Ep",time_s:"Time",score:"Score",max_bucket:"Bucket",lift_bucket:"Lift",final_bin:"Bin",final_spill:"Spill",robot_yaw:"Robot yaw",truck_yaw:"Truck yaw",samples:"Samples",freeze_count:"Freeze"})[key]||key}
function sortHeader(key,label,cls=""){const arrow=episodeSort.key===key?(episodeSort.dir>0?" ▲":" ▼"):""; return `<th class="sortable ${esc(cls)}" data-action="sort-episodes" data-sort-key="${esc(key)}" title="Click to sort by ${esc(label)}">${esc(label)}${arrow}</th>`}
function episodeTableHtml(rows){const head=`<thead><tr>${sortHeader("episode_index","Ep","epCol")}<th>Status</th>${sortHeader("time_s","Time","num")}${sortHeader("score","Score","num")}${sortHeader("max_bucket","Bucket","num")}${sortHeader("lift_bucket","Lift","num")}${sortHeader("final_bin","Bin","num")}${sortHeader("final_spill","Spill","num")}${sortHeader("robot_yaw","Robot yaw","num")}${sortHeader("truck_yaw","Truck yaw","num")}<th>Reason</th></tr></thead>`; const body=rows.map(ep=>episodeRowHtml(ep)).join(""); return `<table id="episodeTable" class="episodeDataSheet">${head}<tbody>${body}</tbody></table>`}
function exportReadyState(ep){
  if(ep.export_status==="time_policy_mismatch"||ep.export_status==="export_config_mismatch") return {cls:"stale", title:`VLA export stale: settings mismatch - ${ep.export_reason||"re-export required"}`};
  if(ep.export_ready===true) return {cls:"ready", title:`VLA export ready${ep.export_video_count!=null?` - videos=${ep.export_video_count}`:""}`};
  if(ep.export_ready===false) return {cls:"notReady", title:`VLA export missing/not ready - ${ep.export_reason||"not exported or missing"}`};
  return {cls:"unknown", title:`VLA export unknown - ${ep.export_reason||"load .dashboard_success or export first"}`};
}
function episodeRowHtml(ep){const s=ep.scene||{}; const status=statusKey(ep.status); const reason=ep.reason||ep.warning_reason||""; const time=Number(ep.time_s); const timeText=Number.isFinite(time)?`${fmt(time,2)}s`:""; const exportState=exportReadyState(ep); const epCell=`<span class="epCellInner"><span>${esc(ep.episode_index)}</span><span class="exportReadyDot ${esc(exportState.cls)}" title="${esc(exportState.title)}"></span></span>`; return `<tr data-ep="${esc(ep.episode_index)}"><td class="epCol">${epCell}</td><td><span class="pill ${esc(status)}">${esc(status)}</span></td><td class="num">${esc(timeText)}</td><td class="num">${fmt(ep.score,1)}</td><td class="num">${esc(ep.max_bucket??"")}</td><td class="num">${esc(ep.lift_bucket??"")}</td><td class="num">${esc(ep.final_bin??"")}</td><td class="num">${esc(ep.final_spill??"")}</td><td class="num">${fmt(s.robot_body_yaw_deg,1)}</td><td class="num">${fmt(s.truck_yaw_deg,1)}</td><td class="reasonCell" title="${esc(reason)}">${esc(shortText(reason||"no reason",150))}</td></tr>`}
function sortEpisodes(key){if(episodeSort.key===key){episodeSort.dir*=-1}else{episodeSort={key,dir:key==="episode_index"?1:-1}} renderEpisodes(filteredEpisodes())}
function sortValue(ep,key){if(key==="robot_yaw") return numericOrNull((ep.scene||{}).robot_body_yaw_deg); if(key==="truck_yaw") return numericOrNull((ep.scene||{}).truck_yaw_deg); return numericOrNull(ep[key])}
function numericOrNull(v){const n=Number(v); return Number.isFinite(n)?n:null}
function markSelectedTab(index){document.querySelectorAll("#episodeTabs tr[data-ep]").forEach(row=>row.classList.toggle("selected",row.dataset.ep==String(index)))}

function renderEpisode(data){
  const ep=data.episode||{}; $("episodeTitle").textContent=`Attempt ${ep.episode_index} timeline`; $("episodeMeta").textContent=`${statusKey(ep.status)} · score=${fmt(ep.score,1)} · samples=${data.sample_count} · shown=${data.returned_points} · ${shortText(ep.dataset_skip_reason || ep.reason||ep.warning_reason||"",260)}`; $("rawBox").textContent=JSON.stringify({episode:ep,stage_spans:data.stage_spans,camera_preview:data.camera_preview},null,2);
  const tp=data.time_policy||ep.export_time_policy||{};
  const tpPolicy=normalizeExportTimePolicyJS(tp.policy||tp.time_policy||{});
  const tpInfo=`speed=${fmt(tpPolicy.speed_scale,3)}x, uniform time`;
  $("episodeMeta").textContent=`${statusKey(ep.status)} | score=${fmt(ep.score,1)} | samples=${data.sample_count} | shown=${data.returned_points} | ${tpInfo} | ${shortText(ep.dataset_skip_reason || ep.reason||ep.warning_reason||"",260)}`;
  $("rawBox").textContent=JSON.stringify({episode:ep,time_policy:data.time_policy,stage_spans:data.stage_spans,camera_preview:data.camera_preview},null,2);
  const promptEl=$("episodeTaskPrompt"); const promptText=String(ep.task_prompt||"").trim(); if(promptEl){promptEl.style.display=promptText?"block":"none"; promptEl.innerHTML=promptText?`<span class="taskPromptLabel">Task prompt</span>${esc(promptText)}`:"";}
  renderCameraPreview(data);
  const s=data.series||{}; drawLineChart("bucketChart","Bucket sand holding",s.t,[{name:"bucket_from_pile",values:s.bucket_from_pile},{name:"bucket_total",values:s.bucket_total}],data.stage_spans,"particles"); drawVectorChart("qChart","Joint angles",s.t,s.q_deg,data.stage_spans,"deg"); drawVectorChart("dqChart","Joint velocity",s.t,s.dq_deg_s,data.stage_spans,"deg/s"); drawVectorChart("ddqChart","Joint acceleration",s.t,s.ddq_deg_s2,data.stage_spans,"deg/s²"); drawVectorChart("effortChart","Measured joint effort",s.t,s.effort,data.stage_spans,"effort");
  updateTimelineCursors();
  markSelectedTab(ep.episode_index);
  syncEpisodeInspectorHeight();
}

function cameraDisplayName(key){
  const k=String(key||"");
  if(k.endsWith(".0")) return "Cam 0";
  if(k.endsWith(".1")) return "Cam 1";
  if(k.endsWith(".2")) return "Cam 2";
  return k || "camera";
}
function cameraFrameMeta(preview, frameIndex){
  const frames=(preview&&preview.frames)||[];
  const idx=Number(frameIndex)||0;
  const row=frames[idx]||{};
  const tp=(preview&&preview.time_policy)||{};
  const policy=normalizeExportTimePolicyJS(tp.policy||tp.time_policy||tp||{});
  const mode=String(policy.time_mode||"");
  const t=Number(row.t);
  const rawT=Number(row.raw_t);
  let label="t";
  if(mode==="uniform_fps") label="uniform t";
  const tText=Number.isFinite(t)?`${label}=${fmt(t,2)}s`:"";
  const rawText=(Number.isFinite(rawT)&&Number.isFinite(t)&&Math.abs(rawT-t)>1e-4)?` · raw=${fmt(rawT,2)}s`:"";
  const phase=row.phase?` · ${row.phase}`:"";
  return `frame ${idx}${tText?` · ${tText}`:""}${rawText}${phase}`;
}
function frameCacheToken(){
  const cache=(currentRun&&currentRun.analysis_cache)||{};
  return String(cache.signature_hash || (currentRun&&currentRun.generated_at) || "frames");
}
function cameraImageUrl(cameraKey, frameIndex){
  const qs=new URLSearchParams({
    run_dir:(currentRun&&currentRun.run_dir)||$("runInput").value||"",
    episode_index:String(currentEpisodeIndex ?? ""),
    camera:String(cameraKey),
    frame_index:String(frameIndex),
    v:frameCacheToken()
  });
  return `/api/frame?${qs.toString()}`;
}
function stopCameraPlayer(){
  cameraPlayerPlaying=false;
  if(cameraPlayerTimer){
    clearTimeout(cameraPlayerTimer);
    cameraPlayerTimer=null;
  }
  const btn=$("cameraPlayBtn");
  if(btn){
    btn.textContent="Play";
    btn.classList.remove("playing");
  }
}
function updateCameraPlayButton(){
  const btn=$("cameraPlayBtn");
  if(!btn) return;
  btn.textContent=cameraPlayerPlaying?"Pause":"Play";
  btn.classList.toggle("playing",cameraPlayerPlaying);
}
function cameraPlayerDelayMs(){
  const fps=Math.max(1,Math.min(60,Number(CAMERA_PLAYER_FIXED_FPS)||10));
  return Math.round(1000/fps);
}
function cameraPlayerTick(){
  if(!cameraPlayerPlaying) return;
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const frameCount=Number(preview.frame_count||0);
  if(!frameCount){
    stopCameraPlayer();
    return;
  }
  updateCameraPreviewFrame((cameraFrameIndex+1)%frameCount,{immediate:true});
  cameraPlayerTimer=setTimeout(cameraPlayerTick,cameraPlayerDelayMs());
}
function toggleCameraPlayer(){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  if(!Number(preview.frame_count||0)) return;
  cameraPlayerPlaying=!cameraPlayerPlaying;
  updateCameraPlayButton();
  if(cameraPlayerPlaying){
    if(cameraPlayerTimer) clearTimeout(cameraPlayerTimer);
    cameraPlayerTimer=setTimeout(cameraPlayerTick,cameraPlayerDelayMs());
  }else if(cameraPlayerTimer){
    clearTimeout(cameraPlayerTimer);
    cameraPlayerTimer=null;
  }
}
function stepCameraPreview(delta){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const frameCount=Number(preview.frame_count||0);
  if(!frameCount) return;
  updateCameraPreviewFrame((cameraFrameIndex+Number(delta)+frameCount)%frameCount,{immediate:true});
}

function currentCameraTime(){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const frames=preview.frames||[];
  const row=frames[cameraFrameIndex]||{};
  const t=Number(row.t);
  return Number.isFinite(t)?t:0;
}
function nearestCameraFrameByTime(timeValue){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const frames=preview.frames||[];
  if(!frames.length) return 0;
  const target=Number(timeValue);
  if(!Number.isFinite(target)) return cameraFrameIndex;
  let best=0,bestDist=Infinity;
  frames.forEach((row,i)=>{
    const t=Number(row&&row.t);
    if(!Number.isFinite(t)) return;
    const d=Math.abs(t-target);
    if(d<bestDist){best=i;bestDist=d;}
  });
  return best;
}
function jumpCameraToTime(timeValue, opts={}){
  const idx=nearestCameraFrameByTime(timeValue);
  updateCameraPreviewFrame(idx,{immediate:true});
}
function timelineSvgMoveToEvent(evt){
  const svg=evt&&evt.currentTarget || timelineDragState.svg;
  if(!svg) return;
  const left=Number(svg.dataset.left||88);
  const width=Number(svg.dataset.plotWidth||858);
  const x0=Number(svg.dataset.x0||0);
  const x1=Number(svg.dataset.x1||0);
  if(!Number.isFinite(width)||width<=0||!Number.isFinite(x0)||!Number.isFinite(x1)) return;
  const rect=svg.getBoundingClientRect();
  const vb=svg.viewBox&&svg.viewBox.baseVal;
  const viewW=vb&&vb.width?vb.width:rect.width;
  const x=(evt.clientX-rect.left)*viewW/Math.max(1,rect.width);
  const ratio=Math.max(0,Math.min(1,(x-left)/width));
  jumpCameraToTime(x0+(x1-x0)*ratio,{immediate:true});
}
function timelineSvgMouseDown(evt){
  if(evt.button!==0) return;
  timelineDragState.active=true;
  timelineDragState.svg=evt.currentTarget;
  timelineDragState.moved=false;
  timelineDragState.suppressClick=false;
  evt.preventDefault();
  timelineSvgMoveToEvent(evt);
}
function timelineSvgMouseMove(evt){
  if(!timelineDragState.active||evt.buttons!==1) return;
  timelineDragState.moved=true;
  evt.preventDefault();
  timelineSvgMoveToEvent(evt);
}
function timelineSvgMouseUp(evt){
  if(!timelineDragState.active) return;
  timelineDragState.active=false;
  timelineDragState.suppressClick=timelineDragState.moved;
  timelineDragState.svg=null;
}
document.addEventListener("mousemove", evt=>{
  if(!timelineDragState.active||evt.buttons!==1) return;
  timelineDragState.moved=true;
  evt.preventDefault();
  timelineSvgMoveToEvent({currentTarget:timelineDragState.svg, clientX:evt.clientX, clientY:evt.clientY});
});
document.addEventListener("mouseup", ()=>{
  if(!timelineDragState.active) return;
  timelineDragState.active=false;
  timelineDragState.suppressClick=timelineDragState.moved;
  timelineDragState.svg=null;
});
function timelineSvgClick(evt){
  if(timelineDragState.suppressClick){timelineDragState.suppressClick=false; return;}
  timelineSvgMoveToEvent(evt);
}
function bindTimelineSvgInteractions(container){
  const root=container||document;
  root.querySelectorAll(".timelineSvgClickable").forEach(svg=>{
    if(svg.dataset.boundTimelineEvents==="1") return;
    svg.dataset.boundTimelineEvents="1";
    svg.addEventListener("mousedown",timelineSvgMouseDown);
    svg.addEventListener("mousemove",timelineSvgMouseMove);
    svg.addEventListener("mouseup",timelineSvgMouseUp);
    svg.addEventListener("click",timelineSvgClick);
  });
}
function cameraTimelineMarkup(preview, spans){
  const f=chartFrame(980,112);
  const frames=(preview&&preview.frames)||[];
  const times=frames.map(r=>Number(r&&r.t)).filter(Number.isFinite);
  const x0=times.length?Math.min(...times):0;
  const x1=times.length?Math.max(...times):Math.max(1,Number(preview&&preview.frame_count||1)/Math.max(1,CAMERA_PLAYER_FIXED_FPS));
  const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw;
  const parts=[`<div class="cameraTimeline"><svg class="timelineSvgClickable" viewBox="0 0 ${f.w} ${f.h}" data-left="${f.l}" data-plot-width="${f.pw}" data-x0="${x0}" data-x1="${x1}" role="img">`];
  parts.push(`<text x="14" y="23" font-size="13" font-weight="850" fill="#101828">Camera timeline</text>`);
  parts.push(`<text id="cameraTimelineFrameMeta" x="150" y="23" font-size="11" fill="#667085">${esc(cameraFrameMeta(preview, cameraFrameIndex))}</text>`);
  drawStageRects(parts,spans||[],sx,36,38);
  parts.push(`<rect x="${f.l}" y="36" width="${f.pw}" height="38" fill="none" stroke="#d0d5dd"/>`);
  for(let i=0;i<=4;i++){
    const x=f.l+f.pw*i/4;
    const t=x0+(x1-x0)*i/4;
    parts.push(`<line x1="${x.toFixed(1)}" y1="36" x2="${x.toFixed(1)}" y2="78" stroke="#eef2f6"/>`);
    parts.push(`<text x="${x.toFixed(1)}" y="96" text-anchor="middle" font-size="11" fill="#667085">${fmtAxis(t)}s</text>`);
  }
  const cx=sx(currentCameraTime());
  parts.push(`<line class="cameraTimelineCursor" data-left="${f.l}" data-plot-width="${f.pw}" data-x0="${x0}" data-x1="${x1}" x1="${cx.toFixed(1)}" y1="32" x2="${cx.toFixed(1)}" y2="82" stroke="#f04438" stroke-width="2.2"/>`);
  parts.push(`</svg></div>`);
  return parts.join("");
}
function updateTimelineCursors(){
  const t=currentCameraTime();
  document.querySelectorAll(".timelineCursor,.cameraTimelineCursor").forEach(line=>{
    const left=Number(line.dataset.left||88);
    const width=Number(line.dataset.plotWidth||858);
    const x0=Number(line.dataset.x0||0);
    const x1=Number(line.dataset.x1||0);
    const ratio=(t-x0)/(x1-x0||1);
    const x=left+Math.max(0,Math.min(1,ratio))*width;
    line.setAttribute("x1",x.toFixed(1));
    line.setAttribute("x2",x.toFixed(1));
  });
}
function trimCameraFrameImageCache(){
  while(cameraFrameImageCacheOrder.length>cameraFrameImageCacheLimit){
    const key=cameraFrameImageCacheOrder.shift();
    if(key) cameraFrameImageCache.delete(key);
  }
}
function cachedCameraImagePromise(url){
  if(cameraFrameImageCache.has(url)) return cameraFrameImageCache.get(url);
  const promise=new Promise(resolve=>{
    const img=new Image();
    img.addEventListener("load",()=>resolve({ok:true,url,img}),{once:true});
    img.addEventListener("error",()=>resolve({ok:false,url,img}),{once:true});
    img.decoding="async";
    img.src=url;
  });
  cameraFrameImageCache.set(url,promise);
  cameraFrameImageCacheOrder.push(url);
  trimCameraFrameImageCache();
  return promise;
}
function prefetchCameraFrames(startFrame, count=6){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const cameras=preview.cameras||[];
  const frameCount=Number(preview.frame_count||0);
  if(!frameCount) return;
  for(let step=1; step<=count; step++){
    const frame=(Number(startFrame)+step)%frameCount;
    for(const cam of cameras){
      if(!cam.available) continue;
      cachedCameraImagePromise(cameraImageUrl(cam.key,frame));
    }
  }
}
function bindCameraPreviewControls(container){
  const root=container||document;
  const slider=root.querySelector("#cameraFrameSlider");
  if(slider && slider.dataset.boundCameraPreview!=="1"){
    slider.dataset.boundCameraPreview="1";
    slider.addEventListener("input",()=>updateCameraPreviewFrame(slider.value));
  }
  const play=root.querySelector("#cameraPlayBtn");
  if(play && play.dataset.boundCameraPreview!=="1"){
    play.dataset.boundCameraPreview="1";
    play.addEventListener("click",toggleCameraPlayer);
  }
  const prev=root.querySelector("#cameraPrevBtn");
  if(prev && prev.dataset.boundCameraPreview!=="1"){
    prev.dataset.boundCameraPreview="1";
    prev.addEventListener("click",()=>stepCameraPreview(-1));
  }
  const next=root.querySelector("#cameraNextBtn");
  if(next && next.dataset.boundCameraPreview!=="1"){
    next.dataset.boundCameraPreview="1";
    next.addEventListener("click",()=>stepCameraPreview(1));
  }
  root.querySelectorAll(".cameraCard img").forEach(img=>{
    if(img.dataset.boundCameraPreview==="1") return;
    img.dataset.boundCameraPreview="1";
    img.addEventListener("error",()=>{
      const card=img.closest(".cameraCard");
      if(card) card.classList.add("missing");
    });
  });
  bindTimelineSvgInteractions(root);
}
function renderCameraPreview(data){
  const box=$("cameraPreview"); if(!box) return;
  stopCameraPlayer();
  const preview=data.camera_preview||{};
  preview.time_policy=data.time_policy||{};
  if(currentRun) currentRun._lastEpisodePreview=preview;
  const frameCount=Number(preview.frame_count||data.sample_count||0);
  const cameras=preview.cameras||[];
  if(!frameCount || !cameras.length){
    box.innerHTML='<div class="empty">No camera frames for this attempt.</div>';
    return;
  }
  const preferred=Number(preview.initial_frame_index||0);
  cameraFrameIndex=Math.max(0, Math.min(frameCount-1, Number.isFinite(preferred)?preferred:0));
  const header=`<div class="cameraPreviewHeader"><div class="cameraFrameControls"><span class="muted nowrap">frame</span><input id="cameraFrameSlider" type="range" min="0" max="${Math.max(0,frameCount-1)}" value="${cameraFrameIndex}"><span id="cameraFrameText" class="mono small nowrap">${cameraFrameIndex}/${Math.max(0,frameCount-1)}</span></div></div>`;
  const cards=cameras.map((cam,idx)=>{
    const key=cam.key; const available=!!cam.available; const existing=Number(cam.existing_frames||0); const present=Number(cam.present_frames||0);
    const cls=available?"cameraCard":"cameraCard missing";
    const src=available?cameraImageUrl(key,cameraFrameIndex):"";
    const blackWarn=cam.looks_all_black?`<span class="pill failed" title="sampled frames are black">black</span>`:"";
    return `<div class="${cls}" data-camera-key="${esc(key)}"><div class="cameraCardHeader"><span class="cameraCardTitle">${esc(cam.label||cameraDisplayName(key))} ${blackWarn}</span><span title="existing/present frames">${existing}/${present}</span></div><img id="cameraImg${idx}" src="${esc(src)}" alt="${esc(cam.label||cameraDisplayName(key))}"><div class="cameraMissing">missing image for this camera/frame</div><div style="padding:6px 8px"><a id="cameraOpen${idx}" class="cameraOpenLink" href="${esc(src)}" target="_blank">open image</a></div></div>`;
  }).join("");
  const timeline=cameraTimelineMarkup(preview, data.stage_spans||[]);
  box.innerHTML=header+timeline+`<div class="cameraGrid">${cards}</div>`;
  const headerEl=box.querySelector(".cameraPreviewHeader");
  const frameControls=box.querySelector(".cameraFrameControls");
  if(headerEl&&frameControls){
    const player=document.createElement("div");
    player.className="cameraPlayerControls";
    player.innerHTML=`<button id="cameraPlayBtn" class="cameraPlayerBtn primary" type="button">Play</button><button id="cameraPrevBtn" class="cameraPlayerBtn" type="button" title="Previous frame">-1</button><button id="cameraNextBtn" class="cameraPlayerBtn" type="button" title="Next frame">+1</button>`;
    headerEl.insertBefore(player,frameControls);
  }
  bindCameraPreviewControls(box);
  prefetchCameraFrames(cameraFrameIndex, 8);
}
function updateCameraPreviewFrame(value,opts){
  const frameCount=Number(((currentRun&&currentRun._lastEpisodePreview)||{}).frame_count||0);
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  if(!frameCount) return;
  cameraFrameIndex=Math.max(0,Math.min(frameCount-1,Number(value)||0));
  const slider=$("cameraFrameSlider"); if(slider && Number(slider.value)!==cameraFrameIndex) slider.value=String(cameraFrameIndex);
  const text=$("cameraFrameText"); if(text) text.textContent=`${cameraFrameIndex}/${Math.max(0,frameCount-1)}`;
  const meta=$("cameraFrameMeta"); if(meta) meta.textContent=cameraFrameMeta(preview,cameraFrameIndex);
  const timelineMeta=$("cameraTimelineFrameMeta"); if(timelineMeta) timelineMeta.textContent=cameraFrameMeta(preview,cameraFrameIndex);
  updateTimelineCursors();
  prefetchCameraFrames(cameraFrameIndex, cameraPlayerPlaying?10:4);
  if(cameraPreviewTimer) clearTimeout(cameraPreviewTimer);
  if(opts&&opts.immediate){
    refreshCameraPreviewImages(cameraFrameIndex);
  }else{
    cameraPreviewTimer=setTimeout(()=>refreshCameraPreviewImages(cameraFrameIndex),70);
  }
}
function refreshCameraPreviewImages(frameIndex){
  const preview=(currentRun&&currentRun._lastEpisodePreview)||{};
  const cameras=preview.cameras||[];
  const seq=++cameraPreviewLoadSeq;
  const jobs=cameras.map((cam,idx)=>{
    if(!cam.available) return Promise.resolve({idx,cam,url:"",ok:false});
    const url=cameraImageUrl(cam.key,frameIndex);
    const img=$("cameraImg"+idx);
    const card=img&&img.closest(".cameraCard");
    if(card) card.classList.add("loading");
    return cachedCameraImagePromise(url).then(result=>({idx,cam,url,ok:!!(result&&result.ok)}));
  });
  Promise.all(jobs).then(results=>{
    if(seq!==cameraPreviewLoadSeq) return;
    results.forEach(({idx,cam,url,ok})=>{
      const img=$("cameraImg"+idx), open=$("cameraOpen"+idx);
      if(!img || !cam.available) return;
      const card=img.closest(".cameraCard");
      if(card){
        card.classList.remove("loading");
        card.classList.toggle("missing",!ok);
      }
      if(ok && img.getAttribute("src")!==url) img.src=url;
      if(open) open.href=url;
    });
  });
}

function chartFrame(width=760,height=220){const l=88,r=34,t=36,b=58; return {w:width,h:height,l,r,t,b,pw:width-l-r,ph:height-t-b}}
function extent(vals){const arr=numeric(vals).filter(v=>Math.abs(v)<1e12); if(!arr.length)return[0,1]; let lo=Math.min(...arr),hi=Math.max(...arr); if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1} const pad=(hi-lo)*0.08; return[lo-pad,hi+pad]}
function drawAxes(parts,f,x0,x1,y0,y1,opts={}){parts.push(`<rect x="${f.l}" y="${f.t}" width="${f.pw}" height="${f.ph}" fill="#fcfcfd" stroke="#d0d5dd"/>`); for(let i=0;i<=4;i++){const x=f.l+f.pw*i/4,y=f.t+f.ph*i/4; parts.push(`<line x1="${x.toFixed(1)}" y1="${f.t}" x2="${x.toFixed(1)}" y2="${f.t+f.ph}" stroke="#eef2f6"/>`); parts.push(`<line x1="${f.l}" y1="${y.toFixed(1)}" x2="${f.l+f.pw}" y2="${y.toFixed(1)}" stroke="#eef2f6"/>`)} const xs=opts.xSuffix||""; parts.push(`<text x="${f.l}" y="${f.h-22}" font-size="11" fill="#667085">${fmtAxis(x0)}${esc(xs)}</text>`); parts.push(`<text x="${f.l+f.pw}" y="${f.h-22}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(x1)}${esc(xs)}</text>`); parts.push(`<text x="${f.l-8}" y="${f.t+f.ph}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(y0)}</text>`); parts.push(`<text x="${f.l-8}" y="${f.t+10}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(y1)}</text>`)}
function drawStageRects(parts,spans,scaleX,top,height){const seen=new Map(); let next=0; for(const span of spans||[]){if(!seen.has(span.stage))seen.set(span.stage,stagePalette[next++%stagePalette.length]); const x=scaleX(span.start),w=Math.max(1,scaleX(span.end)-x); parts.push(`<rect x="${x.toFixed(1)}" y="${top}" width="${w.toFixed(1)}" height="${height}" fill="${seen.get(span.stage)}" opacity="0.44"><title>${esc(span.stage)}</title></rect>`)}}
function drawLineChart(targetId,title,xs,lines,spans,unit){const f=chartFrame(980,270); xs=xs||[]; const xVals=numeric(xs); const x0=xVals.length?Math.min(...xVals):0,x1=xVals.length?Math.max(...xVals):1; const allY=[]; for(const line of lines){for(const v of line.values||[]) if(finite(Number(v))) allY.push(Number(v))} const [y0,y1]=extent(allY); const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw; const sy=y=>f.t+f.ph-(Number(y)-y0)/(y1-y0||1)*f.ph; const currentX=sx(currentCameraTime()); const parts=[`<svg class="chart timelineSvgClickable" viewBox="0 0 ${f.w} ${f.h}" data-left="${f.l}" data-plot-width="${f.pw}" data-x0="${x0}" data-x1="${x1}" role="img">`,`<text x="14" y="22" font-size="15" font-weight="750" fill="#101828">${esc(title)}</text>`]; let lx=230; if(unit){parts.push(`<rect x="${lx}" y="9" width="${Math.max(48,unit.length*7+26)}" height="18" rx="9" fill="#ffffff" stroke="#d0d5dd"/><text x="${lx+10}" y="22" font-size="11" font-weight="750" fill="#475467">unit: ${esc(unit)}</text>`); lx+=Math.max(58,unit.length*7+36);} lines.forEach((line,li)=>{const color=lineColors[li%lineColors.length]; parts.push(`<circle cx="${lx}" cy="17" r="4.5" fill="${color}"/><text x="${lx+8}" y="21" font-size="11" fill="#475467">${esc(line.name)}</text>`); lx+=Math.max(76,String(line.name||"").length*6+22);}); drawAxes(parts,f,x0,x1,y0,y1,{xSuffix:"s"}); drawStageRects(parts,spans,sx,f.t,f.ph); lines.forEach((line,li)=>{const pts=[];(line.values||[]).forEach((v,i)=>{if(finite(Number(v))&&finite(Number(xs[i])))pts.push(`${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`)}); if(pts.length) parts.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="${lineColors[li%lineColors.length]}" stroke-width="1.9"/>`)}); parts.push(`<line class="timelineCursor" data-left="${f.l}" data-plot-width="${f.pw}" data-x0="${x0}" data-x1="${x1}" x1="${currentX.toFixed(1)}" y1="${f.t}" x2="${currentX.toFixed(1)}" y2="${f.t+f.ph}" stroke="#f04438" stroke-width="2.1" stroke-dasharray="4 3"/>`); parts.push(`</svg>`); const target=$(targetId); if(target){target.innerHTML=parts.join(""); bindTimelineSvgInteractions(target);}}
function drawVectorChart(targetId,title,xs,vectors,spans,unit){drawLineChart(targetId,title,xs,jointNames.map((name,j)=>({name,values:(vectors||[]).map(row=>Array.isArray(row)?row[j]:null)})),spans,unit)}

function cleanPolygon(poly){if(!Array.isArray(poly))return[]; const out=[]; for(const pt of poly){if(Array.isArray(pt)&&finite(Number(pt[0]))&&finite(Number(pt[1]))) out.push([Number(pt[0]),Number(pt[1])])} return out}
function polygonCenter(poly){const clean=cleanPolygon(poly); if(!clean.length)return null; let sx=0,sy=0; for(const p of clean){sx+=p[0];sy+=p[1]} return[sx/clean.length,sy/clean.length]}
function polygonEdgeAngle(poly){const clean=cleanPolygon(poly); if(clean.length<2)return 0; let best=0,bestD=-1; for(let i=0;i<clean.length;i++){const a=clean[i],b=clean[(i+1)%clean.length]; const dx=b[0]-a[0],dy=b[1]-a[1],d=dx*dx+dy*dy; if(d>bestD){bestD=d; best=Math.atan2(dy,dx)}} return best}
function normalizeAnglePi(angle){let a=Number(angle)||0; while(a<=-Math.PI/2)a+=Math.PI; while(a>Math.PI/2)a-=Math.PI; return a}
function transformToLocal(pt,origin,angle){const dx=Number(pt[0])-origin[0],dy=Number(pt[1])-origin[1]; const c=Math.cos(-angle),s=Math.sin(-angle); return[dx*c-dy*s,dx*s+dy*c]}
function localBounds(poly){let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity; for(const p of poly){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1])} return {minX,minY,maxX,maxY,w:maxX-minX,h:maxY-minY}}
function meshLocalFrame(scene){const poly=cleanPolygon((scene.unload_polygon_xy&&scene.unload_polygon_xy.length)?scene.unload_polygon_xy:scene.unload_hull_xy); const origin=(Array.isArray(scene.unload_xy)&&finite(Number(scene.unload_xy[0]))&&finite(Number(scene.unload_xy[1])))?[Number(scene.unload_xy[0]),Number(scene.unload_xy[1])]:(polygonCenter(poly)||null); if(!origin) return null; let angle=normalizeAnglePi(polygonEdgeAngle(poly)); let localPoly=poly.map(pt=>transformToLocal(pt,origin,angle)); if(localPoly.length){const b=localBounds(localPoly); if(b.h>b.w){angle=normalizeAnglePi(angle+Math.PI/2); localPoly=poly.map(pt=>transformToLocal(pt,origin,angle));}} return {origin,angle,localPoly,meshPath:scene.unload_mesh||"",shape:scene.unload_shape||""}}
function pointInPolygon(pt,poly){let inside=false; if(!poly||poly.length<3)return false; const x=pt[0],y=pt[1]; for(let i=0,j=poly.length-1;i<poly.length;j=i++){const xi=poly[i][0],yi=poly[i][1],xj=poly[j][0],yj=poly[j][1]; const intersect=((yi>y)!=(yj>y))&&(x<(xj-xi)*(y-yi)/(yj-yi+1e-12)+xi); if(intersect)inside=!inside} return inside}
function canonicalUnloadPoints(scenePoints){const out=[]; for(const p of scenePoints||[]){const frame=meshLocalFrame(p); if(!frame) continue; const source=p.unload_point_xy||p.unload_landing_xy||p.unload_xy; if(!source||!finite(Number(source[0]))||!finite(Number(source[1]))) continue; const local=transformToLocal(source,frame.origin,frame.angle); const inside=pointInPolygon(local,frame.localPoly); out.push({x:local[0],y:local[1],status:p.status,label:p.episode_index,polygon:frame.localPoly,inside,meshPath:frame.meshPath,angleDeg:frame.angle*180/Math.PI,sourceKind:p.unload_point_xy?"unload_point":(p.unload_landing_xy?"landing":"center")})} return out}

function equalAspectExtents(x0,x1,y0,y1,f){
  let cx=(x0+x1)/2, cy=(y0+y1)/2;
  let dx=Math.max(Math.abs(x1-x0),1e-9), dy=Math.max(Math.abs(y1-y0),1e-9);
  const unitsPerPixel=Math.max(dx/Math.max(f.pw,1), dy/Math.max(f.ph,1));
  const newDx=unitsPerPixel*f.pw, newDy=unitsPerPixel*f.ph;
  return {x0:cx-newDx/2, x1:cx+newDx/2, y0:cy-newDy/2, y1:cy+newDy/2, pxPerUnit:1/unitsPerPixel};
}

function drawScatter(targetId, pts, xLabel, yLabel, opts={}){
  const f=chartFrame(opts.width||640, opts.height||420);
  const rows=(pts||[]).map(p=>Object.assign({},p,{poly:cleanPolygon(p.polygon)}));
  const clean=rows.filter(p=>finite(Number(p.x))&&finite(Number(p.y)));
  const xs=[],ys=[];
  for(const p of rows){
    if(finite(Number(p.x))&&finite(Number(p.y))){xs.push(Number(p.x));ys.push(Number(p.y))}
    if((opts.polygons||opts.rangeFromPolygons||opts.meshLocal||opts.truckMeshOverlay)&&p.poly.length){for(const pt of p.poly){xs.push(pt[0]);ys.push(pt[1])}}
  }
  if(opts.origin){xs.push(0);ys.push(0)}
  if(!xs.length||!ys.length){$(targetId).innerHTML='<div class="empty">no data</div>';return}
  let [x0,x1]=extent(xs), [y0,y1]=extent(ys);
  let pxPerUnit=null;
  if(opts.equalAspect){
    const eq=equalAspectExtents(x0,x1,y0,y1,f);
    x0=eq.x0; x1=eq.x1; y0=eq.y0; y1=eq.y1; pxPerUnit=eq.pxPerUnit;
  }
  const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw, sy=y=>f.t+f.ph-(Number(y)-y0)/(y1-y0||1)*f.ph;
  const parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}">`];
  drawAxes(parts,f,x0,x1,y0,y1,{unit:opts.unit||""});
  if(opts.origin){
    parts.push(`<line x1="${sx(0).toFixed(1)}" y1="${f.t}" x2="${sx(0).toFixed(1)}" y2="${f.t+f.ph}" stroke="#101828" stroke-width="1.1" stroke-dasharray="4 4" opacity="0.38"/>`);
    parts.push(`<line x1="${f.l}" y1="${sy(0).toFixed(1)}" x2="${f.l+f.pw}" y2="${sy(0).toFixed(1)}" stroke="#101828" stroke-width="1.1" stroke-dasharray="4 4" opacity="0.38"/>`);
    parts.push(`<circle cx="${sx(0).toFixed(1)}" cy="${sy(0).toFixed(1)}" r="4.2" fill="#101828"><title>${opts.bodyFrame?"excavator body origin":"mesh/world origin"}</title></circle>`)
  }
  const maxPolygons=opts.maxPolygons ?? 80;
  if(opts.polygons || opts.meshLocal || opts.truckMeshOverlay){
    const polyRows=rows.filter(p=>p.poly.length);
    const drawRows=polyRows.length>maxPolygons?sampleEven(polyRows,maxPolygons):polyRows;
    for(const p of drawRows){
      const color=opts.meshLocal?"#101828":statusColor(p.status);
      const opacity=opts.meshLocal?0.18:0.50;
      const fill=opts.meshLocal?"#667085":color;
      const polyTitle=opts.truckMeshOverlay
        ? `ep ${esc(p.label)} selected truck dump-bed mesh transformed into excavator body frame`
        : `ep ${esc(p.label)} selected unload mesh, local angle ${fmt(p.angleDeg,1)} deg`;
      parts.push(`<polygon points="${p.poly.map(pt=>`${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`).join(" ")}" fill="${fill}" fill-opacity="${opts.meshLocal?0.035:0.04}" stroke="${color}" stroke-width="${opts.meshLocal?1.15:1.15}" stroke-opacity="${opacity}"><title>${polyTitle}</title></polygon>`)
    }
  }
  if(opts.heading){
    const span=Math.max(Math.abs(x1-x0),Math.abs(y1-y0)),len=Math.max(0.35,span*0.045);
    const headingRows=clean.length>260?sampleEven(clean,260):clean;
    for(const p of headingRows){
      if(!finite(Number(p.yaw)))continue;
      const a=Number(p.yaw)*Math.PI/180,x2=Number(p.x)+Math.cos(a)*len,y2=Number(p.y)+Math.sin(a)*len;
      parts.push(`<line x1="${sx(p.x).toFixed(1)}" y1="${sy(p.y).toFixed(1)}" x2="${sx(x2).toFixed(1)}" y2="${sy(y2).toFixed(1)}" stroke="${statusColor(p.status)}" stroke-width="1.65" opacity="0.55"><title>yaw ${fmt(p.yaw,1)} deg</title></line>`)
    }
  }
  const useDensity=opts.aggregate && clean.length>(opts.densityThreshold||120);
  let densityInfo="";
  if(useDensity){densityInfo=drawDensity(parts, clean, sx, sy, f, opts)}
  const pointRows=useDensity?sampleEven(clean,Math.min(260,clean.length)):clean;
  const r=useDensity?2.2:(clean.length>400?2.3:clean.length>150?3.0:4.2);
  const op=useDensity?0.45:(clean.length>300?0.48:0.86);
  for(const p of pointRows){
    const color=statusColor(p.status); const ring=(opts.meshLocal && p.inside===false);
    parts.push(`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="${ring?r+1.4:r}" fill="${ring?'#fff':color}" stroke="${color}" stroke-width="${ring?2.0:0.5}" opacity="${op}"><title>ep ${esc(p.label)} ${esc(statusKey(p.status))} (${fmt(p.x,2)}, ${fmt(p.y,2)})${opts.meshLocal?` · ${p.inside?'inside':'outside'} mesh · ${esc(p.sourceKind||'source')}`:""}</title></circle>`)
  }
  parts.push(`<text x="${f.l+f.pw/2}" y="${f.h-10}" text-anchor="middle" font-size="11" fill="#667085">${esc(xLabel)}</text>`);
  parts.push(`<text transform="translate(18 ${f.t+f.ph/2}) rotate(-90)" text-anchor="middle" font-size="11" fill="#667085">${esc(yLabel)}</text>`);
  parts.push(`</svg>`);
  const statuses=[...new Set(clean.map(p=>statusKey(p.status)))].sort((a,b)=>statusOrder(a)-statusOrder(b));
  const unitBadge=opts.unit?`<span class="legendItem unitLegend">unit: ${esc(opts.unit)}</span>`:"";
  const aspectBadge=opts.equalAspect?`<span class="legendItem unitLegend">XY scale: 1:1${pxPerUnit?` · ${fmt(pxPerUnit,1)} px/${esc(opts.unit||'unit')}`:""}</span>`:"";
  const legend=`<div class="chartLegend">${unitBadge}${aspectBadge}${statuses.map(s=>`<span class="legendItem"><span class="legendDot" style="background:${statusColor(s)}"></span>${esc(s)}</span>`).join("")}${useDensity?'<span class="densityBadge">density + sampled points</span>':''}</div>`;
  const densityNote=densityInfo?`<div class="plotNote">${densityInfo}</div>`:"";
  const truckNote=opts.truckMeshOverlay?`<div class="plotNote"><span class="meshFrameBadge">Excavator body frame</span> Coordinates use body_xy = R(-robot_body_yaw) · (world_xy - robot_origin_xy). Truck center, selected dump-bed mesh, and yaw arrows are transformed with the same frame; display scale is locked to 1:1.</div>`:"";
  const meshNote=opts.meshLocal?`<div class="plotNote"><span class="meshFrameBadge">Mesh-local frame</span> origin = selected unload mesh center when available; X-axis = longest consecutive mesh edge; each unload point is transformed with the same frame as its own mesh polygon. Hollow rings mean the projected point is outside that polygon.</div>`:"";
  $(targetId).innerHTML=parts.join("")+legend+densityNote+truckNote+meshNote;
}
function sampleEven(rows, maxN){if(rows.length<=maxN)return rows; const out=[]; const step=(rows.length-1)/(maxN-1); let last=-1; for(let i=0;i<maxN;i++){const idx=Math.round(i*step); if(idx!==last){out.push(rows[idx]); last=idx}} return out}
function drawDensity(parts, clean, sx, sy, f, opts){const cell=opts.cellSize||22; const bins=new Map(); for(const p of clean){const px=sx(p.x),py=sy(p.y); const ix=Math.floor((px-f.l)/cell),iy=Math.floor((py-f.t)/cell); const key=`${ix}:${iy}`; let b=bins.get(key); if(!b){b={ix,iy,count:0,statusCounts:new Map(),sx:0,sy:0}; bins.set(key,b)} b.count++; b.sx+=Number(p.x); b.sy+=Number(p.y); const st=statusKey(p.status); b.statusCounts.set(st,(b.statusCounts.get(st)||0)+1)} const max=Math.max(...[...bins.values()].map(b=>b.count),1); for(const b of bins.values()){let st="unknown",sc=-1; for(const [k,c] of b.statusCounts.entries()){if(c>sc){st=k;sc=c}} const x=f.l+b.ix*cell+1,y=f.t+b.iy*cell+1,w=cell-2,h=cell-2,alpha=0.12+0.5*Math.sqrt(b.count/max); parts.push(`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="${statusColor(st)}" opacity="${alpha.toFixed(2)}"><title>${b.count} attempts, majority ${esc(st)}</title></rect>`); if(b.count>=Math.max(5, max*0.45)){parts.push(`<text x="${(x+w/2).toFixed(1)}" y="${(y+h/2+4).toFixed(1)}" text-anchor="middle" font-size="10" font-weight="700" fill="#101828" opacity="0.72">${b.count}</text>`)}} return `Overlap handled with ${bins.size} density cells; ${Math.min(clean.length,260)} representative points are drawn above the density layer.`}
function drawHistogram(targetId, values, unit, opts={}){const nums=numeric(values); if(!nums.length){$(targetId).innerHTML='<div class="empty">no data</div>';return} let lo=Math.min(...nums),hi=Math.max(...nums); if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1} const bins=opts.bins||12,step=(hi-lo)/bins,counts=Array(bins).fill(0); for(const v of nums){let i=Math.floor((v-lo)/step); if(i>=bins)i=bins-1; if(i<0)i=0; counts[i]++} const f=chartFrame(opts.mini?360:640, opts.mini?195:340),max=Math.max(...counts,1),parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}">`]; drawAxes(parts,f,lo,hi,0,max,{}); counts.forEach((c,i)=>{const x=f.l+f.pw*i/bins+2,w=f.pw/bins-4,h=f.ph*c/max; parts.push(`<rect x="${x.toFixed(1)}" y="${(f.t+f.ph-h).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="#175cd3" opacity="0.78"><title>${c}</title></rect>`)}); parts.push(`</svg>`); const legend=unit?`<div class="chartLegend"><span class="legendItem unitLegend">unit: ${esc(unit)}</span></div>`:""; $(targetId).innerHTML=parts.join("")+legend}

function syncEpisodeInspectorHeight(){
  const inspector=document.querySelector(".episodeInspector");
  const grid=document.querySelector(".episodeInspector .timelineGrid");
  if(!inspector||!grid) return;
  requestAnimationFrame(()=>{
    const h=Math.round(grid.getBoundingClientRect().height || 0);
    if(h>0){
      inspector.style.setProperty("--episodeAsideHeight", Math.max(420,h)+"px");
    }
  });
}

document.addEventListener("click", evt=>{
  const target=evt.target;
  if(!target || !target.closest) return;
  const chooseRunBtn=target.closest('[data-action="choose-run"]');
  if(chooseRunBtn){
    evt.preventDefault();
    chooseRun(chooseRunBtn.dataset.runPath||"");
    return;
  }
  const sortBtn=target.closest('[data-action="sort-episodes"]');
  if(sortBtn){
    evt.preventDefault();
    sortEpisodes(sortBtn.dataset.sortKey||"episode_index");
    return;
  }
  const episodeRow=target.closest("#episodeTabs tr[data-ep]");
  if(episodeRow){
    evt.preventDefault();
    loadEpisode(episodeRow.dataset.ep).catch(e=>setStatus(e.message,"error"));
  }
});
document.addEventListener("change", evt=>{
  const target=evt.target;
  if(!target || !target.matches) return;
  if(target.matches('[data-action="toggle-all-runs"]')){
    toggleAllRuns(!!target.checked);
    return;
  }
  if(target.matches('[data-action="toggle-run-selection"]')){
    toggleRunSelection(target.dataset.runPath||"", !!target.checked);
  }
});

function bindStaticControl(id,event,handler){
  const el=$(id);
  if(el) el.addEventListener(event,handler);
}
bindStaticControl("loadRunsBtn","click",()=>loadRuns().catch(e=>setStatus(e.message,"error")));
bindStaticControl("loadSuccessPoolBtn","click",()=>loadSuccessPool().catch(e=>setStatus(e.message,"error")));
bindStaticControl("exportSuccessPoolBtn","click",()=>exportSuccessPool().catch(e=>setStatus(e.message,"error")));
bindStaticControl("loadRunBtn","click",()=>loadRun(false).catch(e=>setStatus(e.message,"error")));
bindStaticControl("reloadBtn","click",()=>loadRun(true).catch(e=>setStatus(e.message,"error")));
bindStaticControl("copyPathBtn","click",()=>navigator.clipboard&&navigator.clipboard.writeText($("runInput").value).then(()=>setStatus("Run path copied","ok")).catch(()=>setStatus("Copy failed","error")));
bindStaticControl("clearTerminalBtn","click",()=>{const box=$("terminalBox"); if(box) box.textContent=""; terminalWrite("terminal cleared","muted");});
bindStaticControl("copyRawAttemptBtn","click",()=>{
  const text=$("rawBox")?.textContent||"";
  if(!navigator.clipboard){setStatus("Clipboard unavailable","error");return;}
  navigator.clipboard.writeText(text).then(()=>setStatus("Raw attempt copied","ok")).catch(()=>setStatus("Copy failed","error"));
});
bindStaticControl("runSelect","change",()=>{$("runInput").value=$("runSelect").value});
bindStaticControl("selectAllRunsBtn","click",()=>toggleAllRuns(true));
bindStaticControl("selectZeroSuccessBtn","click",()=>selectRuns(r=>Number(r.success||0)===0));
bindStaticControl("selectSuccessRunsBtn","click",()=>selectRuns(r=>Number(r.success||0)>0));
bindStaticControl("clearRunSelectionBtn","click",()=>toggleAllRuns(false));
bindStaticControl("refreshSelectedSizesBtn","click",()=>refreshSelectedSizes().catch(e=>setStatus(e.message,"error")));
bindStaticControl("deleteSelectedRunsBtn","click",()=>deleteSelectedZeroSuccessRuns().catch(e=>setStatus(e.message,"error")));
bindStaticControl("copySuccessBtn","click",()=>transferSuccessRecords("copy", false).catch(e=>setStatus(e.message,"error")));
bindStaticControl("moveSuccessBtn","click",()=>transferSuccessRecords("move", false).catch(e=>setStatus(e.message,"error")));
bindStaticControl("cancelExportTimeBtn","click",()=>hideExportTimeModal());
bindStaticControl("resetTimePolicyBtn","click",()=>saveExportTimePolicyForPreview(true, true).catch(e=>setStatus(e.message,"error")));
bindStaticControl("applyTimePreviewBtn","click",()=>applyTimePreviewAndClose().catch(e=>setStatus(e.message,"error")));
bindStaticControl("exportTimeNowBtn","click",()=>exportSuccessPoolWithPolicy().catch(e=>setStatus(e.message,"error")));
["exportSpeedScaleInput"].forEach(id=>{const el=$(id); if(el) el.addEventListener("input",()=>updateExportTimePolicyPreview());});
bindStaticControl("exportTimeModal","click", e=>{const modal=$("exportTimeModal"); if(e.target===modal && !modal.classList.contains("loading")) hideExportTimeModal();});
bindStaticControl("darkModeToggle","click",()=>toggleDarkMode());
initDarkMode();
const episodeSortSelect=$("episodeSortSelect"); if(episodeSortSelect) episodeSortSelect.addEventListener("change",()=>refreshFilteredViews());
$("rootInput").addEventListener("input", ()=>syncSuccessPoolPath());
$("rootInput").addEventListener("keydown", e=>{if(e.key==="Enter") loadRuns().then(()=>loadRun()).catch(err=>setStatus(err.message,"error"))});
$("runInput").addEventListener("keydown", e=>{if(e.key==="Enter") loadRun().catch(err=>setStatus(err.message,"error"))});
window.addEventListener("resize",()=>syncEpisodeInspectorHeight());
loadRuns().then(()=>{
  loadRun();
  if(runMonitorTimer) clearInterval(runMonitorTimer);
  runMonitorTimer=setInterval(()=>loadRuns({silent:true}).catch(()=>{}), 30000);
}).catch(e=>setStatus(e.message,"error"));

</script>
</body>
</html>
"""



def serve_dashboard(
    dataset_root: Union[str, os.PathLike] = "excavator_auto_dataset",
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    import http.server
    import socketserver
    import webbrowser

    default_root = normalize_dashboard_client_path(dataset_root or "excavator_auto_dataset")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_bytes(self, data: bytes, content_type: str = "application/json", status: int = 200, cache_control: str = "no-store"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", cache_control)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, data: object, status: int = 200):
            self.send_bytes(json.dumps(data, ensure_ascii=True).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlparse(self.path)
            params = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                if parsed.path == "/":
                    self.send_bytes(dashboard_html().encode("utf-8"), "text/html; charset=utf-8")
                    return
                if parsed.path == "/api/manage/job":
                    job_id = params.get("job_id") or ""
                    self.send_json(dashboard_job_snapshot(job_id))
                    return
                if parsed.path == "/api/runs":
                    root = normalize_dashboard_client_path(params.get("root") or default_root)
                    runs = list_dashboard_runs(root)
                    self.send_json({"root": root, "runs": runs, "activity_summary": summarize_run_activity(runs)})
                    return
                if parsed.path == "/api/run":
                    run_dir = normalize_dashboard_client_path(params.get("run_dir") or latest_run(default_root))
                    if not run_dir or not os.path.isdir(run_dir):
                        self.send_json({"error": f"run_dir_not_found:{run_dir}"}, status=404)
                        return
                    force_refresh = str(params.get("force") or params.get("refresh") or "").lower() in {"1", "true", "yes", "y", "force"}
                    self.send_json(dashboard_run_payload(run_dir, force_refresh=force_refresh))
                    return
                if parsed.path == "/api/episode":
                    run_dir = normalize_dashboard_client_path(params.get("run_dir") or latest_run(default_root))
                    episode = params.get("episode_index") or "1"
                    max_points = int(params.get("max_points") or 1800)
                    self.send_json(dashboard_episode_payload(run_dir, episode, max_points=max_points))
                    return
                if parsed.path == "/api/frame":
                    run_dir = normalize_dashboard_client_path(params.get("run_dir") or latest_run(default_root))
                    episode = params.get("episode_index") or "1"
                    camera = params.get("camera") or "0"
                    frame_index = params.get("frame_index") or "0"
                    result = dashboard_episode_frame_image(run_dir, episode, camera, frame_index)
                    if not result.get("ok"):
                        self.send_json({"error": result.get("error", "frame_not_found")}, status=int(result.get("status", 404) or 404))
                        return
                    self.send_bytes(
                        result.get("data", b""),
                        str(result.get("content_type") or "application/octet-stream"),
                        cache_control="public, max-age=3600, immutable",
                    )
                    return
                self.send_json({"error": "not_found"}, status=404)
            except Exception as exc:
                self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)


        def read_json_body(self):
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except Exception:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                return json.loads(raw.decode("utf-8")) if raw else {}
            except Exception as exc:
                raise ValueError(f"invalid_json_body:{exc}")

        def do_POST(self):
            parsed = urlparse(self.path)
            try:
                body = self.read_json_body()
                if parsed.path == "/api/manage/refresh_sizes":
                    root = normalize_dashboard_client_path(body.get("root") or default_root)
                    result = dashboard_refresh_folder_sizes(root, body.get("paths") or [])
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/delete_runs":
                    root = normalize_dashboard_client_path(body.get("root") or default_root)
                    result = dashboard_delete_runs(
                        root,
                        body.get("paths") or [],
                        zero_success_only=bool(body.get("zero_success_only", True)),
                        allow_active=bool(body.get("allow_active", False)),
                    )
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/success_records":
                    root = normalize_dashboard_client_path(body.get("root") or default_root)
                    result = dashboard_start_success_transfer_job(
                        root,
                        body.get("paths") or [],
                        body.get("dest_dir") or dashboard_success_pool_dir(root),
                        mode=str(body.get("mode") or "copy"),
                    )
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/export_success_vla":
                    root = normalize_dashboard_client_path(body.get("root") or default_root)
                    time_policy = body.get("time_policy")
                    result = dashboard_start_success_pool_export_job(
                        root,
                        overwrite=bool(body.get("overwrite", True)),
                        require_vla=bool(body.get("require_vla", True)),
                        time_policy=time_policy,
                    )
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/export_time_policy":
                    root = normalize_dashboard_client_path(body.get("root") or default_root)
                    pool_dir = dashboard_success_pool_dir(root)
                    if not os.path.isdir(pool_dir):
                        self.send_json({"error": f"success_pool_not_found:{pool_dir}"}, status=404)
                        return
                    policy = default_export_time_policy() if body.get("reset") else body.get("time_policy")
                    result = save_dashboard_export_time_policy(pool_dir, policy)
                    result["effective_fps"] = dashboard_effective_export_fps(pool_dir, result.get("policy"))
                    result["pool_dir"] = pool_dir
                    self.send_json(result)
                    return
                self.send_json({"error": "not_found"}, status=404)
            except Exception as exc:
                self.send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadingServer((host, int(port)), Handler)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"[DASHBOARD] {url}")
    print(f"[DASHBOARD] dataset_root={default_root}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[DASHBOARD] stopped")
    finally:
        server.server_close()


def print_summary(run_dir: Union[str, os.PathLike]) -> None:
    summary = summarize_run(run_dir)
    print("[INFO]", json.dumps(summary, ensure_ascii=True, indent=2))


def print_analysis(run_dir: Union[str, os.PathLike], include_timeline: bool = True, compact: bool = False) -> None:
    report = analyze_run(run_dir, include_timeline=include_timeline)
    if compact:
        report = compact_analysis(report)
    print("[ANALYSIS]", json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect excavator auto dataset runs.")
    parser.add_argument("run_dir", nargs="?", help="Path to a run_* directory.")
    parser.add_argument("--root", default="excavator_auto_dataset", help="Dataset root used with --latest.")
    parser.add_argument("--latest", action="store_true", help="Analyze the newest run under --root.")
    parser.add_argument("--analysis", action="store_true", help="Print detailed failure and quality analysis.")
    parser.add_argument("--compact", action="store_true", help="Print compact analysis fields only.")
    parser.add_argument("--no-timeline", action="store_true", help="Skip debug_timeline.jsonl analysis.")
    parser.add_argument("--plots", action="store_true", help="Write SVG analysis plots for the selected run.")
    parser.add_argument("--plot-dir", default=None, help="Output directory for --plots. Defaults to run_dir/analysis_plots.")
    parser.add_argument("--export-lerobot", action="store_true", help="Create run_dir/lerobot_v3 with clean LeRobot v3 VLA training data.")
    parser.add_argument("--export-lerobot-v3", action="store_true", help="Alias for --export-lerobot.")
    parser.add_argument("--export-dir", default=None, help="Output directory for --export-lerobot. Defaults to run_dir/lerobot_v3.")
    parser.add_argument("--export-split", default="trainable", help="Episode index split to export, default: trainable.")
    parser.add_argument("--export-limit", type=int, default=None, help="Limit exported episodes for smoke tests.")
    parser.add_argument("--export-overwrite", action="store_true", help="Delete and rebuild the export directory if it already exists.")
    parser.add_argument("--export-require-standard", action="store_true", help="Fail if parquet/mp4 standard export cannot be produced.")
    parser.add_argument("--export-require-vla", action="store_true", help="Fail unless image + state + action + task VLA export is ready.")
    parser.add_argument("--dashboard", action="store_true", help="Serve a local web dashboard for browsing run folders.")
    parser.add_argument("--dashboard-host", "--host", default="127.0.0.1", help="Host for --dashboard.")
    parser.add_argument("--dashboard-port", "--port", type=int, default=8765, help="Port for --dashboard.")
    parser.add_argument("--dashboard-open", action="store_true", help="Open the dashboard URL in the default browser.")
    args = parser.parse_args()
    if args.dashboard:
        serve_dashboard(
            dataset_root=args.root,
            host=args.dashboard_host,
            port=args.dashboard_port,
            open_browser=args.dashboard_open,
        )
        raise SystemExit(0)
    run_dir = latest_run(args.root) if args.latest else (args.run_dir or "")
    if not run_dir:
        raise SystemExit("run_dir is required unless --latest finds a run.")
    did_action = False
    if args.plots:
        print_plots(run_dir, output_dir=args.plot_dir)
        did_action = True
    if args.export_lerobot or args.export_lerobot_v3:
        print_lerobot_export(
            run_dir,
            output_dir=args.export_dir,
            split=args.export_split,
            limit_episodes=args.export_limit,
            overwrite=args.export_overwrite,
            require_standard=args.export_require_standard,
            require_vla=args.export_require_vla,
        )
        did_action = True
    if args.analysis:
        print_analysis(run_dir, include_timeline=not args.no_timeline, compact=args.compact)
        did_action = True
    if not did_action:
        print_summary(run_dir)
