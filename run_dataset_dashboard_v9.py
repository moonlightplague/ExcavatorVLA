import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qs, unquote, urlparse


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
LEROBOT_IMAGE_SHAPE = [256, 256, 3]
LEROBOT_IMAGE_KEYS = [
    "observation.images.0",
    "observation.images.1",
    "observation.images.2",
]
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
        path = row_or_path.get("trajectory", "")
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
    counts = {name: len(load_index(run_dir, name)) for name in INDEX_FILES}
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


def sample_image_value(sample: dict, canonical_key: str) -> object:
    for key in LEROBOT_IMAGE_KEY_ALIASES.get(canonical_key, [canonical_key]):
        value = sample.get(key)
        if value:
            return value
    return None


def episode_dir_from_row(row: dict) -> str:
    for key in ["trajectory", "meta", "score_path", "events"]:
        path = row_path_value(row, key)
        if path:
            return os.path.dirname(path)
    return ""


def resolve_episode_file(episode_dir: str, value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if os.path.isabs(text):
        return text
    return os.path.normpath(os.path.join(episode_dir, text))


def relpath_posix(path: Union[str, os.PathLike], base: Union[str, os.PathLike]) -> str:
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def safe_copy_file(src: str, dst: str) -> bool:
    if not src or not os.path.isfile(src):
        return False
    ensure_dir(os.path.dirname(dst) or ".")
    shutil.copy2(src, dst)
    return True


def infer_export_fps(run_dir: Union[str, os.PathLike], explicit_fps: Optional[float] = None) -> float:
    if explicit_fps is not None and float(explicit_fps) > 0:
        return float(explicit_fps)
    camera_config = read_json(os.path.join(str(run_dir), "camera_config.json"), default={}) or {}
    try:
        fps = float(camera_config.get("frequency", 10) or 10)
        if fps > 0:
            return fps
    except Exception:
        pass
    return 10.0


def lerobot_task_text(sample: dict, episode_meta: dict) -> str:
    for value in [
        sample.get("task"),
        episode_meta.get("task"),
        episode_meta.get("dataset_task_text"),
    ]:
        text = str(value or "").strip()
        if text:
            return text
    return "Dig soil from the marked area and dump it into the target container."


def try_write_parquet(rows: Sequence[dict], path: str) -> Tuple[bool, str]:
    if not rows:
        return False, "no_rows"
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        return False, f"pyarrow_unavailable:{type(exc).__name__}:{exc}"
    try:
        ensure_dir(os.path.dirname(path) or ".")
        table = pa.Table.from_pylist(list(rows))
        pq.write_table(table, path)
        return True, "ok"
    except Exception as exc:
        return False, f"parquet_write_failed:{type(exc).__name__}:{exc}"


def try_write_dataframe_parquet(df, path: str, index: bool = False) -> Tuple[bool, str]:
    try:
        ensure_dir(os.path.dirname(path) or ".")
        df.to_parquet(path, index=index)
        return True, "ok"
    except Exception as exc:
        return False, f"dataframe_parquet_write_failed:{type(exc).__name__}:{exc}"


def resize_rgb_frame(frame, target_size: Optional[Tuple[int, int]] = None):
    if target_size is None:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    target_width, target_height = int(target_size[0]), int(target_size[1])
    channels = int(frame.shape[2]) if len(frame.shape) >= 3 else 1
    if width == target_width and height == target_height and channels == 3:
        return frame
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pillow_unavailable_for_resize:{type(exc).__name__}:{exc}") from exc
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
    image = image.resize((target_width, target_height), resample)
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"numpy_unavailable_for_resize:{type(exc).__name__}:{exc}") from exc
    return np.asarray(image)


def try_encode_mp4_imageio(
    image_paths: Sequence[str],
    output_path: str,
    fps: float,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        return False, f"imageio_unavailable:{type(exc).__name__}:{exc}"
    try:
        ensure_dir(os.path.dirname(output_path) or ".")
        writer = imageio.get_writer(output_path, fps=float(fps), codec="libx264", quality=8, macro_block_size=1)
        try:
            for image_path in image_paths:
                writer.append_data(resize_rgb_frame(imageio.imread(image_path), target_size=target_size))
        finally:
            writer.close()
        return True, "ok"
    except Exception as exc:
        return False, f"imageio_mp4_failed:{type(exc).__name__}:{exc}"


def try_encode_mp4_cv2(
    image_paths: Sequence[str],
    output_path: str,
    fps: float,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return False, f"cv2_unavailable:{type(exc).__name__}:{exc}"
    try:
        first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            return False, "cv2_first_frame_unreadable"
        height, width = int(first.shape[0]), int(first.shape[1])
        if target_size is not None:
            width, height = int(target_size[0]), int(target_size[1])
        ensure_dir(os.path.dirname(output_path) or ".")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            return False, "cv2_writer_not_opened"
        try:
            for image_path in image_paths:
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    return False, f"cv2_frame_unreadable:{image_path}"
                if int(frame.shape[1]) != width or int(frame.shape[0]) != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
        finally:
            writer.release()
        return True, "ok"
    except Exception as exc:
        return False, f"cv2_mp4_failed:{type(exc).__name__}:{exc}"


def encode_mp4(
    image_paths: Sequence[str],
    output_path: str,
    fps: float,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str, str]:
    if not image_paths:
        return False, "none", "no_images"
    missing = [path for path in image_paths if not path or not os.path.isfile(path)]
    if missing:
        return False, "none", f"missing_images:{len(missing)}"
    ok, reason = try_encode_mp4_imageio(image_paths, output_path, fps, target_size=target_size)
    if ok:
        return True, "imageio", reason
    first_reason = reason
    ok, reason = try_encode_mp4_cv2(image_paths, output_path, fps, target_size=target_size)
    if ok:
        return True, "cv2", reason
    return False, "none", f"{first_reason}; {reason}"


def copy_image_fallback(
    image_paths: Sequence[str],
    export_dir: str,
    image_key: str,
    rows: List[dict],
) -> Tuple[bool, str, List[str]]:
    if not image_paths:
        return False, "no_images", []
    missing = [path for path in image_paths if not path or not os.path.isfile(path)]
    if missing:
        return False, f"missing_images:{len(missing)}", []
    target_dir = os.path.join(export_dir, "images", "chunk-000", image_key, "file-000")
    copied = []
    for frame_index, src in enumerate(image_paths):
        ext = os.path.splitext(src)[1].lower() or ".png"
        dst = os.path.join(target_dir, f"{frame_index:06d}{ext}")
        if not safe_copy_file(src, dst):
            return False, f"copy_failed:{src}", copied
        rel = relpath_posix(dst, export_dir)
        rows[frame_index][image_key] = rel
        copied.append(rel)
    return True, "ok", copied


def merge_image_storage(current: str, new_value: str) -> str:
    current = str(current or "none")
    new_value = str(new_value or "none")
    if current == "none":
        return new_value
    if current == new_value:
        return current
    return "mixed"


def vector_stats_for_rows(rows: Sequence[dict], key: str, dim: int) -> Dict[str, object]:
    vectors = []
    for row in rows:
        vec = vector_or_none(row.get(key), dim)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return {"count": [0]}
    count = len(vectors)
    mins = [float("inf")] * dim
    maxs = [float("-inf")] * dim
    sums = [0.0] * dim
    sums_sq = [0.0] * dim
    for vec in vectors:
        for index, value in enumerate(vec):
            mins[index] = min(mins[index], value)
            maxs[index] = max(maxs[index], value)
            sums[index] += value
            sums_sq[index] += value * value
    means = [value / count for value in sums]
    stds = []
    for index in range(dim):
        variance = max(0.0, (sums_sq[index] / count) - (means[index] * means[index]))
        std = math.sqrt(variance)
        stds.append(1.0 if std < 1.0e-6 else std)
    return {
        "count": [count],
        "mean": means,
        "std": stds,
        "min": mins,
        "max": maxs,
    }


def scalar_stats_for_rows(rows: Sequence[dict], key: str) -> Dict[str, object]:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(float(value))
        except Exception:
            continue
    if not values:
        return {"count": [0]}
    count = len(values)
    minv = min(values)
    maxv = max(values)
    meanv = sum(values) / max(1, count)
    mean_sq = sum(value * value for value in values) / max(1, count)
    std = math.sqrt(max(0.0, mean_sq - meanv * meanv))
    if std < 1.0e-6:
        std = 1.0
    return {
        "count": [count],
        "mean": [meanv],
        "std": [std],
        "min": [minv],
        "max": [maxv],
    }


def visual_identity_stats() -> Dict[str, object]:
    return {
        "count": [0],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 1.0],
    }


def build_lerobot_v3_stats(
    rows: Sequence[dict],
    state_dim: int,
    action_dim: int,
    effort_dim: Optional[int] = None,
    image_keys: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    stats = {
        "observation.state": vector_stats_for_rows(rows, "observation.state", state_dim),
        "action": vector_stats_for_rows(rows, "action", action_dim),
    }
    if effort_dim is not None and effort_dim > 0:
        stats["observation.effort"] = vector_stats_for_rows(rows, "observation.effort", effort_dim)
    for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        stats[key] = scalar_stats_for_rows(rows, key)
    for key in (list(image_keys) if image_keys is not None else LEROBOT_IMAGE_KEYS):
        stats[str(key)] = visual_identity_stats()
    return stats


def lerobot_v3_required_paths(export_dir: str, image_keys: Sequence[str]) -> List[str]:
    required = [
        os.path.join(export_dir, "meta", "info.json"),
        os.path.join(export_dir, "meta", "tasks.parquet"),
        os.path.join(export_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
        os.path.join(export_dir, "data", "chunk-000", "file-000.parquet"),
    ]
    for key in image_keys:
        required.append(os.path.join(export_dir, "videos", key, "chunk-000", "file-000.mp4"))
    return required


def validate_lerobot_v3_export(export_dir: str, image_keys: Sequence[str]) -> Dict[str, object]:
    required = lerobot_v3_required_paths(export_dir, image_keys)
    missing = [relpath_posix(path, export_dir) for path in required if not os.path.exists(path)]
    info = read_json(os.path.join(export_dir, "meta", "info.json"), default={}) or {}
    reasons = []
    if missing:
        reasons.append(f"missing:{','.join(missing)}")
    if info.get("codebase_version") != LEROBOT_CODEBASE_VERSION:
        reasons.append("info/codebase_version_not_v3")
    if info.get("video_path") != "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4":
        reasons.append("info/video_path_not_v3")
    if info.get("data_path") != "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet":
        reasons.append("info/data_path_not_v3")
    features = info.get("features", {}) if isinstance(info.get("features"), dict) else {}
    stats = read_json(os.path.join(export_dir, "meta", "stats.json"), default={}) or {}
    if not isinstance(stats, dict):
        reasons.append("stats/not_dict")
        stats = {}
    if "task" in features:
        reasons.append("info/features_contains_task")
    required_stats = [
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    if "observation.effort" in features:
        required_stats.append("observation.effort")
    for key in image_keys:
        ft = features.get(key, {}) if isinstance(features.get(key), dict) else {}
        if ft.get("dtype") != "video":
            reasons.append(f"{key}/dtype_not_video")
        if list(ft.get("shape", [])) != LEROBOT_IMAGE_SHAPE:
            reasons.append(f"{key}/shape_not_256")
        if "storage" in ft:
            reasons.append(f"{key}/storage_should_be_absent")
        required_stats.append(key)
    for key in required_stats:
        row = stats.get(key)
        if not isinstance(row, dict):
            reasons.append(f"stats/missing_{key}")
            continue
        for stat_name in ["mean", "std", "min", "max"]:
            if stat_name not in row:
                reasons.append(f"stats/{key}_missing_{stat_name}")
    episodes_path = os.path.join(export_dir, "meta", "episodes", "chunk-000", "file-000.parquet")
    if os.path.exists(episodes_path):
        try:
            import pandas as pd  # type: ignore

            episodes_df = pd.read_parquet(episodes_path)
            for key in image_keys:
                for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
                    column = f"videos/{key}/{suffix}"
                    if column not in episodes_df.columns:
                        reasons.append(f"episodes/missing_{column}")
        except Exception as exc:
            reasons.append(f"episodes/read_failed:{type(exc).__name__}:{exc}")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "missing": missing,
    }


def collect_lerobot_rows(
    run_dir: Union[str, os.PathLike],
    split: str = "trainable",
    limit_episodes: Optional[int] = None,
) -> Dict[str, object]:
    run_dir = str(run_dir)
    episode_rows = load_index(run_dir, split)
    if limit_episodes is not None:
        episode_rows = episode_rows[: max(0, int(limit_episodes))]
    run_meta = read_json(os.path.join(run_dir, "run_meta.json"), default={}) or {}
    state_names = run_meta.get("state_names") or [
        "base_x",
        "base_y",
        "base_yaw",
        "swing",
        "boom",
        "arm",
        "bucket",
        "bucket_load_estimate",
        "bucket_tip_x",
        "bucket_tip_y",
        "bucket_tip_z",
        "bucket_load_x",
        "bucket_load_y",
        "bucket_load_z",
    ]
    action_names = run_meta.get("action_names") or [
        "swing_cmd_velocity",
        "boom_cmd_velocity",
        "arm_cmd_velocity",
        "bucket_cmd_velocity",
    ]
    effort_names = run_meta.get("effort_names") or [
        "swing_measured_effort",
        "boom_measured_effort",
        "arm_measured_effort",
        "bucket_measured_effort",
    ]
    rows = []
    episodes = []
    episode_stats = []
    tasks_by_text: Dict[str, int] = {}
    image_paths = {key: [] for key in LEROBOT_IMAGE_KEYS}
    skipped_frames = 0
    skipped_state_action_frames = 0
    skipped_missing_camera_frames = 0
    missing_camera_by_key: Counter = Counter()
    missing_camera_examples: List[dict] = []
    global_frame = 0
    for source_episode_index, episode in enumerate(episode_rows):
        trajectory = load_trajectory(episode)
        episode_dir = episode_dir_from_row(episode)
        meta = read_json(row_path_value(episode, "meta"), default={}) or {}
        if not trajectory:
            continue
        export_episode_index = len(episodes)
        first_t = safe_float_value(trajectory[0].get("t"), 0.0) or 0.0
        episode_start_frame = global_frame
        episode_length = 0
        task_index = 0
        task_text = ""
        score = safe_float_value(episode.get("score"), None)
        for sample in trajectory:
            state = vector_or_none(sample.get("observation.state"), len(state_names))
            if state is None:
                state = vector_or_none(sample.get("obs.state"), len(state_names))
            action = vector_or_none(sample.get("action"), len(action_names))
            if state is None or action is None:
                skipped_frames += 1
                skipped_state_action_frames += 1
                continue
            resolved_images: Dict[str, Tuple[object, str]] = {}
            missing_image_keys: List[str] = []
            for key in LEROBOT_IMAGE_KEYS:
                image_value = sample_image_value(sample, key)
                abs_image = resolve_episode_file(episode_dir, image_value)
                if abs_image and os.path.isfile(abs_image):
                    resolved_images[key] = (image_value, abs_image)
                else:
                    missing_image_keys.append(key)
                    missing_camera_by_key[key] += 1
            if missing_image_keys:
                skipped_frames += 1
                skipped_missing_camera_frames += 1
                if len(missing_camera_examples) < 20:
                    missing_camera_examples.append(
                        {
                            "source_episode_index": int(source_episode_index),
                            "raw_episode_index": episode.get("episode_index"),
                            "raw_episode_id": episode.get("episode_id", ""),
                            "raw_sample_index": sample.get("i"),
                            "phase": str(sample.get("phase", "")),
                            "missing": list(missing_image_keys),
                        }
                    )
                continue
            effort = vector_or_none(sample.get("observation.effort"), len(effort_names))
            task_text = lerobot_task_text(sample, meta)
            if task_text not in tasks_by_text:
                tasks_by_text[task_text] = len(tasks_by_text)
            task_index = tasks_by_text[task_text]
            sample_t = safe_float_value(sample.get("t"), first_t) or first_t
            row = {
                "index": global_frame,
                "episode_index": export_episode_index,
                "frame_index": episode_length,
                "timestamp": float(sample_t - first_t),
                "task_index": int(task_index),
                "task": task_text,
                "observation.state": state,
                "action": action,
                "observation.effort": effort,
                "phase": str(sample.get("phase", "")),
                "raw_episode_index": episode.get("episode_index"),
                "raw_episode_id": episode.get("episode_id", sample.get("id", "")),
                "raw_sample_index": sample.get("i"),
            }
            for key in LEROBOT_IMAGE_KEYS:
                image_value, abs_image = resolved_images[key]
                image_paths[key].append(abs_image)
                row[key] = image_value
                row[f"{key}.available"] = True
            rows.append(row)
            episode_length += 1
            global_frame += 1
        if episode_length <= 0:
            continue
        episodes.append(
            {
                "episode_index": export_episode_index,
                "tasks": [int(task_index)],
                "length": int(episode_length),
                "raw_episode_index": episode.get("episode_index"),
                "raw_episode_id": episode.get("episode_id", ""),
                "status": episode.get("status", ""),
                "score": score,
                "from_frame": int(episode_start_frame),
                "to_frame": int(episode_start_frame + episode_length),
            }
        )
        episode_stats.append(
            {
                "episode_index": export_episode_index,
                "length": int(episode_length),
                "score": score,
                "max_bucket_from_pile_particles": episode.get("max_bucket_from_pile_particles"),
                "lift_bucket_from_pile_particles": episode.get("lift_bucket_from_pile_particles"),
                "final_bin_from_pile_particles": episode.get("final_bin_from_pile_particles"),
                "final_spill_from_pile_particles": episode.get("final_spill_from_pile_particles"),
                "freeze_count": episode.get("freeze_count"),
            }
        )
    tasks = [{"task_index": index, "task": text} for text, index in sorted(tasks_by_text.items(), key=lambda item: item[1])]
    return {
        "rows": rows,
        "episodes": episodes,
        "episode_stats": episode_stats,
        "tasks": tasks,
        "image_paths": image_paths,
        "state_names": state_names,
        "action_names": action_names,
        "effort_names": effort_names,
        "run_meta": run_meta,
        "skipped_frames": skipped_frames,
        "skipped_state_action_frames": skipped_state_action_frames,
        "skipped_missing_camera_frames": skipped_missing_camera_frames,
        "missing_camera_by_key": dict(missing_camera_by_key),
        "missing_camera_examples": missing_camera_examples,
        "source_episode_count": len(episode_rows),
    }


def export_lerobot_dataset(
    run_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    split: str = "trainable",
    fps: Optional[float] = None,
    limit_episodes: Optional[int] = None,
    overwrite: bool = False,
    require_standard: bool = False,
    require_vla: bool = False,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    export_dir = os.path.abspath(str(output_dir or os.path.join(run_dir, LEROBOT_DEFAULT_EXPORT_DIRNAME)))
    if os.path.exists(export_dir):
        if not overwrite:
            raise FileExistsError(f"{export_dir} already exists; pass --export-overwrite to rebuild it")
        if os.path.normcase(export_dir) == os.path.normcase(run_dir):
            raise ValueError("refusing to overwrite run_dir as export_dir")
        shutil.rmtree(export_dir)
    ensure_dir(export_dir)

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"pandas is required for strict LeRobot v3 export: {type(exc).__name__}:{exc}") from exc

    collected = collect_lerobot_rows(run_dir, split=split, limit_episodes=limit_episodes)
    rows: List[dict] = list(collected["rows"])  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"no exportable frames found for split={split}")

    meta_dir = ensure_dir(os.path.join(export_dir, "meta"))
    data_dir = ensure_dir(os.path.join(export_dir, "data", "chunk-000"))
    episodes_dir = ensure_dir(os.path.join(meta_dir, "episodes", "chunk-000"))
    ensure_dir(os.path.join(export_dir, "videos"))

    export_fps = int(round(infer_export_fps(run_dir, fps)))
    if export_fps <= 0:
        export_fps = 10
    video_results = {}
    image_features = list(LEROBOT_IMAGE_KEYS)
    image_paths: Dict[str, List[str]] = collected["image_paths"]  # type: ignore[assignment]
    for key in LEROBOT_IMAGE_KEYS:
        paths = image_paths.get(key, [])
        if not paths or not any(paths):
            video_results[key] = {"available": False, "reason": "no_images"}
            continue
        video_path = os.path.join(export_dir, "videos", key, "chunk-000", "file-000.mp4")
        ok, encoder, reason = encode_mp4(
            paths,
            video_path,
            export_fps,
            target_size=(LEROBOT_IMAGE_SHAPE[1], LEROBOT_IMAGE_SHAPE[0]),
        )
        if ok:
            video_results[key] = {
                "available": True,
                "encoder": encoder,
                "path": relpath_posix(video_path, export_dir),
                "frames": len(paths),
                "shape": LEROBOT_IMAGE_SHAPE,
            }
            continue
        video_results[key] = {
            "available": False,
            "reason": reason,
        }

    tasks = collected["tasks"]
    episodes = collected["episodes"]
    state_names = list(collected["state_names"])  # type: ignore[arg-type]
    action_names = list(collected["action_names"])  # type: ignore[arg-type]
    effort_names = list(collected.get("effort_names", []))  # type: ignore[arg-type]
    effort_dim = len(effort_names)
    effort_available = bool(
        effort_dim > 0
        and rows
        and all(vector_or_none(row.get("observation.effort"), effort_dim) is not None for row in rows)
    )
    task_text_by_index = {int(task["task_index"]): str(task["task"]) for task in tasks}  # type: ignore[index]

    data_rows = []
    for row in rows:
        frame_index = int(row["frame_index"])
        data_row = {
            "index": int(row["index"]),
            "episode_index": int(row["episode_index"]),
            "frame_index": frame_index,
            "timestamp": float(frame_index) / float(export_fps),
            "task_index": int(row["task_index"]),
            "observation.state": row["observation.state"],
            "action": row["action"],
        }
        if effort_available:
            data_row["observation.effort"] = row["observation.effort"]
        data_rows.append(data_row)
    parquet_path = os.path.join(data_dir, "file-000.parquet")
    data_df = pd.DataFrame(data_rows)
    parquet_ok, parquet_reason = try_write_dataframe_parquet(data_df, parquet_path, index=False)

    tasks_path = os.path.join(meta_dir, "tasks.parquet")
    tasks_df = pd.DataFrame(
        {"task_index": [int(task["task_index"]) for task in tasks]},  # type: ignore[index]
        index=pd.Index([str(task["task"]) for task in tasks]),  # type: ignore[index]
    )
    tasks_ok, tasks_reason = try_write_dataframe_parquet(tasks_df, tasks_path, index=True)

    episode_meta_rows = []
    for episode in episodes:  # type: ignore[assignment]
        task_indices = [int(value) for value in episode.get("tasks", [])]
        episode_task_texts = [task_text_by_index.get(index, "") for index in task_indices]
        start = int(episode.get("from_frame", 0))
        end = int(episode.get("to_frame", start + int(episode.get("length", 0))))
        meta_row = {
            "episode_index": int(episode.get("episode_index", 0)),
            "tasks": episode_task_texts,
            "length": int(episode.get("length", 0)),
            "dataset_from_index": start,
            "dataset_to_index": end,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
        }
        for key in LEROBOT_IMAGE_KEYS:
            meta_row[f"videos/{key}/chunk_index"] = 0
            meta_row[f"videos/{key}/file_index"] = 0
            meta_row[f"videos/{key}/from_timestamp"] = float(start) / float(export_fps)
            meta_row[f"videos/{key}/to_timestamp"] = float(end) / float(export_fps)
        episode_meta_rows.append(meta_row)
    episodes_path = os.path.join(episodes_dir, "file-000.parquet")
    episodes_df = pd.DataFrame(episode_meta_rows)
    episodes_ok, episodes_reason = try_write_dataframe_parquet(episodes_df, episodes_path, index=False)

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [len(state_names)],
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": [len(action_names)],
            "names": action_names,
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if effort_available:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": [effort_dim],
            "names": effort_names,
        }
    for key in image_features:
        features[key] = {
            "dtype": "video",
            "shape": list(LEROBOT_IMAGE_SHAPE),
            "names": ["height", "width", "channels"],
        }

    info = {
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "robot_type": "excavator",
        "total_episodes": len(episodes),
        "total_frames": len(data_rows),
        "total_tasks": len(tasks),
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": export_fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    write_json(os.path.join(meta_dir, "info.json"), info)

    stats = build_lerobot_v3_stats(
        data_rows,
        len(state_names),
        len(action_names),
        effort_dim if effort_available else None,
        image_features,
    )
    write_json(os.path.join(meta_dir, "stats.json"), stats)

    video_ready = all(video_results.get(key, {}).get("available") is True for key in image_features)
    parquet_ready = bool(parquet_ok and tasks_ok and episodes_ok)
    validation = validate_lerobot_v3_export(export_dir, image_features)
    vla_training_ready = bool(parquet_ready and video_ready and validation["ok"])
    manifest = {
        "schema": LEROBOT_EXPORT_SCHEMA,
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "export_dir": export_dir,
        "source_run_dir": run_dir,
        "source_split": split,
        "created_at": time.time(),
        "standard_lerobot_ready": vla_training_ready,
        "state_action_ready": bool(parquet_ok),
        "effort_available": bool(effort_available),
        "vla_training_ready": vla_training_ready,
        "parquet": {
            "data_available": parquet_ok,
            "tasks_available": tasks_ok,
            "episodes_available": episodes_ok,
            "path": relpath_posix(parquet_path, export_dir) if parquet_ok else None,
            "data_reason": parquet_reason,
            "tasks_reason": tasks_reason,
            "episodes_reason": episodes_reason,
        },
        "videos": video_results,
        "validation": validation,
        "fps": export_fps,
        "total_frames": len(data_rows),
        "total_episodes": len(episodes),
        "total_tasks": len(tasks),
        "skipped_frames": collected["skipped_frames"],
        "skipped_state_action_frames": collected.get("skipped_state_action_frames", 0),
        "skipped_missing_camera_frames": collected.get("skipped_missing_camera_frames", 0),
        "missing_camera_by_key": collected.get("missing_camera_by_key", {}),
        "missing_camera_examples": collected.get("missing_camera_examples", []),
        "notes": [
            "Original auto-collection debug data remains outside this subfolder.",
            "This folder follows the LeRobot v3.0 offline layout for VLA/SmolVLA training.",
            "Camera streams are observation.images.0, observation.images.1, observation.images.2.",
            "Rows missing any camera frame are skipped during export so state/action/video stay aligned.",
            "If vla_training_ready is false, install pandas/pyarrow plus a video encoder, then rerun the exporter.",
        ],
    }
    write_json(os.path.join(export_dir, "manifest.json"), manifest)
    readme = [
        "# Excavator LeRobot v3 Export",
        "",
        f"Source run: `{run_dir}`",
        f"Split: `{split}`",
        f"Frames: `{len(data_rows)}`",
        f"Episodes: `{len(episodes)}`",
        f"FPS: `{export_fps}`",
        f"LeRobot v3 / VLA ready: `{vla_training_ready}`",
        "",
        "This subfolder is generated from the raw auto-collection run and keeps trainable data separate from debug logs.",
        "",
        "Files:",
        "- `meta/info.json`: feature schema and dataset totals",
        "- `meta/tasks.parquet`: task text index to task_index mapping",
        "- `meta/episodes/chunk-000/file-000.parquet`: episode metadata and video/data chunk indices",
        "- `meta/stats.json`: state/action/effort/scalar statistics plus video normalization entries",
        "- `data/chunk-000/file-000.parquet`: frame table",
        "- `observation.effort` is included in the frame table only when Isaac measured joint efforts were available for every exported frame.",
        "- `videos/observation.images.0/chunk-000/file-000.mp4`: camera 0 stream",
        "- `videos/observation.images.1/chunk-000/file-000.mp4`: camera 1 stream",
        "- `videos/observation.images.2/chunk-000/file-000.mp4`: camera 2 stream",
        "",
    ]
    write_text(os.path.join(export_dir, "README.md"), "\n".join(readme))
    if require_standard and not vla_training_ready:
        raise RuntimeError(f"LeRobot export incomplete: {json.dumps(manifest, ensure_ascii=True)}")
    if require_vla and not vla_training_ready:
        raise RuntimeError(f"VLA export incomplete: {json.dumps(manifest, ensure_ascii=True)}")
    return manifest


def print_lerobot_export(
    run_dir: Union[str, os.PathLike],
    output_dir: Optional[Union[str, os.PathLike]] = None,
    split: str = "trainable",
    fps: Optional[float] = None,
    limit_episodes: Optional[int] = None,
    overwrite: bool = False,
    require_standard: bool = False,
    require_vla: bool = False,
) -> None:
    result = export_lerobot_dataset(
        run_dir,
        output_dir=output_dir,
        split=split,
        fps=fps,
        limit_episodes=limit_episodes,
        overwrite=overwrite,
        require_standard=require_standard,
        require_vla=require_vla,
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
    # For dashboard-level triage we check whether camera fields are present; export still
    # does the stricter file-exists check in collect_lerobot_rows().
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

def dashboard_episode_summary(row: dict, dataset_tag: Optional[str] = None, dataset_skip_reason: str = "") -> Dict[str, object]:
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
        "samples": row.get("samples"),
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


def run_is_under_root(dataset_root: Union[str, os.PathLike], run_dir: Union[str, os.PathLike]) -> bool:
    root = os.path.abspath(str(dataset_root))
    path = os.path.abspath(str(run_dir))
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


def rewrite_row_paths_for_transfer(row: dict, src_dir: str, dst_dir: str) -> dict:
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)
    out = dict(row)
    for key, value in list(out.items()):
        if not isinstance(value, str) or not value:
            continue
        text = value
        try:
            abs_value = os.path.abspath(text) if os.path.isabs(text) else os.path.abspath(os.path.join(src_dir, text))
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


def dashboard_transfer_success_records(
    dataset_root: Union[str, os.PathLike],
    run_paths: Sequence[object],
    dest_dir: Union[str, os.PathLike],
    mode: str = "copy",
) -> Dict[str, object]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    dest_root = ensure_dir(os.path.abspath(str(dest_dir or os.path.join(root, "success_records"))))
    mode = "move" if str(mode).lower() in {"move", "cut"} else "copy"
    transferred = []
    skipped = []
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
        run_dest = ensure_dir(os.path.join(dest_root, "success_records", run_name))
        rewritten_rows = []
        run_manifest = {"source_run_dir": run_dir, "mode": mode, "records": [], "skipped": []}
        for row in rows:
            src_dir = episode_dir_from_row(row)
            if not src_dir or not os.path.isdir(src_dir):
                item = {"episode_index": row.get("episode_index"), "reason": "episode_dir_missing", "source_episode_dir": src_dir}
                skipped.append({"path": run_dir, **item})
                run_manifest["skipped"].append(item)
                continue
            ep_name = os.path.basename(os.path.normpath(src_dir)) or f"episode_{row.get('episode_index', 'unknown')}"
            dst_dir = unique_path(os.path.join(run_dest, ep_name))
            try:
                if mode == "move":
                    shutil.move(src_dir, dst_dir)
                else:
                    shutil.copytree(src_dir, dst_dir)
                rewritten = rewrite_row_paths_for_transfer(row, src_dir, dst_dir)
                rewritten_rows.append(rewritten)
                record = {"episode_index": row.get("episode_index"), "source_episode_dir": src_dir, "dest_episode_dir": dst_dir}
                run_manifest["records"].append(record)
            except Exception as exc:
                item = {"episode_index": row.get("episode_index"), "reason": f"{mode}_failed:{type(exc).__name__}:{exc}", "source_episode_dir": src_dir}
                skipped.append({"path": run_dir, **item})
                run_manifest["skipped"].append(item)
        if rewritten_rows:
            write_jsonl(os.path.join(run_dest, "successful_episodes.jsonl"), rewritten_rows)
        run_manifest.update({"created_at": time.time(), "record_count": len(rewritten_rows), "run_dest": run_dest})
        write_json(os.path.join(run_dest, "transfer_manifest.json"), run_manifest)
        transferred.append({"run": run_name, "source_run_dir": run_dir, "dest_dir": run_dest, "records": len(rewritten_rows), "mode": mode})
    summary = {"ok": True, "mode": mode, "dest_root": dest_root, "transferred": transferred, "skipped": skipped}
    write_json(os.path.join(dest_root, "success_records", f"transfer_summary_{int(time.time())}.json"), summary)
    return summary

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
        success_count = summary.get("success")
        if success_count is None:
            success_count = fast_jsonl_count(index_path(path, "success"))
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
                "attempts": summary.get("attempts"),
                "success": success_count,
                "trainable": summary.get("trainable"),
                "rejected": summary.get("rejected"),
                "failed": summary.get("failed"),
                "requested": summary.get("requested"),
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


def dashboard_run_payload(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    report = analyze_run(run_dir, include_timeline=False)
    compact = compact_analysis(report)
    rows = load_index(run_dir, "all")
    dataset_metrics = compute_dataset_generation_metrics(rows)
    tag_by_episode = {}
    skip_reason_by_episode = {}
    for row in rows:
        trajectory = load_trajectory(row)
        tag, reason = dataset_training_tag_for_row(row, trajectory)
        tag_by_episode[str(row.get("episode_index"))] = tag
        skip_reason_by_episode[str(row.get("episode_index"))] = reason if tag == "skip" else ""
    episodes = [
        dashboard_episode_summary(
            row,
            dataset_tag=tag_by_episode.get(str(row.get("episode_index"))),
            dataset_skip_reason=skip_reason_by_episode.get(str(row.get("episode_index")), ""),
        )
        for row in rows
    ]
    scene_points = [episode["scene"] for episode in episodes]
    status_counts = Counter(str(episode.get("status", "unknown")) for episode in episodes)
    diagnosis = build_report_context(compact, report=report, all_rows=rows)
    diagnosis["dataset_metrics"] = dataset_metrics
    diagnosis["skip_reasons"] = dataset_metrics.get("skip_reasons", [])
    diagnosis["skipped_episodes"] = dataset_metrics.get("skipped_episodes", 0)
    diagnosis["usable_frames"] = dataset_metrics.get("usable_frames", 0)
    diagnosis["total_frames"] = dataset_metrics.get("total_frames", 0)
    diagnosis["data_efficiency_score"] = dataset_metrics.get("data_efficiency_score", 0)
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
        "status_counts": dict(status_counts),
        "episodes": episodes,
        "scene_points": scene_points,
        "generated_at": time.time(),
    }


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


def dashboard_episode_payload(
    run_dir: Union[str, os.PathLike],
    episode_index: Union[int, str],
    max_points: int = 1800,
) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    rows = load_index(run_dir, "all")
    selected = None
    wanted = str(episode_index)
    for row in rows:
        if str(row.get("episode_index")) == wanted or str(row.get("episode_id")) == wanted:
            selected = row
            break
    if selected is None:
        return {"ok": False, "reason": f"episode_not_found:{episode_index}", "run_dir": run_dir}
    trajectory = load_trajectory(selected)
    if not trajectory:
        episode = dashboard_episode_summary(selected, dataset_tag="skip", dataset_skip_reason="trajectory_empty")
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
        }
    first_t = safe_float_value(trajectory[0].get("t"), 0.0) or 0.0
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
        "episode": dashboard_episode_summary(selected),
        "sample_count": len(trajectory),
        "returned_points": len(indices),
        "stage_spans": contiguous_stage_spans(trajectory, first_t),
        "joint_names": ["swing", "boom", "arm", "bucket"],
        "series": series,
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
.shell{max-width:1720px;margin:0 auto;padding:16px 18px 28px}
.topbar{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--border);box-shadow:0 6px 18px rgba(16,24,40,.06)}
.topbar .shell{padding-top:12px;padding-bottom:12px}
h1{font-size:21px;line-height:1.2;margin:0 0 10px;color:#101828}
h2{font-size:15px;line-height:1.25;margin:0;color:#101828}
h3{font-size:13px;margin:0 0 8px;color:#344054;text-transform:uppercase;letter-spacing:.04em}
.controls{display:grid;grid-template-columns:auto minmax(280px,1fr) auto minmax(320px,560px) auto;gap:8px;align-items:center}
label{font-size:12px;color:#475467;font-weight:600;white-space:nowrap}
input,select,button{min-height:34px;border:1px solid #cbd5e1;border-radius:8px;background:#fff;padding:0 10px;font-size:13px;min-width:0}
input.path{width:100%}select{width:100%}
button{background:#1f2937;color:#fff;border-color:#1f2937;cursor:pointer;font-weight:600}
button.secondary{background:#fff;color:#111827;border-color:#cbd5e1}.linkBtn{border:0;background:transparent;color:#175cd3;padding:0;min-height:0;font-weight:800;text-align:left;cursor:pointer}.linkBtn:hover{text-decoration:underline}
.statusLine{margin-top:8px;display:flex;gap:10px;align-items:center;min-height:18px}.runMonitor{margin-top:9px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:#475467}.runMonitorTitle{font-weight:800;color:#101828}.runBadge{display:inline-flex;align-items:center;gap:6px;border:1px solid #d0d5dd;border-radius:999px;background:#fff;color:#344054;padding:4px 9px;min-height:26px;font-size:12px;cursor:pointer}.runBadge.active{border-color:#12b76a;background:#ecfdf3;color:#027a48}.runBadge.recent{border-color:#fdb022;background:#fffaeb;color:#b54708}.activityDot{width:9px;height:9px;border-radius:999px;display:inline-block;background:#98a2b3;box-shadow:0 0 0 2px rgba(152,162,179,.14)}.activityDot.active{background:#12b76a;box-shadow:0 0 0 3px rgba(18,183,106,.18)}.activityDot.recent{background:#fdb022;box-shadow:0 0 0 3px rgba(253,176,34,.18)}.activityDot.idle{background:#f04438;box-shadow:0 0 0 3px rgba(240,68,56,.14)}.activityDot.missing,.activityDot.unknown{background:#98a2b3}.managerPanel{margin-bottom:14px}.managerToolbar{display:grid;grid-template-columns:repeat(3,auto) minmax(280px,1fr) repeat(4,auto);gap:8px;align-items:center;margin-bottom:10px}.managerToolbar .danger{background:#b42318;border-color:#b42318;color:#fff}.managerToolbar .warn{background:#b54708;border-color:#b54708;color:#fff}.managerSummary{font-size:12px;color:#475467;margin-bottom:8px;min-height:18px}.managerTableWrap{max-height:260px;overflow:auto;border:1px solid #eaecf0;border-radius:10px;background:#fff}.managerTable{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}.managerTable th,.managerTable td{padding:7px 8px;border-bottom:1px solid #eef2f6;white-space:nowrap;vertical-align:middle}.managerTable th{position:sticky;top:0;background:#f8fafc;z-index:2;text-transform:uppercase;letter-spacing:.04em;font-size:10.5px;color:#475467}.managerTable .nameCell{font-weight:800;color:#101828}.managerTable .num{text-align:right;font-variant-numeric:tabular-nums}.managerTable .zeroSuccess{color:#b42318;font-weight:850}.managerTable .successRun{color:#067647;font-weight:850}.managerTable .dataSizeCell{max-width:280px;overflow:hidden;text-overflow:ellipsis}.ok{color:#047857}.error{color:#b91c1c}.muted{color:var(--muted);font-size:12px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.small{font-size:12px}.nowrap{white-space:nowrap}
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
.statusFilterTitle{font-size:12px;color:#475467;font-weight:700;margin-right:4px}.statusFilter{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.filterBtn{min-height:28px;border:1px solid #d0d5dd;border-radius:999px;background:#fff;color:#344054;padding:0 10px;font-size:12px;font-weight:700;cursor:pointer}.filterBtn.active{color:#fff;border-color:transparent}.filterBtn.all.active{background:#344054}.filterBtn.trainable.active,.filterBtn.success.active{background:#067647}.filterBtn.rejected.active{background:#d92d20}.filterBtn.failed.active,.filterBtn.fail.active{background:#f79009;color:#111827}.filterBtn.diagnostic.active{background:#6941c6}.filterBtn.planning.active{background:#175cd3}.filterBtn.skip.active{background:#475467}.filterBtn.unknown.active{background:#667085}.filterCount{font-size:12px;color:#667085;white-space:nowrap}.chartMeta{font-size:11px;color:#667085;line-height:1.35;margin-top:6px}.miniChartGrid{display:grid;grid-template-columns:1fr;gap:10px}.miniChart{border:1px solid #eaecf0;border-radius:12px;background:#fcfcfd;padding:10px;min-width:0}.miniChart h3{margin:0 0 6px;font-size:12px;color:#344054;text-transform:none;letter-spacing:0}.miniChart .chart{max-height:165px}.smallChartBox{min-height:0}.pill.failed,.pill.fail{background:#fff7ed;color:#c2410c}.pill.rejected{background:#fef3f2;color:#b42318}.pill.diagnostic{background:#f4f3ff;color:#5925dc}.pill.skip{background:#f2f4f7;color:#344054}.episodeInspector{display:grid;grid-template-columns:minmax(560px,42%) minmax(0,1fr);gap:14px;align-items:start;min-height:0}.episodeSide{background:#f8fafc;border:1px solid #eaecf0;border-radius:12px;padding:10px;display:flex;flex-direction:column;min-height:0;height:var(--episodeAsideHeight,640px);max-height:var(--episodeAsideHeight,640px);overflow:hidden;align-self:start}.sideTabsToolbar{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:8px}.sideSortHint{font-size:11px;color:#667085;line-height:1.35;text-align:right;max-width:190px}.episodeTabsList{flex:1;min-height:0;overflow:auto;scrollbar-width:none;-ms-overflow-style:none;border:1px solid #eaecf0;border-radius:10px;background:#fff}.episodeTabsList::-webkit-scrollbar{display:none;width:0;height:0}.episodeDataSheet{min-width:980px;width:100%;border-collapse:separate;border-spacing:0;font-size:11.5px;line-height:1.25}.episodeDataSheet th,.episodeDataSheet td{padding:7px 8px;border-bottom:1px solid #eef2f6;white-space:nowrap;vertical-align:middle}.episodeDataSheet th{position:sticky;top:0;z-index:4;background:#f8fafc;color:#475467;text-transform:uppercase;letter-spacing:.04em;font-size:10.5px}.episodeDataSheet th.sortable{cursor:pointer;color:#175cd3;user-select:none}.episodeDataSheet th.sortable:hover{background:#eff8ff}.episodeDataSheet tbody tr{cursor:pointer}.episodeDataSheet tbody tr:hover td{background:#f8fafc}.episodeDataSheet tbody tr.selected td{background:#e0f2fe}.episodeDataSheet .epCol{position:sticky;left:0;z-index:3;background:#fff;font-weight:800;color:#101828}.episodeDataSheet th.epCol{z-index:5;background:#f8fafc}.episodeDataSheet tbody tr:hover .epCol{background:#f8fafc}.episodeDataSheet tbody tr.selected .epCol{background:#e0f2fe}.episodeDataSheet .num{text-align:right;font-variant-numeric:tabular-nums}.episodeDataSheet .reasonCell{max-width:360px;overflow:hidden;text-overflow:ellipsis}.sideFooter{font-size:11px;color:#98a2b3;margin-top:7px;line-height:1.35}.timelinePane{min-width:0;display:flex;flex-direction:column;height:auto;align-self:start}.timelinePaneHeader{position:relative;top:auto;z-index:2;background:#fff;border:1px solid #eaecf0;border-radius:12px;padding:10px 12px;margin-bottom:16px}.timelinePaneHeader + .timelineGrid{margin-top:0}#bucketChart{margin-top:0}.timelineChart svg{display:block}.timelineGrid{gap:12px;min-width:0}.timelineChart .chart{min-height:230px}.unitLegend{border:1px solid #d0d5dd;border-radius:999px;padding:2px 7px;background:#fff;color:#475467;font-weight:700}.plotNote{font-size:11px;color:#667085;margin-top:6px;line-height:1.35}.densityBadge{display:inline-block;margin-left:6px;border:1px solid #d0d5dd;border-radius:999px;padding:1px 6px;font-size:10px;color:#475467;background:#fff}.meshFrameBadge{display:inline-block;border:1px solid #d0d5dd;border-radius:999px;padding:2px 7px;background:#fff;color:#475467;font-size:11px;margin-top:6px}
@media(max-width:1280px){.episodeInspector{grid-template-columns:1fr;min-height:0}.episodeSide{height:min(560px,var(--episodeAsideHeight,560px));max-height:min(560px,var(--episodeAsideHeight,560px))}.timelinePaneHeader{position:static}.miniChartGrid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:760px){.miniChartGrid{grid-template-columns:1fr}.statusFilterBar{align-items:flex-start}.episodeTabMetrics{grid-template-columns:repeat(2,1fr)}}


/* v8 diagnosis refactor */
.triageGrid{display:grid;grid-template-columns:minmax(380px,1.35fr) minmax(0,1fr) minmax(0,1fr);gap:12px;align-items:stretch}
.compactScore{min-height:132px}.runtimePieCard{min-height:132px}.runtimePieCard .diagStats{grid-template-columns:repeat(4,minmax(0,1fr));margin-top:4px}.runtimePieCard .diagStatValue{font-size:15px}.pieWrap{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:center}.pieLegend{display:grid;gap:5px;font-size:11px;color:#475467}.pieLegendRow{display:flex;align-items:center;justify-content:space-between;gap:8px}.pieSwatch{width:9px;height:9px;border-radius:99px;display:inline-block;margin-right:5px}.pieSvg{width:130px;height:130px;display:block}.datasetScore{font-size:12px;color:#667085;margin-top:6px}.compactScore{min-height:132px}.compactScore .scoreNumber{font-size:46px}.diagCard{border:1px solid #e5e7eb;border-radius:14px;background:#fff;padding:13px;min-width:0;display:flex;flex-direction:column;gap:8px}.diagCardTitle{font-size:11px;color:#667085;text-transform:uppercase;letter-spacing:.05em;font-weight:800}.diagCardValue{font-size:20px;line-height:1.15;font-weight:850;color:#101828;overflow-wrap:anywhere}.diagCardDetail{font-size:12px;line-height:1.4;color:#667085}.diagCard.bad{border-left:4px solid #d92d20}.diagCard.warn{border-left:4px solid #f79009}.diagCard.good{border-left:4px solid #067647}.diagCard.info{border-left:4px solid #175cd3}.diagStats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px;margin-top:auto}.diagStat{border:1px solid #eef2f6;border-radius:10px;background:#f8fafc;padding:7px}.diagStatLabel{font-size:10px;color:#667085;text-transform:uppercase;letter-spacing:.04em}.diagStatValue{font-size:17px;font-weight:800;color:#101828}.actionsStrip{margin-top:12px;border-top:1px solid #eaecf0;padding-top:10px;display:grid;grid-template-columns:130px 1fr;gap:12px;align-items:start}.actionsStrip h3{margin-top:4px}.actionsInline{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}.actionsInline .action{margin:0;min-height:52px;background:#fcfcfd}.detailsPanel{padding:0}.detailsPanel>summary{cursor:pointer;list-style:none;padding:14px 16px;font-weight:850;color:#101828;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #eaecf0}.detailsPanel>summary::-webkit-details-marker{display:none}.detailsPanel>summary:after{content:"展开";font-size:12px;color:#667085;border:1px solid #d0d5dd;border-radius:999px;padding:3px 9px;background:#fff}.detailsPanel[open]>summary:after{content:"收起"}.diagDetailsGrid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px;padding:14px}.diagDetailsGrid>.span3{grid-column:span 3}.diagDetailsGrid>.span4{grid-column:span 4}.diagDetailsGrid>.span6{grid-column:span 6}.diagDetailsGrid>.span12{grid-column:span 12}.subPanel{border:1px solid #eaecf0;border-radius:12px;background:#fff;padding:12px;min-width:0;overflow:hidden}.subPanel h2{font-size:14px;margin:0}.qualityNote{font-size:11px;color:#667085;line-height:1.35;margin-top:6px}
@media(max-width:1280px){.triageGrid{grid-template-columns:1fr 1fr}.actionsInline{grid-template-columns:1fr}.actionsStrip{grid-template-columns:1fr}.diagDetailsGrid>.span3,.diagDetailsGrid>.span4,.diagDetailsGrid>.span6{grid-column:span 12}}
@media(max-width:720px){.triageGrid{grid-template-columns:1fr}.diagStats{grid-template-columns:1fr}}

</style>
</head>
<body>
<header class="topbar">
  <div class="shell">
    <h1>Excavator Training Dataset Generation Dashboard</h1>
    <div class="controls">
      <label for="rootInput">Dataset root</label>
      <input id="rootInput" class="path" value="excavator_auto_dataset">
      <button id="loadRunsBtn">Load runs</button>
      <select id="runSelect"></select>
      <button id="loadRunBtn">Analyze run</button>
    </div>
    <div class="controls" style="margin-top:8px;grid-template-columns:auto minmax(280px,1fr) auto auto auto">
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
  <section class="panel managerPanel">
    <div class="panelHeader">
      <div><h2>Run folder manager</h2><div class="panelHint">批量管理训练数据生成目录：显示 run/data* 大小，选择 0 success run 移入 trash，或复制/剪切 success records 到目标目录。</div></div>
      <div class="panelHint">删除默认移动到 dataset_root/.dashboard_trash；success records 目标为 dest/success_records/&lt;run_name&gt;/...</div>
    </div>
    <div class="managerToolbar">
      <button type="button" class="secondary" id="selectAllRunsBtn">Select all</button>
      <button type="button" class="secondary" id="selectZeroSuccessBtn">Select 0 success</button>
      <button type="button" class="secondary" id="selectSuccessRunsBtn">Select success&gt;0</button>
      <input id="successDestInput" class="path" placeholder="Destination dir for success records, e.g. D:\450\success_dataset_pool">
      <button type="button" id="copySuccessBtn">Copy selected success</button>
      <button type="button" class="warn" id="moveSuccessBtn">Cut selected success</button>
      <button type="button" class="secondary" id="clearRunSelectionBtn">Clear</button>
      <button type="button" class="secondary" id="refreshSelectedSizesBtn">Refresh selected sizes</button>
      <button type="button" class="danger" id="deleteSelectedRunsBtn">Trash selected 0-success</button>
    </div>
    <div id="managerSummary" class="managerSummary">loading run folders...</div>
    <div class="managerTableWrap"><table id="runManagerTable" class="managerTable"></table></div>
  </section>
  <section class="panel hero">
    <div class="panelHeader">
      <div><h2>Run diagnosis</h2><div class="panelHint">面向训练数据生成：先看可训练数据量、skip/无效样本、真正阻塞和质量信号。成功样本的空 reason 不再显示为 ok 失败。</div></div>
      <div id="reportHint" class="reportHint"></div>
    </div>
    <div class="triageGrid">
      <div id="runtimePieCard" class="diagCard runtimePieCard"></div>
      <div id="topBlockerCard" class="diagCard"></div>
      <div id="qualitySignalCard" class="diagCard"></div>
    </div>
    <div class="actionsStrip">
      <h3>Next actions</h3>
      <div id="actionsList" class="actionsInline"></div>
    </div>
  </section>

  <section class="grid">
    <section class="panel span3"><div class="panelHeader"><h2>Outcome mix</h2><span class="panelHint">status counts</span></div><div id="outcomeBars" class="barRows"></div></section>
    <section class="panel span5"><div class="panelHeader"><h2>Blocker reason Pareto</h2><span class="panelHint">only rejected / failed / diagnostic; excludes ok</span></div><div id="reasonBars" class="barRows"></div></section>
    <section class="panel span4"><div class="panelHeader"><h2>Quality signals</h2><span class="panelHint">score_low / spill / low bin grouped</span></div><div id="qualityBars" class="barRows"></div><div class="qualityNote">带数值的 warning 会按类别聚合，例如 score_low:35.0 → quality/score_low。</div></section>
    <details class="panel span12 detailsPanel">
      <summary>Training/export diagnostics <span class="panelHint">readiness gates, material stats, schema, non-quality warnings</span></summary>
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
    <div class="panelHeader"><div><h2>Episode inspection</h2><div class="panelHint">左侧 datasheet 无可见滚动条但可滚动；高度跟随右侧 timeline plots，不再由左侧撑出空白。</div></div></div>
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
        </div>
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
  <section class="panel" style="margin-top:14px">
    <div class="panelHeader"><h2>Raw selected attempt</h2><span class="panelHint">debug only</span></div>
    <pre id="rawBox" class="codeBox">{}</pre>
  </section>
</main>
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

function $(id){return document.getElementById(id)}
function esc(s){return String(s ?? "").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function setStatus(text, cls="muted"){const el=$("status"); if(el){el.className=cls; el.textContent=text}}
async function api(path, params){const qs=new URLSearchParams(params||{}); const r=await fetch(path+"?"+qs.toString()); if(!r.ok) throw new Error(await r.text()); return await r.json()}
async function postJSON(path, payload){const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload||{})}); if(!r.ok) throw new Error(await r.text()); return await r.json()}
function finite(v){return typeof v==="number" && Number.isFinite(v)}
function number(v){const n=Number(v); return Number.isFinite(n)?n:null}
function numeric(values){return (values||[]).map(v=>Number(v)).filter(v=>Number.isFinite(v))}
function statusKey(s){const t=String(s||"unknown").toLowerCase(); if(t==="fail"||t==="failed"||t==="failure") return "failed"; if(t==="successful") return "success"; return t||"unknown"}
function statusColor(s){return statusPalette[statusKey(s)] || statusPalette.unknown}
function cssClassStatus(s){return statusKey(s).replace(/[^a-zA-Z0-9_-]/g,"_")}
function fmt(v,d=1){const n=Number(v); if(!Number.isFinite(n)) return ""; if(Math.abs(n)>=10000) return n.toFixed(0); if(Math.abs(n)>=1000) return n.toFixed(1); return n.toFixed(d)}
function fmtAxis(v){const n=Number(v); if(!Number.isFinite(n)) return ""; if(Math.abs(n)>=10000) return n.toFixed(0); if(Math.abs(n)>=1000) return n.toFixed(0); if(Math.abs(n)>=10) return n.toFixed(1); return n.toFixed(2)}
function pct(v){const n=Number(v); return Number.isFinite(n)?`${(n*100).toFixed(1)}%`:""}
function shortText(s,n){s=String(s||""); return s.length>n?s.slice(0,n-1)+"…":s}
function basename(path){return String(path||"").split(/[\\/]/).pop()}

async function loadRuns(opts={}){
  const silent = !!opts.silent;
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
    opt.textContent=`${dot} ${run.name}  attempts=${run.attempts ?? "-"} success=${run.success ?? 0} trainable=${run.trainable ?? "-"} size=${run.size_human || "-"}  ${label}`;
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
  const summary=$("managerSummary");
  if(summary) summary.textContent=`folders=${runs.length}; selected=${selected}; 0-success=${zero.length}; success>0=${successRuns.length}; cached/displayed size=${humanBytes(totalSize)} · sizes are cache-first; refresh selected for exact scan`;
  if(!runs.length){table.innerHTML='<tbody><tr><td class="muted">no run folders</td></tr></tbody>';return;}
  const rows=[`<thead><tr><th><input type="checkbox" onchange="toggleAllRuns(this.checked)"></th><th>State</th><th>Run</th><th class="num">Attempts</th><th class="num">Success</th><th class="num">Trainable</th><th class="num">Rejected</th><th class="num">Failed</th><th>Run size</th><th>data* size</th><th>Latest write</th></tr></thead><tbody>`];
  for(const run of runs){
    const activity=run.activity||{};
    const state=activity.state||"unknown";
    const checked=selectedRunPaths.has(run.path) ? "checked" : "";
    const success=Number(run.success||0);
    const successCls=success>0?"successRun":"zeroSuccess";
    const activeTitle=activity.latest_file ? `${state}: ${activity.latest_file}` : state;
    rows.push(`<tr>
      <td><input type="checkbox" ${checked} onchange="toggleRunSelection(${esc(JSON.stringify(run.path))}, this.checked)"></td>
      <td title="${esc(activeTitle)}"><span class="activityDot ${esc(state)}"></span> ${esc(state)}</td>
      <td class="nameCell" title="${esc(run.path)}"><button type="button" class="linkBtn" onclick="chooseRun(${esc(JSON.stringify(run.path))})">${esc(run.name)}</button></td>
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
  const dest=$("successDestInput").value.trim();
  if(!dest){setStatus("Destination dir is required for success records", "error"); return;}
  const paths=allSuccess ? runRecords.filter(r=>Number(r.success||0)>0).map(r=>r.path) : selectedRunList();
  if(!paths.length){setStatus(allSuccess?"No success>0 runs found":"No run folders selected", "error"); return;}
  const verb=mode==="move"?"Cut/move":"Copy";
  if(!confirm(`${verb} success records from ${paths.length} run folder(s) to:\n${dest}\n\nThis operates on successful_episodes.jsonl records.`)) return;
  setStatus(`${verb} success records...`);
  const result=await postJSON("/api/manage/success_records", {root:$("rootInput").value, paths, dest_dir:dest, mode});
  const done=(result.transferred||[]).reduce((a,r)=>a+Number(r.records||0),0);
  const skipped=(result.skipped||[]).length;
  setStatus(`${verb} complete: records=${done}, skipped=${skipped}, dest=${result.dest_root||dest}`, skipped?"error":"ok");
  await loadRuns({silent:true});
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
    return `<button type="button" class="runBadge active" onclick="chooseRun(${esc(JSON.stringify(run.path))})" title="${esc(run.path||"")}"><span class="activityDot active"></span>${esc(run.name||"")} <span class="muted">${esc(ageText(run.age_s))} · attempts=${esc(run.attempts ?? "-")} success=${esc(run.success ?? 0)} trainable=${esc(run.trainable ?? "-")} size=${esc(run.size_human||"-")}${esc(latest)}</span></button>`;
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
async function loadRun(){
  const runDir=$("runInput").value || $("runSelect").value;
  if(!runDir){setStatus("Choose a run folder first", "error"); return}
  setStatus("Analyzing run...");
  const data=await api("/api/run", {run_dir:runDir});
  currentRun=data; currentEpisodeIndex=null;
  renderRun(data);
  setStatus("Run loaded", "ok");
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
  renderRuntimePie("runtimePieCard", data.tag_runtime_seconds || datasetMetrics.tag_runtime_seconds || {}, datasetMetrics, diagnosis, {attempts, trainable, rejected, failed, diagnostic, skip});
  renderActions("actionsList", (diagnosis.recommendations||[]).slice(0,3));
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
  $("reportHint").innerHTML = data.run_dir ? `<span class="activityDot ${esc(actState)}"></span> ${esc(actText)} &nbsp;|&nbsp; Static report: analysis_plots/report.html  |  generate with --plots` : "";
  setupStatusFilter(data.episodes||[]);
  refreshFilteredViews();
}

function renderRuntimePie(id, data, metrics, diagnosis={}, counts={}){
  const el=$(id); if(!el) return;
  el.className="diagCard runtimePieCard info";
  const entries=Object.entries(data||{}).map(([key,value])=>({key:statusKey(key), value:Number(value)||0})).filter(r=>r.value>0);
  const total=entries.reduce((a,r)=>a+r.value,0);
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
      el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">no runtime data</div><div class="diagCardDetail">Counts, readiness and data efficiency are consolidated here; no duplicate KPI strip is shown.</div>${statHtml}${framesHtml}`;
      return;
    }
    el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">count fallback</div><div class="diagCardDetail">No runtime seconds were available. Counts/readiness/efficiency are consolidated here instead of repeated in KPI cards.</div>${pieMarkup(countEntries,"episodes")}${statHtml}${framesHtml}`;
    return;
  }
  el.innerHTML=`<div class="diagCardTitle">Runtime distribution</div><div class="diagCardValue">${fmt(total,1)}s total</div><div class="diagCardDetail">Tag runtime share plus dataset balance. Skip is surfaced instead of silently dropped; duplicate KPI cards are removed.</div>${pieMarkup(entries,"s")}${statHtml}${framesHtml}`;
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
  return `<div class="pieWrap"><svg class="pieSvg" viewBox="0 0 128 128" role="img">${paths}<circle cx="${cx}" cy="${cy}" r="25" fill="#fff" opacity=".96"></circle></svg><div class="pieLegend">${legend}</div></div>`;
}
function renderKPIs(items){const el=$("kpiCards"); if(!el) return; el.innerHTML=(items||[]).map(it=>`<div class="kpi ${esc(it.cls||"")}"><div class="kpiLabel">${esc(it.label)}</div><div class="kpiValue">${esc(it.value)}</div><div class="kpiSub">${esc(it.sub||"")}</div></div>`).join("")}
function renderReadiness(diagnosis, topBlocker){const score=Number(diagnosis.readiness_score||0); const blocker=topBlocker?`blocker: ${shortText(topBlocker,26)}`:"no blocker"; $("readinessCard").innerHTML=`<div class="scoreLabel">Training readiness</div><div class="scoreNumber">${fmt(score,0)}</div><div class="scoreBar"><div class="scoreFill" style="width:${Math.max(0,Math.min(100,score))}%"></div></div><div class="chips"><span class="chip">${esc(blocker)}</span><span class="chip">${(diagnosis.readiness_gates||[]).length} gates</span></div>`}
function renderDiagCard(id, cls, title, value, detail, stats){const statHtml=(stats||[]).length?`<div class="diagStats">${stats.map(s=>`<div class="diagStat"><div class="diagStatLabel">${esc(s.label)}</div><div class="diagStatValue">${esc(s.value)}</div></div>`).join("")}</div>`:""; $(id).className=`diagCard ${esc(cls||"info")}`; $(id).innerHTML=`<div class="diagCardTitle">${esc(title)}</div><div class="diagCardValue">${esc(value)}</div><div class="diagCardDetail">${esc(detail||"")}</div>${statHtml}`}
function renderTriageSummary(diagnosis, c){const problem=Number(diagnosis.problem_attempts ?? (c.rejected+c.failed+c.diagnostic)); const rate=c.attempts?pct(c.trainable/Math.max(1,c.attempts)):""; const cls=c.trainable>0?"good":"bad"; renderDiagCard("triageSummaryCard", cls, "Dataset balance", `${c.trainable}/${c.attempts} trainable`, `problem episodes: ${problem}; rejected=${c.rejected}, failed=${c.failed}`, [{label:"trainable",value:rate||"0%"},{label:"problem",value:pct(problem/Math.max(1,c.attempts))},{label:"diagnostic",value:c.diagnostic}])}
function renderTopBlocker(diagnosis, rows){const top=(diagnosis.top_problem_reason&&diagnosis.top_problem_reason.key)?diagnosis.top_problem_reason:(rows||[])[0]; if(!top||!top.key){renderDiagCard("topBlockerCard","good","Top blocker","None","Failure Pareto excludes trainable/success ok rows. No rejected/failed blocker is currently dominant.",[]);return} const count=Number(top.count||0); const denom=Number(diagnosis.problem_attempts||diagnosis.attempts||count||1); const cls=count>=Math.max(1,denom/2)?"bad":"warn"; renderDiagCard("topBlockerCard",cls,"Top blocker",shortText(top.key,58),`${count}/${denom} problem episode(s). This is the first item to debug; it is not an \"ok\" status.`,[{label:"count",value:count},{label:"share",value:pct(count/Math.max(1,denom))},{label:"scope",value:"problem-only"}])}
function renderQualitySignal(diagnosis, rows){const top=(diagnosis.top_quality_signal&&diagnosis.top_quality_signal.key)?diagnosis.top_quality_signal:(rows||[])[0]; if(!top||!top.key){renderDiagCard("qualitySignalCard","good","Quality signal","No dominant signal","score_low / spill / low bin are grouped separately from failure blockers.",[]);return} const count=Number(top.count||0); const cls=String(top.key).includes("spill")||String(top.key).includes("low")?"warn":"info"; renderDiagCard("qualitySignalCard",cls,"Quality signal",shortText(top.key,58),`${count} episode(s). Median bin/spill live here with the quality signal instead of in the KPI strip.`,[{label:"median score",value:fmt(diagnosis.score_median,1)||"—"},{label:"median bin",value:`${fmt(diagnosis.final_bin_median,1)||"0"} particles`},{label:"median spill",value:`${fmt(diagnosis.final_spill_median,0)||"0"} particles`}])}
function renderFindings(id, rows){const el=$(id); if(!el) return; el.innerHTML=(rows||[]).length?(rows||[]).map(r=>`<div class="finding sev-${esc(r.severity||"info")}"><div class="findingTitle">${esc(r.title||r.key||"finding")}</div><div class="findingDetail">${esc(r.detail||r.message||"")}</div></div>`).join(""):'<div class="empty">no findings</div>'}
function renderActions(id, rows){$(id).innerHTML=(rows||[]).length?(rows||[]).map((r,i)=>`<div class="action"><b>${i+1}.</b> ${esc(r)}</div>`).join(""):'<div class="empty">no actions</div>'}
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
  for(const st of availableStatuses){buttons.push(`<button class="filterBtn ${esc(st)} active" data-status="${esc(st)}"><span style="display:inline-block;width:8px;height:8px;border-radius:9px;background:${statusColor(st)};margin-right:5px"></span>${esc(st)} ${counts.get(st)||0}</button>`)}
  el.innerHTML=buttons.join("");
  for(const btn of el.querySelectorAll(".filterBtn")){btn.onclick=()=>toggleStatusFilter(btn.dataset.status)}
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
  $("episodeSideMeta").textContent=`${sorted.length} shown · sorted by ${label} ${arrow}`;
  if(!sorted.length){$("episodeTabs").innerHTML='<div class="empty">No attempts match the current status filter.</div>'; syncEpisodeInspectorHeight(); return}
  $("episodeTabs").innerHTML=episodeTableHtml(sorted);
  markSelectedTab(currentEpisodeIndex);
  syncEpisodeInspectorHeight();
}
function episodeSortLabel(key){return ({episode_index:"Ep",score:"Score",max_bucket:"Bucket",lift_bucket:"Lift",final_bin:"Bin",final_spill:"Spill",robot_yaw:"Robot yaw",truck_yaw:"Truck yaw",samples:"Samples",freeze_count:"Freeze"})[key]||key}
function sortHeader(key,label,cls=""){const arrow=episodeSort.key===key?(episodeSort.dir>0?" ▲":" ▼"):""; return `<th class="sortable ${esc(cls)}" onclick="sortEpisodes('${esc(key)}')" title="Click to sort by ${esc(label)}">${esc(label)}${arrow}</th>`}
function episodeTableHtml(rows){const head=`<thead><tr>${sortHeader("episode_index","Ep","epCol")}<th>Status</th>${sortHeader("score","Score","num")}${sortHeader("max_bucket","Bucket","num")}${sortHeader("lift_bucket","Lift","num")}${sortHeader("final_bin","Bin","num")}${sortHeader("final_spill","Spill","num")}${sortHeader("robot_yaw","Robot yaw","num")}${sortHeader("truck_yaw","Truck yaw","num")}<th>Reason</th></tr></thead>`; const body=rows.map(ep=>episodeRowHtml(ep)).join(""); return `<table id="episodeTable" class="episodeDataSheet">${head}<tbody>${body}</tbody></table>`}
function episodeRowHtml(ep){const s=ep.scene||{}; const status=statusKey(ep.status); const reason=ep.reason||ep.warning_reason||""; return `<tr data-ep="${esc(ep.episode_index)}" onclick="loadEpisode('${esc(ep.episode_index)}')"><td class="epCol">${esc(ep.episode_index)}</td><td><span class="pill ${esc(status)}">${esc(status)}</span></td><td class="num">${fmt(ep.score,1)}</td><td class="num">${esc(ep.max_bucket??"")}</td><td class="num">${esc(ep.lift_bucket??"")}</td><td class="num">${esc(ep.final_bin??"")}</td><td class="num">${esc(ep.final_spill??"")}</td><td class="num">${fmt(s.robot_body_yaw_deg,1)}</td><td class="num">${fmt(s.truck_yaw_deg,1)}</td><td class="reasonCell" title="${esc(reason)}">${esc(shortText(reason||"no reason",150))}</td></tr>`}
function sortEpisodes(key){if(episodeSort.key===key){episodeSort.dir*=-1}else{episodeSort={key,dir:key==="episode_index"?1:-1}} renderEpisodes(filteredEpisodes())}
function sortValue(ep,key){if(key==="robot_yaw") return numericOrNull((ep.scene||{}).robot_body_yaw_deg); if(key==="truck_yaw") return numericOrNull((ep.scene||{}).truck_yaw_deg); return numericOrNull(ep[key])}
function numericOrNull(v){const n=Number(v); return Number.isFinite(n)?n:null}
function markSelectedTab(index){document.querySelectorAll("#episodeTabs tr[data-ep]").forEach(row=>row.classList.toggle("selected",row.dataset.ep==String(index)))}

function renderEpisode(data){
  const ep=data.episode||{}; $("episodeTitle").textContent=`Attempt ${ep.episode_index} timeline`; $("episodeMeta").textContent=`${statusKey(ep.status)} · score=${fmt(ep.score,1)} · samples=${data.sample_count} · shown=${data.returned_points} · ${shortText(ep.dataset_skip_reason || ep.reason||ep.warning_reason||"",260)}`; $("rawBox").textContent=JSON.stringify({episode:ep,stage_spans:data.stage_spans},null,2);
  const s=data.series||{}; drawLineChart("bucketChart","Bucket sand holding",s.t,[{name:"bucket_from_pile",values:s.bucket_from_pile},{name:"bucket_total",values:s.bucket_total}],data.stage_spans,"particles"); drawVectorChart("qChart","Joint angles",s.t,s.q_deg,data.stage_spans,"deg"); drawVectorChart("dqChart","Joint velocity",s.t,s.dq_deg_s,data.stage_spans,"deg/s"); drawVectorChart("ddqChart","Joint acceleration",s.t,s.ddq_deg_s2,data.stage_spans,"deg/s²"); drawVectorChart("effortChart","Measured joint effort",s.t,s.effort,data.stage_spans,"effort");
  markSelectedTab(ep.episode_index);
  syncEpisodeInspectorHeight();
}

function chartFrame(width=760,height=220){const l=88,r=34,t=36,b=58; return {w:width,h:height,l,r,t,b,pw:width-l-r,ph:height-t-b}}
function extent(vals){const arr=numeric(vals).filter(v=>Math.abs(v)<1e12); if(!arr.length)return[0,1]; let lo=Math.min(...arr),hi=Math.max(...arr); if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1} const pad=(hi-lo)*0.08; return[lo-pad,hi+pad]}
function drawAxes(parts,f,x0,x1,y0,y1,opts={}){parts.push(`<rect x="${f.l}" y="${f.t}" width="${f.pw}" height="${f.ph}" fill="#fcfcfd" stroke="#d0d5dd"/>`); for(let i=0;i<=4;i++){const x=f.l+f.pw*i/4,y=f.t+f.ph*i/4; parts.push(`<line x1="${x.toFixed(1)}" y1="${f.t}" x2="${x.toFixed(1)}" y2="${f.t+f.ph}" stroke="#eef2f6"/>`); parts.push(`<line x1="${f.l}" y1="${y.toFixed(1)}" x2="${f.l+f.pw}" y2="${y.toFixed(1)}" stroke="#eef2f6"/>`)} const xs=opts.xSuffix||""; parts.push(`<text x="${f.l}" y="${f.h-22}" font-size="11" fill="#667085">${fmtAxis(x0)}${esc(xs)}</text>`); parts.push(`<text x="${f.l+f.pw}" y="${f.h-22}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(x1)}${esc(xs)}</text>`); parts.push(`<text x="${f.l-8}" y="${f.t+f.ph}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(y0)}</text>`); parts.push(`<text x="${f.l-8}" y="${f.t+10}" text-anchor="end" font-size="11" fill="#667085">${fmtAxis(y1)}</text>`)}
function drawStageRects(parts,spans,scaleX,top,height){const seen=new Map(); let next=0; for(const span of spans||[]){if(!seen.has(span.stage))seen.set(span.stage,stagePalette[next++%stagePalette.length]); const x=scaleX(span.start),w=Math.max(1,scaleX(span.end)-x); parts.push(`<rect x="${x.toFixed(1)}" y="${top}" width="${w.toFixed(1)}" height="${height}" fill="${seen.get(span.stage)}" opacity="0.44"><title>${esc(span.stage)}</title></rect>`)}}
function drawLineChart(targetId,title,xs,lines,spans,unit){const f=chartFrame(980,270); xs=xs||[]; const xVals=numeric(xs); const x0=xVals.length?Math.min(...xVals):0,x1=xVals.length?Math.max(...xVals):1; const allY=[]; for(const line of lines){for(const v of line.values||[]) if(finite(Number(v))) allY.push(Number(v))} const [y0,y1]=extent(allY); const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw; const sy=y=>f.t+f.ph-(Number(y)-y0)/(y1-y0||1)*f.ph; const parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}" role="img">`,`<text x="14" y="22" font-size="15" font-weight="750" fill="#101828">${esc(title)}</text>`]; drawStageRects(parts,spans,sx,f.t,f.ph); drawAxes(parts,f,x0,x1,y0,y1,{xSuffix:"s"}); lines.forEach((line,li)=>{const pts=[];(line.values||[]).forEach((v,i)=>{if(finite(Number(v))&&finite(Number(xs[i])))pts.push(`${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`)}); if(pts.length) parts.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="${lineColors[li%lineColors.length]}" stroke-width="1.9"/>`)}); parts.push(`</svg>`); const unitBadge=unit?`<span class="legendItem unitLegend">unit: ${esc(unit)}</span>`:""; const legend=`<div class="chartLegend">${unitBadge}${lines.map((line,li)=>`<span class="legendItem"><span class="legendDot" style="background:${lineColors[li%lineColors.length]}"></span>${esc(line.name)}</span>`).join("")}</div>`; $(targetId).innerHTML=parts.join("")+legend}
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

$("loadRunsBtn").onclick=()=>loadRuns().catch(e=>setStatus(e.message,"error"));
$("loadRunBtn").onclick=()=>loadRun().catch(e=>setStatus(e.message,"error"));
$("reloadBtn").onclick=()=>loadRun().catch(e=>setStatus(e.message,"error"));
$("copyPathBtn").onclick=()=>navigator.clipboard&&navigator.clipboard.writeText($("runInput").value).then(()=>setStatus("Run path copied","ok")).catch(()=>setStatus("Copy failed","error"));
$("runSelect").onchange=()=>{$("runInput").value=$("runSelect").value};
$("selectAllRunsBtn").onclick=()=>toggleAllRuns(true);
$("selectZeroSuccessBtn").onclick=()=>selectRuns(r=>Number(r.success||0)===0);
$("selectSuccessRunsBtn").onclick=()=>selectRuns(r=>Number(r.success||0)>0);
$("clearRunSelectionBtn").onclick=()=>toggleAllRuns(false);
$("refreshSelectedSizesBtn").onclick=()=>refreshSelectedSizes().catch(e=>setStatus(e.message,"error"));
$("deleteSelectedRunsBtn").onclick=()=>deleteSelectedZeroSuccessRuns().catch(e=>setStatus(e.message,"error"));
$("copySuccessBtn").onclick=()=>transferSuccessRecords("copy", false).catch(e=>setStatus(e.message,"error"));
$("moveSuccessBtn").onclick=()=>transferSuccessRecords("move", false).catch(e=>setStatus(e.message,"error"));
const episodeSortSelect=$("episodeSortSelect"); if(episodeSortSelect) episodeSortSelect.onchange=()=>refreshFilteredViews();
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

    default_root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def send_bytes(self, data: bytes, content_type: str = "application/json", status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
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
                if parsed.path == "/api/runs":
                    root = os.path.abspath(unquote(params.get("root") or default_root))
                    runs = list_dashboard_runs(root)
                    self.send_json({"root": root, "runs": runs, "activity_summary": summarize_run_activity(runs)})
                    return
                if parsed.path == "/api/run":
                    run_dir = os.path.abspath(unquote(params.get("run_dir") or latest_run(default_root)))
                    if not run_dir or not os.path.isdir(run_dir):
                        self.send_json({"error": f"run_dir_not_found:{run_dir}"}, status=404)
                        return
                    self.send_json(dashboard_run_payload(run_dir))
                    return
                if parsed.path == "/api/episode":
                    run_dir = os.path.abspath(unquote(params.get("run_dir") or latest_run(default_root)))
                    episode = params.get("episode_index") or "1"
                    max_points = int(params.get("max_points") or 1800)
                    self.send_json(dashboard_episode_payload(run_dir, episode, max_points=max_points))
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
                    root = os.path.abspath(unquote(str(body.get("root") or default_root)))
                    result = dashboard_refresh_folder_sizes(root, body.get("paths") or [])
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/delete_runs":
                    root = os.path.abspath(unquote(str(body.get("root") or default_root)))
                    result = dashboard_delete_runs(
                        root,
                        body.get("paths") or [],
                        zero_success_only=bool(body.get("zero_success_only", True)),
                        allow_active=bool(body.get("allow_active", False)),
                    )
                    self.send_json(result)
                    return
                if parsed.path == "/api/manage/success_records":
                    root = os.path.abspath(unquote(str(body.get("root") or default_root)))
                    result = dashboard_transfer_success_records(
                        root,
                        body.get("paths") or [],
                        body.get("dest_dir") or os.path.join(root, "success_records"),
                        mode=str(body.get("mode") or "copy"),
                    )
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
    parser.add_argument("--export-fps", type=float, default=None, help="Video fps for exported camera streams; defaults to camera_config frequency.")
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
            fps=args.export_fps,
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
