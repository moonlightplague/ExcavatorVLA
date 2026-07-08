import json
import hashlib
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
EXPORT_TIME_POLICY_FILENAME = "export_time_policy.json"
LEROBOT_IMAGE_SHAPE = [256, 256, 3]
LEROBOT_VIDEO_KEYFRAME_INTERVAL = 4
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
        if os.path.isdir(os.path.join(dataset_root, name))
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
    if "planning_failed" in text:
        return "planning/failed"
    if "preflight_failed" in text or "prepare_failed" in text:
        return "prepare/failed"
    if "score_low" in text:
        return "quality/score_low"
    return text.split(":", 1)[0][:80]


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
    primary_reasons = Counter()
    warning_parts = Counter()
    initial_by_status = Counter()
    plan_by_status = Counter()
    for row in all_rows:
        status = str(row.get("status", "unknown"))
        initial_by_status[f"{row.get('initial_pose_id', 'unknown')}:{status}"] += 1
        plan_by_status[f"{row.get('chosen_plan_id', 'unknown')}:{status}"] += 1
        primary_reasons[classify_reason(row.get("reason", ""))] += 1
        for part in split_reason(row.get("reason", "")):
            reason_parts[part] += 1
        for part in split_reason(row.get("warning_reason", "")):
            warning_parts[part] += 1

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
        "reason_fragments": top_counter(reason_parts, 16),
        "warning_fragments": top_counter(warning_parts, 16),
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
        "top_reason_fragments": report.get("reason_fragments", [])[:8],
        "top_warnings": report.get("warning_fragments", [])[:8],
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


def write_html_report(
    path: Union[str, os.PathLike],
    compact: Dict[str, object],
    plot_files: Sequence[Union[str, os.PathLike]],
) -> str:
    counts = compact.get("counts", {}) if isinstance(compact.get("counts"), dict) else {}
    segments = compact.get("segments", {}) if isinstance(compact.get("segments"), dict) else {}
    schema = compact.get("trajectory_schema", {}) if isinstance(compact.get("trajectory_schema"), dict) else {}
    camera_coverage = schema.get("camera_coverage", {}) if isinstance(schema.get("camera_coverage"), dict) else {}
    run_dir = compact.get("run_dir", "")
    success_rate = safe_float_value(compact.get("success_rate"), 0.0) or 0.0
    rejection_rate = safe_float_value(compact.get("rejection_rate"), 0.0) or 0.0
    cards = [
        ("Attempts", counts.get("all", 0)),
        ("Trainable", counts.get("trainable", 0)),
        ("Rejected", counts.get("rejected", 0)),
        ("Success Rate", f"{success_rate * 100.0:.1f}%"),
        ("Rejection Rate", f"{rejection_rate * 100.0:.1f}%"),
        ("Dig Segments", segments.get("dig", 0)),
        ("Dig+Secure", segments.get("dig_secure", 0)),
        ("Unload Segments", segments.get("unload", 0)),
        ("Camera Samples", camera_coverage.get("observation.camera.available", 0)),
    ]
    plot_titles = {
        "status_counts.svg": "Episode Status",
        "segment_counts.svg": "Segment Counts",
        "primary_reasons.svg": "Primary Reasons",
        "initial_pose_status.svg": "Initial Pose Status",
        "warning_counts.svg": "Warnings",
        "score_histogram.svg": "Score Histogram",
        "bucket_material_medians.svg": "Material Medians",
        "bin_vs_spill_scatter.svg": "Bin vs Spill",
    }
    html = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Excavator Dataset Analysis</title>",
        "<style>",
        ":root{color-scheme:light;font-family:Arial,sans-serif;background:#f3f4f6;color:#111827;}",
        "body{margin:0;padding:24px;}",
        "header{max-width:1280px;margin:0 auto 18px auto;}",
        "h1{font-size:26px;line-height:1.2;margin:0 0 8px 0;}",
        ".run{font-size:13px;color:#4b5563;word-break:break-all;}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;max-width:1280px;margin:0 auto 18px auto;}",
        ".card{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:12px 14px;}",
        ".card .label{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#6b7280;}",
        ".card .value{font-size:24px;font-weight:700;margin-top:5px;}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px;max-width:1280px;margin:0 auto;}",
        "section{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:12px;overflow:hidden;}",
        "section h2{font-size:16px;margin:0 0 10px 0;color:#1f2937;}",
        "img{display:block;width:100%;height:auto;border:1px solid #f3f4f6;border-radius:4px;background:white;}",
        "footer{max-width:1280px;margin:18px auto 0 auto;color:#6b7280;font-size:12px;}",
        "@media (max-width:720px){body{padding:12px}.grid{grid-template-columns:1fr}.card .value{font-size:20px}}",
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Excavator Dataset Analysis</h1>",
        f'<div class="run">{svg_escape(run_dir)}</div>',
        "</header>",
        '<div class="cards">',
    ]
    for label, value in cards:
        html.extend(
            [
                '<div class="card">',
                f'<div class="label">{svg_escape(label)}</div>',
                f'<div class="value">{svg_escape(value)}</div>',
                "</div>",
            ]
        )
    html.extend(['</div>', '<main class="grid">'])
    for plot_file in plot_files:
        name = os.path.basename(str(plot_file))
        if not name.endswith(".svg"):
            continue
        title = plot_titles.get(name, os.path.splitext(name)[0].replace("_", " ").title())
        html.extend(
            [
                "<section>",
                f"<h2>{svg_escape(title)}</h2>",
                f'<img src="{svg_escape(name)}" alt="{svg_escape(title)}">',
                "</section>",
            ]
        )
    html.extend(
        [
            "</main>",
            "<footer>Generated by excavator_dataset_tools.py. Open analysis_compact.json for the raw compact data.</footer>",
            "</body>",
            "</html>",
            "",
        ]
    )
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
    for row in trainable_rows:
        scatter_points.append(
            {
                "x": row.get("final_spill_from_pile_particles"),
                "y": row.get("final_bin_from_pile_particles"),
                "status": "trainable",
            }
        )
    for row in rejected_rows:
        scatter_points.append(
            {
                "x": row.get("final_spill_from_pile_particles"),
                "y": row.get("final_bin_from_pile_particles"),
                "status": "rejected",
            }
        )

    files = []
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "status_counts.svg"),
            "Episode Status Counts",
            [(key, counts.get(key, 0)) for key in ["all", "trainable", "rejected", "failed", "diagnostic", "planning"]],
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
    files.append(
        write_svg_bar_chart(
            os.path.join(output_dir, "primary_reasons.svg"),
            "Primary Episode Reasons",
            counter_rows_to_pairs(compact.get("primary_reasons"), limit=12),
            left=330,
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
        write_svg_scatter(
            os.path.join(output_dir, "bin_vs_spill_scatter.svg"),
            "Final Bin vs Spill",
            scatter_points,
            "final spill particles",
            "final bin particles",
        )
    )
    files.append(write_html_report(os.path.join(output_dir, "report.html"), compact, files))
    files.append(write_text(os.path.join(output_dir, "analysis_compact.json"), json.dumps(compact, ensure_ascii=True, indent=2)))
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


def _success_pool_episode_dir_from_row(run_dir: str, row: dict) -> str:
    episodes_root = os.path.join(str(run_dir), "episodes")
    for key in ["transferred_episode_dir", "dest_episode_dir"]:
        folder = _basename_any_platform(row.get(key))
        if folder:
            candidate = os.path.join(episodes_root, folder)
            if os.path.isdir(candidate):
                return candidate
    for key in ["trajectory", "meta", "score_path", "events"]:
        parent = _basename_any_platform(_dirname_any_platform(row_path_value(row, key)))
        if parent:
            candidate = os.path.join(episodes_root, parent)
            if os.path.isdir(candidate):
                return candidate
    return ""


def sample_image_value(sample: dict, canonical_key: str) -> object:
    for key in LEROBOT_IMAGE_KEY_ALIASES.get(canonical_key, [canonical_key]):
        value = sample.get(key)
        if value:
            return value
    return None


def episode_dir_from_row(row: dict, run_dir: Optional[Union[str, os.PathLike]] = None) -> str:
    if run_dir:
        pool_episode_dir = _success_pool_episode_dir_from_row(str(run_dir), row)
        if pool_episode_dir:
            return pool_episode_dir
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
                        run_relative = os.path.join(str(run_dir), parent_name)
                        if os.path.isdir(run_relative):
                            return run_relative
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
    # A success pool copied between Windows and Linux can contain stale absolute
    # paths such as D:\...\episode_x\trajectory.jsonl.  In that case the current
    # pool's episode directory is authoritative; the stored path is only a
    # filename hint.
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


def normalize_export_time_policy(policy: object) -> Dict[str, object]:
    data = policy if isinstance(policy, dict) else {}
    try:
        speed_scale = float(data.get("speed_scale", 1.0))
    except Exception:
        speed_scale = 1.0
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        speed_scale = 1.0
    time_mode = str(data.get("time_mode") or "uniform_fps").strip() or "uniform_fps"
    if time_mode not in {"uniform_fps", "scaled_raw"}:
        time_mode = "uniform_fps"
    try:
        base_fps = float(data.get("base_fps")) if data.get("base_fps") not in (None, "") else None
    except Exception:
        base_fps = None
    if base_fps is not None and (not math.isfinite(base_fps) or base_fps <= 0):
        base_fps = None
    return {
        "version": 1,
        "speed_scale": float(speed_scale),
        "time_mode": time_mode,
        "base_fps": base_fps,
    }


def default_export_time_policy() -> Dict[str, object]:
    return normalize_export_time_policy({})


def export_time_policy_hash(policy: object) -> str:
    normalized = normalize_export_time_policy(policy)
    payload = json.dumps(normalized, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def export_time_policy_is_default(policy: object) -> bool:
    return normalize_export_time_policy(policy) == default_export_time_policy()


def export_time_policy_path(run_dir: Union[str, os.PathLike]) -> str:
    return os.path.join(os.path.abspath(str(run_dir)), EXPORT_TIME_POLICY_FILENAME)


def load_export_time_policy(run_dir: Union[str, os.PathLike], explicit_policy: object = None) -> Dict[str, object]:
    raw = explicit_policy
    if raw is None:
        raw = read_json(export_time_policy_path(run_dir), default={}) or {}
    policy = normalize_export_time_policy(raw)
    return {
        "policy": policy,
        "hash": export_time_policy_hash(policy),
        "path": export_time_policy_path(run_dir),
        "is_default": export_time_policy_is_default(policy),
    }


def effective_export_fps(base_fps: float, policy: object) -> float:
    normalized = normalize_export_time_policy(policy)
    base = float(normalized.get("base_fps") or base_fps or 10.0)
    if not math.isfinite(base) or base <= 0:
        base = 10.0
    return max(0.001, base * float(normalized.get("speed_scale") or 1.0))


def _unwrap_angle_series(values: Sequence[Optional[Sequence[float]]], dim: int) -> List[List[Optional[float]]]:
    out: List[List[Optional[float]]] = []
    prev: List[Optional[float]] = [None] * int(dim)
    offsets = [0.0] * int(dim)
    two_pi = 2.0 * math.pi
    for row in values:
        current: List[Optional[float]] = []
        src = list(row or [])
        for i in range(int(dim)):
            value = src[i] if i < len(src) else None
            try:
                v = float(value)  # type: ignore[arg-type]
            except Exception:
                current.append(None)
                continue
            if prev[i] is not None:
                delta = v + offsets[i] - float(prev[i])
                while delta > math.pi:
                    offsets[i] -= two_pi
                    delta -= two_pi
                while delta < -math.pi:
                    offsets[i] += two_pi
                    delta += two_pi
            unwrapped = v + offsets[i]
            prev[i] = unwrapped
            current.append(float(unwrapped))
        out.append(current)
    return out


def _finite_difference_vectors(values: Sequence[Sequence[Optional[float]]], timestamps: Sequence[float], dim: int) -> List[List[Optional[float]]]:
    count = len(values)
    if count <= 0:
        return []
    out: List[List[Optional[float]]] = []
    for i in range(count):
        if count == 1:
            out.append([0.0] * int(dim))
            continue
        prev_i = max(0, i - 1)
        next_i = min(count - 1, i + 1)
        if i == 0:
            prev_i, next_i = 0, 1
        elif i == count - 1:
            prev_i, next_i = count - 2, count - 1
        dt = float(timestamps[next_i]) - float(timestamps[prev_i])
        if abs(dt) < 1e-9:
            out.append([0.0] * int(dim))
            continue
        row: List[Optional[float]] = []
        prev_row = list(values[prev_i] or [])
        next_row = list(values[next_i] or [])
        for j in range(int(dim)):
            a = prev_row[j] if j < len(prev_row) else None
            b = next_row[j] if j < len(next_row) else None
            try:
                row.append((float(b) - float(a)) / dt)  # type: ignore[arg-type]
            except Exception:
                row.append(None)
        out.append(row)
    return out


def _rewrap_angles(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for value in values:
        if value is None:
            out.append(None)
            continue
        try:
            v = float(value)
            out.append(float((v + math.pi) % (2.0 * math.pi) - math.pi))
        except Exception:
            out.append(None)
    return out


def _vector_with_fallback(primary: object, fallback: object = None, length: Optional[int] = None) -> Optional[List[float]]:
    vec = vector_or_none(primary, length)
    if vec is not None:
        return vec
    return vector_or_none(fallback, length)


def _set_state_named_values(
    state: Optional[List[float]],
    state_names: Sequence[str],
    replacements: Dict[str, Sequence[Optional[float]]],
) -> Optional[List[float]]:
    if state is None:
        return None
    out = list(state)
    lower_names = [str(name).strip().lower() for name in state_names]
    for group, values in replacements.items():
        vals = list(values or [])
        for axis_i, axis_name in enumerate(["swing", "boom", "arm", "bucket"]):
            if axis_i >= len(vals) or vals[axis_i] is None:
                continue
            candidates: List[str] = []
            if group == "q":
                candidates = [axis_name]
            elif group == "dq":
                candidates = [
                    f"{axis_name}_velocity",
                    f"{axis_name}_vel",
                    f"{axis_name}_dq",
                    f"d{axis_name}",
                    f"{axis_name}_angle_velocity",
                ]
            elif group == "ddq":
                candidates = [
                    f"{axis_name}_acceleration",
                    f"{axis_name}_accel",
                    f"{axis_name}_ddq",
                    f"dd{axis_name}",
                    f"{axis_name}_angle_acceleration",
                ]
            for candidate in candidates:
                if candidate in lower_names:
                    idx = lower_names.index(candidate)
                    if idx < len(out):
                        out[idx] = float(vals[axis_i])  # type: ignore[arg-type]
                    break
    return out


def apply_export_time_policy_to_trajectory(
    trajectory: Sequence[dict],
    policy: object = None,
    base_fps: float = 10.0,
    state_names: Optional[Sequence[str]] = None,
    action_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    samples = [dict(sample) for sample in (trajectory or []) if isinstance(sample, dict)]
    normalized = normalize_export_time_policy(policy)
    raw_first_t = safe_float_value(samples[0].get("t"), 0.0) if samples else 0.0
    raw_first_t = float(raw_first_t or 0.0)
    effective_fps_value = effective_export_fps(base_fps, normalized)
    speed_scale = float(normalized.get("speed_scale") or 1.0)
    raw_times: List[float] = []
    new_times: List[float] = []
    for index, sample in enumerate(samples):
        raw_t = safe_float_value(sample.get("t"), raw_first_t) or raw_first_t
        raw_times.append(float(raw_t) - raw_first_t)
        if normalized.get("time_mode") == "scaled_raw":
            new_times.append(float(raw_t - raw_first_t) / speed_scale)
        else:
            new_times.append(float(index) / float(effective_fps_value))
    dim = 4
    q_raw = [_vector_with_fallback(sample.get("obs.q"), sample.get("goal.q"), dim) for sample in samples]
    q_cmd_raw = [_vector_with_fallback(sample.get("obs.q_cmd"), sample.get("goal.q"), dim) for sample in samples]
    raw_action = [vector_or_none(sample.get("action"), dim) for sample in samples]
    q_unwrapped = _unwrap_angle_series(q_raw, dim)
    q_cmd_unwrapped = _unwrap_angle_series(q_cmd_raw, dim)
    dq = _finite_difference_vectors(q_unwrapped, new_times, dim)
    ddq = _finite_difference_vectors(dq, new_times, dim)
    if any(vec is not None for vec in q_cmd_raw):
        action = _finite_difference_vectors(q_cmd_unwrapped, new_times, dim)
    else:
        action = []
        for vec in raw_action:
            src = list(vec or [])
            row: List[Optional[float]] = []
            for axis_i in range(dim):
                try:
                    row.append(float(src[axis_i]) * speed_scale)
                except Exception:
                    row.append(0.0)
            action.append(row)
    action_ddq = _finite_difference_vectors(action, new_times, dim)
    state_names_list = list(state_names or [])
    transformed: List[dict] = []
    for index, sample in enumerate(samples):
        out = dict(sample)
        out["t_raw"] = sample.get("t")
        out["t"] = float(new_times[index])
        out["export_time_policy"] = dict(normalized)
        q_wrapped = _rewrap_angles(q_unwrapped[index])
        q_cmd_wrapped = _rewrap_angles(q_cmd_unwrapped[index])
        if all(value is not None for value in q_wrapped):
            out["obs.q"] = [float(value) for value in q_wrapped]  # type: ignore[arg-type]
        if all(value is not None for value in q_cmd_wrapped):
            out["obs.q_cmd"] = [float(value) for value in q_cmd_wrapped]  # type: ignore[arg-type]
        out["obs.dq"] = [float(value or 0.0) for value in dq[index]]
        out["obs.ddq"] = [float(value or 0.0) for value in ddq[index]]
        out["action"] = [float(value or 0.0) for value in action[index]]
        out["action.ddq"] = [float(value or 0.0) for value in action_ddq[index]]
        state = vector_or_none(out.get("observation.state")) or vector_or_none(out.get("obs.state"))
        updated_state = _set_state_named_values(
            state,
            state_names_list,
            {"q": q_wrapped, "dq": dq[index], "ddq": ddq[index]},
        )
        if updated_state is not None:
            if "observation.state" in out:
                out["observation.state"] = updated_state
            if "obs.state" in out:
                out["obs.state"] = updated_state
        transformed.append(out)
    return {
        "samples": transformed,
        "time_policy": normalized,
        "time_policy_hash": export_time_policy_hash(normalized),
        "base_fps": float(base_fps or 10.0),
        "effective_fps": float(effective_fps_value),
        "speed_scale": speed_scale,
        "time_mode": normalized.get("time_mode"),
        "raw_duration_s": float(raw_times[-1]) if raw_times else 0.0,
        "duration_s": float(new_times[-1]) if new_times else 0.0,
    }


GENERIC_LEROBOT_TASK_TEXT = "Dig soil from the marked area and dump it into the target container."


def is_generic_lerobot_task_text(text: object) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return True
    generic = re.sub(r"\s+", " ", GENERIC_LEROBOT_TASK_TEXT.lower())
    return normalized == generic or ("marked area" in normalized and "target container" in normalized)


def _nested_value(data: object, path: Sequence[str], default=None):
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _first_vector_xy(*values: object) -> Optional[List[float]]:
    for value in values:
        vec = vector_or_none(value)
        if vec is not None and len(vec) >= 2:
            return [float(vec[0]), float(vec[1])]
    return None


def _first_float(*values: object) -> Optional[float]:
    for value in values:
        out = safe_float_value(value, None)
        if out is not None:
            return float(out)
    return None


def _episode_scene_dict(episode_row: Optional[dict], episode_meta: Optional[dict]) -> Dict[str, object]:
    for source in [episode_row, episode_meta]:
        if isinstance(source, dict) and isinstance(source.get("scene_randomization"), dict):
            return source.get("scene_randomization")  # type: ignore[return-value]
    return {}


def _state_xy_yaw(sample: Optional[dict], state_names: Optional[Sequence[str]] = None) -> Tuple[Optional[List[float]], Optional[float]]:
    if not isinstance(sample, dict):
        return None, None
    state = vector_or_none(sample.get("observation.state")) or vector_or_none(sample.get("obs.state"))
    if state is None:
        return None, None
    names = list(state_names or [])
    x_index = names.index("base_x") if "base_x" in names else 0
    y_index = names.index("base_y") if "base_y" in names else 1
    yaw_index = names.index("base_yaw") if "base_yaw" in names else 2
    xy = None
    yaw = None
    if len(state) > max(x_index, y_index):
        xy = [float(state[x_index]), float(state[y_index])]
    if len(state) > yaw_index:
        yaw = float(state[yaw_index])
    return xy, yaw


def _direction_label_from_xy(
    point_xy: Optional[Sequence[float]],
    origin_xy: Optional[Sequence[float]],
    robot_yaw_rad: Optional[float],
) -> str:
    if point_xy is None or origin_xy is None or len(point_xy) < 2 or len(origin_xy) < 2:
        return "nearby"
    dx = float(point_xy[0]) - float(origin_xy[0])
    dy = float(point_xy[1]) - float(origin_xy[1])
    if abs(dx) + abs(dy) < 1.0e-6:
        return "nearby"
    # In this scene, yaw=0 points roughly toward +Y, so local-front angle is
    # world yaw + 90 degrees.  Positive local angle is left of the excavator.
    forward_angle = (float(robot_yaw_rad or 0.0) + math.pi * 0.5)
    angle = math.atan2(dy, dx) - forward_angle
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    labels = [
        "front",
        "front-left",
        "left",
        "rear-left",
        "rear",
        "rear-right",
        "right",
        "front-right",
    ]
    index = int(math.floor(((math.degrees(angle) + 22.5) % 360.0) / 45.0))
    return labels[index % len(labels)]


def _format_xy_for_task(xy: Optional[Sequence[float]]) -> str:
    if xy is None or len(xy) < 2:
        return "the selected area"
    return f"({float(xy[0]):.2f}, {float(xy[1]):.2f})"


def build_episode_task_text(
    sample: Optional[dict],
    episode_meta: Optional[dict],
    episode_row: Optional[dict] = None,
    state_names: Optional[Sequence[str]] = None,
) -> str:
    explicit_values = []
    for source in [sample, episode_meta, episode_row]:
        if isinstance(source, dict):
            explicit_values.extend([source.get("task"), source.get("dataset_task_text")])
    for value in explicit_values:
        text = str(value or "").strip()
        if text and not is_generic_lerobot_task_text(text):
            return text

    scene = _episode_scene_dict(episode_row, episode_meta)
    candidate = scene.get("candidate") if isinstance(scene.get("candidate"), dict) else {}
    applied = scene.get("applied") if isinstance(scene.get("applied"), dict) else {}
    scene_context = applied.get("scene_context") if isinstance(applied.get("scene_context"), dict) else {}

    state_xy, state_yaw = _state_xy_yaw(sample, state_names=state_names)
    robot_xy = _first_vector_xy(
        _nested_value(applied, ["robot_translation_xyz"]),
        state_xy,
        [0.0, 0.0],
    )
    robot_yaw_deg = _first_float(
        applied.get("robot_body_yaw_deg") if isinstance(applied, dict) else None,
        candidate.get("robot_body_yaw_deg") if isinstance(candidate, dict) else None,
    )
    robot_yaw_rad = math.radians(robot_yaw_deg) if robot_yaw_deg is not None else state_yaw

    sand_xy = _first_vector_xy(
        _nested_value(applied, ["sand_center"]),
        _nested_value(candidate, ["sand_xy"]),
        _nested_value(scene_context, ["pile_center"]),
        episode_row.get("target_xyz") if isinstance(episode_row, dict) else None,
        sample.get("target") if isinstance(sample, dict) else None,
    )
    unload_xy = _first_vector_xy(
        _nested_value(applied, ["unload_point_xyz"]),
        _nested_value(scene_context, ["unload_point"]),
        _nested_value(candidate, ["unload_xy"]),
        episode_row.get("unload_point_xyz") if isinstance(episode_row, dict) else None,
        episode_row.get("unload_release_xyz") if isinstance(episode_row, dict) else None,
        episode_row.get("unload_landing_xyz") if isinstance(episode_row, dict) else None,
    )
    sand_dir = _direction_label_from_xy(sand_xy, robot_xy, robot_yaw_rad)
    unload_dir = _direction_label_from_xy(unload_xy, robot_xy, robot_yaw_rad)
    if sand_xy is not None or unload_xy is not None:
        return (
            f"Dig soil from the sand pile near {_format_xy_for_task(sand_xy)}, {sand_dir} of the excavator, "
            f"and dump it into the truck bed near {_format_xy_for_task(unload_xy)}, {unload_dir} of the excavator."
        )

    for value in explicit_values:
        text = str(value or "").strip()
        if text:
            return text
    return GENERIC_LEROBOT_TASK_TEXT


def lerobot_task_text(
    sample: dict,
    episode_meta: dict,
    episode_row: Optional[dict] = None,
    state_names: Optional[Sequence[str]] = None,
) -> str:
    return build_episode_task_text(sample, episode_meta, episode_row=episode_row, state_names=state_names)


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


def _call_frame_progress(progress_callback, done: int, total: int, message: str) -> None:
    if not progress_callback:
        return
    try:
        progress_callback(int(done), int(max(1, total)), str(message or ""))
    except Exception:
        pass


def _frame_progress_interval(total: int) -> int:
    # Keep dashboard job polling useful without flooding it for long camera streams.
    return max(1, int(max(1, total) // 120))


def try_encode_mp4_imageio(
    image_paths: Sequence[str],
    output_path: str,
    fps: float,
    target_size: Optional[Tuple[int, int]] = None,
    progress_callback=None,
    progress_label: str = "",
) -> Tuple[bool, str]:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception as exc:
        return False, f"imageio_unavailable:{type(exc).__name__}:{exc}"
    total = len(image_paths)
    interval = _frame_progress_interval(total)
    try:
        ensure_dir(os.path.dirname(output_path) or ".")
        keyint = max(1, int(LEROBOT_VIDEO_KEYFRAME_INTERVAL))
        writer = imageio.get_writer(
            output_path,
            fps=float(fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
            output_params=[
                "-g",
                str(keyint),
                "-keyint_min",
                str(keyint),
                "-sc_threshold",
                "0",
            ],
        )
        try:
            for frame_i, image_path in enumerate(image_paths, 1):
                writer.append_data(resize_rgb_frame(imageio.imread(image_path), target_size=target_size))
                if frame_i == 1 or frame_i == total or frame_i % interval == 0:
                    _call_frame_progress(
                        progress_callback,
                        frame_i,
                        total,
                        f"encoding {progress_label or os.path.basename(os.path.dirname(output_path))} with imageio: {frame_i}/{total}",
                    )
        finally:
            writer.close()
        return True, f"ok:keyframe_interval={keyint}"
    except Exception as exc:
        return False, f"imageio_mp4_failed:{type(exc).__name__}:{exc}"


def try_encode_mp4_cv2(
    image_paths: Sequence[str],
    output_path: str,
    fps: float,
    target_size: Optional[Tuple[int, int]] = None,
    progress_callback=None,
    progress_label: str = "",
) -> Tuple[bool, str]:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        return False, f"cv2_unavailable:{type(exc).__name__}:{exc}"
    total = len(image_paths)
    interval = _frame_progress_interval(total)
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
            for frame_i, image_path in enumerate(image_paths, 1):
                frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if frame is None:
                    return False, f"cv2_frame_unreadable:{image_path}"
                if int(frame.shape[1]) != width or int(frame.shape[0]) != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                if frame_i == 1 or frame_i == total or frame_i % interval == 0:
                    _call_frame_progress(
                        progress_callback,
                        frame_i,
                        total,
                        f"encoding {progress_label or os.path.basename(os.path.dirname(output_path))} with cv2: {frame_i}/{total}",
                    )
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
    progress_callback=None,
    progress_label: str = "",
) -> Tuple[bool, str, str]:
    if not image_paths:
        return False, "none", "no_images"
    missing = [path for path in image_paths if not path or not os.path.isfile(path)]
    if missing:
        return False, "none", f"missing_images:{len(missing)}"
    ok, reason = try_encode_mp4_imageio(
        image_paths,
        output_path,
        fps,
        target_size=target_size,
        progress_callback=progress_callback,
        progress_label=progress_label,
    )
    if ok:
        return True, "imageio", reason
    first_reason = reason
    ok, reason = try_encode_mp4_cv2(
        image_paths,
        output_path,
        fps,
        target_size=target_size,
        progress_callback=progress_callback,
        progress_label=progress_label,
    )
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
    return [
        os.path.join(export_dir, "meta", "info.json"),
        os.path.join(export_dir, "meta", "tasks.parquet"),
        os.path.join(export_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
        os.path.join(export_dir, "data", "chunk-000", "file-000.parquet"),
    ]


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
            if len(episodes_df) <= 0:
                reasons.append("episodes/no_rows")
            for key in image_keys:
                for suffix in ["chunk_index", "file_index", "from_timestamp", "to_timestamp"]:
                    column = f"videos/{key}/{suffix}"
                    if column not in episodes_df.columns:
                        reasons.append(f"episodes/missing_{column}")
                chunk_column = f"videos/{key}/chunk_index"
                file_column = f"videos/{key}/file_index"
                if chunk_column in episodes_df.columns and file_column in episodes_df.columns:
                    for row_i, row in episodes_df.iterrows():
                        try:
                            chunk_index = int(row[chunk_column])
                            file_index = int(row[file_column])
                        except Exception:
                            reasons.append(f"episodes/{key}_bad_video_index_row_{row_i}")
                            continue
                        video_path = os.path.join(export_dir, "videos", key, f"chunk-{chunk_index:03d}", f"file-{file_index:03d}.mp4")
                        if not os.path.exists(video_path):
                            reasons.append(f"videos/missing_{key}_file_{file_index:03d}")
        except Exception as exc:
            reasons.append(f"episodes/read_failed:{type(exc).__name__}:{exc}")
    return {
        "ok": not reasons,
        "reasons": reasons,
        "missing": missing,
    }


def lerobot_manifest_reuse_entries(reuse_from_dir: Union[str, os.PathLike, None], export_config_hash: str) -> Dict[str, dict]:
    reuse_dir = os.path.abspath(str(reuse_from_dir or ""))
    if not reuse_dir or not os.path.isdir(reuse_dir):
        return {}
    manifest = read_json(os.path.join(reuse_dir, "manifest.json"), default={}) or {}
    if not isinstance(manifest, dict):
        return {}
    if manifest.get("video_layout") != "per_episode":
        return {}
    if str(manifest.get("export_config_hash") or "") != str(export_config_hash or ""):
        return {}
    out: Dict[str, dict] = {}
    for episode in manifest.get("episodes", []) or []:
        if not isinstance(episode, dict):
            continue
        if episode.get("export_ready") is not True:
            continue
        source_hash = str(episode.get("source_signature_hash") or "")
        if not source_hash:
            continue
        out[source_hash] = episode
    return out


def reusable_episode_video(
    reuse_from_dir: Union[str, os.PathLike, None],
    reusable_episodes: Dict[str, dict],
    episode: dict,
    image_key: str,
    expected_frames: int,
) -> Optional[str]:
    source_hash = str(episode.get("source_signature_hash") or "")
    if not source_hash:
        return None
    reused_episode = reusable_episodes.get(source_hash)
    if not isinstance(reused_episode, dict):
        return None
    if int(reused_episode.get("length", -1) or -1) != int(episode.get("length", -2) or -2):
        return None
    videos = reused_episode.get("videos", {}) if isinstance(reused_episode.get("videos", {}), dict) else {}
    entry = videos.get(image_key, {}) if isinstance(videos.get(image_key, {}), dict) else {}
    if entry.get("available") is not True:
        return None
    if int(entry.get("frames", -1) or -1) != int(expected_frames):
        return None
    rel_path = str(entry.get("path") or "")
    if not rel_path:
        return None
    src_path = os.path.normpath(os.path.join(os.path.abspath(str(reuse_from_dir or "")), rel_path.replace("/", os.sep)))
    if not os.path.isfile(src_path):
        return None
    return src_path


def collect_lerobot_rows(
    run_dir: Union[str, os.PathLike],
    split: str = "trainable",
    limit_episodes: Optional[int] = None,
    time_policy: object = None,
    base_fps: float = 10.0,
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
    episode_image_paths: List[Dict[str, List[str]]] = []
    skipped_frames = 0
    skipped_state_action_frames = 0
    skipped_missing_camera_frames = 0
    missing_camera_by_key: Counter = Counter()
    missing_camera_examples: List[dict] = []
    global_frame = 0
    for source_episode_index, episode in enumerate(episode_rows):
        episode_dir = episode_dir_from_row(episode, run_dir=run_dir)
        trajectory_path = resolve_episode_file(episode_dir, row_path_value(episode, "trajectory"))
        trajectory = read_jsonl(trajectory_path)
        meta = read_json(resolve_episode_file(episode_dir, row_path_value(episode, "meta")), default={}) or {}
        if not trajectory:
            continue
        transformed = apply_export_time_policy_to_trajectory(
            trajectory,
            policy=time_policy,
            base_fps=base_fps,
            state_names=state_names,
            action_names=action_names,
        )
        trajectory = list(transformed.get("samples") or [])
        export_episode_index = len(episodes)
        first_t = safe_float_value(trajectory[0].get("t"), 0.0) or 0.0
        episode_start_frame = global_frame
        episode_length = 0
        task_index = 0
        task_text = ""
        score = safe_float_value(episode.get("score"), None)
        current_episode_image_paths = {key: [] for key in LEROBOT_IMAGE_KEYS}
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
            task_text = lerobot_task_text(sample, meta, episode_row=episode, state_names=state_names)
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
                current_episode_image_paths[key].append(abs_image)
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
                "source_episode_index": int(source_episode_index),
                "source_episode_dir": episode_dir,
                "raw_episode_dir": episode_dir,
                "source_signature_hash": episode.get("dashboard_transfer_source_signature_hash")
                or episode.get("source_signature_hash")
                or "",
                "source_key": episode.get("dashboard_transfer_source_key")
                or episode.get("source_key")
                or "",
                "duration_s": float(transformed.get("duration_s") or 0.0),
                "raw_duration_s": float(transformed.get("raw_duration_s") or 0.0),
            }
        )
        episode_image_paths.append(current_episode_image_paths)
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
        "episode_image_paths": episode_image_paths,
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
    reuse_from_dir: Optional[Union[str, os.PathLike]] = None,
    time_policy: object = None,
    progress_callback=None,
) -> Dict[str, object]:
    def progress(percent: float, message: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(float(max(0.0, min(100.0, percent))), str(message or ""), current, total)
        except Exception:
            pass

    run_dir = os.path.abspath(str(run_dir))
    progress(1.0, "checking source run folder")
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(run_dir)
    export_dir = os.path.abspath(str(output_dir or os.path.join(run_dir, LEROBOT_DEFAULT_EXPORT_DIRNAME)))
    reuse_dir = os.path.abspath(str(reuse_from_dir or ""))
    if not reuse_dir or not os.path.isdir(reuse_dir):
        reuse_dir = ""
    if reuse_dir and os.path.normcase(reuse_dir) == os.path.normcase(export_dir):
        reuse_dir = ""
    progress(2.0, "preparing export directory")
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

    base_export_fps = infer_export_fps(run_dir, fps)
    time_policy_info = load_export_time_policy(run_dir, explicit_policy=time_policy)
    normalized_time_policy = dict(time_policy_info.get("policy") or default_export_time_policy())
    export_fps = effective_export_fps(base_export_fps, normalized_time_policy)
    progress(
        5.0,
        f"collecting trainable rows and validating camera files; speed_scale={normalized_time_policy.get('speed_scale', 1.0)} effective_fps={export_fps:.3f}",
    )
    collected = collect_lerobot_rows(
        run_dir,
        split=split,
        limit_episodes=limit_episodes,
        time_policy=normalized_time_policy,
        base_fps=base_export_fps,
    )
    rows: List[dict] = list(collected["rows"])  # type: ignore[arg-type]
    if not rows:
        raise ValueError(f"no exportable frames found for split={split}")

    meta_dir = ensure_dir(os.path.join(export_dir, "meta"))
    data_dir = ensure_dir(os.path.join(export_dir, "data", "chunk-000"))
    episodes_dir = ensure_dir(os.path.join(meta_dir, "episodes", "chunk-000"))
    ensure_dir(os.path.join(export_dir, "videos"))

    if export_fps <= 0:
        export_fps = 10.0
    video_results = {}
    image_features = list(LEROBOT_IMAGE_KEYS)
    export_config = {
        "video_layout": "per_episode",
        "fps": float(export_fps),
        "video_keyframe_interval": int(LEROBOT_VIDEO_KEYFRAME_INTERVAL),
        "base_fps": float(base_export_fps),
        "time_policy": normalized_time_policy,
        "time_policy_hash": str(time_policy_info.get("hash") or export_time_policy_hash(normalized_time_policy)),
        "image_features": list(image_features),
        "image_shape": list(LEROBOT_IMAGE_SHAPE),
        "source_split": split,
        "schema": LEROBOT_EXPORT_SCHEMA,
    }
    export_config_hash = hashlib.sha1(
        json.dumps(export_config, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    reusable_episodes = lerobot_manifest_reuse_entries(reuse_dir, export_config_hash)
    if reuse_dir:
        progress(10.0, f"reuse manifest entries={len(reusable_episodes)} from {reuse_dir}")
    image_paths: Dict[str, List[str]] = collected["image_paths"]  # type: ignore[assignment]
    episode_image_paths: List[Dict[str, List[str]]] = list(collected.get("episode_image_paths", []))  # type: ignore[arg-type]
    image_frame_counts = {key: len(image_paths.get(key, []) or []) for key in image_features}
    image_missing_file_counts = {
        key: len([path for path in (image_paths.get(key, []) or []) if not path or not os.path.isfile(path)])
        for key in image_features
    }
    progress(12.0, "camera preflight: " + ", ".join(f"{key}={image_frame_counts.get(key, 0)}" for key in image_features))
    episodes = collected["episodes"]
    tasks = collected["tasks"]
    for key in image_features:
        video_results[key] = {
            "available": True,
            "layout": "per_episode",
            "frames": 0,
            "shape": LEROBOT_IMAGE_SHAPE,
            "keyframe_interval": int(LEROBOT_VIDEO_KEYFRAME_INTERVAL),
            "episode_files": [],
        }
    episode_video_manifest: List[dict] = []
    total_video_jobs = max(1, len(episode_image_paths) * len(image_features))
    completed_video_jobs = 0
    reused_video_jobs = 0
    encoded_video_jobs = 0
    video_start_percent = 15.0
    video_end_percent = 75.0
    for episode_i, episode_paths_by_key in enumerate(episode_image_paths):
        current_episode = episodes[episode_i] if 0 <= episode_i < len(episodes) else {"episode_index": episode_i, "length": 0}
        episode_video = {
            "episode_index": int(episode_i),
            "videos": {},
        }
        for key in image_features:
            paths = list(episode_paths_by_key.get(key, []) or [])
            episode_duration_s = float(current_episode.get("duration_s") or (float(len(paths)) / float(export_fps)))
            job_start = video_start_percent + (video_end_percent - video_start_percent) * (completed_video_jobs / total_video_jobs)
            job_end = video_start_percent + (video_end_percent - video_start_percent) * ((completed_video_jobs + 1) / total_video_jobs)
            if not paths or not any(paths):
                entry = {"available": False, "reason": "no_images", "frames": 0, "chunk_index": 0, "file_index": int(episode_i)}
                episode_video["videos"][key] = entry
                video_results[key]["available"] = False
                video_results[key].setdefault("failed_episodes", []).append(entry)
                completed_video_jobs += 1
                progress(job_end, f"{key} episode {episode_i}: no images", completed_video_jobs, total_video_jobs)
                continue
            video_path = os.path.join(export_dir, "videos", key, "chunk-000", f"file-{episode_i:03d}.mp4")
            rel_video_path = relpath_posix(video_path, export_dir)
            reused_src = reusable_episode_video(reuse_dir, reusable_episodes, current_episode, key, len(paths))
            if reused_src and safe_copy_file(reused_src, video_path):
                entry = {
                    "available": True,
                    "encoder": "reused",
                    "path": rel_video_path,
                    "frames": len(paths),
                    "keyframe_interval": int(LEROBOT_VIDEO_KEYFRAME_INTERVAL),
                    "chunk_index": 0,
                    "file_index": int(episode_i),
                    "from_timestamp": 0.0,
                    "to_timestamp": episode_duration_s,
                    "reused_from": relpath_posix(reused_src, reuse_dir),
                }
                video_results[key]["frames"] = int(video_results[key].get("frames", 0) or 0) + len(paths)
                video_results[key]["episode_files"].append(entry)
                episode_video["videos"][key] = entry
                completed_video_jobs += 1
                reused_video_jobs += 1
                progress(job_end, f"reused {key} episode {episode_i}: {len(paths)} frames", completed_video_jobs, total_video_jobs)
                continue

            def camera_progress(done: int, total: int, message: str, _start=job_start, _end=job_end):
                ratio = float(done) / float(max(1, total))
                progress(_start + (_end - _start) * ratio, message, completed_video_jobs, total_video_jobs)

            progress(job_start, f"encoding {key} episode {episode_i}: 0/{len(paths)}", completed_video_jobs, total_video_jobs)
            ok, encoder, reason = encode_mp4(
                paths,
                video_path,
                export_fps,
                target_size=(LEROBOT_IMAGE_SHAPE[1], LEROBOT_IMAGE_SHAPE[0]),
                progress_callback=camera_progress,
                progress_label=f"{key} ep{episode_i}",
            )
            if ok:
                entry = {
                    "available": True,
                    "encoder": encoder,
                    "path": rel_video_path,
                    "frames": len(paths),
                    "keyframe_interval": int(LEROBOT_VIDEO_KEYFRAME_INTERVAL),
                    "encode_reason": reason,
                    "chunk_index": 0,
                    "file_index": int(episode_i),
                    "from_timestamp": 0.0,
                    "to_timestamp": episode_duration_s,
                }
                video_results[key]["frames"] = int(video_results[key].get("frames", 0) or 0) + len(paths)
                video_results[key]["episode_files"].append(entry)
                episode_video["videos"][key] = entry
                completed_video_jobs += 1
                encoded_video_jobs += 1
                progress(job_end, f"encoded {key} episode {episode_i}: {len(paths)} frames", completed_video_jobs, total_video_jobs)
                continue
            entry = {
                "available": False,
                "reason": reason,
                "path": rel_video_path,
                "frames": len(paths),
                "chunk_index": 0,
                "file_index": int(episode_i),
            }
            episode_video["videos"][key] = entry
            video_results[key]["available"] = False
            video_results[key].setdefault("failed_episodes", []).append(entry)
            completed_video_jobs += 1
            encoded_video_jobs += 1
            progress(job_end, f"{key} episode {episode_i}: video encode failed: {reason}", completed_video_jobs, total_video_jobs)
        episode_video_manifest.append(episode_video)

    progress(78.0, "building parquet tables and metadata")
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
            "timestamp": float(row.get("timestamp", float(frame_index) / float(export_fps))),
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
    progress(82.0, "wrote data parquet" if parquet_ok else f"data parquet failed: {parquet_reason}")

    tasks_path = os.path.join(meta_dir, "tasks.parquet")
    tasks_df = pd.DataFrame(
        {"task_index": [int(task["task_index"]) for task in tasks]},  # type: ignore[index]
        index=pd.Index([str(task["task"]) for task in tasks]),  # type: ignore[index]
    )
    tasks_ok, tasks_reason = try_write_dataframe_parquet(tasks_df, tasks_path, index=True)
    progress(84.0, "wrote tasks parquet" if tasks_ok else f"tasks parquet failed: {tasks_reason}")

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
            meta_row[f"videos/{key}/file_index"] = int(episode.get("episode_index", 0))
            meta_row[f"videos/{key}/from_timestamp"] = 0.0
            meta_row[f"videos/{key}/to_timestamp"] = float(
                episode.get("duration_s") or (float(int(episode.get("length", 0))) / float(export_fps))
            )
        episode_meta_rows.append(meta_row)
    episodes_path = os.path.join(episodes_dir, "file-000.parquet")
    episodes_df = pd.DataFrame(episode_meta_rows)
    episodes_ok, episodes_reason = try_write_dataframe_parquet(episodes_df, episodes_path, index=False)
    progress(86.0, "wrote episode metadata" if episodes_ok else f"episode metadata failed: {episodes_reason}")

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
    progress(90.0, "wrote info.json")

    stats = build_lerobot_v3_stats(
        data_rows,
        len(state_names),
        len(action_names),
        effort_dim if effort_available else None,
        image_features,
    )
    write_json(os.path.join(meta_dir, "stats.json"), stats)
    progress(93.0, "wrote stats.json")

    episode_export_manifest = []
    for episode in episodes:  # type: ignore[assignment]
        episode_index = int(episode.get("episode_index", 0))
        task_indices = [int(value) for value in episode.get("tasks", [])]
        episode_task_texts = [task_text_by_index.get(index, "") for index in task_indices]
        video_entry = episode_video_manifest[episode_index] if 0 <= episode_index < len(episode_video_manifest) else {"videos": {}}
        missing_or_failed = []
        for key in image_features:
            key_video = video_entry.get("videos", {}).get(key, {}) if isinstance(video_entry.get("videos", {}), dict) else {}
            if key_video.get("available") is not True:
                missing_or_failed.append(f"{key}:{key_video.get('reason', 'not_available')}")
        episode_export_manifest.append(
            {
                "episode_index": episode_index,
                "raw_episode_index": episode.get("raw_episode_index"),
                "raw_episode_id": episode.get("raw_episode_id", ""),
                "source_episode_index": episode.get("source_episode_index"),
                "source_episode_dir": episode.get("source_episode_dir", ""),
                "source_signature_hash": episode.get("source_signature_hash", ""),
                "source_key": episode.get("source_key", ""),
                "task_texts": episode_task_texts,
                "length": int(episode.get("length", 0)),
                "from_frame": int(episode.get("from_frame", 0)),
                "to_frame": int(episode.get("to_frame", 0)),
                "video_layout": "per_episode",
                "videos": video_entry.get("videos", {}),
                "export_ready": not missing_or_failed,
                "export_status": "ready" if not missing_or_failed else "not_ready",
                "not_ready_reasons": missing_or_failed,
                "export_config_hash": export_config_hash,
                "time_policy": normalized_time_policy,
                "time_policy_hash": str(time_policy_info.get("hash") or export_time_policy_hash(normalized_time_policy)),
            }
        )

    video_ready = all(video_results.get(key, {}).get("available") is True for key in image_features)
    parquet_ready = bool(parquet_ok and tasks_ok and episodes_ok)
    progress(95.0, "validating LeRobot/VLA export")
    validation = validate_lerobot_v3_export(export_dir, image_features)
    vla_training_ready = bool(parquet_ready and video_ready and validation["ok"])
    manifest = {
        "schema": LEROBOT_EXPORT_SCHEMA,
        "codebase_version": LEROBOT_CODEBASE_VERSION,
        "export_dir": export_dir,
        "source_run_dir": run_dir,
        "source_split": split,
        "created_at": time.time(),
        "export_config": export_config,
        "export_config_hash": export_config_hash,
        "time_policy": normalized_time_policy,
        "time_policy_hash": str(time_policy_info.get("hash") or export_time_policy_hash(normalized_time_policy)),
        "base_fps": float(base_export_fps),
        "effective_fps": float(export_fps),
        "video_keyframe_interval": int(LEROBOT_VIDEO_KEYFRAME_INTERVAL),
        "reuse_from_dir": reuse_dir,
        "reused_video_jobs": int(reused_video_jobs),
        "encoded_video_jobs": int(encoded_video_jobs),
        "total_video_jobs": int(total_video_jobs),
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
        "video_layout": "per_episode",
        "videos": video_results,
        "episodes": episode_export_manifest,
        "image_frame_counts": image_frame_counts,
        "image_missing_file_counts": image_missing_file_counts,
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
            "Camera media is segmented per episode so training can open the relevant episode video directly.",
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
        "- `videos/observation.images.0/chunk-000/file-XYZ.mp4`: camera 0 stream for episode file_index XYZ",
        "- `videos/observation.images.1/chunk-000/file-XYZ.mp4`: camera 1 stream for episode file_index XYZ",
        "- `videos/observation.images.2/chunk-000/file-XYZ.mp4`: camera 2 stream for episode file_index XYZ",
        "- `manifest.json`: export status per episode, including source signature and segmented video paths",
        "",
    ]
    write_text(os.path.join(export_dir, "README.md"), "\n".join(readme))
    progress(98.0, "VLA export ready" if vla_training_ready else "VLA export incomplete; see manifest.validation")
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
    time_policy: object = None,
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


def dashboard_scene_from_episode(row: dict) -> Dict[str, object]:
    scene = row.get("scene_randomization") if isinstance(row.get("scene_randomization"), dict) else {}
    candidate = scene.get("candidate") if isinstance(scene.get("candidate"), dict) else {}
    applied = scene.get("applied") if isinstance(scene.get("applied"), dict) else {}
    scene_context = applied.get("scene_context") if isinstance(applied.get("scene_context"), dict) else {}
    sand_xy = (
        vector_xy(applied.get("sand_center"))
        or vector_xy(candidate.get("sand_xy"))
        or vector_xy(row.get("target_xyz"))
    )
    truck_xy = (
        vector_xy(candidate.get("truck_center_xy"))
        or vector_xy(applied.get("truck_translation_xyz"))
    )
    unload_xy = (
        vector_xy(scene_context.get("unload_bin_center"))
        or vector_xy(candidate.get("unload_xy"))
        or vector_xy(row.get("unload_landing_xyz"))
    )
    unload_landing_xy = vector_xy(row.get("unload_landing_xyz"))
    unload_point_xy = vector_xy(row.get("unload_point_xyz")) or vector_xy(row.get("unload_release_xyz"))
    return {
        "episode_index": row.get("episode_index"),
        "episode_id": row.get("episode_id"),
        "status": row.get("status"),
        "score": row.get("score"),
        "sand_xy": sand_xy,
        "truck_xy": truck_xy,
        "unload_xy": unload_xy,
        "unload_landing_xy": unload_landing_xy,
        "unload_point_xy": unload_point_xy,
        "sand_amount_multiplier": safe_float_value(
            applied.get("sand_amount_multiplier", candidate.get("sand_amount_multiplier"))
        ),
        "estimated_particle_count": safe_float_value(
            applied.get("estimated_particle_count", candidate.get("estimated_particle_count"))
        ),
        "truck_yaw_deg": safe_float_value(applied.get("truck_yaw_deg", candidate.get("truck_yaw_deg"))),
        "robot_body_yaw_deg": safe_float_value(
            applied.get("robot_body_yaw_deg", candidate.get("robot_body_yaw_deg"))
        ),
        "truck_radius_m": safe_float_value(candidate.get("truck_radius_m")),
        "unload_radius_m": safe_float_value(candidate.get("unload_radius_m")),
        "sand_radius_m": safe_float_value(candidate.get("sand_radius_m")),
        "unload_polygon_xy": scene_context.get("unload_polygon_xy") if isinstance(scene_context.get("unload_polygon_xy"), list) else [],
        "unload_hull_xy": scene_context.get("unload_hull_xy") if isinstance(scene_context.get("unload_hull_xy"), list) else [],
        "unload_shape": scene_context.get("manual_unload_range_shape"),
        "unload_mesh": scene_context.get("manual_unload_selected_path"),
    }


def dashboard_episode_summary(row: dict) -> Dict[str, object]:
    scene = dashboard_scene_from_episode(row)
    return {
        "episode_index": row.get("episode_index"),
        "episode_id": row.get("episode_id"),
        "status": row.get("status"),
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


def list_dashboard_runs(dataset_root: Union[str, os.PathLike], limit: int = 80) -> List[dict]:
    root = os.path.abspath(str(dataset_root or "excavator_auto_dataset"))
    if not os.path.isdir(root):
        return []
    runs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if not name.startswith("run_"):
            continue
        summary = read_json(os.path.join(path, "summary.json"), default={}) or {}
        runs.append(
            {
                "name": name,
                "path": path,
                "mtime": os.path.getmtime(path),
                "attempts": summary.get("attempts"),
                "trainable": summary.get("trainable"),
                "requested": summary.get("requested"),
            }
        )
    runs.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    return runs[: max(1, int(limit))]


def dashboard_run_payload(run_dir: Union[str, os.PathLike]) -> Dict[str, object]:
    run_dir = os.path.abspath(str(run_dir))
    report = analyze_run(run_dir, include_timeline=False)
    compact = compact_analysis(report)
    rows = load_index(run_dir, "all")
    episodes = [dashboard_episode_summary(row) for row in rows]
    scene_points = [episode["scene"] for episode in episodes]
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    return {
        "run_dir": run_dir,
        "compact": compact,
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
        return {
            "ok": False,
            "reason": "trajectory_empty",
            "episode": dashboard_episode_summary(selected),
            "run_dir": run_dir,
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
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Excavator Auto Dataset Dashboard</title>
<style>
:root{font-family:Arial,sans-serif;color:#111827;background:#eef2f7;--border:#dbe3ee;}
*{box-sizing:border-box}
body{margin:0;padding:16px}
header,.panel{background:#fff;border:1px solid var(--border);border-radius:8px}
header{padding:14px 16px;margin-bottom:12px}
h1{font-size:22px;margin:0 0 8px}
h2{font-size:15px;margin:0 0 10px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,select,button{height:32px;border:1px solid #cbd5e1;border-radius:6px;background:#fff;padding:0 9px;font-size:13px}
input.path{min-width:420px;flex:1}
button{background:#1f2937;color:#fff;border-color:#1f2937;cursor:pointer}
button.secondary{background:#fff;color:#111827}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
.panel{padding:12px;overflow:hidden}
.span3{grid-column:span 3}.span4{grid-column:span 4}.span6{grid-column:span 6}.span8{grid-column:span 8}.span12{grid-column:span 12}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px}
.card{border:1px solid #e5e7eb;border-radius:8px;padding:10px;background:#f8fafc}
.label{font-size:11px;text-transform:uppercase;color:#64748b;letter-spacing:.04em}.value{font-size:22px;font-weight:700;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:12px}
th,td{padding:6px;border-bottom:1px solid #e5e7eb;text-align:left;vertical-align:top}
th.sortable{cursor:pointer;user-select:none;color:#1d4ed8}
th.sortable:hover{text-decoration:underline;background:#eff6ff}
tbody tr{cursor:pointer}tbody tr:hover{background:#f1f5f9}tbody tr.selected{background:#dbeafe}
.muted{color:#64748b;font-size:12px}.error{color:#b91c1c}.ok{color:#047857}
.chart{width:100%;height:auto;border:1px solid #edf2f7;border-radius:6px;background:#fff}
.timelineChart{margin-top:6px}
.timelineChart .chart{max-height:210px}
.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:6px;font-size:12px;color:#475569}
.dot{display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:5px}
pre{white-space:pre-wrap;font-size:12px;max-height:220px;overflow:auto;background:#0f172a;color:#e2e8f0;padding:10px;border-radius:6px}
@media(max-width:980px){.span3,.span4,.span6,.span8,.span12{grid-column:span 12}input.path{min-width:240px}}
</style>
</head>
<body>
<header>
  <h1>Excavator Auto Dataset Dashboard</h1>
  <div class="row">
    <label>Dataset root</label>
    <input id="rootInput" class="path" value="excavator_auto_dataset">
    <button id="loadRunsBtn">Load runs</button>
    <select id="runSelect"></select>
    <button id="loadRunBtn">Analyze folder</button>
  </div>
  <div class="row" style="margin-top:8px">
    <label>Run folder</label>
    <input id="runInput" class="path" placeholder="D:\450\assets\usd\URDF_real3\excavator_auto_dataset\run_...">
    <span id="status" class="muted"></span>
  </div>
</header>
<main class="grid">
  <section class="panel span12">
    <h2>Run Summary</h2>
    <div id="cards" class="cards"></div>
  </section>
  <section class="panel span4">
    <h2>Sand Position Distribution</h2>
    <div id="sandScatter"></div>
  </section>
  <section class="panel span4">
    <h2>Truck Position Distribution</h2>
    <div id="truckScatter"></div>
  </section>
  <section class="panel span4">
    <h2>Unload Point Distribution</h2>
    <div id="unloadScatter"></div>
  </section>
  <section class="panel span4">
    <h2>Robot Initial Yaw</h2>
    <div id="robotYawHist"></div>
  </section>
  <section class="panel span4">
    <h2>Truck Yaw</h2>
    <div id="truckYawHist"></div>
  </section>
  <section class="panel span4">
    <h2>Sand Amount</h2>
    <div id="sandAmountHist"></div>
  </section>
  <section class="panel span4">
    <h2>Attempts</h2>
    <div class="muted">Click an attempt to inspect bucket sand, joints, velocity, acceleration and stage background.</div>
    <div style="max-height:760px;overflow:auto;margin-top:8px"><table id="episodeTable"></table></div>
  </section>
  <section class="panel span8">
    <h2 id="episodeTitle">Attempt Timeline</h2>
    <div id="episodeMeta" class="muted"></div>
    <div id="bucketChart" class="timelineChart"></div>
    <div id="qChart" class="timelineChart"></div>
    <div id="dqChart" class="timelineChart"></div>
    <div id="ddqChart" class="timelineChart"></div>
    <div id="effortChart" class="timelineChart"></div>
  </section>
  <section class="panel span12">
    <h2>Raw Selected Attempt</h2>
    <pre id="rawBox">{}</pre>
  </section>
</main>
<script>
const jointNames = ["swing","boom","arm","bucket"];
const colors = {trainable:"#059669",success:"#059669",rejected:"#dc2626",failed:"#7c2d12",diagnostic:"#d97706",planning:"#2563eb",unknown:"#64748b"};
const lineColors = ["#2563eb","#059669","#d97706","#dc2626","#7c3aed","#0891b2"];
const stagePalette = ["#dbeafe","#dcfce7","#fef3c7","#fee2e2","#ede9fe","#cffafe","#fce7f3","#e2e8f0"];
let currentRun = null;
let currentEpisodeIndex = null;
let episodeSort = {key:"episode_index", dir:1};

function $(id){return document.getElementById(id)}
function esc(s){return String(s ?? "").replace(/[&<>"']/g, c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
function setStatus(text, cls="muted"){const el=$("status"); el.className=cls; el.textContent=text}
async function api(path, params){
  const qs = new URLSearchParams(params || {});
  const r = await fetch(path + "?" + qs.toString());
  if(!r.ok) throw new Error(await r.text());
  return await r.json();
}
function finite(v){return typeof v==="number" && Number.isFinite(v)}
function numeric(values){return values.map(v=>Number(v)).filter(v=>Number.isFinite(v))}
function statusColor(s){return colors[s] || colors.unknown}

async function loadRuns(){
  setStatus("Loading runs...");
  const data = await api("/api/runs", {root:$("rootInput").value});
  const sel = $("runSelect");
  sel.innerHTML = "";
  for(const run of data.runs){
    const opt = document.createElement("option");
    opt.value = run.path;
    opt.textContent = `${run.name}  attempts=${run.attempts ?? "-"} trainable=${run.trainable ?? "-"}`;
    sel.appendChild(opt);
  }
  if(data.runs.length){
    $("runInput").value = data.runs[0].path;
    setStatus(`Loaded ${data.runs.length} runs`, "ok");
  }else{
    setStatus("No run_* folders found", "error");
  }
}

async function loadRun(){
  const runDir = $("runInput").value || $("runSelect").value;
  if(!runDir){setStatus("Choose a run folder first", "error"); return}
  setStatus("Analyzing run...");
  const data = await api("/api/run", {run_dir:runDir});
  currentRun = data;
  currentEpisodeIndex = null;
  renderRun(data);
  setStatus("Run loaded", "ok");
  if(data.episodes && data.episodes.length) loadEpisode(data.episodes[0].episode_index);
}

async function loadEpisode(index){
  if(!currentRun) return;
  currentEpisodeIndex = index;
  markSelectedRow(index);
  setStatus(`Loading attempt ${index}...`);
  const data = await api("/api/episode", {run_dir:currentRun.run_dir, episode_index:index, max_points:2200});
  if(!data.ok){setStatus(data.reason || "episode load failed", "error"); return}
  renderEpisode(data);
  setStatus(`Attempt ${index} loaded`, "ok");
}

function renderRun(data){
  const counts = (data.compact && data.compact.counts) || {};
  const cards = [
    ["Attempts", counts.all || 0],
    ["Trainable", counts.trainable || 0],
    ["Rejected", counts.rejected || 0],
    ["Failed", counts.failed || 0],
    ["Diagnostic", counts.diagnostic || 0],
    ["Success rate", `${(((data.compact && data.compact.success_rate)||0)*100).toFixed(1)}%`],
    ["Run", (data.run_dir||"").split(/[\\/]/).pop()],
  ];
  $("cards").innerHTML = cards.map(c=>`<div class="card"><div class="label">${esc(c[0])}</div><div class="value">${esc(c[1])}</div></div>`).join("");
  const pts = data.scene_points || [];
  drawScatter(
    "sandScatter",
    pts.map(p=>({x:p.sand_xy&&p.sand_xy[0], y:p.sand_xy&&p.sand_xy[1], status:p.status, label:p.episode_index})),
    "x from excavator origin", "y from excavator origin", {origin:true}
  );
  drawScatter(
    "truckScatter",
    pts.map(p=>({x:p.truck_xy&&p.truck_xy[0], y:p.truck_xy&&p.truck_xy[1], status:p.status, label:p.episode_index, yaw:p.truck_yaw_deg, polygon:(p.unload_polygon_xy&&p.unload_polygon_xy.length?p.unload_polygon_xy:p.unload_hull_xy)})),
    "x from excavator origin", "y from excavator origin", {origin:true, polygons:true, heading:true}
  );
  drawScatter(
    "unloadScatter",
    canonicalUnloadPoints(pts),
    "bin local x", "bin local y", {canonicalPolygon:true, origin:true}
  );
  drawHistogram("robotYawHist", pts.map(p=>p.robot_body_yaw_deg), "deg");
  drawHistogram("truckYawHist", pts.map(p=>p.truck_yaw_deg), "deg");
  drawHistogram("sandAmountHist", pts.map(p=>p.sand_amount_multiplier), "x");
  renderEpisodes(data.episodes || []);
}

function renderEpisodes(episodes){
  const sortable = {
    episode_index: "Ep",
    score: "Score",
    max_bucket: "Bucket",
    lift_bucket: "Lift",
    final_bin: "Bin",
    final_spill: "Spill",
    robot_yaw: "Robot yaw",
    truck_yaw: "Truck yaw",
  };
  const sorted = [...episodes].sort((a,b)=>{
    const av = sortValue(a, episodeSort.key);
    const bv = sortValue(b, episodeSort.key);
    if(av === bv) return Number(a.episode_index||0) - Number(b.episode_index||0);
    if(av === null) return 1;
    if(bv === null) return -1;
    return (av < bv ? -1 : 1) * episodeSort.dir;
  });
  const head = key => {
    const label = sortable[key];
    const arrow = episodeSort.key === key ? (episodeSort.dir > 0 ? " ▲" : " ▼") : "";
    return `<th class="sortable" onclick="sortEpisodes('${key}')" title="Click to sort">${esc(label)}${arrow}</th>`;
  };
  const rows = [`<thead><tr>${head("episode_index")}<th>Status</th>${head("score")}${head("max_bucket")}${head("lift_bucket")}${head("final_bin")}${head("final_spill")}${head("robot_yaw")}${head("truck_yaw")}<th>Reason</th></tr></thead><tbody>`];
  for(const ep of sorted){
    const s = ep.scene || {};
    rows.push(`<tr data-ep="${esc(ep.episode_index)}" onclick="loadEpisode('${esc(ep.episode_index)}')">
      <td>${esc(ep.episode_index)}</td><td style="color:${statusColor(ep.status)}">${esc(ep.status)}</td>
      <td>${fmt(ep.score,1)}</td><td>${esc(ep.max_bucket ?? "")}</td><td>${esc(ep.lift_bucket ?? "")}</td>
      <td>${esc(ep.final_bin ?? "")}</td><td>${esc(ep.final_spill ?? "")}</td>
      <td>${fmt(s.robot_body_yaw_deg,1)}</td><td>${fmt(s.truck_yaw_deg,1)}</td>
      <td>${esc(shortText(ep.reason || ep.warning_reason || "", 120))}</td></tr>`);
  }
  rows.push("</tbody>");
  $("episodeTable").innerHTML = rows.join("");
  if(currentEpisodeIndex !== null) markSelectedRow(currentEpisodeIndex);
}
function sortValue(ep, key){
  if(key === "robot_yaw") return numericOrNull((ep.scene||{}).robot_body_yaw_deg);
  if(key === "truck_yaw") return numericOrNull((ep.scene||{}).truck_yaw_deg);
  return numericOrNull(ep[key]);
}
function numericOrNull(v){ const n=Number(v); return Number.isFinite(n) ? n : null; }
function sortEpisodes(key){
  if(episodeSort.key === key) episodeSort.dir *= -1;
  else episodeSort = {key, dir: key === "episode_index" ? 1 : -1};
  if(currentRun) renderEpisodes(currentRun.episodes || []);
}
function markSelectedRow(index){
  document.querySelectorAll("#episodeTable tbody tr").forEach(tr=>tr.classList.toggle("selected", tr.dataset.ep == String(index)));
}
function shortText(s,n){s=String(s||""); return s.length>n?s.slice(0,n-1)+"...":s}
function fmt(v,d=2){return finite(Number(v)) ? Number(v).toFixed(d) : ""}

function renderEpisode(data){
  const ep = data.episode || {};
  $("episodeTitle").textContent = `Attempt ${ep.episode_index} Timeline`;
  $("episodeMeta").textContent = `${ep.status} score=${fmt(ep.score,1)} samples=${data.sample_count} shown=${data.returned_points} reason=${shortText(ep.reason || ep.warning_reason || "", 280)}`;
  $("rawBox").textContent = JSON.stringify({episode:ep, stage_spans:data.stage_spans}, null, 2);
  const s = data.series || {};
  drawLineChart("bucketChart", "Bucket sand holding", s.t, [
    {name:"bucket_from_pile", values:s.bucket_from_pile},
    {name:"bucket_total", values:s.bucket_total},
  ], data.stage_spans, "particles");
  drawVectorChart("qChart", "Joint angles", s.t, s.q_deg, data.stage_spans, "deg");
  drawVectorChart("dqChart", "Joint velocity", s.t, s.dq_deg_s, data.stage_spans, "deg/s");
  drawVectorChart("ddqChart", "Joint acceleration", s.t, s.ddq_deg_s2, data.stage_spans, "deg/s^2");
  drawVectorChart("effortChart", "Measured joint effort", s.t, s.effort, data.stage_spans, "effort");
}

function chartFrame(width=760,height=190){
  return {w:width,h:height,l:50,r:14,t:22,b:32,pw:width-64,ph:height-54};
}
function extent(vals){
  const arr = numeric(vals).filter(v=>Math.abs(v)<1e12);
  if(!arr.length) return [0,1];
  let lo=Math.min(...arr), hi=Math.max(...arr);
  if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1}
  const pad=(hi-lo)*0.08; return [lo-pad,hi+pad];
}
function drawStageRects(parts, spans, scaleX, top, height){
  const seen = new Map(); let next = 0;
  for(const span of spans||[]){
    if(!seen.has(span.stage)) seen.set(span.stage, stagePalette[next++ % stagePalette.length]);
    const x = scaleX(span.start), w = Math.max(1, scaleX(span.end)-x);
    parts.push(`<rect x="${x.toFixed(1)}" y="${top}" width="${w.toFixed(1)}" height="${height}" fill="${seen.get(span.stage)}" opacity="0.48"><title>${esc(span.stage)}</title></rect>`);
  }
}
function drawAxes(parts, f, x0, x1, y0, y1, unit){
  parts.push(`<rect x="${f.l}" y="${f.t}" width="${f.pw}" height="${f.ph}" fill="none" stroke="#cbd5e1"/>`);
  for(let i=0;i<=4;i++){
    const x=f.l+f.pw*i/4, y=f.t+f.ph*i/4;
    parts.push(`<line x1="${x}" y1="${f.t}" x2="${x}" y2="${f.t+f.ph}" stroke="#e5e7eb"/>`);
    parts.push(`<line x1="${f.l}" y1="${y}" x2="${f.l+f.pw}" y2="${y}" stroke="#e5e7eb"/>`);
  }
  parts.push(`<text x="${f.l}" y="${f.h-16}" font-size="11" fill="#64748b">${fmt(x0,1)}s</text>`);
  parts.push(`<text x="${f.l+f.pw}" y="${f.h-16}" text-anchor="end" font-size="11" fill="#64748b">${fmt(x1,1)}s</text>`);
  parts.push(`<text x="${f.l-6}" y="${f.t+f.ph}" text-anchor="end" font-size="11" fill="#64748b">${fmt(y0,1)}</text>`);
  parts.push(`<text x="${f.l-6}" y="${f.t+10}" text-anchor="end" font-size="11" fill="#64748b">${fmt(y1,1)}</text>`);
  parts.push(`<text x="${f.w-20}" y="${f.h-16}" text-anchor="end" font-size="11" fill="#64748b">${esc(unit||"")}</text>`);
}
function drawLineChart(targetId, title, xs, lines, spans, unit){
  const f=chartFrame(); xs=xs||[];
  const xVals=numeric(xs); const x0=xVals.length?Math.min(...xVals):0, x1=xVals.length?Math.max(...xVals):1;
  const allY=[]; for(const line of lines){ for(const v of line.values||[]) if(finite(Number(v))) allY.push(Number(v)); }
  const [y0,y1]=extent(allY);
  const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw;
  const sy=y=>f.t+f.ph-(Number(y)-y0)/(y1-y0||1)*f.ph;
  const parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}" role="img">`,`<text x="12" y="17" font-size="14" font-weight="700" fill="#111827">${esc(title)}</text>`];
  drawStageRects(parts, spans, sx, f.t, f.ph); drawAxes(parts,f,x0,x1,y0,y1,unit);
  lines.forEach((line,li)=>{
    const pts=[]; (line.values||[]).forEach((v,i)=>{ if(finite(Number(v)) && finite(Number(xs[i]))) pts.push(`${sx(xs[i]).toFixed(1)},${sy(v).toFixed(1)}`); });
    if(pts.length) parts.push(`<polyline points="${pts.join(" ")}" fill="none" stroke="${lineColors[li%lineColors.length]}" stroke-width="1.8"/>`);
  });
  let lx=f.l+8; lines.forEach((line,li)=>{parts.push(`<circle cx="${lx}" cy="${f.h-28}" r="4" fill="${lineColors[li%lineColors.length]}"/><text x="${lx+7}" y="${f.h-24}" font-size="11" fill="#334155">${esc(line.name)}</text>`); lx+=86;});
  parts.push(`</svg>`);
  $(targetId).innerHTML=parts.join("");
}
function drawVectorChart(targetId, title, xs, vectors, spans, unit){
  const lines = jointNames.map((name, j)=>({name, values:(vectors||[]).map(row=>Array.isArray(row)?row[j]:null)}));
  drawLineChart(targetId, title, xs, lines, spans, unit);
}
function cleanPolygon(poly){
  if(!Array.isArray(poly)) return [];
  const out=[];
  for(const pt of poly){
    if(Array.isArray(pt) && finite(Number(pt[0])) && finite(Number(pt[1]))) out.push([Number(pt[0]), Number(pt[1])]);
  }
  return out;
}
function polygonCenter(poly){
  const clean=cleanPolygon(poly);
  if(!clean.length) return null;
  let sx=0, sy=0;
  for(const p of clean){sx+=p[0]; sy+=p[1];}
  return [sx/clean.length, sy/clean.length];
}
function polygonPrincipalAngle(poly){
  const clean=cleanPolygon(poly);
  if(clean.length < 2) return 0;
  let bestA=0, bestD=-1;
  for(let i=0;i<clean.length;i++){
    for(let j=i+1;j<clean.length;j++){
      const dx=clean[j][0]-clean[i][0], dy=clean[j][1]-clean[i][1];
      const d=dx*dx+dy*dy;
      if(d>bestD){bestD=d; bestA=Math.atan2(dy,dx);}
    }
  }
  return bestA;
}
function transformToLocal(pt, center, angle){
  const dx=Number(pt[0])-center[0], dy=Number(pt[1])-center[1];
  const c=Math.cos(-angle), s=Math.sin(-angle);
  return [dx*c - dy*s, dx*s + dy*c];
}
function canonicalUnloadPoints(scenePoints){
  const out=[];
  for(const p of scenePoints||[]){
    const poly=cleanPolygon((p.unload_polygon_xy&&p.unload_polygon_xy.length)?p.unload_polygon_xy:p.unload_hull_xy);
    const center=polygonCenter(poly) || p.unload_xy;
    if(!center) continue;
    const angle=polygonPrincipalAngle(poly);
    const source=p.unload_point_xy || p.unload_landing_xy || p.unload_xy;
    if(!source) continue;
    const local=transformToLocal(source, center, angle);
    const localPoly=poly.map(pt=>transformToLocal(pt, center, angle));
    out.push({x:local[0], y:local[1], status:p.status, label:p.episode_index, polygon:localPoly});
  }
  return out;
}
function drawScatter(targetId, pts, xLabel, yLabel, opts={}){
  const f=chartFrame(430,320);
  const rows=(pts||[]).map(p=>Object.assign({}, p, {poly:cleanPolygon(p.polygon)}));
  const clean=rows.filter(p=>finite(Number(p.x))&&finite(Number(p.y)));
  const xs=[], ys=[];
  for(const p of rows){
    if(finite(Number(p.x))&&finite(Number(p.y))){ xs.push(Number(p.x)); ys.push(Number(p.y)); }
    if((opts.polygons || opts.rangeFromPolygons || opts.canonicalPolygon) && p.poly.length){
      for(const pt of p.poly){ xs.push(pt[0]); ys.push(pt[1]); }
    }
  }
  if(opts.origin){ xs.push(0); ys.push(0); }
  if(!xs.length || !ys.length){$(targetId).innerHTML='<div class="muted">no data</div>';return}
  const [x0,x1]=extent(xs), [y0,y1]=extent(ys);
  const sx=x=>f.l+(Number(x)-x0)/(x1-x0||1)*f.pw, sy=y=>f.t+f.ph-(Number(y)-y0)/(y1-y0||1)*f.ph;
  const parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}">`]; drawAxes(parts,f,x0,x1,y0,y1,"m");
  if(opts.origin){
    parts.push(`<line x1="${sx(0).toFixed(1)}" y1="${f.t}" x2="${sx(0).toFixed(1)}" y2="${f.t+f.ph}" stroke="#111827" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.45"/>`);
    parts.push(`<line x1="${f.l}" y1="${sy(0).toFixed(1)}" x2="${f.l+f.pw}" y2="${sy(0).toFixed(1)}" stroke="#111827" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.45"/>`);
    parts.push(`<circle cx="${sx(0).toFixed(1)}" cy="${sy(0).toFixed(1)}" r="5" fill="#111827"><title>excavator origin</title></circle>`);
  }
  if(opts.canonicalPolygon){
    let bestPoly=[];
    let bestArea=-1;
    for(const p of rows){
      if(!p.poly.length) continue;
      let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
      for(const pt of p.poly){minX=Math.min(minX,pt[0]);maxX=Math.max(maxX,pt[0]);minY=Math.min(minY,pt[1]);maxY=Math.max(maxY,pt[1]);}
      const area=(maxX-minX)*(maxY-minY);
      if(area>bestArea){bestArea=area;bestPoly=p.poly;}
    }
    if(bestPoly.length){
      const d=bestPoly.map(pt=>`${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`).join(" ");
      parts.push(`<polygon points="${d}" fill="#64748b" fill-opacity="0.055" stroke="#0f172a" stroke-width="1.7" stroke-opacity="0.82"><title>canonical unload bin projection</title></polygon>`);
    }
  }
  if(opts.polygons){
    for(const p of rows){
      if(!p.poly.length) continue;
      const color=statusColor(p.status);
      const d=p.poly.map(pt=>`${sx(pt[0]).toFixed(1)},${sy(pt[1]).toFixed(1)}`).join(" ");
      parts.push(`<polygon points="${d}" fill="${color}" fill-opacity="0.045" stroke="${color}" stroke-width="1.2" stroke-opacity="0.55"><title>ep ${esc(p.label)} unload mesh / dump bed polygon</title></polygon>`);
    }
  }
  if(opts.heading){
    const span=Math.max(Math.abs(x1-x0), Math.abs(y1-y0));
    const len=Math.max(0.45, span*0.055);
    for(const p of clean){
      if(!finite(Number(p.yaw))) continue;
      const a=Number(p.yaw)*Math.PI/180.0;
      const x2=Number(p.x)+Math.cos(a)*len;
      const y2=Number(p.y)+Math.sin(a)*len;
      parts.push(`<line x1="${sx(p.x).toFixed(1)}" y1="${sy(p.y).toFixed(1)}" x2="${sx(x2).toFixed(1)}" y2="${sy(y2).toFixed(1)}" stroke="${statusColor(p.status)}" stroke-width="2.2" opacity="0.85"><title>truck yaw ${fmt(p.yaw,1)} deg</title></line>`);
    }
  }
  for(const p of clean){
    parts.push(`<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="4.5" fill="${statusColor(p.status)}" opacity="0.9"><title>ep ${esc(p.label)} ${esc(p.status)} (${fmt(p.x,2)}, ${fmt(p.y,2)})${finite(Number(p.yaw)) ? " yaw " + fmt(p.yaw,1) + " deg" : ""}</title></circle>`);
  }
  parts.push(`<text x="${f.l+f.pw/2}" y="${f.h-4}" text-anchor="middle" font-size="11" fill="#64748b">${esc(xLabel)}</text>`);
  parts.push(`<text transform="translate(13 ${f.t+f.ph/2}) rotate(-90)" text-anchor="middle" font-size="11" fill="#64748b">${esc(yLabel)}</text>`);
  parts.push(`</svg>`); $(targetId).innerHTML=parts.join("");
}
function drawHistogram(targetId, values, unit){
  const nums=numeric(values); if(!nums.length){$(targetId).innerHTML='<div class="muted">no data</div>';return}
  let lo=Math.min(...nums), hi=Math.max(...nums); if(Math.abs(hi-lo)<1e-9){lo-=1;hi+=1}
  const bins=12, step=(hi-lo)/bins, counts=Array(bins).fill(0);
  for(const v of nums){let i=Math.floor((v-lo)/step); if(i>=bins)i=bins-1; if(i<0)i=0; counts[i]++}
  const f=chartFrame(430,260), max=Math.max(...counts,1), parts=[`<svg class="chart" viewBox="0 0 ${f.w} ${f.h}">`];
  drawAxes(parts,f,lo,hi,0,max,unit);
  counts.forEach((c,i)=>{const x=f.l+f.pw*i/bins+2, w=f.pw/bins-4, h=f.ph*c/max; parts.push(`<rect x="${x}" y="${f.t+f.ph-h}" width="${w}" height="${h}" fill="#2563eb" opacity="0.78"/>`)});
  parts.push(`</svg>`); $(targetId).innerHTML=parts.join("");
}

$("loadRunsBtn").onclick=()=>loadRuns().catch(e=>setStatus(e.message,"error"));
$("loadRunBtn").onclick=()=>loadRun().catch(e=>setStatus(e.message,"error"));
$("runSelect").onchange=()=>{$("runInput").value=$("runSelect").value};
loadRuns().then(()=>loadRun()).catch(e=>setStatus(e.message,"error"));
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
                    self.send_json({"root": root, "runs": list_dashboard_runs(root)})
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
    parser.add_argument("--export-speed-scale", type=float, default=None, help="Speed multiplier for export timestamps and recomputed dq/ddq/action.")
    parser.add_argument("--export-time-mode", default="uniform_fps", choices=["uniform_fps", "scaled_raw"], help="Time policy for export rows.")
    parser.add_argument("--export-base-fps", type=float, default=None, help="Base fps before speed scaling; defaults to camera_config frequency.")
    parser.add_argument("--export-limit", type=int, default=None, help="Limit exported episodes for smoke tests.")
    parser.add_argument("--export-overwrite", action="store_true", help="Delete and rebuild the export directory if it already exists.")
    parser.add_argument("--export-require-standard", action="store_true", help="Fail if parquet/mp4 standard export cannot be produced.")
    parser.add_argument("--export-require-vla", action="store_true", help="Fail unless image + state + action + task VLA export is ready.")
    parser.add_argument("--dashboard", action="store_true", help="Serve a local web dashboard for browsing run folders.")
    parser.add_argument("--dashboard-host", default="127.0.0.1", help="Host for --dashboard.")
    parser.add_argument("--dashboard-port", type=int, default=8765, help="Port for --dashboard.")
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
        export_time_policy = None
        if args.export_speed_scale is not None or args.export_base_fps is not None or args.export_time_mode != "uniform_fps":
            export_time_policy = {
                "speed_scale": args.export_speed_scale if args.export_speed_scale is not None else 1.0,
                "time_mode": args.export_time_mode,
                "base_fps": args.export_base_fps,
            }
        print_lerobot_export(
            run_dir,
            output_dir=args.export_dir,
            split=args.export_split,
            fps=args.export_fps,
            limit_episodes=args.export_limit,
            overwrite=args.export_overwrite,
            require_standard=args.export_require_standard,
            require_vla=args.export_require_vla,
            time_policy=export_time_policy,
        )
        did_action = True
    if args.analysis:
        print_analysis(run_dir, include_timeline=not args.no_timeline, compact=args.compact)
        did_action = True
    if not did_action:
        print_summary(run_dir)
