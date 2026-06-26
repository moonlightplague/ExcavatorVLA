import json
import math
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


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
    args = parser.parse_args()
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
