#!/usr/bin/env python3
"""Audit causal, deployment-available VLA observation candidates.

The analysis intentionally reads raw trajectory rows, not exported parquet
files. Observation features use only values available at or before each row.
The action target is the forward commanded joint velocity to the next row.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PHASE_NAMES = [
    "pre_dig",
    "approach_contact",
    "insert_cut",
    "pull_mid_cut",
    "curl_to_hold_material",
    "pull_exit_cut",
    "secure_load",
    "lift_carry",
    "loaded_transit",
    "unload_to_bin",
]

GROUP_NAMES = {
    "q": ["swing", "boom", "arm", "bucket"],
    "dq": ["swing_velocity", "boom_velocity", "arm_velocity", "bucket_velocity"],
    "q_error": ["swing_tracking_error", "boom_tracking_error", "arm_tracking_error", "bucket_tracking_error"],
    "previous_action": [
        "previous_swing_action",
        "previous_boom_action",
        "previous_arm_action",
        "previous_bucket_action",
    ],
    "q_cmd": ["swing_command", "boom_command", "arm_command", "bucket_command"],
    "tip_base": ["bucket_tip_base_x", "bucket_tip_base_y", "bucket_tip_height"],
    "load_base": ["bucket_load_base_x", "bucket_load_base_y", "bucket_load_height"],
    "pour_base": ["bucket_pour_base_x", "bucket_pour_base_y", "bucket_pour_height"],
    "load_height": ["bucket_load_height"],
    "carry_orientation": [
        "bucket_pour_axis_upper_horizontal",
        "bucket_pour_axis_vertical",
    ],
    "dig_delta": ["dig_delta_upper_x", "dig_delta_upper_y", "dig_delta_z"],
    "unload_delta": ["unload_delta_upper_x", "unload_delta_upper_y", "unload_delta_z"],
    "truck_heading": ["truck_heading_upper_sin", "truck_heading_upper_cos"],
    "fill": ["bucket_fill_fraction"],
    "fill_rate": ["bucket_fill_rate_fraction_per_s"],
}

BASELINE_GROUPS = [
    "q",
    "dq",
    "q_error",
    "tip_base",
    "load_base",
    "dig_delta",
    "unload_delta",
    "truck_heading",
    "fill",
    "fill_rate",
]


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def vector(value: object, length: int) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < length:
        return None
    arr = arr[:length]
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def nested(mapping: object, *keys: str) -> object:
    value = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def first_vector(length: int, *values: object) -> Optional[np.ndarray]:
    for value in values:
        found = vector(value, length)
        if found is not None:
            return found
    return None


def first_float(*values: object) -> Optional[float]:
    for value in values:
        try:
            found = float(value)
        except Exception:
            continue
        if math.isfinite(found):
            return found
    return None


def phase_index(value: object) -> Optional[int]:
    text = str(value or "").strip().lower()
    if text == "loaded_transit" or any(token in text for token in ("clearance_route_post", "staged_unload", "high_carry")):
        return 8
    if any(token in text for token in ("unload_to_bin", "unload_pre_dump_align", "unload", "dump")):
        return 9
    if any(token in text for token in ("pre_dig", "clearance_route", "travel", "align")):
        return 0
    if "approach_contact" in text or ("approach" in text and "contact" in text):
        return 1
    if "insert" in text:
        return 2
    if "pull_mid" in text:
        return 3
    if "curl" in text:
        return 4
    if "pull_exit" in text:
        return 5
    if "secure" in text:
        return 6
    if "lift" in text or "carry" in text:
        return 7
    return None


def wrap_delta(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def joint_delta(q1: np.ndarray, q0: np.ndarray) -> np.ndarray:
    out = np.asarray(q1 - q0, dtype=np.float64)
    out[0] = wrap_delta(out[0])
    return out


def world_to_frame_xy(delta_xy: Sequence[float], yaw: float) -> np.ndarray:
    x, y = float(delta_xy[0]), float(delta_xy[1])
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.asarray([c * x + s * y, -s * x + c * y], dtype=np.float64)


def resolve_episode_file(pool: Path, row: dict, key: str, filename: str) -> Optional[Path]:
    raw = str(row.get(key) or "").strip()
    candidates: List[Path] = []
    if raw:
        candidates.append(Path(raw))
    transferred = str(row.get("transferred_episode_dir") or "").strip()
    if transferred:
        candidates.append(Path(transferred) / filename)
        candidates.append(pool / "episodes" / Path(transferred.replace("\\", "/")).name / filename)
    if raw:
        parent_name = Path(raw.replace("\\", "/")).parent.name
        if parent_name:
            candidates.append(pool / "episodes" / parent_name / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def episode_scene(row: dict) -> Tuple[dict, dict]:
    scene = row.get("scene_randomization")
    scene = scene if isinstance(scene, dict) else {}
    candidate = scene.get("candidate")
    applied = scene.get("applied")
    return (
        candidate if isinstance(candidate, dict) else {},
        applied if isinstance(applied, dict) else {},
    )


def episode_context(row: dict) -> Optional[dict]:
    candidate, applied = episode_scene(row)
    unload = first_vector(
        3,
        row.get("unload_landing_xyz"),
        applied.get("unload_landing_xyz"),
        nested(applied, "scene_context", "unload_point"),
        row.get("unload_point_xyz"),
    )
    truck_yaw_deg = first_float(applied.get("truck_yaw_deg"), candidate.get("truck_yaw_deg"))
    if unload is None or truck_yaw_deg is None:
        return None
    return {
        "unload": unload,
        "truck_yaw": math.radians(truck_yaw_deg),
    }


def camera_age_ms(sample: dict) -> Optional[float]:
    camera = sample.get("observation.camera")
    if not isinstance(camera, dict):
        return None
    return first_float(camera.get("frame_age_ms"))


def raw_row(sample: dict, context: dict) -> Optional[dict]:
    state = first_vector(14, sample.get("observation.state"), sample.get("obs.state"))
    q = first_vector(4, sample.get("obs.q"), None if state is None else state[3:7])
    q_cmd = first_vector(4, sample.get("obs.q_cmd"), sample.get("goal.q"))
    dq = first_vector(4, sample.get("obs.dq"))
    effort = first_vector(4, sample.get("observation.effort"))
    tip = first_vector(3, sample.get("bucket.tip"), None if state is None else state[8:11])
    load = first_vector(3, sample.get("bucket.load"), None if state is None else state[11:14])
    pour = first_vector(3, sample.get("bucket.pour"))
    target = first_vector(3, sample.get("target"))
    pidx = phase_index(sample.get("phase") or sample.get("label"))
    t = first_float(sample.get("t"))
    if any(value is None for value in (state, q, q_cmd, dq, effort, tip, load, target, t, pidx)):
        return None
    q_error = first_vector(4, sample.get("obs.q_err"))
    if q_error is None:
        q_error = joint_delta(q_cmd, q)
    previous_action = first_vector(4, sample.get("action"))
    if previous_action is None:
        previous_action = np.zeros(4, dtype=np.float64)
    return {
        "t": float(t),
        "state": state,
        "q": q,
        "q_cmd": q_cmd,
        "dq": dq,
        "q_error": q_error,
        "previous_action": previous_action,
        "effort": effort,
        "tip": tip,
        "load": load,
        "pour": pour,
        "target": target,
        "unload": context["unload"],
        "truck_yaw": float(context["truck_yaw"]),
        "phase_index": int(pidx),
        "bucket_count": max(0.0, float(state[7])),
        "camera_age_ms": camera_age_ms(sample),
    }


def provisional_capacity(episodes: Sequence[List[dict]]) -> float:
    maxima = [max((row["bucket_count"] for row in rows), default=0.0) for rows in episodes]
    positive = np.asarray([value for value in maxima if value > 0.0], dtype=np.float64)
    if positive.size == 0:
        return 1.0
    return max(1.0, float(np.percentile(positive, 95.0)))


def add_causal_features(
    rows: List[dict],
    capacity: float,
    step_seconds: float,
    tau_seconds: float = 0.4,
) -> None:
    smoothed: Optional[float] = None
    previous_smoothed: Optional[float] = None
    previous_q: Optional[np.ndarray] = None
    previous_q_cmd: Optional[np.ndarray] = None
    for row in rows:
        current = float(row["bucket_count"]) / max(1.0, float(capacity))
        current = float(np.clip(current, 0.0, 1.5))
        if smoothed is None:
            smoothed = current
            rate = 0.0
        else:
            alpha = 1.0 - math.exp(-step_seconds / max(1.0e-4, float(tau_seconds)))
            previous_smoothed = smoothed
            smoothed = smoothed + alpha * (current - smoothed)
            rate = (smoothed - previous_smoothed) / step_seconds
        if previous_q is None:
            causal_dq = np.zeros(4, dtype=np.float64)
        else:
            causal_dq = joint_delta(row["q"], previous_q) / step_seconds
        if previous_q_cmd is None:
            previous_action = np.zeros(4, dtype=np.float64)
        else:
            previous_action = joint_delta(row["q_cmd"], previous_q_cmd) / step_seconds
        row["fill_fraction"] = current
        row["fill_rate"] = float(rate)
        row["causal_dq"] = causal_dq
        row["causal_previous_action"] = previous_action
        previous_q = row["q"].copy()
        previous_q_cmd = row["q_cmd"].copy()


def build_groups(row: dict) -> Dict[str, np.ndarray]:
    state = row["state"]
    base_xy = state[:2]
    base_yaw = float(state[2])
    q = row["q"]
    tip = row["tip"]
    load = row["load"]
    pour = row["pour"] if row["pour"] is not None else load
    upper_yaw = base_yaw + float(q[0])

    tip_base_xy = world_to_frame_xy(tip[:2] - base_xy, base_yaw)
    load_base_xy = world_to_frame_xy(load[:2] - base_xy, base_yaw)
    pour_base_xy = world_to_frame_xy(pour[:2] - base_xy, base_yaw)
    pour_axis_xy = world_to_frame_xy(pour[:2] - load[:2], upper_yaw)
    pour_axis_horizontal = float(pour_axis_xy[0])
    pour_axis_vertical = float(pour[2] - load[2])
    pour_axis_norm = max(1.0e-8, math.hypot(pour_axis_horizontal, pour_axis_vertical))
    dig_delta_xy = world_to_frame_xy(row["target"][:2] - tip[:2], upper_yaw)
    unload_delta_xy = world_to_frame_xy(row["unload"][:2] - load[:2], upper_yaw)
    truck_relative = float(row["truck_yaw"]) - upper_yaw
    return {
        "q": q,
        "dq": row["causal_dq"],
        "q_error": row["q_error"],
        "previous_action": row["causal_previous_action"],
        "q_cmd": row["q_cmd"],
        "tip_base": np.asarray([tip_base_xy[0], tip_base_xy[1], tip[2]], dtype=np.float64),
        "load_base": np.asarray([load_base_xy[0], load_base_xy[1], load[2]], dtype=np.float64),
        "pour_base": np.asarray([pour_base_xy[0], pour_base_xy[1], pour[2]], dtype=np.float64),
        "load_height": np.asarray([load[2]], dtype=np.float64),
        "carry_orientation": np.asarray(
            [pour_axis_horizontal / pour_axis_norm, pour_axis_vertical / pour_axis_norm],
            dtype=np.float64,
        ),
        "dig_delta": np.asarray([dig_delta_xy[0], dig_delta_xy[1], row["target"][2] - tip[2]], dtype=np.float64),
        "unload_delta": np.asarray(
            [unload_delta_xy[0], unload_delta_xy[1], row["unload"][2] - load[2]],
            dtype=np.float64,
        ),
        "truck_heading": np.asarray([math.sin(truck_relative), math.cos(truck_relative)], dtype=np.float64),
        "fill": np.asarray([row["fill_fraction"]], dtype=np.float64),
        "fill_rate": np.asarray([row["fill_rate"]], dtype=np.float64),
    }


def feature_matrix(records: Sequence[dict], groups: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    names = [name for group in groups for name in GROUP_NAMES[group]]
    matrix = np.stack(
        [np.concatenate([record["groups"][group] for group in groups]) for record in records],
        axis=0,
    )
    return matrix, names


def ridge_fold_metrics(
    records: Sequence[dict],
    groups: Sequence[str],
    fold_count: int = 5,
    alpha: float = 1.0,
) -> dict:
    if not records:
        return {"available": False, "reason": "no_records"}
    state_x, names = feature_matrix(records, groups)
    effort = np.stack([record["effort"] for record in records], axis=0)
    phase = np.zeros((len(records), len(PHASE_NAMES)), dtype=np.float64)
    for i, record in enumerate(records):
        phase[i, int(record["phase_index"])] = 1.0
    x = np.concatenate([state_x, effort, phase], axis=1)
    y = np.stack([record["target_action"] for record in records], axis=0)
    episode_ids = np.asarray([int(record["episode_id"]) for record in records], dtype=np.int64)
    unique_episodes = np.asarray(sorted(set(int(value) for value in episode_ids.tolist())), dtype=np.int64)
    np.random.default_rng(20260718).shuffle(unique_episodes)
    folds: List[dict] = []
    phase_scores: Dict[int, List[float]] = {index: [] for index in range(len(PHASE_NAMES))}

    for fold in range(min(fold_count, len(unique_episodes))):
        test_episodes = {
            int(episode)
            for index, episode in enumerate(unique_episodes.tolist())
            if index % fold_count == fold
        }
        test_mask = np.asarray([episode in test_episodes for episode in episode_ids], dtype=bool)
        train_mask = ~test_mask
        if int(np.count_nonzero(train_mask)) < 20 or int(np.count_nonzero(test_mask)) < 5:
            continue
        x_train, x_test = x[train_mask], x[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        x_mean = x_train.mean(axis=0)
        x_std = x_train.std(axis=0)
        x_std[x_std < 1.0e-8] = 1.0
        y_mean = y_train.mean(axis=0)
        y_std = y_train.std(axis=0)
        y_std[y_std < 1.0e-8] = 1.0
        xn = (x_train - x_mean) / x_std
        xt = (x_test - x_mean) / x_std
        yn = (y_train - y_mean) / y_std
        design = np.concatenate([xn, np.ones((xn.shape[0], 1), dtype=np.float64)], axis=1)
        design_test = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float64)], axis=1)
        reg = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
        reg[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + reg, design.T @ yn)
        pred = (design_test @ weights) * y_std + y_mean
        rmse = np.sqrt(np.mean((pred - y_test) ** 2, axis=0))
        nrmse = rmse / y_std
        fold_phase_scores: Dict[str, float] = {}
        test_phase_indices = np.asarray(
            [int(record["phase_index"]) for index, record in enumerate(records) if test_mask[index]],
            dtype=np.int64,
        )
        for phase_idx, phase_name in enumerate(PHASE_NAMES):
            phase_mask = test_phase_indices == phase_idx
            if int(np.count_nonzero(phase_mask)) < 5:
                continue
            phase_rmse = np.sqrt(np.mean((pred[phase_mask] - y_test[phase_mask]) ** 2, axis=0))
            phase_nrmse = float(np.mean(phase_rmse / y_std))
            phase_scores[phase_idx].append(phase_nrmse)
            fold_phase_scores[phase_name] = phase_nrmse
        folds.append(
            {
                "fold": fold,
                "test_episodes": len(test_episodes),
                "rows": int(y_test.shape[0]),
                "nrmse": nrmse.tolist(),
                "mean_nrmse": float(np.mean(nrmse)),
                "phase_mean_nrmse": fold_phase_scores,
            }
        )
    return {
        "available": bool(folds),
        "state_dim": len(names),
        "state_names": names,
        "input_dim_with_effort_and_phase_one_hot": len(names) + 4 + len(PHASE_NAMES),
        "folds": folds,
        "mean_nrmse": float(np.mean([fold["mean_nrmse"] for fold in folds])) if folds else None,
        "median_nrmse": float(np.median([fold["mean_nrmse"] for fold in folds])) if folds else None,
        "phase_mean_nrmse": {
            PHASE_NAMES[index]: float(np.mean(values))
            for index, values in phase_scores.items()
            if values
        },
    }


def distribution(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p01": float(np.percentile(arr, 1.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p95": float(np.percentile(arr, 95.0)),
        "p99": float(np.percentile(arr, 99.0)),
        "max": float(np.max(arr)),
    }


def redundancy_report(x: np.ndarray, names: Sequence[str]) -> dict:
    std = np.std(x, axis=0)
    valid = std > 1.0e-10
    corr = np.eye(x.shape[1], dtype=np.float64)
    if int(np.count_nonzero(valid)) > 1:
        corr_valid = np.corrcoef(x[:, valid], rowvar=False)
        valid_indices = np.flatnonzero(valid)
        corr[np.ix_(valid_indices, valid_indices)] = corr_valid
    pairs: List[dict] = []
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            value = abs(float(corr[i, j]))
            if math.isfinite(value) and value >= 0.95:
                pairs.append({"a": names[i], "b": names[j], "abs_pearson": value})
    pairs.sort(key=lambda item: item["abs_pearson"], reverse=True)
    return {
        "near_zero_variance": [names[i] for i, value in enumerate(std) if value <= 1.0e-10],
        "high_correlation_pairs": pairs[:50],
    }


def write_markdown(path: Path, report: dict) -> None:
    models = report["ridge_models"]
    lines = [
        "# VLA Observation Candidate Analysis",
        "",
        f"Episodes: {report['episodes_used']}",
        f"Causal rows: {report['causal_rows']}",
        f"Provisional bucket capacity: {report['provisional_bucket_capacity_particles']:.3f} particles",
        f"Screening sample rate: {report['analysis_fps']:.3f} Hz",
        "",
        "## Time And Synchronization",
        "",
        f"- Raw dt p50: {report['raw_dt_ms'].get('p50', 0.0):.3f} ms",
        f"- Camera age p50/p95/p99: "
        f"{report['camera_age_ms'].get('p50', 0.0):.3f} / "
        f"{report['camera_age_ms'].get('p95', 0.0):.3f} / "
        f"{report['camera_age_ms'].get('p99', 0.0):.3f} ms",
        "- Observation derivatives are causal; action labels use the next commanded position.",
        "",
        "## Linear Grouped-Episode Screening",
        "",
        "| Candidate | State dim | Mean NRMSE | Median NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, value in sorted(models.items(), key=lambda item: float(item[1].get("median_nrmse") or 999.0)):
        lines.append(
            f"| {name} | {value.get('state_dim')} | "
            f"{float(value.get('mean_nrmse') or 0.0):.5f} | "
            f"{float(value.get('median_nrmse') or 0.0):.5f} |"
        )
    lines.extend(
        [
            "",
            "This ridge result is a screening metric, not proof of final VLA rollout quality.",
            "The final contract still requires nonlinear episode-level ablation and simulator rollout.",
            "",
            "## Per-Phase Screening",
            "",
            "| Candidate | " + " | ".join(PHASE_NAMES) + " |",
            "| --- | " + " | ".join(["---:"] * len(PHASE_NAMES)) + " |",
        ]
    )
    for name, value in sorted(models.items(), key=lambda item: float(item[1].get("median_nrmse") or 999.0)):
        phase_values = value.get("phase_mean_nrmse") or {}
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{float(phase_values.get(phase_name, float('nan'))):.3f}"
                for phase_name in PHASE_NAMES
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Baseline 28D",
            "",
        ]
    )
    for index, name in enumerate(report["baseline_state_names"]):
        lines.append(f"{index}. `{name}`")
    lines.extend(["", "## Effort Robust Statistics", ""])
    for index, stats in enumerate(report["effort_abs_by_joint"]):
        lines.append(
            f"- {GROUP_NAMES['q'][index]} absolute effort p50/p95/p99/max: "
            f"{stats.get('p50', 0.0):.3f} / {stats.get('p95', 0.0):.3f} / "
            f"{stats.get('p99', 0.0):.3f} / {stats.get('max', 0.0):.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(pool: Path, max_episodes: Optional[int] = None, analysis_fps: float = 2.0) -> dict:
    index_path = pool / "trainable_episodes.jsonl"
    episode_index = read_jsonl(index_path)
    if max_episodes is not None:
        episode_index = episode_index[: max(0, int(max_episodes))]
    episodes: List[List[dict]] = []
    skipped: Dict[str, int] = {}

    for episode_id, episode_row in enumerate(episode_index):
        trajectory_path = resolve_episode_file(pool, episode_row, "trajectory", "trajectory.jsonl")
        context = episode_context(episode_row)
        if trajectory_path is None or context is None:
            key = "missing_trajectory" if trajectory_path is None else "missing_context"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        raw_samples = read_jsonl(trajectory_path)
        parsed = [raw_row(sample, context) for sample in raw_samples]
        rows = [row for row in parsed if row is not None]
        rows.sort(key=lambda row: float(row["t"]))
        if len(rows) < 2:
            skipped["too_few_rows"] = skipped.get("too_few_rows", 0) + 1
            continue
        for row in rows:
            row["episode_id"] = episode_id
        episodes.append(rows)

    analysis_fps = max(0.1, float(analysis_fps))
    step_seconds = 1.0 / analysis_fps
    capacity = provisional_capacity(episodes)
    for rows in episodes:
        add_causal_features(rows, capacity, step_seconds=step_seconds)

    records: List[dict] = []
    raw_dts: List[float] = []
    camera_ages: List[float] = []
    for rows in episodes:
        for index in range(len(rows) - 1):
            row = rows[index]
            next_row = rows[index + 1]
            dt = float(next_row["t"]) - float(row["t"])
            if not math.isfinite(dt) or dt <= 1.0e-4:
                continue
            target_action = joint_delta(next_row["q_cmd"], row["q_cmd"]) / step_seconds
            groups = build_groups(row)
            record = dict(row)
            record["groups"] = groups
            record["target_action"] = target_action
            records.append(record)
            raw_dts.append(dt * 1000.0)
            if row["camera_age_ms"] is not None:
                camera_ages.append(float(row["camera_age_ms"]))

    model_groups = {
        "baseline_q_error_28d": BASELINE_GROUPS,
        "replace_q_error_with_previous_action_28d": [
            "q",
            "dq",
            "previous_action",
            "tip_base",
            "load_base",
            "dig_delta",
            "unload_delta",
            "truck_heading",
            "fill",
            "fill_rate",
        ],
        "replace_q_error_with_q_cmd_28d": [
            "q",
            "dq",
            "q_cmd",
            "tip_base",
            "load_base",
            "dig_delta",
            "unload_delta",
            "truck_heading",
            "fill",
            "fill_rate",
        ],
        "replace_load_pose_with_pour_pose_28d": [
            "q",
            "dq",
            "q_error",
            "tip_base",
            "pour_base",
            "dig_delta",
            "unload_delta",
            "truck_heading",
            "fill",
            "fill_rate",
        ],
        "retention_geometry_28d": [
            "q",
            "dq",
            "q_error",
            "tip_base",
            "carry_orientation",
            "load_height",
            "dig_delta",
            "unload_delta",
            "truck_heading",
            "fill",
            "fill_rate",
        ],
        "dynamics_memory_28d": [
            "q",
            "dq",
            "q_error",
            "previous_action",
            "carry_orientation",
            "dig_delta",
            "unload_delta",
            "truck_heading",
            "fill",
            "fill_rate",
        ],
    }
    ridge_models = {name: ridge_fold_metrics(records, groups) for name, groups in model_groups.items()}
    baseline_x, baseline_names = feature_matrix(records, BASELINE_GROUPS)
    effort = np.stack([record["effort"] for record in records], axis=0) if records else np.empty((0, 4))
    report = {
        "schema": "vla_observation_candidate_analysis_v1",
        "pool": str(pool),
        "episodes_indexed": len(episode_index),
        "episodes_used": len(episodes),
        "causal_rows": len(records),
        "analysis_fps": float(analysis_fps),
        "analysis_step_seconds": float(step_seconds),
        "skipped_episodes": skipped,
        "provisional_bucket_capacity_particles": capacity,
        "raw_dt_ms": distribution(np.asarray(raw_dts, dtype=np.float64)),
        "camera_age_ms": distribution(np.asarray(camera_ages, dtype=np.float64)),
        "baseline_state_names": baseline_names,
        "baseline_feature_distributions": {
            name: distribution(baseline_x[:, index]) for index, name in enumerate(baseline_names)
        },
        "baseline_redundancy": redundancy_report(baseline_x, baseline_names),
        "effort_abs_by_joint": [
            distribution(np.abs(effort[:, index])) for index in range(4)
        ]
        if effort.size
        else [],
        "target_action_by_joint": [
            distribution(np.asarray([record["target_action"][index] for record in records], dtype=np.float64))
            for index in range(4)
        ],
        "ridge_models": ridge_models,
        "notes": [
            "All observation candidates are causal and available from simulator runtime state.",
            "Causal derivatives and forward action labels use the explicit screening sample rate, not wall-clock dt.",
            "Phase one-hot and measured effort are included in every ridge screening model but do not count toward 28D state.",
            "The provisional capacity is for screening only and must become a versioned physical/configuration constant.",
            "Ridge NRMSE is not sufficient to claim final VLA optimality; nonlinear ablation and rollout remain required.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pool",
        default=str(Path(__file__).resolve().parents[1] / "excavator_auto_dataset" / ".dashboard_success"),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-md", default="")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--analysis-fps", type=float, default=2.0)
    args = parser.parse_args()

    pool = Path(args.pool).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else Path(__file__).resolve().parents[1] / "docs" / "analysis" / "vla_observation_candidate_report.json"
    )
    output_md = (
        Path(args.output_md).expanduser().resolve()
        if args.output_md
        else Path(__file__).resolve().parents[1] / "docs" / "analysis" / "vla_observation_candidate_report.md"
    )
    report = analyze(pool, max_episodes=args.max_episodes, analysis_fps=args.analysis_fps)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(output_md, report)
    print(json.dumps(
        {
            "episodes_used": report["episodes_used"],
            "causal_rows": report["causal_rows"],
            "capacity": report["provisional_bucket_capacity_particles"],
            "camera_age_ms": report["camera_age_ms"],
            "models": {
                key: value.get("mean_nrmse") for key, value in report["ridge_models"].items()
            },
            "output_json": str(output_json),
            "output_md": str(output_md),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
