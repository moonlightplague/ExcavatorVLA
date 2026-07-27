#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = ("swing", "boom", "arm", "bucket")
STAGE_NAMES = (
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
)
DIRECTION_LABELS = (
    "front",
    "front-left",
    "left",
    "rear-left",
    "rear",
    "rear-right",
    "right",
    "front-right",
)

DEFAULT_TRACE = Path(
    "/root/gpufree-data/excavator_logs/"
    "smolvla_policy_steps_trace.jsonl"
)
DEFAULT_CHUNKS = Path(
    "/root/gpufree-data/excavator_logs/"
    "smolvla_action_stage_chunks.jsonl"
)
DEFAULT_PRIOR = Path(
    "/root/gpufree-data/excavator_stage_action_analysis/"
    "stage_action_prior_training.json"
)
DEFAULT_OUTPUT = Path(
    "/root/gpufree-data/excavator_200step_analysis"
)

PROMPT_PATTERN = re.compile(
    r"Dig soil from the sand pile near "
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*\),\s*"
    r"([a-z-]+)\s+of the excavator,\s*"
    r"and dump it into the truck bed near "
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*\),\s*"
    r"([a-z-]+)\s+of the excavator\.",
    re.IGNORECASE,
)


def load_jsonl(path: Path, expected_type: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if expected_type is None or row.get("type") == expected_type:
                rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No usable records found in {path}"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def finite_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def world_from_local_xy(
    base_x: float,
    base_y: float,
    base_yaw: float,
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    cosine = math.cos(base_yaw)
    sine = math.sin(base_yaw)
    dx = cosine * local_x - sine * local_y
    dy = sine * local_x + cosine * local_y
    return base_x + dx, base_y + dy


def direction_label(
    point_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    base_yaw: float,
) -> str:
    dx = point_xy[0] - origin_xy[0]
    dy = point_xy[1] - origin_xy[1]
    if abs(dx) + abs(dy) < 1e-9:
        return "nearby"

    # This reproduces the simulation prompt's current convention.
    forward_heading = base_yaw + math.pi * 0.5
    relative_angle = math.atan2(dy, dx) - forward_heading
    index = int(
        math.floor(
            (
                math.degrees(relative_angle)
                + 22.5
            )
            % 360.0
            / 45.0
        )
    )
    return DIRECTION_LABELS[index % len(DIRECTION_LABELS)]


def parse_prompt(task_text: str) -> dict[str, Any]:
    match = PROMPT_PATTERN.fullmatch(task_text.strip())
    if match is None:
        return {
            "parsed": False,
            "task_text": task_text,
        }

    return {
        "parsed": True,
        "task_text": task_text,
        "prompt_sand_x": float(match.group(1)),
        "prompt_sand_y": float(match.group(2)),
        "prompt_sand_direction": match.group(3).lower(),
        "prompt_unload_x": float(match.group(4)),
        "prompt_unload_y": float(match.group(5)),
        "prompt_unload_direction": match.group(6).lower(),
    }


def compress_stage_sequence(stage_ids: np.ndarray) -> list[int]:
    compressed: list[int] = []
    for value in stage_ids.tolist():
        value = int(value)
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def stage_progression_summary(stage_ids: np.ndarray) -> dict[str, Any]:
    compressed = compress_stage_sequence(stage_ids)
    transitions = list(zip(compressed[:-1], compressed[1:]))

    backward = []
    forward_skips = []
    allowed_cycle_resets = []

    for old, new in transitions:
        if old == 9 and new == 0:
            allowed_cycle_resets.append((old, new))
        elif new < old:
            backward.append((old, new))
        elif new > old + 1:
            forward_skips.append((old, new))

    return {
        "compressed_stage_ids": compressed,
        "compressed_stage_names": [
            STAGE_NAMES[value]
            if 0 <= value < len(STAGE_NAMES)
            else f"unknown_{value}"
            for value in compressed
        ],
        "num_stage_segments": len(compressed),
        "num_backward_transitions": len(backward),
        "backward_transitions": backward,
        "num_forward_skips": len(forward_skips),
        "forward_skips": forward_skips,
        "num_allowed_cycle_resets": len(allowed_cycle_resets),
    }


def load_prior(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"[WARN] prior file not found: {path}")
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))

    raw_stage_names = payload.get("stage_names")
    source_stage_names: tuple[str, ...] | None = None
    if isinstance(raw_stage_names, dict):
        try:
            source_stage_names = tuple(
                str(raw_stage_names[str(stage_id)])
                if str(stage_id) in raw_stage_names
                else str(raw_stage_names[stage_id])
                for stage_id in range(len(STAGE_NAMES))
            )
        except (KeyError, TypeError):
            source_stage_names = None
    elif isinstance(raw_stage_names, (list, tuple)):
        source_stage_names = tuple(str(name) for name in raw_stage_names)

    labels_match_canonical = source_stage_names == STAGE_NAMES
    if source_stage_names is None:
        print(
            "[WARN] prior has no complete stage_names metadata; "
            "interpreting its rows by canonical v4 stage ID"
        )
    elif not labels_match_canonical:
        print(
            "[WARN] prior stage_names metadata is not canonical v4; "
            "interpreting its rows by numeric stage ID and reporting "
            "canonical v4 names"
        )

    direction = np.asarray(
        payload.get("direction", []),
        dtype=np.float64,
    )
    weight = np.asarray(
        payload.get("weight", []),
        dtype=np.float64,
    )
    reliable = np.asarray(
        payload.get("reliable", []),
        dtype=bool,
    )
    scale = np.asarray(
        payload.get("action_scale", []),
        dtype=np.float64,
    ).reshape(-1)

    if direction.shape != (len(STAGE_NAMES), len(ACTION_NAMES)):
        print(
            "[WARN] unexpected prior direction shape:",
            direction.shape,
        )
        return None
    if weight.shape != direction.shape:
        weight = np.ones_like(direction)
    if reliable.shape != direction.shape:
        reliable = direction != 0
    if scale.size < len(ACTION_NAMES):
        scale = np.ones(len(ACTION_NAMES), dtype=np.float64)

    strong = reliable & (weight >= 0.8) & (direction != 0)
    return {
        "direction": direction[:, : len(ACTION_NAMES)],
        "strong": strong[:, : len(ACTION_NAMES)],
        "scale": np.maximum(scale[: len(ACTION_NAMES)], 1e-8),
        "source_stage_labels_present": source_stage_names is not None,
        "source_stage_labels_match_canonical": labels_match_canonical,
    }


def prior_action_diagnostics(
    stages: np.ndarray,
    actions: np.ndarray,
    prior: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if prior is None:
        return {
            "available": False,
        }, []

    rows = []
    all_nonviolating = []
    low_swing_motion = []

    for step, (stage_id, action) in enumerate(
        zip(stages.tolist(), actions, strict=True)
    ):
        if not 0 <= int(stage_id) < len(STAGE_NAMES):
            continue

        stage_id = int(stage_id)
        for action_index, action_name in enumerate(ACTION_NAMES):
            if not prior["strong"][stage_id, action_index]:
                continue

            direction = float(
                prior["direction"][stage_id, action_index]
            )
            value = float(action[action_index])
            nonviolating = direction * value >= 0.0
            all_nonviolating.append(float(nonviolating))
            rows.append(
                {
                    "policy_step": step,
                    "stage_id": stage_id,
                    "stage_name": STAGE_NAMES[stage_id],
                    "action_name": action_name,
                    "expected_direction": direction,
                    "action_value": value,
                    "direction_nonviolating": int(nonviolating),
                }
            )

        if 1 <= stage_id <= 7:
            normalized_abs_swing = (
                abs(float(action[0]))
                / float(prior["scale"][0])
            )
            low_swing_motion.append(
                float(normalized_abs_swing <= 0.10)
            )

    by_stage: dict[str, Any] = {}
    for stage_id in range(len(STAGE_NAMES)):
        stage_rows = [
            row for row in rows
            if row["stage_id"] == stage_id
        ]
        if not stage_rows:
            continue
        by_stage[STAGE_NAMES[stage_id]] = {
            "num_checked_stage_action_pairs": len(stage_rows),
            "direction_nonviolation_rate": finite_mean(
                [row["direction_nonviolating"] for row in stage_rows]
            ),
        }

    return {
        "available": True,
        "source_stage_labels_present": prior[
            "source_stage_labels_present"
        ],
        "source_stage_labels_match_canonical": prior[
            "source_stage_labels_match_canonical"
        ],
        "stage_row_interpretation": "canonical_v4_numeric_stage_id",
        "num_checked_stage_action_pairs": len(all_nonviolating),
        "overall_direction_nonviolation_rate": finite_mean(
            all_nonviolating
        ),
        "low_swing_motion_stage_1_to_7_rate": finite_mean(
            low_swing_motion
        ),
        "by_stage": by_stage,
    }, rows


def action_summary_rows(
    steps: np.ndarray,
    stages: np.ndarray,
    actions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for action_index, action_name in enumerate(ACTION_NAMES):
        values = actions[:, action_index]
        delta = np.diff(values)
        rows.append(
            {
                "scope": "overall",
                "stage_id": "",
                "stage_name": "all",
                "action_name": action_name,
                "count": len(values),
                "mean": float(np.mean(values)),
                "mean_abs": float(np.mean(np.abs(values))),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "near_zero_fraction": float(np.mean(np.abs(values) <= 1e-4)),
                "mean_abs_step_delta": (
                    float(np.mean(np.abs(delta)))
                    if len(delta)
                    else 0.0
                ),
                "max_abs_step_delta": (
                    float(np.max(np.abs(delta)))
                    if len(delta)
                    else 0.0
                ),
            }
        )

    for stage_id in sorted(set(int(value) for value in stages.tolist())):
        mask = stages == stage_id
        if not np.any(mask):
            continue

        for action_index, action_name in enumerate(ACTION_NAMES):
            values = actions[mask, action_index]
            rows.append(
                {
                    "scope": "stage",
                    "stage_id": stage_id,
                    "stage_name": (
                        STAGE_NAMES[stage_id]
                        if 0 <= stage_id < len(STAGE_NAMES)
                        else f"unknown_{stage_id}"
                    ),
                    "action_name": action_name,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "negative_fraction": float(np.mean(values < 0)),
                    "near_zero_fraction": float(
                        np.mean(np.abs(values) <= 1e-4)
                    ),
                    "mean_abs_step_delta": "",
                    "max_abs_step_delta": "",
                }
            )

    return rows


def temporal_ensemble_summary(
    step_rows: list[dict[str, Any]],
    executed_actions: np.ndarray,
) -> dict[str, Any]:
    """Summarize continuity changes from logged overlapping predictions."""
    if not step_rows or not all(
        isinstance(row.get("latest_action"), dict)
        and isinstance(row.get("temporal_ensemble"), dict)
        for row in step_rows
    ):
        return {"available": False}

    latest_actions = np.asarray(
        [
            [
                float(row["latest_action"][action_name])
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    metadata = [row["temporal_ensemble"] for row in step_rows]
    widths = [int(row.get("configured_width", 1)) for row in metadata]
    decays = [float(row.get("decay", 0.0)) for row in metadata]
    prediction_counts = [
        int(row.get("num_predictions", 0)) for row in metadata
    ]
    configured_width = widths[0]
    full_width_steps = [
        int(step_rows[index]["policy_step"])
        for index, count in enumerate(prediction_counts)
        if count == configured_width
    ]

    by_action = {}
    for action_index, action_name in enumerate(ACTION_NAMES):
        latest = latest_actions[:, action_index]
        executed = executed_actions[:, action_index]
        latest_delta = float(np.mean(np.abs(np.diff(latest))))
        executed_delta = float(np.mean(np.abs(np.diff(executed))))
        latest_flips = int(np.sum(np.sign(latest[1:]) != np.sign(latest[:-1])))
        executed_flips = int(
            np.sum(np.sign(executed[1:]) != np.sign(executed[:-1]))
        )
        by_action[action_name] = {
            "latest_prediction_sign_flips": latest_flips,
            "executed_ensemble_sign_flips": executed_flips,
            "sign_flip_reduction": latest_flips - executed_flips,
            "latest_prediction_mean_abs_step_delta": latest_delta,
            "executed_ensemble_mean_abs_step_delta": executed_delta,
            "mean_abs_step_delta_reduction_fraction": (
                (latest_delta - executed_delta) / latest_delta
                if latest_delta > 0.0
                else 0.0
            ),
        }

    return {
        "available": True,
        "configured_width": configured_width,
        "decay": decays[0],
        "configuration_consistent": (
            len(set(widths)) == 1 and len(set(decays)) == 1
        ),
        "minimum_predictions": min(prediction_counts),
        "maximum_predictions": max(prediction_counts),
        "num_full_width_steps": len(full_width_steps),
        "first_full_width_step": (
            full_width_steps[0] if full_width_steps else None
        ),
        "by_action": by_action,
    }


def build_prompt_diagnostics(
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for row in step_rows:
        step = int(row["policy_step"])
        state = np.asarray(
            row.get("observation_state", []),
            dtype=np.float64,
        ).reshape(-1)
        parsed = parse_prompt(str(row.get("task_text", "")))

        diagnostic: dict[str, Any] = {
            "policy_step": step,
            "prompt_parsed": int(bool(parsed.get("parsed"))),
            "task_text": str(row.get("task_text", "")),
        }

        if state.size != 27:
            diagnostic["state27_valid"] = 0
            diagnostic["state_dim"] = int(state.size)
            output.append(diagnostic)
            continue

        diagnostic["state27_valid"] = 1
        diagnostic["state_dim"] = 27

        base_x = float(state[0])
        base_y = float(state[1])
        base_yaw = float(state[2])

        state_dig_world = world_from_local_xy(
            base_x,
            base_y,
            base_yaw,
            float(state[18]),
            float(state[19]),
        )
        state_unload_world = world_from_local_xy(
            base_x,
            base_y,
            base_yaw,
            float(state[21]),
            float(state[22]),
        )

        diagnostic.update(
            {
                "base_x": base_x,
                "base_y": base_y,
                "base_yaw": base_yaw,
                "state_dig_world_x": state_dig_world[0],
                "state_dig_world_y": state_dig_world[1],
                "state_dig_world_z": float(state[20]),
                "state_unload_world_x": state_unload_world[0],
                "state_unload_world_y": state_unload_world[1],
                "state_unload_world_z": float(state[23]),
                "state_dig_direction": direction_label(
                    state_dig_world,
                    (base_x, base_y),
                    base_yaw,
                ),
                "state_unload_direction": direction_label(
                    state_unload_world,
                    (base_x, base_y),
                    base_yaw,
                ),
            }
        )

        if not parsed.get("parsed"):
            output.append(diagnostic)
            continue

        prompt_sand = (
            float(parsed["prompt_sand_x"]),
            float(parsed["prompt_sand_y"]),
        )
        prompt_unload = (
            float(parsed["prompt_unload_x"]),
            float(parsed["prompt_unload_y"]),
        )

        prompt_sand_expected_direction = direction_label(
            prompt_sand,
            (base_x, base_y),
            base_yaw,
        )
        prompt_unload_expected_direction = direction_label(
            prompt_unload,
            (base_x, base_y),
            base_yaw,
        )

        diagnostic.update(
            {
                "prompt_sand_x": prompt_sand[0],
                "prompt_sand_y": prompt_sand[1],
                "prompt_sand_direction": parsed[
                    "prompt_sand_direction"
                ],
                "prompt_unload_x": prompt_unload[0],
                "prompt_unload_y": prompt_unload[1],
                "prompt_unload_direction": parsed[
                    "prompt_unload_direction"
                ],
                "prompt_sand_expected_direction": (
                    prompt_sand_expected_direction
                ),
                "prompt_unload_expected_direction": (
                    prompt_unload_expected_direction
                ),
                "prompt_sand_direction_matches_coordinate": int(
                    parsed["prompt_sand_direction"]
                    == prompt_sand_expected_direction
                ),
                "prompt_unload_direction_matches_coordinate": int(
                    parsed["prompt_unload_direction"]
                    == prompt_unload_expected_direction
                ),
                "prompt_sand_direction_matches_state_target": int(
                    parsed["prompt_sand_direction"]
                    == diagnostic["state_dig_direction"]
                ),
                "prompt_unload_direction_matches_state_target": int(
                    parsed["prompt_unload_direction"]
                    == diagnostic["state_unload_direction"]
                ),
                "prompt_sand_vs_state_target_error_m": math.hypot(
                    prompt_sand[0] - state_dig_world[0],
                    prompt_sand[1] - state_dig_world[1],
                ),
                "prompt_unload_vs_state_target_error_m": math.hypot(
                    prompt_unload[0] - state_unload_world[0],
                    prompt_unload[1] - state_unload_world[1],
                ),
            }
        )
        output.append(diagnostic)

    return output


def prompt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("prompt_parsed") == 1]
    state_valid = [row for row in rows if row.get("state27_valid") == 1]

    summary: dict[str, Any] = {
        "num_steps": len(rows),
        "num_state27_valid": len(state_valid),
        "num_prompt_parsed": len(parsed),
        "prompt_parse_rate": (
            len(parsed) / len(rows) if rows else float("nan")
        ),
    }

    keys = (
        "prompt_sand_direction_matches_coordinate",
        "prompt_unload_direction_matches_coordinate",
        "prompt_sand_direction_matches_state_target",
        "prompt_unload_direction_matches_state_target",
        "prompt_sand_vs_state_target_error_m",
        "prompt_unload_vs_state_target_error_m",
    )
    for key in keys:
        values = [
            float(row[key])
            for row in parsed
            if key in row
        ]
        summary[f"{key}_mean"] = finite_mean(values)
        summary[f"{key}_median"] = finite_median(values)

    unique_prompts = sorted(
        set(str(row.get("task_text", "")) for row in rows)
    )
    summary["num_unique_prompts"] = len(unique_prompts)
    summary["unique_prompts"] = unique_prompts

    return summary


def chunk_matrices(
    chunk_rows: list[dict[str, Any]],
    max_policy_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [
            row for row in chunk_rows
            if 0 <= int(row.get("replan_step", -1)) < max_policy_step
        ],
        key=lambda row: int(row["replan_step"]),
    )
    if not selected:
        raise RuntimeError("No chunk records overlap requested steps")

    replan_steps = np.asarray(
        [int(row["replan_step"]) for row in selected],
        dtype=np.int64,
    )
    chunk_size = min(
        int(row.get("chunk_size", len(row.get("items", []))))
        for row in selected
    )

    stage_matrix = np.full(
        (len(selected), chunk_size),
        np.nan,
        dtype=np.float64,
    )
    confidence_matrix = np.full_like(stage_matrix, np.nan)
    action_matrix = np.full(
        (len(selected), chunk_size, len(ACTION_NAMES)),
        np.nan,
        dtype=np.float64,
    )

    for row_index, row in enumerate(selected):
        items = row.get("items", [])[:chunk_size]
        for horizon, item in enumerate(items):
            stage_matrix[row_index, horizon] = float(
                item["stage_id"]
            )
            confidence_matrix[row_index, horizon] = float(
                item["stage_confidence"]
            )
            for action_index, action_name in enumerate(ACTION_NAMES):
                action_matrix[row_index, horizon, action_index] = float(
                    item["action"][action_name]
                )

    return (
        replan_steps,
        stage_matrix,
        confidence_matrix,
        action_matrix,
    )


def plot_stage_and_action(
    output_dir: Path,
    steps: np.ndarray,
    stages: np.ndarray,
    confidence: np.ndarray,
    actions: np.ndarray,
    chunk_rows: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(16, 5))
    plt.step(
        steps,
        stages,
        where="post",
        label="executed predicted stage",
    )
    plt.yticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    plt.xlabel("policy step")
    plt.ylabel("predicted stage")
    plt.title("Executed predicted stage over the 200-step rollout")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "01_executed_stage_timeline.png")

    plt.figure(figsize=(16, 4))
    plt.plot(steps, confidence)
    plt.ylim(0.0, 1.05)
    plt.xlabel("policy step")
    plt.ylabel("stage confidence")
    plt.title("Confidence of the executed predicted stage")
    plt.grid(True, alpha=0.3)
    save_figure(output_dir / "02_executed_stage_confidence.png")

    counts = Counter(int(value) for value in stages.tolist())
    stage_count_values = [
        counts.get(stage_id, 0)
        for stage_id in range(len(STAGE_NAMES))
    ]
    plt.figure(figsize=(13, 5))
    plt.bar(np.arange(len(STAGE_NAMES)), stage_count_values)
    plt.xticks(
        np.arange(len(STAGE_NAMES)),
        STAGE_NAMES,
        rotation=45,
        ha="right",
    )
    plt.xlabel("predicted stage")
    plt.ylabel("executed step count")
    plt.title("Predicted-stage occupancy in the rollout")
    plt.grid(True, axis="y", alpha=0.3)
    save_figure(output_dir / "03_stage_occupancy.png")

    for action_index, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(16, 4))
        plt.plot(
            steps,
            actions[:, action_index],
            label=f"executed {action_name}",
        )
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("policy step")
        plt.ylabel("executed joint velocity action")
        plt.title(
            f"Executed action: {action_name}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_figure(
            output_dir
            / f"{4 + action_index:02d}_executed_action_{action_name}.png"
        )

    replan_steps, stage_matrix, confidence_matrix, action_matrix = (
        chunk_matrices(
            chunk_rows,
            max_policy_step=int(steps.max()) + 1,
        )
    )

    plt.figure(figsize=(16, 7))
    plt.imshow(
        stage_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    plt.colorbar(label="predicted stage ID")
    plt.xticks(np.arange(0, stage_matrix.shape[1], 5))
    plt.yticks(
        np.arange(len(replan_steps)),
        [str(value) for value in replan_steps],
    )
    plt.xlabel("future horizon within the predicted 50-step chunk")
    plt.ylabel("replan policy step")
    plt.title("Predicted stage for every future position in every chunk")
    save_figure(output_dir / "08_stage_chunk_heatmap.png")

    plt.figure(figsize=(16, 7))
    plt.imshow(
        confidence_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="stage confidence")
    plt.xticks(np.arange(0, confidence_matrix.shape[1], 5))
    plt.yticks(
        np.arange(len(replan_steps)),
        [str(value) for value in replan_steps],
    )
    plt.xlabel("future horizon within the predicted 50-step chunk")
    plt.ylabel("replan policy step")
    plt.title("Stage confidence for every predicted future position")
    save_figure(output_dir / "09_stage_chunk_confidence_heatmap.png")

    for action_index, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(16, 7))
        plt.imshow(
            action_matrix[:, :, action_index],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        plt.colorbar(label=f"raw {action_name} action")
        plt.xticks(np.arange(0, action_matrix.shape[1], 5))
        plt.yticks(
            np.arange(len(replan_steps)),
            [str(value) for value in replan_steps],
        )
        plt.xlabel("future horizon within the predicted 50-step chunk")
        plt.ylabel("replan policy step")
        plt.title(
            f"Full predicted action chunks: {action_name}"
        )
        save_figure(
            output_dir
            / f"{10 + action_index:02d}_chunk_action_{action_name}.png"
        )


def plot_prompt_diagnostics(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    valid = [
        row for row in rows
        if row.get("prompt_parsed") == 1
        and row.get("state27_valid") == 1
    ]
    if not valid:
        print("[WARN] no valid prompt diagnostic rows; skipping prompt plots")
        return

    steps = np.asarray(
        [row["policy_step"] for row in valid],
        dtype=np.int64,
    )
    base_x = np.asarray([row["base_x"] for row in valid])
    base_y = np.asarray([row["base_y"] for row in valid])
    prompt_sand_x = np.asarray(
        [row["prompt_sand_x"] for row in valid]
    )
    prompt_sand_y = np.asarray(
        [row["prompt_sand_y"] for row in valid]
    )
    prompt_unload_x = np.asarray(
        [row["prompt_unload_x"] for row in valid]
    )
    prompt_unload_y = np.asarray(
        [row["prompt_unload_y"] for row in valid]
    )
    state_dig_x = np.asarray(
        [row["state_dig_world_x"] for row in valid]
    )
    state_dig_y = np.asarray(
        [row["state_dig_world_y"] for row in valid]
    )
    state_unload_x = np.asarray(
        [row["state_unload_world_x"] for row in valid]
    )
    state_unload_y = np.asarray(
        [row["state_unload_world_y"] for row in valid]
    )

    plt.figure(figsize=(10, 9))
    plt.plot(base_x, base_y, marker=".", label="robot base path")
    plt.scatter(
        prompt_sand_x,
        prompt_sand_y,
        marker="x",
        label="prompt sand XY",
    )
    plt.scatter(
        state_dig_x,
        state_dig_y,
        marker="+",
        label="State27 dig target XY",
    )
    plt.scatter(
        prompt_unload_x,
        prompt_unload_y,
        marker="x",
        label="prompt unload XY",
    )
    plt.scatter(
        state_unload_x,
        state_unload_y,
        marker="+",
        label="State27 unload target XY",
    )
    plt.xlabel("world X")
    plt.ylabel("world Y")
    plt.title("Prompt coordinates versus State27 target coordinates")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "14_prompt_and_state_targets_world_xy.png")

    sand_error = np.asarray(
        [
            row["prompt_sand_vs_state_target_error_m"]
            for row in valid
        ]
    )
    unload_error = np.asarray(
        [
            row["prompt_unload_vs_state_target_error_m"]
            for row in valid
        ]
    )

    plt.figure(figsize=(16, 4))
    plt.plot(
        steps,
        sand_error,
        label="prompt sand XY vs State27 dig target XY",
    )
    plt.plot(
        steps,
        unload_error,
        label="prompt unload XY vs State27 unload target XY",
    )
    plt.xlabel("policy step")
    plt.ylabel("XY distance error (m)")
    plt.title("Prompt-coordinate consistency with State27 targets")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "15_prompt_target_coordinate_error.png")

    sand_match = np.asarray(
        [
            row["prompt_sand_direction_matches_state_target"]
            for row in valid
        ]
    )
    unload_match = np.asarray(
        [
            row["prompt_unload_direction_matches_state_target"]
            for row in valid
        ]
    )

    plt.figure(figsize=(16, 3.5))
    plt.step(
        steps,
        sand_match,
        where="post",
        label="sand prompt direction matches State27 target",
    )
    plt.step(
        steps,
        unload_match,
        where="post",
        label="unload prompt direction matches State27 target",
    )
    plt.yticks([0, 1], ["mismatch", "match"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("policy step")
    plt.title("Prompt direction consistency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "16_prompt_direction_consistency.png")


def nullable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: nullable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nullable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a SmolVLA rollout: prompt targets, predicted "
            "stage flow, complete 50-step chunks, and executed actions."
        )
    )
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--chunk-log", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows = load_jsonl(
        args.trace_log.expanduser().resolve(),
        expected_type="smolvla_policy_step",
    )
    step_rows = sorted(
        step_rows,
        key=lambda row: int(row["policy_step"]),
    )
    step_rows = [
        row for row in step_rows
        if 0 <= int(row["policy_step"]) < args.steps
    ]
    if not step_rows:
        raise RuntimeError("No step rows within requested range")

    unique_steps = [int(row["policy_step"]) for row in step_rows]
    if len(unique_steps) != len(set(unique_steps)):
        raise RuntimeError("Trace log contains duplicate policy_step values")

    expected = list(range(min(unique_steps), min(unique_steps) + len(unique_steps)))
    if unique_steps != expected:
        print(
            "[WARN] policy steps are not strictly consecutive:",
            unique_steps[:10],
            "...",
            unique_steps[-10:],
        )

    chunk_rows = load_jsonl(
        args.chunk_log.expanduser().resolve(),
        expected_type="smolvla_action_stage_chunk",
    )

    steps = np.asarray(
        [int(row["policy_step"]) for row in step_rows],
        dtype=np.int64,
    )
    stages = np.asarray(
        [int(row["stage_id"]) for row in step_rows],
        dtype=np.int64,
    )
    confidence = np.asarray(
        [float(row["stage_confidence"]) for row in step_rows],
        dtype=np.float64,
    )
    model_actions = np.asarray(
        [
            [
                float(row["action"][action_name])
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    actions = np.asarray(
        [
            [
                float(
                    row.get("executed_action", row["action"])[action_name]
                )
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    temporal_ensemble = temporal_ensemble_summary(step_rows, model_actions)

    constraint_rows = [
        row.get("execution_constraints") or {}
        for row in step_rows
    ]
    supervisor_rows = [
        row
        for row in constraint_rows
        if row.get("mode") == "excavation_sequence_supervisor_v1"
    ]
    supervisor_summary = {
        "available": bool(supervisor_rows),
        "mode": (
            "excavation_sequence_supervisor_v1"
            if supervisor_rows
            else None
        ),
        "modified_steps": int(
            sum(bool(row.get("modified", False)) for row in supervisor_rows)
        ),
        "phase_sequence": list(
            dict.fromkeys(str(row.get("phase")) for row in supervisor_rows)
        ),
        "transitions": [
            {
                "policy_step": int(step_rows[index]["policy_step"]),
                "from": row.get("phase_before"),
                "to": row.get("phase"),
                "reason": row.get("transition_reason"),
            }
            for index, row in enumerate(constraint_rows)
            if bool(row.get("transition", False))
        ],
    }

    prompt_rows = build_prompt_diagnostics(step_rows)
    prompt_result = prompt_summary(prompt_rows)

    progression = stage_progression_summary(stages)
    prior = load_prior(args.prior.expanduser().resolve())
    prior_summary, prior_rows = prior_action_diagnostics(
        stages,
        model_actions,
        prior,
    )

    stage_counts = Counter(int(value) for value in stages.tolist())
    stage_summary = {
        "schema": "canonical_v4",
        "stage_names": list(STAGE_NAMES),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "minimum_confidence": float(np.min(confidence)),
        "maximum_confidence": float(np.max(confidence)),
        "low_confidence_below_0_5_fraction": float(
            np.mean(confidence < 0.5)
        ),
        "stage_counts": {
            (
                STAGE_NAMES[stage_id]
                if 0 <= stage_id < len(STAGE_NAMES)
                else f"unknown_{stage_id}"
            ): count
            for stage_id, count in sorted(stage_counts.items())
        },
        **progression,
    }

    action_rows = action_summary_rows(steps, stages, actions)
    write_csv(output_dir / "action_summary.csv", action_rows)
    write_csv(output_dir / "prompt_diagnostics.csv", prompt_rows)
    if prior_rows:
        write_csv(
            output_dir / "stage_action_prior_diagnostics.csv",
            prior_rows,
        )

    plot_stage_and_action(
        output_dir,
        steps,
        stages,
        confidence,
        actions,
        chunk_rows,
    )
    plot_prompt_diagnostics(output_dir, prompt_rows)

    summary = {
        "num_policy_steps": len(step_rows),
        "step_min": int(steps.min()),
        "step_max": int(steps.max()),
        "prompt": prompt_result,
        "stage": stage_summary,
        "temporal_ensemble": temporal_ensemble,
        "excavation_sequence_supervisor": supervisor_summary,
        "stage_action_prior": prior_summary,
        "important_limit": (
            "This rollout has no expert ground-truth action or true stage. "
            "The analysis checks internal workflow consistency, confidence, "
            "stage progression, prompt/State27 target consistency, and action "
            "direction non-violation against the training prior; it does not by "
            "itself prove task success."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            nullable(summary),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# SmolVLA rollout analysis",
        "",
        f"- Steps analyzed: {len(step_rows)}",
        "- Stage schema: canonical v4",
        f"- Stage mean confidence: {stage_summary['mean_confidence']:.4f}",
        f"- Stage minimum confidence: {stage_summary['minimum_confidence']:.4f}",
        (
            "- Compressed stage flow: "
            + " -> ".join(stage_summary["compressed_stage_names"])
        ),
        (
            "- Backward stage transitions: "
            f"{stage_summary['num_backward_transitions']}"
        ),
        (
            "- Forward stage skips: "
            f"{stage_summary['num_forward_skips']}"
        ),
        (
            "- Prompt parse rate: "
            f"{prompt_result['prompt_parse_rate']:.4f}"
        ),
        (
            "- Sand prompt coordinate error versus State27 target, mean: "
            f"{prompt_result.get('prompt_sand_vs_state_target_error_m_mean', float('nan')):.4f} m"
        ),
        (
            "- Unload prompt coordinate error versus State27 target, mean: "
            f"{prompt_result.get('prompt_unload_vs_state_target_error_m_mean', float('nan')):.4f} m"
        ),
    ]
    if temporal_ensemble.get("available"):
        report_lines.extend(
            [
                (
                    "- Temporal ensemble: "
                    f"width={temporal_ensemble['configured_width']}, "
                    f"decay={temporal_ensemble['decay']:g}, "
                    "first full-width step="
                    f"{temporal_ensemble['first_full_width_step']}"
                ),
                "- Temporal-ensemble continuity changes:",
            ]
        )
        for action_name in ACTION_NAMES:
            action_result = temporal_ensemble["by_action"][action_name]
            report_lines.append(
                f"  - {action_name}: sign flips "
                f"{action_result['latest_prediction_sign_flips']} -> "
                f"{action_result['executed_ensemble_sign_flips']}; "
                "mean absolute step delta reduction="
                f"{action_result['mean_abs_step_delta_reduction_fraction']:.1%}"
            )

    if supervisor_summary.get("available"):
        report_lines.extend(
            [
                (
                    "- Excavation sequence supervisor phases: "
                    + " -> ".join(supervisor_summary["phase_sequence"])
                ),
                (
                    "- Supervisor-modified steps: "
                    f"{supervisor_summary['modified_steps']}"
                ),
                "- Supervisor transitions:",
            ]
        )
        for transition in supervisor_summary["transitions"]:
            report_lines.append(
                f"  - step {transition['policy_step']}: "
                f"{transition['from']} -> {transition['to']} "
                f"({transition['reason']})"
            )

    if prior_summary.get("available"):
        report_lines.extend(
            [
                (
                    "- Stage-action prior direction non-violation rate: "
                    f"{prior_summary['overall_direction_nonviolation_rate']:.4f}"
                ),
                (
                    "- Low-swing-motion rate for canonical stages 1-7 "
                    "(|swing| / prior scale <= 0.10): "
                    f"{prior_summary['low_swing_motion_stage_1_to_7_rate']:.4f}"
                ),
                (
                    "- Prior rows interpreted by canonical v4 numeric stage ID; "
                    "source labels match canonical v4: "
                    f"{prior_summary['source_stage_labels_match_canonical']}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "## Important interpretation limit",
            "",
            (
                "There is no expert ground-truth stage/action in this online "
                "rollout. The prior metric only checks whether an action violates "
                "a reliable sign constraint; zero-valued actions count as "
                "non-violating. These checks do not measure action accuracy or "
                "establish final physical task success."
            ),
            "",
            f"All figures and CSV files are in `{output_dir}`.",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("SMOLVLA ROLLOUT ANALYSIS")
    print("=" * 78)
    print(f"steps: {len(step_rows)}")
    print(
        "stage flow:",
        " -> ".join(stage_summary["compressed_stage_names"]),
    )
    print(
        f"stage confidence mean/min: "
        f"{stage_summary['mean_confidence']:.4f} / "
        f"{stage_summary['minimum_confidence']:.4f}"
    )
    print(
        f"backward transitions: "
        f"{stage_summary['num_backward_transitions']}"
    )
    print(
        f"forward skips: "
        f"{stage_summary['num_forward_skips']}"
    )
    if temporal_ensemble.get("available"):
        print(
            "temporal ensemble: "
            f"width={temporal_ensemble['configured_width']} "
            f"decay={temporal_ensemble['decay']:g}"
        )
    print(
        f"prompt parse rate: "
        f"{prompt_result['prompt_parse_rate']:.4f}"
    )
    print(
        "sand prompt vs State27 target mean error:",
        prompt_result.get(
            "prompt_sand_vs_state_target_error_m_mean"
        ),
    )
    print(
        "unload prompt vs State27 target mean error:",
        prompt_result.get(
            "prompt_unload_vs_state_target_error_m_mean"
        ),
    )
    if prior_summary.get("available"):
        print(
            "stage-action prior direction non-violation rate:",
            prior_summary["overall_direction_nonviolation_rate"],
        )
        print(
            "canonical stages 1-7 low-swing-motion rate:",
            prior_summary[
                "low_swing_motion_stage_1_to_7_rate"
            ],
        )
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = ("swing", "boom", "arm", "bucket")
STAGE_NAMES = (
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
)
DIRECTION_LABELS = (
    "front",
    "front-left",
    "left",
    "rear-left",
    "rear",
    "rear-right",
    "right",
    "front-right",
)

DEFAULT_TRACE = Path(
    "/root/gpufree-data/excavator_logs/"
    "smolvla_policy_steps_trace.jsonl"
)
DEFAULT_CHUNKS = Path(
    "/root/gpufree-data/excavator_logs/"
    "smolvla_action_stage_chunks.jsonl"
)
DEFAULT_PRIOR = Path(
    "/root/gpufree-data/excavator_stage_action_analysis/"
    "stage_action_prior_training.json"
)
DEFAULT_OUTPUT = Path(
    "/root/gpufree-data/excavator_200step_analysis"
)

PROMPT_PATTERN = re.compile(
    r"Dig soil from the sand pile near "
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*\),\s*"
    r"([a-z-]+)\s+of the excavator,\s*"
    r"and dump it into the truck bed near "
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*\),\s*"
    r"([a-z-]+)\s+of the excavator\.",
    re.IGNORECASE,
)


def load_jsonl(path: Path, expected_type: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            if expected_type is None or row.get("type") == expected_type:
                rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No usable records found in {path}"
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def finite_mean(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_median(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.median(array)) if array.size else float("nan")


def world_from_local_xy(
    base_x: float,
    base_y: float,
    base_yaw: float,
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    cosine = math.cos(base_yaw)
    sine = math.sin(base_yaw)
    dx = cosine * local_x - sine * local_y
    dy = sine * local_x + cosine * local_y
    return base_x + dx, base_y + dy


def direction_label(
    point_xy: tuple[float, float],
    origin_xy: tuple[float, float],
    base_yaw: float,
) -> str:
    dx = point_xy[0] - origin_xy[0]
    dy = point_xy[1] - origin_xy[1]
    if abs(dx) + abs(dy) < 1e-9:
        return "nearby"

    # This reproduces the simulation prompt's current convention.
    forward_heading = base_yaw + math.pi * 0.5
    relative_angle = math.atan2(dy, dx) - forward_heading
    index = int(
        math.floor(
            (
                math.degrees(relative_angle)
                + 22.5
            )
            % 360.0
            / 45.0
        )
    )
    return DIRECTION_LABELS[index % len(DIRECTION_LABELS)]


def parse_prompt(task_text: str) -> dict[str, Any]:
    match = PROMPT_PATTERN.fullmatch(task_text.strip())
    if match is None:
        return {
            "parsed": False,
            "task_text": task_text,
        }

    return {
        "parsed": True,
        "task_text": task_text,
        "prompt_sand_x": float(match.group(1)),
        "prompt_sand_y": float(match.group(2)),
        "prompt_sand_direction": match.group(3).lower(),
        "prompt_unload_x": float(match.group(4)),
        "prompt_unload_y": float(match.group(5)),
        "prompt_unload_direction": match.group(6).lower(),
    }


def compress_stage_sequence(stage_ids: np.ndarray) -> list[int]:
    compressed: list[int] = []
    for value in stage_ids.tolist():
        value = int(value)
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return compressed


def stage_progression_summary(stage_ids: np.ndarray) -> dict[str, Any]:
    compressed = compress_stage_sequence(stage_ids)
    transitions = list(zip(compressed[:-1], compressed[1:]))

    backward = []
    forward_skips = []
    allowed_cycle_resets = []

    for old, new in transitions:
        if old == 9 and new == 0:
            allowed_cycle_resets.append((old, new))
        elif new < old:
            backward.append((old, new))
        elif new > old + 1:
            forward_skips.append((old, new))

    return {
        "compressed_stage_ids": compressed,
        "compressed_stage_names": [
            STAGE_NAMES[value]
            if 0 <= value < len(STAGE_NAMES)
            else f"unknown_{value}"
            for value in compressed
        ],
        "num_stage_segments": len(compressed),
        "num_backward_transitions": len(backward),
        "backward_transitions": backward,
        "num_forward_skips": len(forward_skips),
        "forward_skips": forward_skips,
        "num_allowed_cycle_resets": len(allowed_cycle_resets),
    }


def load_prior(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"[WARN] prior file not found: {path}")
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))

    raw_stage_names = payload.get("stage_names")
    source_stage_names: tuple[str, ...] | None = None
    if isinstance(raw_stage_names, dict):
        try:
            source_stage_names = tuple(
                str(raw_stage_names[str(stage_id)])
                if str(stage_id) in raw_stage_names
                else str(raw_stage_names[stage_id])
                for stage_id in range(len(STAGE_NAMES))
            )
        except (KeyError, TypeError):
            source_stage_names = None
    elif isinstance(raw_stage_names, (list, tuple)):
        source_stage_names = tuple(str(name) for name in raw_stage_names)

    labels_match_canonical = source_stage_names == STAGE_NAMES
    if source_stage_names is None:
        print(
            "[WARN] prior has no complete stage_names metadata; "
            "interpreting its rows by canonical v4 stage ID"
        )
    elif not labels_match_canonical:
        print(
            "[WARN] prior stage_names metadata is not canonical v4; "
            "interpreting its rows by numeric stage ID and reporting "
            "canonical v4 names"
        )

    direction = np.asarray(
        payload.get("direction", []),
        dtype=np.float64,
    )
    weight = np.asarray(
        payload.get("weight", []),
        dtype=np.float64,
    )
    reliable = np.asarray(
        payload.get("reliable", []),
        dtype=bool,
    )
    scale = np.asarray(
        payload.get("action_scale", []),
        dtype=np.float64,
    ).reshape(-1)

    if direction.shape != (len(STAGE_NAMES), len(ACTION_NAMES)):
        print(
            "[WARN] unexpected prior direction shape:",
            direction.shape,
        )
        return None
    if weight.shape != direction.shape:
        weight = np.ones_like(direction)
    if reliable.shape != direction.shape:
        reliable = direction != 0
    if scale.size < len(ACTION_NAMES):
        scale = np.ones(len(ACTION_NAMES), dtype=np.float64)

    strong = reliable & (weight >= 0.8) & (direction != 0)
    return {
        "direction": direction[:, : len(ACTION_NAMES)],
        "strong": strong[:, : len(ACTION_NAMES)],
        "scale": np.maximum(scale[: len(ACTION_NAMES)], 1e-8),
        "source_stage_labels_present": source_stage_names is not None,
        "source_stage_labels_match_canonical": labels_match_canonical,
    }


def prior_action_diagnostics(
    stages: np.ndarray,
    actions: np.ndarray,
    prior: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if prior is None:
        return {
            "available": False,
        }, []

    rows = []
    all_nonviolating = []
    low_swing_motion = []

    for step, (stage_id, action) in enumerate(
        zip(stages.tolist(), actions, strict=True)
    ):
        if not 0 <= int(stage_id) < len(STAGE_NAMES):
            continue

        stage_id = int(stage_id)
        for action_index, action_name in enumerate(ACTION_NAMES):
            if not prior["strong"][stage_id, action_index]:
                continue

            direction = float(
                prior["direction"][stage_id, action_index]
            )
            value = float(action[action_index])
            nonviolating = direction * value >= 0.0
            all_nonviolating.append(float(nonviolating))
            rows.append(
                {
                    "policy_step": step,
                    "stage_id": stage_id,
                    "stage_name": STAGE_NAMES[stage_id],
                    "action_name": action_name,
                    "expected_direction": direction,
                    "action_value": value,
                    "direction_nonviolating": int(nonviolating),
                }
            )

        if 1 <= stage_id <= 7:
            normalized_abs_swing = (
                abs(float(action[0]))
                / float(prior["scale"][0])
            )
            low_swing_motion.append(
                float(normalized_abs_swing <= 0.10)
            )

    by_stage: dict[str, Any] = {}
    for stage_id in range(len(STAGE_NAMES)):
        stage_rows = [
            row for row in rows
            if row["stage_id"] == stage_id
        ]
        if not stage_rows:
            continue
        by_stage[STAGE_NAMES[stage_id]] = {
            "num_checked_stage_action_pairs": len(stage_rows),
            "direction_nonviolation_rate": finite_mean(
                [row["direction_nonviolating"] for row in stage_rows]
            ),
        }

    return {
        "available": True,
        "source_stage_labels_present": prior[
            "source_stage_labels_present"
        ],
        "source_stage_labels_match_canonical": prior[
            "source_stage_labels_match_canonical"
        ],
        "stage_row_interpretation": "canonical_v4_numeric_stage_id",
        "num_checked_stage_action_pairs": len(all_nonviolating),
        "overall_direction_nonviolation_rate": finite_mean(
            all_nonviolating
        ),
        "low_swing_motion_stage_1_to_7_rate": finite_mean(
            low_swing_motion
        ),
        "by_stage": by_stage,
    }, rows


def action_summary_rows(
    steps: np.ndarray,
    stages: np.ndarray,
    actions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for action_index, action_name in enumerate(ACTION_NAMES):
        values = actions[:, action_index]
        delta = np.diff(values)
        rows.append(
            {
                "scope": "overall",
                "stage_id": "",
                "stage_name": "all",
                "action_name": action_name,
                "count": len(values),
                "mean": float(np.mean(values)),
                "mean_abs": float(np.mean(np.abs(values))),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "positive_fraction": float(np.mean(values > 0)),
                "negative_fraction": float(np.mean(values < 0)),
                "near_zero_fraction": float(np.mean(np.abs(values) <= 1e-4)),
                "mean_abs_step_delta": (
                    float(np.mean(np.abs(delta)))
                    if len(delta)
                    else 0.0
                ),
                "max_abs_step_delta": (
                    float(np.max(np.abs(delta)))
                    if len(delta)
                    else 0.0
                ),
            }
        )

    for stage_id in sorted(set(int(value) for value in stages.tolist())):
        mask = stages == stage_id
        if not np.any(mask):
            continue

        for action_index, action_name in enumerate(ACTION_NAMES):
            values = actions[mask, action_index]
            rows.append(
                {
                    "scope": "stage",
                    "stage_id": stage_id,
                    "stage_name": (
                        STAGE_NAMES[stage_id]
                        if 0 <= stage_id < len(STAGE_NAMES)
                        else f"unknown_{stage_id}"
                    ),
                    "action_name": action_name,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "positive_fraction": float(np.mean(values > 0)),
                    "negative_fraction": float(np.mean(values < 0)),
                    "near_zero_fraction": float(
                        np.mean(np.abs(values) <= 1e-4)
                    ),
                    "mean_abs_step_delta": "",
                    "max_abs_step_delta": "",
                }
            )

    return rows


def temporal_ensemble_summary(
    step_rows: list[dict[str, Any]],
    executed_actions: np.ndarray,
) -> dict[str, Any]:
    """Summarize continuity changes from logged overlapping predictions."""
    if not step_rows or not all(
        isinstance(row.get("latest_action"), dict)
        and isinstance(row.get("temporal_ensemble"), dict)
        for row in step_rows
    ):
        return {"available": False}

    latest_actions = np.asarray(
        [
            [
                float(row["latest_action"][action_name])
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    metadata = [row["temporal_ensemble"] for row in step_rows]
    widths = [int(row.get("configured_width", 1)) for row in metadata]
    decays = [float(row.get("decay", 0.0)) for row in metadata]
    prediction_counts = [
        int(row.get("num_predictions", 0)) for row in metadata
    ]
    configured_width = widths[0]
    full_width_steps = [
        int(step_rows[index]["policy_step"])
        for index, count in enumerate(prediction_counts)
        if count == configured_width
    ]

    by_action = {}
    for action_index, action_name in enumerate(ACTION_NAMES):
        latest = latest_actions[:, action_index]
        executed = executed_actions[:, action_index]
        latest_delta = float(np.mean(np.abs(np.diff(latest))))
        executed_delta = float(np.mean(np.abs(np.diff(executed))))
        latest_flips = int(np.sum(np.sign(latest[1:]) != np.sign(latest[:-1])))
        executed_flips = int(
            np.sum(np.sign(executed[1:]) != np.sign(executed[:-1]))
        )
        by_action[action_name] = {
            "latest_prediction_sign_flips": latest_flips,
            "executed_ensemble_sign_flips": executed_flips,
            "sign_flip_reduction": latest_flips - executed_flips,
            "latest_prediction_mean_abs_step_delta": latest_delta,
            "executed_ensemble_mean_abs_step_delta": executed_delta,
            "mean_abs_step_delta_reduction_fraction": (
                (latest_delta - executed_delta) / latest_delta
                if latest_delta > 0.0
                else 0.0
            ),
        }

    return {
        "available": True,
        "configured_width": configured_width,
        "decay": decays[0],
        "configuration_consistent": (
            len(set(widths)) == 1 and len(set(decays)) == 1
        ),
        "minimum_predictions": min(prediction_counts),
        "maximum_predictions": max(prediction_counts),
        "num_full_width_steps": len(full_width_steps),
        "first_full_width_step": (
            full_width_steps[0] if full_width_steps else None
        ),
        "by_action": by_action,
    }


def build_prompt_diagnostics(
    step_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output = []

    for row in step_rows:
        step = int(row["policy_step"])
        state = np.asarray(
            row.get("observation_state", []),
            dtype=np.float64,
        ).reshape(-1)
        parsed = parse_prompt(str(row.get("task_text", "")))

        diagnostic: dict[str, Any] = {
            "policy_step": step,
            "prompt_parsed": int(bool(parsed.get("parsed"))),
            "task_text": str(row.get("task_text", "")),
        }

        if state.size != 27:
            diagnostic["state27_valid"] = 0
            diagnostic["state_dim"] = int(state.size)
            output.append(diagnostic)
            continue

        diagnostic["state27_valid"] = 1
        diagnostic["state_dim"] = 27

        base_x = float(state[0])
        base_y = float(state[1])
        base_yaw = float(state[2])

        state_dig_world = world_from_local_xy(
            base_x,
            base_y,
            base_yaw,
            float(state[18]),
            float(state[19]),
        )
        state_unload_world = world_from_local_xy(
            base_x,
            base_y,
            base_yaw,
            float(state[21]),
            float(state[22]),
        )

        diagnostic.update(
            {
                "base_x": base_x,
                "base_y": base_y,
                "base_yaw": base_yaw,
                "state_dig_world_x": state_dig_world[0],
                "state_dig_world_y": state_dig_world[1],
                "state_dig_world_z": float(state[20]),
                "state_unload_world_x": state_unload_world[0],
                "state_unload_world_y": state_unload_world[1],
                "state_unload_world_z": float(state[23]),
                "state_dig_direction": direction_label(
                    state_dig_world,
                    (base_x, base_y),
                    base_yaw,
                ),
                "state_unload_direction": direction_label(
                    state_unload_world,
                    (base_x, base_y),
                    base_yaw,
                ),
            }
        )

        if not parsed.get("parsed"):
            output.append(diagnostic)
            continue

        prompt_sand = (
            float(parsed["prompt_sand_x"]),
            float(parsed["prompt_sand_y"]),
        )
        prompt_unload = (
            float(parsed["prompt_unload_x"]),
            float(parsed["prompt_unload_y"]),
        )

        prompt_sand_expected_direction = direction_label(
            prompt_sand,
            (base_x, base_y),
            base_yaw,
        )
        prompt_unload_expected_direction = direction_label(
            prompt_unload,
            (base_x, base_y),
            base_yaw,
        )

        diagnostic.update(
            {
                "prompt_sand_x": prompt_sand[0],
                "prompt_sand_y": prompt_sand[1],
                "prompt_sand_direction": parsed[
                    "prompt_sand_direction"
                ],
                "prompt_unload_x": prompt_unload[0],
                "prompt_unload_y": prompt_unload[1],
                "prompt_unload_direction": parsed[
                    "prompt_unload_direction"
                ],
                "prompt_sand_expected_direction": (
                    prompt_sand_expected_direction
                ),
                "prompt_unload_expected_direction": (
                    prompt_unload_expected_direction
                ),
                "prompt_sand_direction_matches_coordinate": int(
                    parsed["prompt_sand_direction"]
                    == prompt_sand_expected_direction
                ),
                "prompt_unload_direction_matches_coordinate": int(
                    parsed["prompt_unload_direction"]
                    == prompt_unload_expected_direction
                ),
                "prompt_sand_direction_matches_state_target": int(
                    parsed["prompt_sand_direction"]
                    == diagnostic["state_dig_direction"]
                ),
                "prompt_unload_direction_matches_state_target": int(
                    parsed["prompt_unload_direction"]
                    == diagnostic["state_unload_direction"]
                ),
                "prompt_sand_vs_state_target_error_m": math.hypot(
                    prompt_sand[0] - state_dig_world[0],
                    prompt_sand[1] - state_dig_world[1],
                ),
                "prompt_unload_vs_state_target_error_m": math.hypot(
                    prompt_unload[0] - state_unload_world[0],
                    prompt_unload[1] - state_unload_world[1],
                ),
            }
        )
        output.append(diagnostic)

    return output


def prompt_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [row for row in rows if row.get("prompt_parsed") == 1]
    state_valid = [row for row in rows if row.get("state27_valid") == 1]

    summary: dict[str, Any] = {
        "num_steps": len(rows),
        "num_state27_valid": len(state_valid),
        "num_prompt_parsed": len(parsed),
        "prompt_parse_rate": (
            len(parsed) / len(rows) if rows else float("nan")
        ),
    }

    keys = (
        "prompt_sand_direction_matches_coordinate",
        "prompt_unload_direction_matches_coordinate",
        "prompt_sand_direction_matches_state_target",
        "prompt_unload_direction_matches_state_target",
        "prompt_sand_vs_state_target_error_m",
        "prompt_unload_vs_state_target_error_m",
    )
    for key in keys:
        values = [
            float(row[key])
            for row in parsed
            if key in row
        ]
        summary[f"{key}_mean"] = finite_mean(values)
        summary[f"{key}_median"] = finite_median(values)

    unique_prompts = sorted(
        set(str(row.get("task_text", "")) for row in rows)
    )
    summary["num_unique_prompts"] = len(unique_prompts)
    summary["unique_prompts"] = unique_prompts

    return summary


def chunk_matrices(
    chunk_rows: list[dict[str, Any]],
    max_policy_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = sorted(
        [
            row for row in chunk_rows
            if 0 <= int(row.get("replan_step", -1)) < max_policy_step
        ],
        key=lambda row: int(row["replan_step"]),
    )
    if not selected:
        raise RuntimeError("No chunk records overlap requested steps")

    replan_steps = np.asarray(
        [int(row["replan_step"]) for row in selected],
        dtype=np.int64,
    )
    chunk_size = min(
        int(row.get("chunk_size", len(row.get("items", []))))
        for row in selected
    )

    stage_matrix = np.full(
        (len(selected), chunk_size),
        np.nan,
        dtype=np.float64,
    )
    confidence_matrix = np.full_like(stage_matrix, np.nan)
    action_matrix = np.full(
        (len(selected), chunk_size, len(ACTION_NAMES)),
        np.nan,
        dtype=np.float64,
    )

    for row_index, row in enumerate(selected):
        items = row.get("items", [])[:chunk_size]
        for horizon, item in enumerate(items):
            stage_matrix[row_index, horizon] = float(
                item["stage_id"]
            )
            confidence_matrix[row_index, horizon] = float(
                item["stage_confidence"]
            )
            for action_index, action_name in enumerate(ACTION_NAMES):
                action_matrix[row_index, horizon, action_index] = float(
                    item["action"][action_name]
                )

    return (
        replan_steps,
        stage_matrix,
        confidence_matrix,
        action_matrix,
    )


def plot_stage_and_action(
    output_dir: Path,
    steps: np.ndarray,
    stages: np.ndarray,
    confidence: np.ndarray,
    actions: np.ndarray,
    chunk_rows: list[dict[str, Any]],
) -> None:
    plt.figure(figsize=(16, 5))
    plt.step(
        steps,
        stages,
        where="post",
        label="executed predicted stage",
    )
    plt.yticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    plt.xlabel("policy step")
    plt.ylabel("predicted stage")
    plt.title("Executed predicted stage over the 200-step rollout")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "01_executed_stage_timeline.png")

    plt.figure(figsize=(16, 4))
    plt.plot(steps, confidence)
    plt.ylim(0.0, 1.05)
    plt.xlabel("policy step")
    plt.ylabel("stage confidence")
    plt.title("Confidence of the executed predicted stage")
    plt.grid(True, alpha=0.3)
    save_figure(output_dir / "02_executed_stage_confidence.png")

    counts = Counter(int(value) for value in stages.tolist())
    stage_count_values = [
        counts.get(stage_id, 0)
        for stage_id in range(len(STAGE_NAMES))
    ]
    plt.figure(figsize=(13, 5))
    plt.bar(np.arange(len(STAGE_NAMES)), stage_count_values)
    plt.xticks(
        np.arange(len(STAGE_NAMES)),
        STAGE_NAMES,
        rotation=45,
        ha="right",
    )
    plt.xlabel("predicted stage")
    plt.ylabel("executed step count")
    plt.title("Predicted-stage occupancy in the rollout")
    plt.grid(True, axis="y", alpha=0.3)
    save_figure(output_dir / "03_stage_occupancy.png")

    for action_index, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(16, 4))
        plt.plot(
            steps,
            actions[:, action_index],
            label=f"executed {action_name}",
        )
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("policy step")
        plt.ylabel("executed joint velocity action")
        plt.title(
            f"Executed action: {action_name}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_figure(
            output_dir
            / f"{4 + action_index:02d}_executed_action_{action_name}.png"
        )

    replan_steps, stage_matrix, confidence_matrix, action_matrix = (
        chunk_matrices(
            chunk_rows,
            max_policy_step=int(steps.max()) + 1,
        )
    )

    plt.figure(figsize=(16, 7))
    plt.imshow(
        stage_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    plt.colorbar(label="predicted stage ID")
    plt.xticks(np.arange(0, stage_matrix.shape[1], 5))
    plt.yticks(
        np.arange(len(replan_steps)),
        [str(value) for value in replan_steps],
    )
    plt.xlabel("future horizon within the predicted 50-step chunk")
    plt.ylabel("replan policy step")
    plt.title("Predicted stage for every future position in every chunk")
    save_figure(output_dir / "08_stage_chunk_heatmap.png")

    plt.figure(figsize=(16, 7))
    plt.imshow(
        confidence_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(label="stage confidence")
    plt.xticks(np.arange(0, confidence_matrix.shape[1], 5))
    plt.yticks(
        np.arange(len(replan_steps)),
        [str(value) for value in replan_steps],
    )
    plt.xlabel("future horizon within the predicted 50-step chunk")
    plt.ylabel("replan policy step")
    plt.title("Stage confidence for every predicted future position")
    save_figure(output_dir / "09_stage_chunk_confidence_heatmap.png")

    for action_index, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(16, 7))
        plt.imshow(
            action_matrix[:, :, action_index],
            aspect="auto",
            interpolation="nearest",
            origin="upper",
        )
        plt.colorbar(label=f"raw {action_name} action")
        plt.xticks(np.arange(0, action_matrix.shape[1], 5))
        plt.yticks(
            np.arange(len(replan_steps)),
            [str(value) for value in replan_steps],
        )
        plt.xlabel("future horizon within the predicted 50-step chunk")
        plt.ylabel("replan policy step")
        plt.title(
            f"Full predicted action chunks: {action_name}"
        )
        save_figure(
            output_dir
            / f"{10 + action_index:02d}_chunk_action_{action_name}.png"
        )


def plot_prompt_diagnostics(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    valid = [
        row for row in rows
        if row.get("prompt_parsed") == 1
        and row.get("state27_valid") == 1
    ]
    if not valid:
        print("[WARN] no valid prompt diagnostic rows; skipping prompt plots")
        return

    steps = np.asarray(
        [row["policy_step"] for row in valid],
        dtype=np.int64,
    )
    base_x = np.asarray([row["base_x"] for row in valid])
    base_y = np.asarray([row["base_y"] for row in valid])
    prompt_sand_x = np.asarray(
        [row["prompt_sand_x"] for row in valid]
    )
    prompt_sand_y = np.asarray(
        [row["prompt_sand_y"] for row in valid]
    )
    prompt_unload_x = np.asarray(
        [row["prompt_unload_x"] for row in valid]
    )
    prompt_unload_y = np.asarray(
        [row["prompt_unload_y"] for row in valid]
    )
    state_dig_x = np.asarray(
        [row["state_dig_world_x"] for row in valid]
    )
    state_dig_y = np.asarray(
        [row["state_dig_world_y"] for row in valid]
    )
    state_unload_x = np.asarray(
        [row["state_unload_world_x"] for row in valid]
    )
    state_unload_y = np.asarray(
        [row["state_unload_world_y"] for row in valid]
    )

    plt.figure(figsize=(10, 9))
    plt.plot(base_x, base_y, marker=".", label="robot base path")
    plt.scatter(
        prompt_sand_x,
        prompt_sand_y,
        marker="x",
        label="prompt sand XY",
    )
    plt.scatter(
        state_dig_x,
        state_dig_y,
        marker="+",
        label="State27 dig target XY",
    )
    plt.scatter(
        prompt_unload_x,
        prompt_unload_y,
        marker="x",
        label="prompt unload XY",
    )
    plt.scatter(
        state_unload_x,
        state_unload_y,
        marker="+",
        label="State27 unload target XY",
    )
    plt.xlabel("world X")
    plt.ylabel("world Y")
    plt.title("Prompt coordinates versus State27 target coordinates")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "14_prompt_and_state_targets_world_xy.png")

    sand_error = np.asarray(
        [
            row["prompt_sand_vs_state_target_error_m"]
            for row in valid
        ]
    )
    unload_error = np.asarray(
        [
            row["prompt_unload_vs_state_target_error_m"]
            for row in valid
        ]
    )

    plt.figure(figsize=(16, 4))
    plt.plot(
        steps,
        sand_error,
        label="prompt sand XY vs State27 dig target XY",
    )
    plt.plot(
        steps,
        unload_error,
        label="prompt unload XY vs State27 unload target XY",
    )
    plt.xlabel("policy step")
    plt.ylabel("XY distance error (m)")
    plt.title("Prompt-coordinate consistency with State27 targets")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "15_prompt_target_coordinate_error.png")

    sand_match = np.asarray(
        [
            row["prompt_sand_direction_matches_state_target"]
            for row in valid
        ]
    )
    unload_match = np.asarray(
        [
            row["prompt_unload_direction_matches_state_target"]
            for row in valid
        ]
    )

    plt.figure(figsize=(16, 3.5))
    plt.step(
        steps,
        sand_match,
        where="post",
        label="sand prompt direction matches State27 target",
    )
    plt.step(
        steps,
        unload_match,
        where="post",
        label="unload prompt direction matches State27 target",
    )
    plt.yticks([0, 1], ["mismatch", "match"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("policy step")
    plt.title("Prompt direction consistency")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(output_dir / "16_prompt_direction_consistency.png")


def nullable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: nullable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [nullable(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a SmolVLA rollout: prompt targets, predicted "
            "stage flow, complete 50-step chunks, and executed actions."
        )
    )
    parser.add_argument("--trace-log", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--chunk-log", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be positive")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    step_rows = load_jsonl(
        args.trace_log.expanduser().resolve(),
        expected_type="smolvla_policy_step",
    )
    step_rows = sorted(
        step_rows,
        key=lambda row: int(row["policy_step"]),
    )
    step_rows = [
        row for row in step_rows
        if 0 <= int(row["policy_step"]) < args.steps
    ]
    if not step_rows:
        raise RuntimeError("No step rows within requested range")

    unique_steps = [int(row["policy_step"]) for row in step_rows]
    if len(unique_steps) != len(set(unique_steps)):
        raise RuntimeError("Trace log contains duplicate policy_step values")

    expected = list(range(min(unique_steps), min(unique_steps) + len(unique_steps)))
    if unique_steps != expected:
        print(
            "[WARN] policy steps are not strictly consecutive:",
            unique_steps[:10],
            "...",
            unique_steps[-10:],
        )

    chunk_rows = load_jsonl(
        args.chunk_log.expanduser().resolve(),
        expected_type="smolvla_action_stage_chunk",
    )

    steps = np.asarray(
        [int(row["policy_step"]) for row in step_rows],
        dtype=np.int64,
    )
    stages = np.asarray(
        [int(row["stage_id"]) for row in step_rows],
        dtype=np.int64,
    )
    confidence = np.asarray(
        [float(row["stage_confidence"]) for row in step_rows],
        dtype=np.float64,
    )
    model_actions = np.asarray(
        [
            [
                float(row["action"][action_name])
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    actions = np.asarray(
        [
            [
                float(
                    row.get("executed_action", row["action"])[action_name]
                )
                for action_name in ACTION_NAMES
            ]
            for row in step_rows
        ],
        dtype=np.float64,
    )
    temporal_ensemble = temporal_ensemble_summary(step_rows, model_actions)

    constraint_rows = [
        row.get("execution_constraints") or {}
        for row in step_rows
    ]
    supervisor_rows = [
        row
        for row in constraint_rows
        if row.get("mode") == "excavation_sequence_supervisor_v1"
    ]
    supervisor_summary = {
        "available": bool(supervisor_rows),
        "mode": (
            "excavation_sequence_supervisor_v1"
            if supervisor_rows
            else None
        ),
        "modified_steps": int(
            sum(bool(row.get("modified", False)) for row in supervisor_rows)
        ),
        "phase_sequence": list(
            dict.fromkeys(str(row.get("phase")) for row in supervisor_rows)
        ),
        "transitions": [
            {
                "policy_step": int(step_rows[index]["policy_step"]),
                "from": row.get("phase_before"),
                "to": row.get("phase"),
                "reason": row.get("transition_reason"),
            }
            for index, row in enumerate(constraint_rows)
            if bool(row.get("transition", False))
        ],
    }

    prompt_rows = build_prompt_diagnostics(step_rows)
    prompt_result = prompt_summary(prompt_rows)

    progression = stage_progression_summary(stages)
    prior = load_prior(args.prior.expanduser().resolve())
    prior_summary, prior_rows = prior_action_diagnostics(
        stages,
        model_actions,
        prior,
    )

    stage_counts = Counter(int(value) for value in stages.tolist())
    stage_summary = {
        "schema": "canonical_v4",
        "stage_names": list(STAGE_NAMES),
        "mean_confidence": float(np.mean(confidence)),
        "median_confidence": float(np.median(confidence)),
        "minimum_confidence": float(np.min(confidence)),
        "maximum_confidence": float(np.max(confidence)),
        "low_confidence_below_0_5_fraction": float(
            np.mean(confidence < 0.5)
        ),
        "stage_counts": {
            (
                STAGE_NAMES[stage_id]
                if 0 <= stage_id < len(STAGE_NAMES)
                else f"unknown_{stage_id}"
            ): count
            for stage_id, count in sorted(stage_counts.items())
        },
        **progression,
    }

    action_rows = action_summary_rows(steps, stages, actions)
    write_csv(output_dir / "action_summary.csv", action_rows)
    write_csv(output_dir / "prompt_diagnostics.csv", prompt_rows)
    if prior_rows:
        write_csv(
            output_dir / "stage_action_prior_diagnostics.csv",
            prior_rows,
        )

    plot_stage_and_action(
        output_dir,
        steps,
        stages,
        confidence,
        actions,
        chunk_rows,
    )
    plot_prompt_diagnostics(output_dir, prompt_rows)

    summary = {
        "num_policy_steps": len(step_rows),
        "step_min": int(steps.min()),
        "step_max": int(steps.max()),
        "prompt": prompt_result,
        "stage": stage_summary,
        "temporal_ensemble": temporal_ensemble,
        "excavation_sequence_supervisor": supervisor_summary,
        "stage_action_prior": prior_summary,
        "important_limit": (
            "This rollout has no expert ground-truth action or true stage. "
            "The analysis checks internal workflow consistency, confidence, "
            "stage progression, prompt/State27 target consistency, and action "
            "direction non-violation against the training prior; it does not by "
            "itself prove task success."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(
            nullable(summary),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# SmolVLA rollout analysis",
        "",
        f"- Steps analyzed: {len(step_rows)}",
        "- Stage schema: canonical v4",
        f"- Stage mean confidence: {stage_summary['mean_confidence']:.4f}",
        f"- Stage minimum confidence: {stage_summary['minimum_confidence']:.4f}",
        (
            "- Compressed stage flow: "
            + " -> ".join(stage_summary["compressed_stage_names"])
        ),
        (
            "- Backward stage transitions: "
            f"{stage_summary['num_backward_transitions']}"
        ),
        (
            "- Forward stage skips: "
            f"{stage_summary['num_forward_skips']}"
        ),
        (
            "- Prompt parse rate: "
            f"{prompt_result['prompt_parse_rate']:.4f}"
        ),
        (
            "- Sand prompt coordinate error versus State27 target, mean: "
            f"{prompt_result.get('prompt_sand_vs_state_target_error_m_mean', float('nan')):.4f} m"
        ),
        (
            "- Unload prompt coordinate error versus State27 target, mean: "
            f"{prompt_result.get('prompt_unload_vs_state_target_error_m_mean', float('nan')):.4f} m"
        ),
    ]
    if temporal_ensemble.get("available"):
        report_lines.extend(
            [
                (
                    "- Temporal ensemble: "
                    f"width={temporal_ensemble['configured_width']}, "
                    f"decay={temporal_ensemble['decay']:g}, "
                    "first full-width step="
                    f"{temporal_ensemble['first_full_width_step']}"
                ),
                "- Temporal-ensemble continuity changes:",
            ]
        )
        for action_name in ACTION_NAMES:
            action_result = temporal_ensemble["by_action"][action_name]
            report_lines.append(
                f"  - {action_name}: sign flips "
                f"{action_result['latest_prediction_sign_flips']} -> "
                f"{action_result['executed_ensemble_sign_flips']}; "
                "mean absolute step delta reduction="
                f"{action_result['mean_abs_step_delta_reduction_fraction']:.1%}"
            )

    if supervisor_summary.get("available"):
        report_lines.extend(
            [
                (
                    "- Excavation sequence supervisor phases: "
                    + " -> ".join(supervisor_summary["phase_sequence"])
                ),
                (
                    "- Supervisor-modified steps: "
                    f"{supervisor_summary['modified_steps']}"
                ),
                "- Supervisor transitions:",
            ]
        )
        for transition in supervisor_summary["transitions"]:
            report_lines.append(
                f"  - step {transition['policy_step']}: "
                f"{transition['from']} -> {transition['to']} "
                f"({transition['reason']})"
            )

    if prior_summary.get("available"):
        report_lines.extend(
            [
                (
                    "- Stage-action prior direction non-violation rate: "
                    f"{prior_summary['overall_direction_nonviolation_rate']:.4f}"
                ),
                (
                    "- Low-swing-motion rate for canonical stages 1-7 "
                    "(|swing| / prior scale <= 0.10): "
                    f"{prior_summary['low_swing_motion_stage_1_to_7_rate']:.4f}"
                ),
                (
                    "- Prior rows interpreted by canonical v4 numeric stage ID; "
                    "source labels match canonical v4: "
                    f"{prior_summary['source_stage_labels_match_canonical']}"
                ),
            ]
        )

    report_lines.extend(
        [
            "",
            "## Important interpretation limit",
            "",
            (
                "There is no expert ground-truth stage/action in this online "
                "rollout. The prior metric only checks whether an action violates "
                "a reliable sign constraint; zero-valued actions count as "
                "non-violating. These checks do not measure action accuracy or "
                "establish final physical task success."
            ),
            "",
            f"All figures and CSV files are in `{output_dir}`.",
        ]
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 78)
    print("SMOLVLA ROLLOUT ANALYSIS")
    print("=" * 78)
    print(f"steps: {len(step_rows)}")
    print(
        "stage flow:",
        " -> ".join(stage_summary["compressed_stage_names"]),
    )
    print(
        f"stage confidence mean/min: "
        f"{stage_summary['mean_confidence']:.4f} / "
        f"{stage_summary['minimum_confidence']:.4f}"
    )
    print(
        f"backward transitions: "
        f"{stage_summary['num_backward_transitions']}"
    )
    print(
        f"forward skips: "
        f"{stage_summary['num_forward_skips']}"
    )
    if temporal_ensemble.get("available"):
        print(
            "temporal ensemble: "
            f"width={temporal_ensemble['configured_width']} "
            f"decay={temporal_ensemble['decay']:g}"
        )
    print(
        f"prompt parse rate: "
        f"{prompt_result['prompt_parse_rate']:.4f}"
    )
    print(
        "sand prompt vs State27 target mean error:",
        prompt_result.get(
            "prompt_sand_vs_state_target_error_m_mean"
        ),
    )
    print(
        "unload prompt vs State27 target mean error:",
        prompt_result.get(
            "prompt_unload_vs_state_target_error_m_mean"
        ),
    )
    if prior_summary.get("available"):
        print(
            "stage-action prior direction non-violation rate:",
            prior_summary["overall_direction_nonviolation_rate"],
        )
        print(
            "canonical stages 1-7 low-swing-motion rate:",
            prior_summary[
                "low_swing_motion_stage_1_to_7_rate"
            ],
        )
    print(f"outputs: {output_dir}")


if __name__ == "__main__":
    main()
