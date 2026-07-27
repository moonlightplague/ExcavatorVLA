#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_smolvla_random50_episodes import (
    ACTION_NAMES,
    aggregate_episode_metrics,
    evaluate_arrays,
    sanitize_metrics,
)


DEFAULT_CHUNKS = (1, 2, 3, 5, 10, 15, 20, 30, 50)
ARRAY_KEYS = (
    "pred_action",
    "true_action",
    "action_valid",
    "pred_stage",
    "true_stage",
    "stage_valid",
    "stage_probability",
)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_prior(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    direction = np.asarray(payload["direction"], dtype=np.float32)
    weight = np.asarray(payload["weight"], dtype=np.float32)
    reliable = np.asarray(payload["reliable"], dtype=bool)
    scale = np.asarray(payload["action_scale"], dtype=np.float32)
    strong = reliable & (weight >= 0.8) & (direction != 0)
    return {
        "direction": direction,
        "weight": weight,
        "reliable": reliable,
        "strong": strong,
        "scale": scale,
    }


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in ARRAY_KEYS if key not in payload]
        if missing:
            raise RuntimeError(f"{path} is missing arrays: {missing}")
        arrays = {key: np.asarray(payload[key]) for key in ARRAY_KEYS}
        for key in ("absolute_index", "frame_index", "timestamp"):
            if key in payload:
                arrays[key] = np.asarray(payload[key])
    return arrays


def validate_episode_arrays(arrays: dict[str, np.ndarray], path: Path) -> None:
    pred_action = arrays["pred_action"]
    pred_stage = arrays["pred_stage"]
    if pred_action.ndim != 3 or pred_action.shape[-1] != 4:
        raise RuntimeError(
            f"{path}: expected pred_action [frames,horizon,4], got {pred_action.shape}"
        )
    if pred_stage.ndim != 2:
        raise RuntimeError(
            f"{path}: expected pred_stage [frames,horizon], got {pred_stage.shape}"
        )
    if pred_action.shape[:2] != pred_stage.shape:
        raise RuntimeError(
            f"{path}: action/stage horizon mismatch: "
            f"{pred_action.shape} versus {pred_stage.shape}"
        )
    for key in ARRAY_KEYS:
        if arrays[key].shape[0] != pred_action.shape[0]:
            raise RuntimeError(f"{path}: inconsistent frame count for {key}")


def stitched_episode(
    arrays: dict[str, np.ndarray],
    execution_chunk_size: int,
    *,
    truth_atol: float = 1e-5,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    num_frames, model_horizon, action_dim = arrays["pred_action"].shape
    if execution_chunk_size < 1 or execution_chunk_size > model_horizon:
        raise ValueError(
            f"Execution chunk {execution_chunk_size} is outside [1, {model_horizon}]"
        )

    pieces: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    source_frames: list[np.ndarray] = []
    chunk_offsets: list[np.ndarray] = []

    for source_frame in range(0, num_frames, execution_chunk_size):
        take = min(execution_chunk_size, num_frames - source_frame)
        target_slice = slice(source_frame, source_frame + take)

        chunk_true_action = arrays["true_action"][source_frame, :take]
        frame_true_action = arrays["true_action"][target_slice, 0]
        valid_action = (
            arrays["action_valid"][source_frame, :take]
            & arrays["action_valid"][target_slice, 0]
        )
        if valid_action.any():
            max_error = float(
                np.max(
                    np.abs(
                        chunk_true_action[valid_action]
                        - frame_true_action[valid_action]
                    )
                )
            )
            if max_error > truth_atol:
                raise RuntimeError(
                    "Action target alignment failed at source frame "
                    f"{source_frame}: max error={max_error}"
                )

        chunk_true_stage = arrays["true_stage"][source_frame, :take]
        frame_true_stage = arrays["true_stage"][target_slice, 0]
        valid_stage = (
            arrays["stage_valid"][source_frame, :take]
            & arrays["stage_valid"][target_slice, 0]
        )
        if valid_stage.any() and not np.array_equal(
            chunk_true_stage[valid_stage],
            frame_true_stage[valid_stage],
        ):
            raise RuntimeError(
                f"Stage target alignment failed at source frame {source_frame}"
            )

        for key in ARRAY_KEYS:
            pieces[key].append(arrays[key][source_frame, :take])
        source_frames.append(np.full(take, source_frame, dtype=np.int64))
        chunk_offsets.append(np.arange(take, dtype=np.int64))

    stitched_flat = {
        key: np.concatenate(value, axis=0)
        for key, value in pieces.items()
    }
    if stitched_flat["pred_action"].shape[0] != num_frames:
        raise RuntimeError("The stitched sequence does not cover every episode frame")

    stitched = {
        "pred_action": stitched_flat["pred_action"][:, None, :],
        "true_action": stitched_flat["true_action"][:, None, :],
        "action_valid": stitched_flat["action_valid"][:, None],
        "pred_stage": stitched_flat["pred_stage"][:, None],
        "true_stage": stitched_flat["true_stage"][:, None],
        "stage_valid": stitched_flat["stage_valid"][:, None],
        "stage_probability": stitched_flat["stage_probability"][:, None, :],
    }

    source_frame_values = np.concatenate(source_frames)
    chunk_offset_values = np.concatenate(chunk_offsets)
    frame_values = arrays.get("frame_index", np.arange(num_frames))
    sequence_rows: list[dict[str, Any]] = []
    for index in range(num_frames):
        pred_action = stitched["pred_action"][index, 0]
        true_action = stitched["true_action"][index, 0]
        row: dict[str, Any] = {
            "frame_index": int(frame_values[index]),
            "source_frame_index": int(source_frame_values[index]),
            "chunk_offset": int(chunk_offset_values[index]),
            "pred_stage": int(stitched["pred_stage"][index, 0]),
            "true_stage": int(stitched["true_stage"][index, 0]),
            "stage_valid": bool(stitched["stage_valid"][index, 0]),
            "stage_correct": bool(
                stitched["pred_stage"][index, 0]
                == stitched["true_stage"][index, 0]
            ),
            "action_valid": bool(stitched["action_valid"][index, 0]),
        }
        for action_id, action_name in enumerate(ACTION_NAMES[:action_dim]):
            row[f"pred_{action_name}"] = float(pred_action[action_id])
            row[f"true_{action_name}"] = float(true_action[action_id])
            row[f"abs_error_{action_name}"] = float(
                abs(pred_action[action_id] - true_action[action_id])
            )
        sequence_rows.append(row)

    return stitched, sequence_rows


def additional_accuracy_metrics(
    arrays: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    action_tolerance_fraction: float,
) -> dict[str, float]:
    pred_action = arrays["pred_action"][:, 0].astype(np.float64)
    true_action = arrays["true_action"][:, 0].astype(np.float64)
    action_valid = arrays["action_valid"][:, 0].astype(bool)
    pred_stage = arrays["pred_stage"][:, 0].astype(np.int64)
    true_stage = arrays["true_stage"][:, 0].astype(np.int64)
    stage_valid = arrays["stage_valid"][:, 0].astype(bool)

    tolerance = (
        action_tolerance_fraction
        * prior["scale"][: pred_action.shape[-1]].astype(np.float64)
    )
    component_correct = np.abs(pred_action - true_action) <= tolerance[None, :]
    vector_correct = np.all(component_correct, axis=1)
    stage_correct = pred_stage == true_stage
    joint_valid = action_valid & stage_valid

    def mean(masked_values: np.ndarray) -> float:
        values = np.asarray(masked_values, dtype=np.float64)
        return float(values.mean()) if values.size else float("nan")

    return {
        "action_vector_within_tolerance_accuracy": mean(
            vector_correct[action_valid]
        ),
        "joint_stage_action_vector_accuracy": mean(
            (stage_correct & vector_correct)[joint_valid]
        ),
    }


def finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def best_chunk(rows: list[dict[str, Any]], metric: str, maximize: bool = True) -> dict[str, Any]:
    candidates = [row for row in rows if finite(row.get(metric)) is not None]
    if not candidates:
        raise RuntimeError(f"No finite values for ranking metric {metric}")
    key = lambda row: (float(row[metric]), -int(row["execution_chunk_size"]))
    winner = max(candidates, key=key) if maximize else min(
        candidates,
        key=lambda row: (float(row[metric]), int(row["execution_chunk_size"])),
    )
    return {
        "execution_chunk_size": int(winner["execution_chunk_size"]),
        "value": float(winner[metric]),
    }


def make_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib is unavailable; plot generation was skipped")
        return

    chunks = [int(row["execution_chunk_size"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    axes[0, 0].plot(
        chunks,
        [row["stage_current_accuracy"] for row in rows],
        marker="o",
        label="stage accuracy",
    )
    axes[0, 0].set_title("Stitched stage accuracy")

    axes[0, 1].plot(
        chunks,
        [row["action_current_within_tolerance_accuracy"] for row in rows],
        marker="o",
        label="action components",
    )
    axes[0, 1].plot(
        chunks,
        [row["action_vector_within_tolerance_accuracy"] for row in rows],
        marker="o",
        label="full action vector",
    )
    axes[0, 1].legend()
    axes[0, 1].set_title("Stitched action accuracy")

    axes[1, 0].plot(
        chunks,
        [row["joint_stage_action_vector_accuracy"] for row in rows],
        marker="o",
        color="tab:purple",
    )
    axes[1, 0].set_title("Joint stage + action-vector accuracy")

    axes[1, 1].plot(
        chunks,
        [row["representative_score"] for row in rows],
        marker="o",
        color="tab:green",
    )
    axes[1, 1].set_title("Evaluator representative score")

    for axis in axes.flat:
        axis.set_xlabel("Executed steps before replanning")
        axis.set_ylabel("Score")
        axis.grid(True, alpha=0.3)
        axis.set_xticks(chunks)
        axis.tick_params(axis="x", rotation=45)

    fig.suptitle("Action/stage chunk stitching sweep")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stitch saved action and stage predictions using different execution "
            "chunk sizes and compare offline replay accuracy."
        )
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=list(DEFAULT_CHUNKS),
    )
    parser.add_argument("--action-tolerance-fraction", type=float, default=0.10)
    parser.add_argument("--action-active-fraction", type=float, default=0.05)
    parser.add_argument("--low-motion-margin", type=float, default=0.10)
    args = parser.parse_args()

    eval_dir = args.eval_dir.expanduser().resolve()
    prior_path = args.prior.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    chunks = sorted(set(args.chunks))
    if chunks != list(DEFAULT_CHUNKS):
        print(f"[INFO] using custom chunk list: {chunks}")

    summary_path = eval_dir / "summary.json"
    episode_dir = eval_dir / "episodes"
    if not summary_path.is_file() or not episode_dir.is_dir():
        raise FileNotFoundError(
            f"Evaluation directory lacks summary.json or episodes/: {eval_dir}"
        )
    if not prior_path.is_file():
        raise FileNotFoundError(prior_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("chunk_size", -1)) < max(chunks):
        raise RuntimeError(
            f"Model horizon {summary.get('chunk_size')} is smaller than {max(chunks)}"
        )

    episode_paths = sorted(episode_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise RuntimeError(f"No episode NPZ files found in {episode_dir}")

    episodes: dict[int, dict[str, np.ndarray]] = {}
    for path in episode_paths:
        episode_id = int(path.stem.split("_")[-1])
        arrays = load_episode(path)
        validate_episode_arrays(arrays, path)
        episodes[episode_id] = arrays

    prior = load_prior(prior_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    sequence_root = output_dir / "stitched_sequences"
    sequence_root.mkdir()

    micro_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for chunk in chunks:
        stitched_by_episode: dict[int, dict[str, np.ndarray]] = {}
        chunk_episode_metrics: list[dict[str, Any]] = []

        for episode_id, arrays in sorted(episodes.items()):
            stitched, sequence_rows = stitched_episode(arrays, chunk)
            metrics, _ = evaluate_arrays(
                stitched,
                prior,
                args.action_tolerance_fraction,
                args.action_active_fraction,
                args.low_motion_margin,
            )
            metrics.update(
                additional_accuracy_metrics(
                    stitched,
                    prior,
                    args.action_tolerance_fraction,
                )
            )
            metrics["episode_id"] = episode_id
            metrics["execution_chunk_size"] = chunk
            chunk_episode_metrics.append(metrics)
            episode_rows.append(sanitize_metrics(metrics))
            stitched_by_episode[episode_id] = stitched

            for row in sequence_rows:
                row["episode_id"] = episode_id
                row["execution_chunk_size"] = chunk
            write_csv(
                sequence_root
                / f"chunk_{chunk:03d}_episode_{episode_id:06d}.csv",
                sequence_rows,
            )

        concatenated = {
            key: np.concatenate(
                [stitched_by_episode[eid][key] for eid in sorted(stitched_by_episode)],
                axis=0,
            )
            for key in ARRAY_KEYS
        }
        micro, _ = evaluate_arrays(
            concatenated,
            prior,
            args.action_tolerance_fraction,
            args.action_active_fraction,
            args.low_motion_margin,
        )
        micro.update(
            additional_accuracy_metrics(
                concatenated,
                prior,
                args.action_tolerance_fraction,
            )
        )
        micro["execution_chunk_size"] = chunk
        micro_rows.append(sanitize_metrics(micro))

        macro = aggregate_episode_metrics(chunk_episode_metrics)["macro_episode_mean"]
        macro["execution_chunk_size"] = chunk
        macro_rows.append(sanitize_metrics(macro))

    baseline = next(row for row in micro_rows if row["execution_chunk_size"] == 1)
    original_micro = summary["micro_all_frames"]
    baseline_checks = {}
    for key in (
        "stage_current_accuracy",
        "action_current_within_tolerance_accuracy",
        "action_current_sign_accuracy",
        "representative_score",
    ):
        difference = abs(float(baseline[key]) - float(original_micro[key]))
        baseline_checks[key] = difference
        if difference > 1e-12:
            raise RuntimeError(
                f"Chunk-1 baseline does not reproduce {key}: difference={difference}"
            )

    ranking_specs = {
        "representative_score": True,
        "joint_stage_action_vector_accuracy": True,
        "stage_current_accuracy": True,
        "action_current_within_tolerance_accuracy": True,
        "action_vector_within_tolerance_accuracy": True,
        "action_current_sign_accuracy": True,
        "swing_current_mae": False,
        "boom_current_mae": False,
        "arm_current_mae": False,
        "bucket_current_mae": False,
    }
    winners = {
        metric: best_chunk(micro_rows, metric, maximize)
        for metric, maximize in ranking_specs.items()
    }

    write_csv(output_dir / "chunk_sweep_micro.csv", micro_rows)
    write_csv(output_dir / "chunk_sweep_macro.csv", macro_rows)
    write_csv(output_dir / "chunk_sweep_episode_metrics.csv", episode_rows)
    make_plot(micro_rows, output_dir / "chunk_stitch_accuracy.png")

    payload = {
        "source_evaluation": str(eval_dir),
        "checkpoint": summary["checkpoint"],
        "checkpoint_step": int(summary["checkpoint_step"]),
        "dataset": summary["dataset"],
        "episode_ids": sorted(episodes),
        "model_output_chunk_size": int(summary["chunk_size"]),
        "execution_chunk_sizes": chunks,
        "protocol": (
            "At source frames 0,K,2K,... take the first K action and stage "
            "predictions from the saved 50-step model output, concatenate them, "
            "and compare them with the aligned expert targets. Replanning uses "
            "recorded expert observations, so this is teacher-forced offline replay, "
            "not closed-loop simulation."
        ),
        "action_tolerance_fraction": args.action_tolerance_fraction,
        "action_active_fraction": args.action_active_fraction,
        "low_motion_margin": args.low_motion_margin,
        "chunk_1_baseline_absolute_differences": baseline_checks,
        "best_micro_chunk_by_metric": winners,
        "micro_results": micro_rows,
        "macro_results": macro_rows,
    }
    json_dump(output_dir / "chunk_sweep_summary.json", payload)

    display_metrics = (
        "execution_chunk_size",
        "representative_score",
        "stage_current_accuracy",
        "action_current_within_tolerance_accuracy",
        "action_vector_within_tolerance_accuracy",
        "joint_stage_action_vector_accuracy",
        "action_current_sign_accuracy",
        "swing_current_mae",
        "boom_current_mae",
        "arm_current_mae",
        "bucket_current_mae",
    )
    print(",".join(display_metrics))
    for row in micro_rows:
        print(",".join(str(row.get(key, "")) for key in display_metrics))

    print()
    print("BEST EXECUTION CHUNKS")
    for metric, winner in winners.items():
        print(
            f"{metric}: chunk={winner['execution_chunk_size']} "
            f"value={winner['value']}"
        )
    print()
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
