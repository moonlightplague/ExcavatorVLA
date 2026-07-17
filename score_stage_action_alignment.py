#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_NAMES = {
    0: "pre_dig_align",
    1: "approach_contact",
    2: "insert",
    3: "pull_mid",
    4: "pull_exit",
    5: "curl",
    6: "secure_load",
    7: "lift_carry",
    8: "unload_to_bin",
    9: "unload_dump",
}

ACTION_NAMES = {
    0: "swing",
    1: "boom",
    2: "arm",
    3: "bucket",
}


@dataclass
class Prior:
    direction: np.ndarray       # [num_stages, action_dim], values in {-1, 0, +1}
    magnitude: np.ndarray       # normalized positive magnitude
    weight: np.ndarray          # reliability/activity weight in [0, 1]
    reliable: np.ndarray        # boolean mask
    active_fraction: np.ndarray
    dominant_fraction: np.ndarray
    median_normalized: np.ndarray
    support: np.ndarray
    action_scale: np.ndarray    # raw-unit normalization per action dimension


def scalar_int(value: Any) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected scalar stage/episode value, got shape {array.shape}")
    return int(array.reshape(-1)[0])


def vector_float(value: Any, expected_dim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_dim is not None and array.size != expected_dim:
        raise ValueError(
            f"Expected vector with {expected_dim} values, got shape {np.asarray(value).shape}"
        )
    return array


def load_dataset_columns(
    dataset_root: Path,
    stage_key: str,
    action_key: str,
    episode_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parquet_files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    all_stages: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_episodes: list[np.ndarray] = []

    action_dim: int | None = None

    for index, path in enumerate(parquet_files, start=1):
        frame = pd.read_parquet(
            path,
            columns=[stage_key, action_key, episode_key],
        )

        stages = np.asarray(
            [scalar_int(value) for value in frame[stage_key]],
            dtype=np.int64,
        )
        episodes = np.asarray(
            [scalar_int(value) for value in frame[episode_key]],
            dtype=np.int64,
        )

        first_action = vector_float(frame[action_key].iloc[0])
        if action_dim is None:
            action_dim = int(first_action.size)
        actions = np.stack(
            [
                vector_float(value, expected_dim=action_dim)
                for value in frame[action_key]
            ],
            axis=0,
        )

        all_stages.append(stages)
        all_actions.append(actions)
        all_episodes.append(episodes)

        print(
            f"[{index:03d}/{len(parquet_files):03d}] "
            f"{path.relative_to(dataset_root)}: {len(frame)} rows"
        )

    stages = np.concatenate(all_stages)
    actions = np.concatenate(all_actions)
    episodes = np.concatenate(all_episodes)

    if actions.shape[1] != 4:
        raise ValueError(
            f"This diagnostic expects 4 excavator actions, got {actions.shape[1]}"
        )

    return stages, actions, episodes


def robust_action_scale(actions: np.ndarray) -> np.ndarray:
    """Use the 90th percentile absolute value to normalize each action dimension."""
    scale = np.quantile(np.abs(actions), 0.90, axis=0)
    fallback = np.quantile(np.abs(actions), 0.75, axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return scale.astype(np.float64)


def fit_prior(
    stages: np.ndarray,
    actions: np.ndarray,
    *,
    num_stages: int,
    zero_threshold: float,
    min_active_fraction: float,
    min_dominant_fraction: float,
    min_support: int,
) -> Prior:
    action_dim = actions.shape[1]
    action_scale = robust_action_scale(actions)
    normalized = actions / action_scale[None, :]

    direction = np.zeros((num_stages, action_dim), dtype=np.int64)
    magnitude = np.ones((num_stages, action_dim), dtype=np.float64)
    weight = np.zeros((num_stages, action_dim), dtype=np.float64)
    reliable = np.zeros((num_stages, action_dim), dtype=bool)
    active_fraction = np.zeros((num_stages, action_dim), dtype=np.float64)
    dominant_fraction = np.zeros((num_stages, action_dim), dtype=np.float64)
    median_normalized = np.zeros((num_stages, action_dim), dtype=np.float64)
    support = np.zeros((num_stages, action_dim), dtype=np.int64)

    for stage_id in range(num_stages):
        stage_mask = stages == stage_id
        stage_actions = normalized[stage_mask]

        if len(stage_actions) == 0:
            continue

        for action_id in range(action_dim):
            values = stage_actions[:, action_id]
            active = np.abs(values) > zero_threshold
            active_values = values[active]

            support[stage_id, action_id] = len(values)
            active_fraction[stage_id, action_id] = float(active.mean())

            if len(active_values) == 0:
                continue

            positive_fraction = float((active_values > 0).mean())
            negative_fraction = float((active_values < 0).mean())

            if positive_fraction >= negative_fraction:
                sign = 1
                dominant = positive_fraction
            else:
                sign = -1
                dominant = negative_fraction

            median_value = float(np.median(active_values))
            target_magnitude = float(np.median(np.abs(active_values)))
            target_magnitude = max(target_magnitude, zero_threshold)

            direction[stage_id, action_id] = sign
            magnitude[stage_id, action_id] = target_magnitude
            dominant_fraction[stage_id, action_id] = dominant
            median_normalized[stage_id, action_id] = median_value

            is_reliable = (
                len(values) >= min_support
                and active_fraction[stage_id, action_id] >= min_active_fraction
                and dominant >= min_dominant_fraction
            )
            reliable[stage_id, action_id] = is_reliable

            if is_reliable:
                sign_strength = max(0.0, (dominant - 0.5) / 0.5)
                activity_strength = min(
                    1.0,
                    active_fraction[stage_id, action_id]
                    / max(min_active_fraction, 1e-8),
                )
                weight[stage_id, action_id] = (
                    sign_strength * activity_strength
                )

    return Prior(
        direction=direction,
        magnitude=magnitude,
        weight=weight,
        reliable=reliable,
        active_fraction=active_fraction,
        dominant_fraction=dominant_fraction,
        median_normalized=median_normalized,
        support=support,
        action_scale=action_scale,
    )


def evaluate_prior(
    prior: Prior,
    stages: np.ndarray,
    actions: np.ndarray,
    *,
    zero_threshold: float,
) -> dict[str, Any]:
    normalized = actions / prior.action_scale[None, :]

    total_weight = 0.0
    weighted_score_sum = 0.0
    weighted_logistic_loss_sum = 0.0
    weighted_sign_correct = 0.0
    weighted_sign_total = 0.0
    covered_positions = 0
    total_positions = int(len(actions) * actions.shape[1])

    stage_accumulators: dict[int, dict[str, float]] = {}

    for stage_id in range(prior.direction.shape[0]):
        row_mask = stages == stage_id
        stage_values = normalized[row_mask]

        accumulator = {
            "weighted_score_sum": 0.0,
            "weighted_loss_sum": 0.0,
            "weight_sum": 0.0,
            "sign_correct": 0.0,
            "sign_total": 0.0,
            "covered_positions": 0.0,
            "total_positions": float(len(stage_values) * actions.shape[1]),
        }

        if len(stage_values) == 0:
            stage_accumulators[stage_id] = accumulator
            continue

        for action_id in range(actions.shape[1]):
            if not prior.reliable[stage_id, action_id]:
                continue

            values = stage_values[:, action_id]
            active = np.abs(values) > zero_threshold
            if not np.any(active):
                continue

            active_values = values[active]
            sign = float(prior.direction[stage_id, action_id])
            magnitude = float(prior.magnitude[stage_id, action_id])
            reliability_weight = float(prior.weight[stage_id, action_id])

            z = sign * active_values / max(magnitude, 1e-8)

            # Dataset alignment score: sigmoid(z). A value of 0.5 is neutral,
            # values above 0.5 indicate the expected direction.
            bounded_score = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

            # Candidate training loss, normalized so zero command has loss 1.
            logistic_loss = np.logaddexp(0.0, -z) / math.log(2.0)

            sample_weight = np.full(
                active_values.shape,
                reliability_weight,
                dtype=np.float64,
            )

            weight_sum = float(sample_weight.sum())
            score_sum = float((sample_weight * bounded_score).sum())
            loss_sum = float((sample_weight * logistic_loss).sum())
            sign_correct = float(
                (sample_weight * (z > 0).astype(np.float64)).sum()
            )

            total_weight += weight_sum
            weighted_score_sum += score_sum
            weighted_logistic_loss_sum += loss_sum
            weighted_sign_correct += sign_correct
            weighted_sign_total += weight_sum
            covered_positions += int(active.sum())

            accumulator["weight_sum"] += weight_sum
            accumulator["weighted_score_sum"] += score_sum
            accumulator["weighted_loss_sum"] += loss_sum
            accumulator["sign_correct"] += sign_correct
            accumulator["sign_total"] += weight_sum
            accumulator["covered_positions"] += float(active.sum())

        stage_accumulators[stage_id] = accumulator

    overall_score = (
        100.0 * weighted_score_sum / total_weight
        if total_weight > 0
        else float("nan")
    )
    directional_loss = (
        weighted_logistic_loss_sum / total_weight
        if total_weight > 0
        else float("nan")
    )
    sign_agreement = (
        100.0 * weighted_sign_correct / weighted_sign_total
        if weighted_sign_total > 0
        else float("nan")
    )
    coverage = (
        100.0 * covered_positions / total_positions
        if total_positions > 0
        else 0.0
    )

    per_stage: list[dict[str, Any]] = []
    for stage_id, acc in stage_accumulators.items():
        weight_sum = acc["weight_sum"]
        sign_total = acc["sign_total"]
        total_stage_positions = acc["total_positions"]

        per_stage.append(
            {
                "stage_id": stage_id,
                "stage_name": STAGE_NAMES.get(stage_id, str(stage_id)),
                "alignment_score": (
                    100.0 * acc["weighted_score_sum"] / weight_sum
                    if weight_sum > 0
                    else float("nan")
                ),
                "directional_logistic_loss": (
                    acc["weighted_loss_sum"] / weight_sum
                    if weight_sum > 0
                    else float("nan")
                ),
                "sign_agreement_percent": (
                    100.0 * acc["sign_correct"] / sign_total
                    if sign_total > 0
                    else float("nan")
                ),
                "coverage_percent": (
                    100.0 * acc["covered_positions"] / total_stage_positions
                    if total_stage_positions > 0
                    else 0.0
                ),
            }
        )

    return {
        "alignment_score": overall_score,
        "directional_logistic_loss": directional_loss,
        "sign_agreement_percent": sign_agreement,
        "coverage_percent": coverage,
        "covered_positions": covered_positions,
        "total_positions": total_positions,
        "per_stage": per_stage,
    }


def prior_table(prior: Prior) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for stage_id in range(prior.direction.shape[0]):
        for action_id in range(prior.direction.shape[1]):
            sign = int(prior.direction[stage_id, action_id])
            rows.append(
                {
                    "stage_id": stage_id,
                    "stage_name": STAGE_NAMES.get(stage_id, str(stage_id)),
                    "action_id": action_id,
                    "action_name": ACTION_NAMES.get(action_id, str(action_id)),
                    "direction_sign": sign,
                    "direction_label": (
                        "positive" if sign > 0 else "negative" if sign < 0 else "none"
                    ),
                    "reliable": bool(prior.reliable[stage_id, action_id]),
                    "weight": float(prior.weight[stage_id, action_id]),
                    "active_fraction": float(
                        prior.active_fraction[stage_id, action_id]
                    ),
                    "dominant_sign_fraction": float(
                        prior.dominant_fraction[stage_id, action_id]
                    ),
                    "median_normalized_action": float(
                        prior.median_normalized[stage_id, action_id]
                    ),
                    "target_magnitude_normalized": float(
                        prior.magnitude[stage_id, action_id]
                    ),
                    "support": int(prior.support[stage_id, action_id]),
                    "raw_action_scale": float(prior.action_scale[action_id]),
                }
            )

    return pd.DataFrame(rows)


def json_ready_prior(prior: Prior) -> dict[str, Any]:
    return {
        "num_stages": int(prior.direction.shape[0]),
        "action_dim": int(prior.direction.shape[1]),
        "stage_names": STAGE_NAMES,
        "action_names": ACTION_NAMES,
        "direction": prior.direction.tolist(),
        "magnitude": prior.magnitude.tolist(),
        "weight": prior.weight.tolist(),
        "reliable": prior.reliable.astype(int).tolist(),
        "action_scale": prior.action_scale.tolist(),
        "candidate_loss": {
            "formula": (
                "weighted_mean(softplus(-direction * action / "
                "(action_scale * magnitude)) / log(2))"
            ),
            "note": (
                "Evaluate this loss only on reliable stage-action pairs. "
                "For SmolVLA flow matching, do not apply it directly to the "
                "predicted velocity field; first form an estimated clean action."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Learn stage-conditioned action-direction priors and evaluate them "
            "with episode-level cross-validation."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--stage-key",
        default="observation.stage_current_id",
    )
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--episode-key", default="episode_index")
    parser.add_argument("--num-stages", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument(
        "--zero-threshold",
        type=float,
        default=0.05,
        help="Normalized absolute action below this value is treated as zero.",
    )
    parser.add_argument(
        "--min-active-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--min-dominant-fraction",
        type=float,
        default=0.65,
    )
    parser.add_argument("--min-support", type=int, default=100)
    args = parser.parse_args()

    dataset_root = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("LOADING TRAINING DATASET")
    print("=" * 92)
    stages, actions, episodes = load_dataset_columns(
        dataset_root,
        args.stage_key,
        args.action_key,
        args.episode_key,
    )

    print()
    print("rows:", len(actions))
    print("episodes:", len(np.unique(episodes)))
    print("action shape:", actions.shape)
    print("stage range:", int(stages.min()), "..", int(stages.max()))

    unique_episodes = np.unique(episodes)
    rng = np.random.default_rng(args.seed)
    shuffled_episodes = unique_episodes.copy()
    rng.shuffle(shuffled_episodes)
    episode_folds = np.array_split(shuffled_episodes, args.folds)

    fold_results: list[dict[str, Any]] = []

    print()
    print("=" * 92)
    print("EPISODE-LEVEL CROSS-VALIDATION")
    print("=" * 92)

    for fold_id, eval_episodes in enumerate(episode_folds):
        eval_mask = np.isin(episodes, eval_episodes)
        train_mask = ~eval_mask

        prior = fit_prior(
            stages[train_mask],
            actions[train_mask],
            num_stages=args.num_stages,
            zero_threshold=args.zero_threshold,
            min_active_fraction=args.min_active_fraction,
            min_dominant_fraction=args.min_dominant_fraction,
            min_support=args.min_support,
        )
        result = evaluate_prior(
            prior,
            stages[eval_mask],
            actions[eval_mask],
            zero_threshold=args.zero_threshold,
        )
        result["fold"] = fold_id
        result["train_rows"] = int(train_mask.sum())
        result["eval_rows"] = int(eval_mask.sum())
        result["eval_episodes"] = int(len(eval_episodes))
        fold_results.append(result)

        print(
            f"fold={fold_id} "
            f"score={result['alignment_score']:.2f} "
            f"sign={result['sign_agreement_percent']:.2f}% "
            f"coverage={result['coverage_percent']:.2f}% "
            f"loss={result['directional_logistic_loss']:.4f}"
        )

    weights = np.asarray(
        [result["covered_positions"] for result in fold_results],
        dtype=np.float64,
    )
    weights = np.where(weights > 0, weights, 0.0)

    def weighted_metric(name: str) -> float:
        values = np.asarray(
            [result[name] for result in fold_results],
            dtype=np.float64,
        )
        valid = np.isfinite(values) & (weights > 0)
        if not np.any(valid):
            return float("nan")
        return float(np.average(values[valid], weights=weights[valid]))

    cv_summary = {
        "alignment_score": weighted_metric("alignment_score"),
        "directional_logistic_loss": weighted_metric(
            "directional_logistic_loss"
        ),
        "sign_agreement_percent": weighted_metric(
            "sign_agreement_percent"
        ),
        "coverage_percent": weighted_metric("coverage_percent"),
        "folds": args.folds,
    }

    print()
    print("=" * 92)
    print("CROSS-VALIDATED DATASET SCORE")
    print("=" * 92)
    print(f"alignment score:       {cv_summary['alignment_score']:.2f} / 100")
    print(
        "sign agreement:       "
        f"{cv_summary['sign_agreement_percent']:.2f}%"
    )
    print(
        "reliable-pair coverage: "
        f"{cv_summary['coverage_percent']:.2f}%"
    )
    print(
        "normalized prior loss: "
        f"{cv_summary['directional_logistic_loss']:.4f}"
    )
    print()
    print("Interpretation:")
    print("  score > 70: strong stage/action directional structure")
    print("  score 60-70: useful but should remain a soft auxiliary loss")
    print("  score 50-60: weak structure; inspect stage/action definitions")
    print("  score near 50: little directional evidence")

    print()
    print("=" * 92)
    print("FITTING FINAL PRIOR ON THE FULL DATASET")
    print("=" * 92)

    final_prior = fit_prior(
        stages,
        actions,
        num_stages=args.num_stages,
        zero_threshold=args.zero_threshold,
        min_active_fraction=args.min_active_fraction,
        min_dominant_fraction=args.min_dominant_fraction,
        min_support=args.min_support,
    )

    table = prior_table(final_prior)
    table_path = output_dir / "stage_action_prior_table.csv"
    table.to_csv(table_path, index=False)

    prior_json_path = output_dir / "stage_action_prior.json"
    with prior_json_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready_prior(final_prior), f, indent=2)

    fold_table = pd.DataFrame(
        [
            {
                "fold": result["fold"],
                "train_rows": result["train_rows"],
                "eval_rows": result["eval_rows"],
                "eval_episodes": result["eval_episodes"],
                "alignment_score": result["alignment_score"],
                "sign_agreement_percent": result[
                    "sign_agreement_percent"
                ],
                "coverage_percent": result["coverage_percent"],
                "directional_logistic_loss": result[
                    "directional_logistic_loss"
                ],
            }
            for result in fold_results
        ]
    )
    fold_path = output_dir / "stage_action_cv_folds.csv"
    fold_table.to_csv(fold_path, index=False)

    per_stage_rows: list[dict[str, Any]] = []
    for result in fold_results:
        for row in result["per_stage"]:
            per_stage_rows.append({"fold": result["fold"], **row})
    per_stage_path = output_dir / "stage_action_cv_per_stage.csv"
    pd.DataFrame(per_stage_rows).to_csv(per_stage_path, index=False)

    report = {
        "dataset": str(dataset_root),
        "rows": int(len(actions)),
        "episodes": int(len(unique_episodes)),
        "parameters": {
            "folds": args.folds,
            "seed": args.seed,
            "zero_threshold": args.zero_threshold,
            "min_active_fraction": args.min_active_fraction,
            "min_dominant_fraction": args.min_dominant_fraction,
            "min_support": args.min_support,
        },
        "cross_validation_summary": cv_summary,
        "fold_results": fold_results,
    }
    report_path = output_dir / "stage_action_alignment_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("prior table:", table_path)
    print("prior JSON:", prior_json_path)
    print("fold scores:", fold_path)
    print("per-stage scores:", per_stage_path)
    print("full report:", report_path)


if __name__ == "__main__":
    main()
