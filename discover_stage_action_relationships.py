#!/usr/bin/env python3
"""
Discover and validate stage-conditioned excavator action relationships.

The script:
1. Loads LeRobot parquet data.
2. Normalizes each action by its global 95th-percentile absolute magnitude.
3. Generates several candidate relationship families.
4. Validates them with episode-level K-fold splits.
5. Scores and selects relationships that are stable across episodes.

Candidate families:
- direction: one action has a stable sign in a stage
- low_motion: one action stays near zero in a stage
- active: one action is frequently active in a stage
- relative_small: one action is consistently smaller than another
- sign_coupling: two actions usually have the same/opposite sign
- coactivation: two actions are usually active together or mutually exclusive
- dominant_action: one action dominates total command magnitude
- correlation: two action values are strongly correlated
- smoothness: one action changes unusually slowly inside a stage
- monotonic_progress: an action tends to rise/fall through a contiguous stage segment

Only direction, low_motion, and relative_small are marked as immediately
recommended loss candidates. The others are evidence/diagnostic candidates
that should be reviewed before adding a loss.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DEFAULT_STAGE_NAMES = [
    "pre_dig_align",
    "approach_contact",
    "insert",
    "pull_mid",
    "pull_exit",
    "curl",
    "secure_load",
    "lift_carry",
    "unload_to_bin",
    "unload_dump",
]

DEFAULT_ACTION_NAMES = ["arm", "boom", "bucket", "swing"]


@dataclass
class Config:
    folds: int = 5
    seed: int = 17
    active_threshold: float = 0.10
    zero_threshold: float = 0.10
    min_frames: int = 100
    min_episodes: int = 8
    min_fold_frames: int = 20
    min_segment_length: int = 5


def finite_mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def finite_std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.std(vals)) if vals else float("nan")


def finite_min(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.min(vals)) if vals else float("nan")


def finite_max(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(np.max(vals)) if vals else float("nan")


def clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def unwrap_scalar(value):
    arr = np.asarray(value)
    if arr.size == 0:
        return np.nan
    return arr.reshape(-1)[0]


def unpack_vectors(series: pd.Series, name: str) -> np.ndarray:
    vectors = []
    width = None
    for value in series:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if width is None:
            width = arr.size
        if arr.size != width:
            raise ValueError(
                f"Inconsistent vector width in {name}: expected {width}, got {arr.size}"
            )
        vectors.append(arr)
    if not vectors:
        raise ValueError(f"No vectors found for {name}")
    return np.vstack(vectors)


def find_data_parquets(dataset_root: Path, action_col: str, stage_col: str) -> list[Path]:
    preferred = sorted((dataset_root / "data").rglob("*.parquet"))
    candidates = preferred or sorted(dataset_root.rglob("*.parquet"))

    usable: list[Path] = []
    for path in candidates:
        try:
            names = set(pq.ParquetFile(path).schema_arrow.names)
        except Exception:
            continue
        if action_col in names and stage_col in names:
            usable.append(path)

    if not usable:
        raise FileNotFoundError(
            f"No parquet files containing both {action_col!r} and {stage_col!r} "
            f"under {dataset_root}"
        )
    return usable


def load_dataset(
    dataset_root: Path,
    action_col: str,
    stage_col: str,
    episode_col: str,
    frame_col: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    files = find_data_parquets(dataset_root, action_col, stage_col)
    parts = []
    fallback_frame = 0

    for path in files:
        schema_names = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [action_col, stage_col]
        if episode_col in schema_names:
            columns.append(episode_col)
        if frame_col in schema_names:
            columns.append(frame_col)

        part = pq.read_table(path, columns=columns).to_pandas()
        if episode_col not in part:
            part[episode_col] = 0
        if frame_col not in part:
            part[frame_col] = np.arange(fallback_frame, fallback_frame + len(part))
            fallback_frame += len(part)
        parts.append(part)

    frame = pd.concat(parts, ignore_index=True)

    actions = unpack_vectors(frame[action_col], action_col)
    frame["stage_id"] = frame[stage_col].map(unwrap_scalar).astype(int)
    frame["episode_id"] = frame[episode_col].map(unwrap_scalar).astype(int)
    frame["frame_id"] = frame[frame_col].map(unwrap_scalar).astype(int)

    keep = np.isfinite(actions).all(axis=1) & np.isfinite(frame["stage_id"].to_numpy())
    frame = frame.loc[keep, ["stage_id", "episode_id", "frame_id"]].reset_index(drop=True)
    actions = actions[keep]

    order = np.lexsort((frame["frame_id"].to_numpy(), frame["episode_id"].to_numpy()))
    frame = frame.iloc[order].reset_index(drop=True)
    actions = actions[order]

    return frame, actions


def assign_episode_folds(episode_ids: np.ndarray, folds: int, seed: int) -> dict[int, int]:
    unique = np.unique(episode_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    return {int(ep): int(i % folds) for i, ep in enumerate(shuffled)}


def support_summary(mask: np.ndarray, episode_ids: np.ndarray) -> dict:
    return {
        "support_frames": int(mask.sum()),
        "support_episodes": int(np.unique(episode_ids[mask]).size),
    }


def fold_metric_values(
    base_mask: np.ndarray,
    fold_ids: np.ndarray,
    folds: int,
    metric_fn: Callable[[np.ndarray], float],
    min_fold_frames: int,
) -> list[float]:
    values: list[float] = []
    for fold in range(folds):
        mask = base_mask & (fold_ids == fold)
        if int(mask.sum()) < min_fold_frames:
            values.append(float("nan"))
        else:
            values.append(float(metric_fn(mask)))
    return values


def valid_support(mask: np.ndarray, episode_ids: np.ndarray, cfg: Config) -> bool:
    return (
        int(mask.sum()) >= cfg.min_frames
        and int(np.unique(episode_ids[mask]).size) >= cfg.min_episodes
    )


def base_candidate(
    candidate_type: str,
    stage_id: int,
    stage_name: str,
    action_a: str | None,
    action_b: str | None,
    relation: str,
    score: float,
    selected: bool,
    recommended_for_loss: bool,
    mask: np.ndarray,
    episode_ids: np.ndarray,
    fold_values: list[float],
    **metrics,
) -> dict:
    row = {
        "candidate_type": candidate_type,
        "stage_id": int(stage_id),
        "stage_name": stage_name,
        "action_a": action_a or "",
        "action_b": action_b or "",
        "relation": relation,
        "score": float(score),
        "selected": bool(selected),
        "recommended_for_loss": bool(recommended_for_loss and selected),
        **support_summary(mask, episode_ids),
        "fold_values": json.dumps(
            [None if not np.isfinite(v) else round(float(v), 8) for v in fold_values]
        ),
        "fold_mean": finite_mean(fold_values),
        "fold_std": finite_std(fold_values),
        "fold_min": finite_min(fold_values),
        "fold_max": finite_max(fold_values),
    }
    row.update(metrics)
    return row


def discover_direction(
    rows: list[dict],
    x: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    abs_x = np.abs(x)
    active_mask = stage_mask & (abs_x >= cfg.active_threshold)
    if not valid_support(active_mask, episode_ids, cfg):
        return

    positive_fraction = float(np.mean(x[active_mask] > 0))
    direction = 1 if positive_fraction >= 0.5 else -1
    purity = max(positive_fraction, 1.0 - positive_fraction)
    active_rate = float(np.mean(abs_x[stage_mask] >= cfg.active_threshold))

    fold_purity = fold_metric_values(
        active_mask,
        fold_ids,
        cfg.folds,
        lambda m: max(float(np.mean(x[m] > 0)), float(np.mean(x[m] < 0))),
        cfg.min_fold_frames,
    )
    fold_direction = fold_metric_values(
        active_mask,
        fold_ids,
        cfg.folds,
        lambda m: 1.0 if float(np.mean(x[m] > 0)) >= 0.5 else -1.0,
        cfg.min_fold_frames,
    )
    direction_consistency = finite_mean([float(v == direction) for v in fold_direction])

    stability = clip01(1.0 - finite_std(fold_purity) / 0.12)
    strength = math.sqrt(active_rate) * clip01((purity - 0.5) / 0.5)
    score = 100.0 * strength * stability * direction_consistency

    selected = (
        active_rate >= 0.20
        and purity >= 0.85
        and finite_min(fold_purity) >= 0.75
        and direction_consistency >= 0.8
        and score >= 40.0
    )

    rows.append(
        base_candidate(
            "direction",
            stage_id,
            stage_name,
            action_name,
            None,
            "positive" if direction > 0 else "negative",
            score,
            selected,
            True,
            active_mask,
            episode_ids,
            fold_purity,
            active_rate=active_rate,
            dominant_sign_fraction=purity,
            direction_value=direction,
            direction_consistency=direction_consistency,
            median_abs=float(np.median(abs_x[stage_mask])),
            q90_abs=float(np.quantile(abs_x[stage_mask], 0.90)),
        )
    )


def discover_low_motion(
    rows: list[dict],
    x: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    if not valid_support(stage_mask, episode_ids, cfg):
        return

    abs_x = np.abs(x)
    values = abs_x[stage_mask]
    near_zero_rate = float(np.mean(values <= cfg.zero_threshold))
    q50 = float(np.quantile(values, 0.50))
    q90 = float(np.quantile(values, 0.90))
    q95 = float(np.quantile(values, 0.95))

    fold_near_zero = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.mean(abs_x[m] <= cfg.zero_threshold)),
        cfg.min_fold_frames,
    )
    fold_q90 = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.quantile(abs_x[m], 0.90)),
        cfg.min_fold_frames,
    )

    tightness = clip01(1.0 - q90 / 0.30)
    stability = clip01(1.0 - finite_std(fold_near_zero) / 0.12)
    score = 100.0 * (0.65 * near_zero_rate + 0.35 * tightness) * stability

    selected = (
        near_zero_rate >= 0.75
        and q90 <= 0.20
        and finite_min(fold_near_zero) >= 0.65
        and finite_max(fold_q90) <= 0.25
        and score >= 65.0
    )

    rows.append(
        base_candidate(
            "low_motion",
            stage_id,
            stage_name,
            action_name,
            None,
            "near_zero",
            score,
            selected,
            True,
            stage_mask,
            episode_ids,
            fold_near_zero,
            near_zero_rate=near_zero_rate,
            median_abs=q50,
            q90_abs=q90,
            q95_abs=q95,
            allowed_abs_norm=max(cfg.zero_threshold, q90),
            fold_q90_max=finite_max(fold_q90),
        )
    )


def discover_active(
    rows: list[dict],
    x: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    if not valid_support(stage_mask, episode_ids, cfg):
        return
    abs_x = np.abs(x)
    active_rate = float(np.mean(abs_x[stage_mask] >= cfg.active_threshold))
    fold_active = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.mean(abs_x[m] >= cfg.active_threshold)),
        cfg.min_fold_frames,
    )
    stability = clip01(1.0 - finite_std(fold_active) / 0.15)
    score = 100.0 * active_rate * stability
    selected = (
        active_rate >= 0.70
        and finite_min(fold_active) >= 0.60
        and score >= 65.0
    )
    rows.append(
        base_candidate(
            "active",
            stage_id,
            stage_name,
            action_name,
            None,
            "frequently_active",
            score,
            selected,
            False,
            stage_mask,
            episode_ids,
            fold_active,
            active_rate=active_rate,
            median_abs=float(np.median(abs_x[stage_mask])),
            q90_abs=float(np.quantile(abs_x[stage_mask], 0.90)),
        )
    )


def discover_relative_small(
    rows: list[dict],
    xa: np.ndarray,
    xb: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_a: str,
    action_b: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    denom_active = np.abs(xb) >= cfg.active_threshold
    mask = stage_mask & denom_active
    if not valid_support(mask, episode_ids, cfg):
        return

    ratio = np.abs(xa) / (np.abs(xb) + 1e-6)
    q50 = float(np.quantile(ratio[mask], 0.50))
    q90 = float(np.quantile(ratio[mask], 0.90))
    support_rate = float(np.mean(denom_active[stage_mask]))

    fold_q90 = fold_metric_values(
        mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.quantile(ratio[m], 0.90)),
        cfg.min_fold_frames,
    )

    tightness = clip01(1.0 - q90 / 0.75)
    stability = clip01(1.0 - finite_std(fold_q90) / 0.25)
    score = 100.0 * math.sqrt(support_rate) * tightness * stability

    selected = (
        support_rate >= 0.20
        and q90 <= 0.50
        and finite_max(fold_q90) <= 0.75
        and score >= 35.0
    )

    rows.append(
        base_candidate(
            "relative_small",
            stage_id,
            stage_name,
            action_a,
            action_b,
            f"|{action_a}| <= ratio * |{action_b}|",
            score,
            selected,
            True,
            mask,
            episode_ids,
            fold_q90,
            denominator_active_rate=support_rate,
            ratio_median=q50,
            ratio_q90=q90,
            recommended_ratio_limit=finite_max(fold_q90),
        )
    )


def discover_sign_coupling(
    rows: list[dict],
    xa: np.ndarray,
    xb: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_a: str,
    action_b: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    mask = (
        stage_mask
        & (np.abs(xa) >= cfg.active_threshold)
        & (np.abs(xb) >= cfg.active_threshold)
    )
    if not valid_support(mask, episode_ids, cfg):
        return

    same = xa * xb > 0
    same_fraction = float(np.mean(same[mask]))
    relation = "same_sign" if same_fraction >= 0.5 else "opposite_sign"
    purity = max(same_fraction, 1.0 - same_fraction)
    coverage = float(np.mean(mask[stage_mask]))

    fold_purity = fold_metric_values(
        mask,
        fold_ids,
        cfg.folds,
        lambda m: max(float(np.mean(same[m])), float(np.mean(~same[m]))),
        cfg.min_fold_frames,
    )
    fold_relation = fold_metric_values(
        mask,
        fold_ids,
        cfg.folds,
        lambda m: 1.0 if float(np.mean(same[m])) >= 0.5 else -1.0,
        cfg.min_fold_frames,
    )
    target_relation = 1.0 if relation == "same_sign" else -1.0
    consistency = finite_mean([float(v == target_relation) for v in fold_relation])
    stability = clip01(1.0 - finite_std(fold_purity) / 0.12)
    score = (
        100.0
        * math.sqrt(coverage)
        * clip01((purity - 0.5) / 0.5)
        * stability
        * consistency
    )
    selected = (
        coverage >= 0.15
        and purity >= 0.85
        and finite_min(fold_purity) >= 0.75
        and consistency >= 0.8
        and score >= 35.0
    )

    rows.append(
        base_candidate(
            "sign_coupling",
            stage_id,
            stage_name,
            action_a,
            action_b,
            relation,
            score,
            selected,
            False,
            mask,
            episode_ids,
            fold_purity,
            coactive_coverage=coverage,
            relation_purity=purity,
            relation_consistency=consistency,
        )
    )


def discover_coactivation(
    rows: list[dict],
    xa: np.ndarray,
    xb: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_a: str,
    action_b: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    if not valid_support(stage_mask, episode_ids, cfg):
        return

    aa = np.abs(xa) >= cfg.active_threshold
    ab = np.abs(xb) >= cfg.active_threshold
    union = aa | ab
    both = aa & ab

    union_mask = stage_mask & union
    if int(union_mask.sum()) < cfg.min_frames:
        return

    jaccard = float(np.sum(stage_mask & both) / max(np.sum(union_mask), 1))
    active_a = float(np.mean(aa[stage_mask]))
    active_b = float(np.mean(ab[stage_mask]))

    relation = "coactive" if jaccard >= 0.5 else "mutually_exclusive"
    evidence = jaccard if relation == "coactive" else 1.0 - jaccard

    fold_values = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        lambda m: float(
            np.sum(m & both) / max(np.sum(m & union), 1)
        ),
        cfg.min_fold_frames,
    )
    if relation == "mutually_exclusive":
        fold_evidence = [1.0 - v if np.isfinite(v) else v for v in fold_values]
    else:
        fold_evidence = fold_values

    stability = clip01(1.0 - finite_std(fold_evidence) / 0.15)
    score = 100.0 * evidence * math.sqrt(min(active_a, active_b)) * stability
    selected = (
        min(active_a, active_b) >= 0.20
        and evidence >= 0.75
        and finite_min(fold_evidence) >= 0.65
        and score >= 40.0
    )

    rows.append(
        base_candidate(
            "coactivation",
            stage_id,
            stage_name,
            action_a,
            action_b,
            relation,
            score,
            selected,
            False,
            stage_mask,
            episode_ids,
            fold_evidence,
            jaccard=jaccard,
            active_rate_a=active_a,
            active_rate_b=active_b,
            evidence=evidence,
        )
    )


def discover_correlation(
    rows: list[dict],
    xa: np.ndarray,
    xb: np.ndarray,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_a: str,
    action_b: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    if not valid_support(stage_mask, episode_ids, cfg):
        return

    def corr(mask: np.ndarray) -> float:
        a = xa[mask]
        b = xb[mask]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    overall = corr(stage_mask)
    if not np.isfinite(overall):
        return

    fold_corr = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        corr,
        cfg.min_fold_frames,
    )
    sign_consistency = finite_mean(
        [float(np.sign(v) == np.sign(overall)) for v in fold_corr]
    )
    stability = clip01(1.0 - finite_std(fold_corr) / 0.30)
    score = 100.0 * abs(overall) * stability * sign_consistency
    selected = (
        abs(overall) >= 0.70
        and finite_min([abs(v) for v in fold_corr]) >= 0.50
        and sign_consistency >= 0.8
        and score >= 55.0
    )

    rows.append(
        base_candidate(
            "correlation",
            stage_id,
            stage_name,
            action_a,
            action_b,
            "positive_correlation" if overall > 0 else "negative_correlation",
            score,
            selected,
            False,
            stage_mask,
            episode_ids,
            fold_corr,
            correlation=overall,
            sign_consistency=sign_consistency,
        )
    )


def discover_dominant_action(
    rows: list[dict],
    action_matrix: np.ndarray,
    action_idx: int,
    stage_mask: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    if not valid_support(stage_mask, episode_ids, cfg):
        return
    abs_all = np.abs(action_matrix)
    share = abs_all[:, action_idx] / (abs_all.sum(axis=1) + 1e-6)
    dominant_rate = float(np.mean(share[stage_mask] >= 0.50))
    fold_rate = fold_metric_values(
        stage_mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.mean(share[m] >= 0.50)),
        cfg.min_fold_frames,
    )
    stability = clip01(1.0 - finite_std(fold_rate) / 0.15)
    score = 100.0 * dominant_rate * stability
    selected = (
        dominant_rate >= 0.65
        and finite_min(fold_rate) >= 0.55
        and score >= 60.0
    )

    rows.append(
        base_candidate(
            "dominant_action",
            stage_id,
            stage_name,
            action_name,
            None,
            "dominates_l1_command",
            score,
            selected,
            False,
            stage_mask,
            episode_ids,
            fold_rate,
            dominant_rate=dominant_rate,
            median_share=float(np.median(share[stage_mask])),
            q90_share=float(np.quantile(share[stage_mask], 0.90)),
        )
    )


def consecutive_pair_masks(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    ep = frame["episode_id"].to_numpy()
    stage = frame["stage_id"].to_numpy()
    fid = frame["frame_id"].to_numpy()

    valid_prev = np.zeros(len(frame), dtype=bool)
    same_stage_prev = np.zeros(len(frame), dtype=bool)

    valid_prev[1:] = (ep[1:] == ep[:-1]) & (fid[1:] == fid[:-1] + 1)
    same_stage_prev[1:] = valid_prev[1:] & (stage[1:] == stage[:-1])
    return valid_prev, same_stage_prev


def discover_smoothness(
    rows: list[dict],
    x: np.ndarray,
    stage_mask: np.ndarray,
    same_stage_prev: np.ndarray,
    valid_prev: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    cfg: Config,
) -> None:
    delta = np.full_like(x, np.nan, dtype=np.float64)
    delta[1:] = np.abs(x[1:] - x[:-1])

    mask = stage_mask & same_stage_prev
    global_mask = valid_prev
    if not valid_support(mask, episode_ids, cfg):
        return

    global_q90 = float(np.quantile(delta[global_mask], 0.90))
    stage_q50 = float(np.quantile(delta[mask], 0.50))
    stage_q90 = float(np.quantile(delta[mask], 0.90))
    ratio = stage_q90 / max(global_q90, 1e-6)

    fold_ratio = fold_metric_values(
        mask,
        fold_ids,
        cfg.folds,
        lambda m: float(np.quantile(delta[m], 0.90) / max(global_q90, 1e-6)),
        cfg.min_fold_frames,
    )
    stability = clip01(1.0 - finite_std(fold_ratio) / 0.20)
    score = 100.0 * clip01(1.0 - ratio) * stability
    selected = (
        ratio <= 0.60
        and stage_q90 <= 0.20
        and finite_max(fold_ratio) <= 0.80
        and score >= 35.0
    )

    rows.append(
        base_candidate(
            "smoothness",
            stage_id,
            stage_name,
            action_name,
            None,
            "low_within_stage_delta",
            score,
            selected,
            False,
            mask,
            episode_ids,
            fold_ratio,
            median_abs_delta=stage_q50,
            q90_abs_delta=stage_q90,
            global_q90_abs_delta=global_q90,
            q90_ratio_to_global=ratio,
        )
    )


def build_segments(frame: pd.DataFrame, min_length: int) -> list[np.ndarray]:
    ep = frame["episode_id"].to_numpy()
    stage = frame["stage_id"].to_numpy()
    fid = frame["frame_id"].to_numpy()

    segments: list[np.ndarray] = []
    start = 0
    for i in range(1, len(frame) + 1):
        boundary = (
            i == len(frame)
            or ep[i] != ep[i - 1]
            or stage[i] != stage[i - 1]
            or fid[i] != fid[i - 1] + 1
        )
        if boundary:
            if i - start >= min_length:
                segments.append(np.arange(start, i))
            start = i
    return segments


def discover_monotonic_progress(
    rows: list[dict],
    x: np.ndarray,
    segments: list[np.ndarray],
    stage_ids: np.ndarray,
    episode_ids: np.ndarray,
    fold_ids: np.ndarray,
    stage_id: int,
    stage_name: str,
    action_name: str,
    cfg: Config,
) -> None:
    effects = []
    segment_folds = []
    segment_episodes = []

    for idx in segments:
        if int(stage_ids[idx[0]]) != stage_id:
            continue
        y = x[idx]
        progress = np.linspace(0.0, 1.0, len(idx))
        slope = float(np.polyfit(progress, y, 1)[0])
        effects.append(slope)
        segment_folds.append(int(fold_ids[idx[0]]))
        segment_episodes.append(int(episode_ids[idx[0]]))

    if len(effects) < max(20, cfg.min_episodes):
        return

    effects_arr = np.asarray(effects)
    significant = np.abs(effects_arr) >= 0.10
    significant_rate = float(np.mean(significant))
    if int(significant.sum()) < 10:
        return

    positive_fraction = float(np.mean(effects_arr[significant] > 0))
    direction = 1 if positive_fraction >= 0.5 else -1
    purity = max(positive_fraction, 1.0 - positive_fraction)
    median_abs_effect = float(np.median(np.abs(effects_arr[significant])))

    fold_purity = []
    for fold in range(cfg.folds):
        mask = (np.asarray(segment_folds) == fold) & significant
        if int(mask.sum()) < 2:
            fold_purity.append(float("nan"))
        else:
            pos = float(np.mean(effects_arr[mask] > 0))
            fold_purity.append(max(pos, 1.0 - pos))

    stability = clip01(1.0 - finite_std(fold_purity) / 0.15)
    score = (
        100.0
        * significant_rate
        * clip01((purity - 0.5) / 0.5)
        * clip01(median_abs_effect / 0.50)
        * stability
    )
    selected = (
        significant_rate >= 0.40
        and purity >= 0.80
        and finite_min(fold_purity) >= 0.70
        and median_abs_effect >= 0.10
        and score >= 30.0
    )

    mask_for_support = np.isin(
        episode_ids,
        np.unique(np.asarray(segment_episodes, dtype=int)),
    )
    rows.append(
        base_candidate(
            "monotonic_progress",
            stage_id,
            stage_name,
            action_name,
            None,
            "increasing" if direction > 0 else "decreasing",
            score,
            selected,
            False,
            mask_for_support,
            episode_ids,
            fold_purity,
            segment_count=len(effects),
            significant_segment_rate=significant_rate,
            direction_purity=purity,
            median_abs_full_segment_change=median_abs_effect,
        )
    )


def write_report(
    output_dir: Path,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    scales: np.ndarray,
    action_names: list[str],
    stage_names: list[str],
    frame_count: int,
    episode_count: int,
) -> None:
    lines = [
        "# Stage–Action Relationship Discovery",
        "",
        f"- Frames: {frame_count:,}",
        f"- Episodes: {episode_count:,}",
        f"- Selected relationships: {len(selected):,}",
        "",
        "## Action normalization",
        "",
        "Each action is divided by its global 95th-percentile absolute magnitude.",
        "",
        "| Action | Raw scale |",
        "|---|---:|",
    ]
    for name, scale in zip(action_names, scales):
        lines.append(f"| {name} | {scale:.8g} |")

    lines.extend(
        [
            "",
            "## Selected relationships by family",
            "",
        ]
    )

    if selected.empty:
        lines.append("No relationship passed all support, score, and fold-stability thresholds.")
    else:
        for family in sorted(selected["candidate_type"].unique()):
            subset = selected[selected["candidate_type"] == family].sort_values(
                "score", ascending=False
            )
            lines.append(f"### {family}")
            lines.append("")
            for _, row in subset.head(30).iterrows():
                pair = row["action_a"]
                if row["action_b"]:
                    pair += f" / {row['action_b']}"
                recommendation = (
                    " — recommended loss candidate"
                    if bool(row["recommended_for_loss"])
                    else ""
                )
                lines.append(
                    f"- **{row['stage_id']} {row['stage_name']} — {pair}: "
                    f"{row['relation']}**; score={row['score']:.1f}, "
                    f"frames={int(row['support_frames'])}, "
                    f"episodes={int(row['support_episodes'])}{recommendation}"
                )
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- `direction`, `low_motion`, and `relative_small` are the safest first loss candidates.",
            "- Other selected families are evidence that can guide later ablations, but should not be converted into losses without inspection.",
            "- Selection requires episode-level fold stability; a high full-dataset score alone is insufficient.",
            "- A relationship being absent does not mean the model cannot learn it through imitation loss; it only means this script did not find sufficiently stable explicit evidence.",
            "",
            "## Files",
            "",
            "- `all_candidates.csv`: every generated candidate",
            "- `selected_relationships.csv`: candidates passing thresholds",
            "- `recommended_loss_rules.json`: selected candidates considered safe enough for an initial loss ablation",
            "- `stage_action_descriptive_stats.csv`: direct stage/action descriptive statistics",
            "- `analysis_config.json`: thresholds, names, and normalization scales",
        ]
    )

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-col", default="action")
    parser.add_argument("--stage-col", default="observation.stage_current_id")
    parser.add_argument("--episode-col", default="episode_index")
    parser.add_argument("--frame-col", default="frame_index")
    parser.add_argument(
        "--action-names",
        nargs="+",
        default=DEFAULT_ACTION_NAMES,
    )
    parser.add_argument(
        "--stage-names",
        nargs="+",
        default=DEFAULT_STAGE_NAMES,
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--active-threshold", type=float, default=0.10)
    parser.add_argument("--zero-threshold", type=float, default=0.10)
    parser.add_argument("--min-frames", type=int, default=100)
    parser.add_argument("--min-episodes", type=int, default=8)
    args = parser.parse_args()

    cfg = Config(
        folds=args.folds,
        seed=args.seed,
        active_threshold=args.active_threshold,
        zero_threshold=args.zero_threshold,
        min_frames=args.min_frames,
        min_episodes=args.min_episodes,
    )

    dataset_root = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, raw_actions = load_dataset(
        dataset_root,
        args.action_col,
        args.stage_col,
        args.episode_col,
        args.frame_col,
    )

    if raw_actions.shape[1] != len(args.action_names):
        raise ValueError(
            f"Dataset action width is {raw_actions.shape[1]}, but "
            f"{len(args.action_names)} action names were supplied: {args.action_names}"
        )

    max_stage = int(frame["stage_id"].max())
    if max_stage >= len(args.stage_names):
        raise ValueError(
            f"Found stage id {max_stage}, but only {len(args.stage_names)} stage names supplied"
        )

    # Robust per-action normalization.
    scales = np.quantile(np.abs(raw_actions), 0.95, axis=0)
    scales = np.maximum(scales, 1e-6)
    actions = raw_actions / scales

    episode_ids = frame["episode_id"].to_numpy()
    stage_ids = frame["stage_id"].to_numpy()
    fold_map = assign_episode_folds(episode_ids, cfg.folds, cfg.seed)
    fold_ids = np.asarray([fold_map[int(ep)] for ep in episode_ids], dtype=int)

    valid_prev, same_stage_prev = consecutive_pair_masks(frame)
    segments = build_segments(frame, cfg.min_segment_length)

    rows: list[dict] = []
    descriptive_rows: list[dict] = []

    unique_stages = sorted(np.unique(stage_ids).tolist())
    for stage_id in unique_stages:
        stage_name = args.stage_names[stage_id]
        stage_mask = stage_ids == stage_id

        for action_idx, action_name in enumerate(args.action_names):
            x = actions[:, action_idx]
            abs_values = np.abs(x[stage_mask])
            descriptive_rows.append(
                {
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "action": action_name,
                    "frames": int(stage_mask.sum()),
                    "episodes": int(np.unique(episode_ids[stage_mask]).size),
                    "mean": float(np.mean(x[stage_mask])),
                    "std": float(np.std(x[stage_mask])),
                    "median": float(np.median(x[stage_mask])),
                    "mean_abs": float(np.mean(abs_values)),
                    "median_abs": float(np.median(abs_values)),
                    "q75_abs": float(np.quantile(abs_values, 0.75)),
                    "q90_abs": float(np.quantile(abs_values, 0.90)),
                    "q95_abs": float(np.quantile(abs_values, 0.95)),
                    "near_zero_rate": float(
                        np.mean(abs_values <= cfg.zero_threshold)
                    ),
                    "active_rate": float(
                        np.mean(abs_values >= cfg.active_threshold)
                    ),
                    "positive_rate": float(np.mean(x[stage_mask] > 0)),
                    "negative_rate": float(np.mean(x[stage_mask] < 0)),
                }
            )

            discover_direction(
                rows, x, stage_mask, stage_id, stage_name, action_name,
                episode_ids, fold_ids, cfg
            )
            discover_low_motion(
                rows, x, stage_mask, stage_id, stage_name, action_name,
                episode_ids, fold_ids, cfg
            )
            discover_active(
                rows, x, stage_mask, stage_id, stage_name, action_name,
                episode_ids, fold_ids, cfg
            )
            discover_dominant_action(
                rows, actions, action_idx, stage_mask, stage_id, stage_name,
                action_name, episode_ids, fold_ids, cfg
            )
            discover_smoothness(
                rows, x, stage_mask, same_stage_prev, valid_prev,
                stage_id, stage_name, action_name,
                episode_ids, fold_ids, cfg
            )
            discover_monotonic_progress(
                rows, x, segments, stage_ids, episode_ids, fold_ids,
                stage_id, stage_name, action_name, cfg
            )

        for i, action_a in enumerate(args.action_names):
            for j, action_b in enumerate(args.action_names):
                if i == j:
                    continue
                discover_relative_small(
                    rows,
                    actions[:, i],
                    actions[:, j],
                    stage_mask,
                    stage_id,
                    stage_name,
                    action_a,
                    action_b,
                    episode_ids,
                    fold_ids,
                    cfg,
                )

        for i in range(len(args.action_names)):
            for j in range(i + 1, len(args.action_names)):
                action_a = args.action_names[i]
                action_b = args.action_names[j]
                xa = actions[:, i]
                xb = actions[:, j]

                discover_sign_coupling(
                    rows, xa, xb, stage_mask, stage_id, stage_name,
                    action_a, action_b, episode_ids, fold_ids, cfg
                )
                discover_coactivation(
                    rows, xa, xb, stage_mask, stage_id, stage_name,
                    action_a, action_b, episode_ids, fold_ids, cfg
                )
                discover_correlation(
                    rows, xa, xb, stage_mask, stage_id, stage_name,
                    action_a, action_b, episode_ids, fold_ids, cfg
                )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No candidates were generated; inspect dataset columns and thresholds")

    candidates = candidates.sort_values(
        ["selected", "score"],
        ascending=[False, False],
    ).reset_index(drop=True)
    selected = candidates[candidates["selected"]].copy()
    recommended = selected[selected["recommended_for_loss"]].copy()
    descriptive = pd.DataFrame(descriptive_rows)

    candidates.to_csv(output_dir / "all_candidates.csv", index=False)
    selected.to_csv(output_dir / "selected_relationships.csv", index=False)
    descriptive.to_csv(
        output_dir / "stage_action_descriptive_stats.csv",
        index=False,
    )

    recommended_rules = []
    for _, row in recommended.iterrows():
        rule = {
            "candidate_type": row["candidate_type"],
            "stage_id": int(row["stage_id"]),
            "stage_name": row["stage_name"],
            "action_a": row["action_a"],
            "action_b": row["action_b"],
            "relation": row["relation"],
            "score": float(row["score"]),
        }
        for key in [
            "direction_value",
            "dominant_sign_fraction",
            "active_rate",
            "allowed_abs_norm",
            "near_zero_rate",
            "ratio_q90",
            "recommended_ratio_limit",
            "denominator_active_rate",
        ]:
            if key in row and pd.notna(row[key]):
                rule[key] = float(row[key])
        recommended_rules.append(rule)

    (output_dir / "recommended_loss_rules.json").write_text(
        json.dumps(recommended_rules, indent=2),
        encoding="utf-8",
    )

    config_payload = {
        "dataset": str(dataset_root),
        "frames": int(len(frame)),
        "episodes": int(np.unique(episode_ids).size),
        "action_names": args.action_names,
        "stage_names": args.stage_names,
        "action_normalization_q95_abs": {
            name: float(scale)
            for name, scale in zip(args.action_names, scales)
        },
        "folds": cfg.folds,
        "seed": cfg.seed,
        "active_threshold_normalized": cfg.active_threshold,
        "zero_threshold_normalized": cfg.zero_threshold,
        "min_frames": cfg.min_frames,
        "min_episodes": cfg.min_episodes,
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config_payload, indent=2),
        encoding="utf-8",
    )

    write_report(
        output_dir,
        candidates,
        selected,
        scales,
        args.action_names,
        args.stage_names,
        len(frame),
        int(np.unique(episode_ids).size),
    )

    print(f"frames: {len(frame):,}")
    print(f"episodes: {np.unique(episode_ids).size:,}")
    print(f"candidates: {len(candidates):,}")
    print(f"selected: {len(selected):,}")
    print(f"recommended loss candidates: {len(recommended):,}")
    print()
    print("Selected relationships by family:")
    if selected.empty:
        print("  none")
    else:
        print(selected.groupby("candidate_type").size().to_string())
    print()
    print("Top recommended loss candidates:")
    if recommended.empty:
        print("  none")
    else:
        cols = [
            "candidate_type",
            "stage_id",
            "stage_name",
            "action_a",
            "action_b",
            "relation",
            "score",
        ]
        print(recommended[cols].head(30).to_string(index=False))
    print()
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
