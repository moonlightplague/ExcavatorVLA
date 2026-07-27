#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    make_att_2d_masks,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)

try:
    from safetensors import safe_open
except ImportError:
    safe_open = None


STAGE_NAMES = [
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
ACTION_NAMES = ["swing", "boom", "arm", "bucket"]

DEFAULT_DATASET = Path(
    "/root/gpufree-data/ExcavatorVLA/"
    "excavator_auto_dataset/.dashboard_success/"
    "lerobot_v3_stage10_h30_obslabels"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/root/gpufree-data/excavator_final_offline_eval"
)
DEFAULT_TRAIN_ROOT = Path("/root/gpufree-data/outputs/train")
DEFAULT_VLM = Path(
    "/root/gpufree-data/checkpoints/"
    "SmolVLM2-500M-Video-Instruct"
)
DEFAULT_PRIOR = Path(
    "/root/gpufree-data/excavator_stage_action_analysis/"
    "stage_action_prior_training.json"
)

CHECKPOINT_GLOBS = (
    "*lowmotion*add30ep",
    "*stage05*prior020*",
    "*dynamic_stage_seq50*",
)


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def patch_local_vlm_paths(checkpoint: Path, vlm_dir: Path) -> None:
    """Replace Hub VLM names in small JSON files only; weights are untouched."""
    if not vlm_dir.exists():
        raise FileNotFoundError(f"Local VLM directory not found: {vlm_dir}")

    old_names = (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
    )
    target = str(vlm_dir.resolve())

    def replace_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: replace_obj(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [replace_obj(value) for value in obj]
        if isinstance(obj, str):
            for old in old_names:
                obj = obj.replace(old, target)
            return obj
        return obj

    for filename in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ):
        path = checkpoint / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = replace_obj(payload)
        if updated != payload:
            json_dump(path, updated)
            print(f"[INFO] patched local VLM path in {path}")


def checkpoint_step(pretrained_model_dir: Path) -> int:
    state_file = (
        pretrained_model_dir.parent
        / "training_state"
        / "training_step.json"
    )
    if state_file.is_file():
        try:
            return int(json.loads(state_file.read_text())["step"])
        except Exception:
            pass

    match = re.fullmatch(r"0*(\d+)", pretrained_model_dir.parent.name)
    return int(match.group(1)) if match else -1


def resolve_pretrained_model(path: Path) -> Path:
    path = path.expanduser().resolve()

    if (path / "model.safetensors").is_file():
        return path

    if path.name == "checkpoints" or (path / "checkpoints").is_dir():
        checkpoints_dir = path if path.name == "checkpoints" else path / "checkpoints"

        last = checkpoints_dir / "last"
        if last.exists():
            last_resolved = last.resolve()
            candidate = (
                last_resolved
                if (last_resolved / "model.safetensors").is_file()
                else last_resolved / "pretrained_model"
            )
            if (candidate / "model.safetensors").is_file():
                return candidate.resolve()

        candidates = []
        for checkpoint in checkpoints_dir.iterdir():
            if checkpoint.is_symlink() or not checkpoint.is_dir():
                continue
            pretrained = checkpoint / "pretrained_model"
            if (pretrained / "model.safetensors").is_file():
                candidates.append(pretrained)

        if candidates:
            return max(
                candidates,
                key=lambda candidate: (
                    checkpoint_step(candidate),
                    candidate.stat().st_mtime,
                ),
            ).resolve()

    candidate = path / "pretrained_model"
    if (candidate / "model.safetensors").is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Could not resolve a pretrained_model checkpoint from: {path}"
    )


def auto_discover_checkpoint(train_root: Path) -> Path:
    candidates: list[Path] = []
    for pattern in CHECKPOINT_GLOBS:
        for run_dir in train_root.glob(pattern):
            if not run_dir.is_dir():
                continue
            try:
                candidates.append(resolve_pretrained_model(run_dir))
            except FileNotFoundError:
                continue

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found below {train_root} using {CHECKPOINT_GLOBS}"
        )

    # Highest saved global step first, then newest file time.
    return max(
        set(candidates),
        key=lambda candidate: (
            checkpoint_step(candidate),
            candidate.stat().st_mtime,
        ),
    ).resolve()


def metadata_episode_ids(meta: LeRobotDatasetMetadata) -> list[int]:
    episodes = meta.episodes
    if episodes is None:
        return list(range(int(meta.total_episodes)))

    if hasattr(episodes, "columns") and "episode_index" in episodes.columns:
        return [int(value) for value in episodes["episode_index"].tolist()]

    if hasattr(episodes, "index"):
        try:
            return [int(value) for value in episodes.index.tolist()]
        except Exception:
            pass

    return list(range(int(meta.total_episodes)))


def episode_lengths(meta: LeRobotDatasetMetadata) -> dict[int, int]:
    episodes = meta.episodes
    result: dict[int, int] = {}

    if episodes is None or not hasattr(episodes, "iterrows"):
        return result

    for index, row in episodes.iterrows():
        episode_id = int(
            row["episode_index"]
            if "episode_index" in row
            else index
        )

        if (
            "dataset_from_index" in row
            and "dataset_to_index" in row
        ):
            result[episode_id] = int(
                row["dataset_to_index"] - row["dataset_from_index"]
            )
        elif "length" in row:
            result[episode_id] = int(row["length"])

    return result


def collated_tensor(
    batch: dict[str, Any],
    keys: tuple[str, ...],
) -> torch.Tensor | None:
    for key in keys:
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            return value
    return None


def action_padding_mask(
    raw_batch: dict[str, Any],
    frame_index: torch.Tensor,
    episode_index: torch.Tensor,
    horizon: int,
    lengths: dict[int, int],
) -> torch.Tensor:
    existing = collated_tensor(
        raw_batch,
        (
            "action_is_pad",
            "actions_id_pad",
            "action_padding_mask",
        ),
    )
    if existing is not None:
        return existing.to(dtype=torch.bool)

    mask = torch.zeros(
        frame_index.shape[0],
        horizon,
        dtype=torch.bool,
    )
    for batch_i, (frame, episode) in enumerate(
        zip(frame_index.tolist(), episode_index.tolist(), strict=True)
    ):
        episode_length = lengths.get(int(episode))
        if episode_length is None:
            continue
        valid = max(0, min(horizon, episode_length - int(frame)))
        if valid < horizon:
            mask[batch_i, valid:] = True
    return mask


def stage_padding_mask(
    raw_batch: dict[str, Any],
    stage_key: str,
    fallback_action_pad: torch.Tensor,
) -> torch.Tensor:
    existing = raw_batch.get(f"{stage_key}_is_pad")
    if isinstance(existing, torch.Tensor):
        return existing.to(dtype=torch.bool)
    return fallback_action_pad.clone()


def load_action_stats(
    checkpoint: Path,
    action_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = None
    std = None

    if safe_open is not None:
        for filename in (
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
        ):
            path = checkpoint / filename
            if not path.is_file():
                continue
            with safe_open(str(path), framework="pt") as handle:
                for key in handle.keys():
                    tensor = handle.get_tensor(key).cpu().numpy()
                    if key == "action.mean":
                        mean = tensor
                    elif key == "action.std":
                        std = tensor

    if mean is None:
        mean = np.zeros(action_dim, dtype=np.float32)
    if std is None:
        std = np.ones(action_dim, dtype=np.float32)

    mean = np.asarray(mean, dtype=np.float32).reshape(-1)[:action_dim]
    std = np.asarray(std, dtype=np.float32).reshape(-1)[:action_dim]
    std = np.where(std < 1e-8, 1.0, std)

    return mean, std


def unnormalize_action_chunk(
    postprocessor: Any,
    normalized: torch.Tensor,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> torch.Tensor:
    try:
        result = postprocessor(normalized)
        if isinstance(result, dict):
            result = result[ACTION]
        if isinstance(result, torch.Tensor):
            return result.detach().to(dtype=torch.float32, device="cpu")
    except Exception as exc:
        print(
            "[WARN] saved postprocessor could not unnormalize a 3D chunk; "
            f"using action mean/std directly: {exc}"
        )

    mean = torch.as_tensor(
        action_mean,
        dtype=normalized.dtype,
        device=normalized.device,
    )
    std = torch.as_tensor(
        action_std,
        dtype=normalized.dtype,
        device=normalized.device,
    )
    return (
        normalized[..., : mean.numel()] * std + mean
    ).detach().to(dtype=torch.float32, device="cpu")


def build_deterministic_noise(
    absolute_indices: torch.Tensor,
    shape_tail: tuple[int, int],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    chunks = []
    for absolute_index in absolute_indices.tolist():
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + int(absolute_index) * 1009)
        chunks.append(
            torch.randn(
                shape_tail,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
        )
    return torch.stack(chunks, dim=0)


@torch.inference_mode()
def predict_chunk_and_stage_logits(
    policy: SmolVLAPolicy,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reproduce SmolVLA flow integration and expose the custom stage head.

    The stage logits are taken from the final denoising iteration. This adds
    no extra transformer call beyond normal action-chunk inference.
    """
    policy.eval()
    batch = policy._prepare_batch(batch)

    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens = batch[OBS_LANGUAGE_TOKENS]
    lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]

    model = policy.model
    batch_size = state.shape[0]
    device = state.device

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state=state,
    )
    prefix_att_2d_masks = make_att_2d_masks(
        prefix_pad_masks,
        prefix_att_masks,
    )
    prefix_position_ids = torch.cumsum(
        prefix_pad_masks,
        dim=1,
    ) - 1

    _, past_key_values = model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )

    x_t = noise
    dt = -1.0 / model.config.num_steps
    final_stage_logits = None

    for denoise_index in range(model.config.num_steps):
        current_time = 1.0 + denoise_index * dt
        timestep = torch.full(
            (batch_size,),
            current_time,
            dtype=torch.float32,
            device=device,
        )

        suffix_embs, suffix_pad_masks, suffix_att_masks = (
            model.embed_suffix(x_t, timestep)
        )
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_len,
            prefix_len,
        )
        suffix_att_2d_masks = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
        )
        full_att_2d_masks = torch.cat(
            [prefix_pad_2d_masks, suffix_att_2d_masks],
            dim=2,
        )

        prefix_offsets = torch.sum(
            prefix_pad_masks,
            dim=-1,
        )[:, None]
        suffix_position_ids = (
            prefix_offsets
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )

        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=model.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -model.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        velocity = model.action_out_proj(suffix_out)

        stage_head_dtype = next(model.stage_head.parameters()).dtype
        final_stage_logits = model.stage_head(
            suffix_out.to(dtype=stage_head_dtype)
        ).to(dtype=torch.float32)

        x_t = x_t + dt * velocity

    if final_stage_logits is None:
        raise RuntimeError("No denoising iteration was executed")

    action_dim = int(policy.config.action_feature.shape[0])
    return x_t[:, :, :action_dim], final_stage_logits


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def binary_f1(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=bool)
    pred = np.asarray(pred, dtype=bool)

    tp = int(np.sum(true & pred))
    fp = int(np.sum(~true & pred))
    fn = int(np.sum(true & ~pred))

    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    if (
        math.isfinite(precision)
        and math.isfinite(recall)
        and precision + recall > 0
    ):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def multiclass_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    num_classes: int,
) -> tuple[dict[str, float], list[dict[str, float]], np.ndarray]:
    true = np.asarray(true, dtype=np.int64).reshape(-1)
    pred = np.asarray(pred, dtype=np.int64).reshape(-1)
    valid = (
        (true >= 0)
        & (true < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    true = true[valid]
    pred = pred[valid]

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )
    np.add.at(confusion, (true, pred), 1)

    per_class = []
    for stage_id in range(num_classes):
        tp = int(confusion[stage_id, stage_id])
        fp = int(confusion[:, stage_id].sum() - tp)
        fn = int(confusion[stage_id, :].sum() - tp)
        support = int(confusion[stage_id, :].sum())

        precision = tp / (tp + fp) if tp + fp else float("nan")
        recall = tp / (tp + fn) if tp + fn else float("nan")
        if (
            math.isfinite(precision)
            and math.isfinite(recall)
            and precision + recall > 0
        ):
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")

        per_class.append(
            {
                "stage_id": stage_id,
                "stage_name": STAGE_NAMES[stage_id],
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    accuracy = float(np.mean(true == pred)) if true.size else float("nan")
    macro_precision = safe_mean(
        np.array([row["precision"] for row in per_class])
    )
    macro_recall = safe_mean(
        np.array([row["recall"] for row in per_class])
    )
    macro_f1 = safe_mean(
        np.array([row["f1"] for row in per_class])
    )

    return (
        {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        },
        per_class,
        confusion,
    )


def load_prior(
    policy: SmolVLAPolicy,
    prior_path: Path,
    action_dim: int,
) -> dict[str, np.ndarray]:
    if prior_path.is_file():
        payload = json.loads(prior_path.read_text(encoding="utf-8"))
        direction = np.asarray(payload["direction"], dtype=np.float32)
        weight = np.asarray(payload["weight"], dtype=np.float32)
        reliable = np.asarray(payload["reliable"], dtype=bool)
        scale = np.asarray(payload["action_scale"], dtype=np.float32)
        strong = reliable & (weight >= 0.8) & (direction != 0)
        return {
            "direction": direction[:, :action_dim],
            "strong": strong[:, :action_dim],
            "scale": np.maximum(scale[:action_dim], 1e-8),
        }

    model_direction = getattr(
        policy,
        "stage_action_prior_direction",
        None,
    )
    model_strong = getattr(
        policy,
        "stage_action_prior_strong_mask",
        None,
    )
    model_scale = getattr(
        policy,
        "stage_action_prior_action_scale",
        None,
    )
    if all(
        isinstance(value, torch.Tensor)
        for value in (model_direction, model_strong, model_scale)
    ):
        return {
            "direction": model_direction.detach().cpu().numpy()[:, :action_dim],
            "strong": model_strong.detach().cpu().numpy()[:, :action_dim],
            "scale": np.maximum(
                model_scale.detach().cpu().numpy()[:action_dim],
                1e-8,
            ),
        }

    return {
        "direction": np.zeros((len(STAGE_NAMES), action_dim), dtype=np.float32),
        "strong": np.zeros((len(STAGE_NAMES), action_dim), dtype=bool),
        "scale": np.ones(action_dim, dtype=np.float32),
    }


def episode_arrays(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    rows = sorted(rows, key=lambda row: int(row["frame_index"]))
    output: dict[str, np.ndarray] = {}

    scalar_keys = (
        "absolute_index",
        "frame_index",
        "timestamp",
    )
    array_keys = (
        "pred_action",
        "true_action",
        "action_valid",
        "pred_stage",
        "true_stage",
        "stage_valid",
        "stage_probability",
    )

    for key in scalar_keys:
        output[key] = np.asarray([row[key] for row in rows])
    for key in array_keys:
        output[key] = np.stack([row[key] for row in rows], axis=0)

    return output


def evaluate_arrays(
    arrays: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    action_tolerance_fraction: float,
    action_active_fraction: float,
    low_motion_margin: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    pred_action = arrays["pred_action"].astype(np.float64)
    true_action = arrays["true_action"].astype(np.float64)
    action_valid = arrays["action_valid"].astype(bool)

    pred_stage = arrays["pred_stage"].astype(np.int64)
    true_stage = arrays["true_stage"].astype(np.int64)
    stage_valid = arrays["stage_valid"].astype(bool)
    stage_probability = arrays["stage_probability"].astype(np.float64)

    action_dim = pred_action.shape[-1]
    scale = prior["scale"][:action_dim].astype(np.float64)
    tolerance = action_tolerance_fraction * scale
    active_threshold = action_active_fraction * scale

    action_valid_3d = np.repeat(
        action_valid[:, :, None],
        action_dim,
        axis=2,
    )

    absolute_error = np.abs(pred_action - true_action)
    squared_error = np.square(pred_action - true_action)

    within_tolerance = absolute_error <= tolerance[None, None, :]
    active_true = np.abs(true_action) > active_threshold[None, None, :]
    sign_correct = np.sign(pred_action) == np.sign(true_action)

    valid_stage_true = true_stage[stage_valid]
    valid_stage_pred = pred_stage[stage_valid]
    stage_metrics, per_stage, confusion = multiclass_metrics(
        valid_stage_true,
        valid_stage_pred,
        len(STAGE_NAMES),
    )

    current_stage_valid = stage_valid[:, 0]
    current_action_valid = action_valid[:, 0]

    current_true_stage = true_stage[:, 0]
    current_pred_stage = pred_stage[:, 0]
    current_pred_action = pred_action[:, 0, :]
    current_true_action = true_action[:, 0, :]

    current_stage_accuracy = safe_mean(
        (
            current_true_stage[current_stage_valid]
            == current_pred_stage[current_stage_valid]
        ).astype(np.float64)
    )

    current_action_valid_2d = np.repeat(
        current_action_valid[:, None],
        action_dim,
        axis=1,
    )
    current_abs_error = np.abs(
        current_pred_action - current_true_action
    )
    current_within = (
        current_abs_error <= tolerance[None, :]
    )
    current_active = (
        np.abs(current_true_action)
        > active_threshold[None, :]
    )
    current_sign_correct = (
        np.sign(current_pred_action)
        == np.sign(current_true_action)
    )

    # Strong stage-conditioned direction-rule correctness.
    clipped_stage = np.clip(
        true_stage,
        0,
        prior["direction"].shape[0] - 1,
    )
    prior_direction = prior["direction"][clipped_stage]
    prior_strong = prior["strong"][clipped_stage]
    prior_valid = (
        prior_strong
        & stage_valid[:, :, None]
        & action_valid[:, :, None]
    )
    prior_correct = prior_direction * pred_action >= 0

    current_prior_valid = prior_valid[:, 0, :]
    current_prior_correct = prior_correct[:, 0, :]

    # Dataset-validated low-motion rule: stage 1..7, action index 0=swing.
    low_motion_valid = (
        stage_valid
        & action_valid
        & (true_stage >= 1)
        & (true_stage <= 7)
    )
    low_motion_correct = (
        np.abs(pred_action[:, :, 0]) / scale[0]
        <= low_motion_margin
    )

    current_low_motion_valid = low_motion_valid[:, 0]
    current_low_motion_correct = low_motion_correct[:, 0]

    transition_true = (
        current_true_stage[1:] != current_true_stage[:-1]
    )
    transition_pred = (
        current_pred_stage[1:] != current_pred_stage[:-1]
    )
    transition_valid = (
        current_stage_valid[1:] & current_stage_valid[:-1]
    )
    transition_result = binary_f1(
        transition_true[transition_valid],
        transition_pred[transition_valid],
    )

    metrics: dict[str, float] = {
        "num_frames": float(pred_action.shape[0]),
        "stage_current_accuracy": current_stage_accuracy,
        "stage_chunk_accuracy": stage_metrics["accuracy"],
        "stage_chunk_macro_f1": stage_metrics["macro_f1"],
        "stage_current_confidence": safe_mean(
            np.max(stage_probability[:, 0, :], axis=-1)[current_stage_valid]
        ),
        "stage_transition_precision": transition_result["precision"],
        "stage_transition_recall": transition_result["recall"],
        "stage_transition_f1": transition_result["f1"],
        "action_current_within_tolerance_accuracy": safe_mean(
            current_within[current_action_valid_2d].astype(np.float64)
        ),
        "action_chunk_within_tolerance_accuracy": safe_mean(
            within_tolerance[action_valid_3d].astype(np.float64)
        ),
        "action_current_sign_accuracy": safe_mean(
            current_sign_correct[
                current_action_valid_2d & current_active
            ].astype(np.float64)
        ),
        "action_chunk_sign_accuracy": safe_mean(
            sign_correct[
                action_valid_3d & active_true
            ].astype(np.float64)
        ),
        "stage_action_prior_current_accuracy": safe_mean(
            current_prior_correct[current_prior_valid].astype(np.float64)
        ),
        "stage_action_prior_chunk_accuracy": safe_mean(
            prior_correct[prior_valid].astype(np.float64)
        ),
        "low_motion_current_satisfaction": safe_mean(
            current_low_motion_correct[
                current_low_motion_valid
            ].astype(np.float64)
        ),
        "low_motion_chunk_satisfaction": safe_mean(
            low_motion_correct[low_motion_valid].astype(np.float64)
        ),
    }

    for action_id, action_name in enumerate(ACTION_NAMES[:action_dim]):
        current_valid = current_action_valid
        chunk_valid = action_valid

        metrics[f"{action_name}_current_mae"] = safe_mean(
            current_abs_error[current_valid, action_id]
        )
        metrics[f"{action_name}_current_rmse"] = math.sqrt(
            safe_mean(
                np.square(
                    current_pred_action[current_valid, action_id]
                    - current_true_action[current_valid, action_id]
                )
            )
        )
        metrics[f"{action_name}_current_corr"] = safe_corr(
            current_pred_action[current_valid, action_id],
            current_true_action[current_valid, action_id],
        )
        metrics[f"{action_name}_current_within_tolerance_accuracy"] = (
            safe_mean(
                current_within[current_valid, action_id].astype(np.float64)
            )
        )

        active_current_mask = current_valid & current_active[:, action_id]
        metrics[f"{action_name}_current_sign_accuracy"] = safe_mean(
            current_sign_correct[
                active_current_mask,
                action_id,
            ].astype(np.float64)
        )

        metrics[f"{action_name}_chunk_mae"] = safe_mean(
            absolute_error[:, :, action_id][chunk_valid]
        )
        metrics[f"{action_name}_chunk_rmse"] = math.sqrt(
            safe_mean(squared_error[:, :, action_id][chunk_valid])
        )
        metrics[f"{action_name}_chunk_corr"] = safe_corr(
            pred_action[:, :, action_id][chunk_valid],
            true_action[:, :, action_id][chunk_valid],
        )
        metrics[f"{action_name}_chunk_within_tolerance_accuracy"] = (
            safe_mean(
                within_tolerance[:, :, action_id][chunk_valid].astype(
                    np.float64
                )
            )
        )

        active_chunk_mask = chunk_valid & active_true[:, :, action_id]
        metrics[f"{action_name}_chunk_sign_accuracy"] = safe_mean(
            sign_correct[:, :, action_id][active_chunk_mask].astype(
                np.float64
            )
        )

    score_components = np.array(
        [
            metrics["stage_current_accuracy"],
            metrics["action_current_within_tolerance_accuracy"],
            metrics["action_current_sign_accuracy"],
            metrics["stage_action_prior_current_accuracy"],
            metrics["low_motion_current_satisfaction"],
        ],
        dtype=np.float64,
    )
    metrics["representative_score"] = safe_mean(score_components)

    detailed = {
        "per_stage": per_stage,
        "confusion": confusion,
        "action_scale": scale,
        "action_tolerance": tolerance,
        "action_active_threshold": active_threshold,
    }
    return metrics, detailed


def aggregate_episode_metrics(
    episode_metrics: list[dict[str, float]],
) -> dict[str, Any]:
    keys = sorted(
        {
            key
            for row in episode_metrics
            for key, value in row.items()
            if key != "episode_id" and isinstance(value, (int, float))
        }
    )

    macro_mean: dict[str, float | None] = {}
    macro_std: dict[str, float | None] = {}
    for key in keys:
        values = np.asarray(
            [row.get(key, float("nan")) for row in episode_metrics],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        macro_mean[key] = (
            float(finite.mean()) if finite.size else None
        )
        macro_std[key] = (
            float(finite.std(ddof=0)) if finite.size else None
        )

    return {
        "num_episodes": len(episode_metrics),
        "macro_episode_mean": macro_mean,
        "macro_episode_std": macro_std,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_metrics(metrics: dict[str, float]) -> dict[str, Any]:
    return {
        key: finite_or_none(value)
        if isinstance(value, (int, float))
        else value
        for key, value in metrics.items()
    }


def print_important_summary(
    aggregate: dict[str, Any],
    micro_metrics: dict[str, float],
    median_episode: int,
    median_score: float,
    output_dir: Path,
) -> None:
    macro = aggregate["macro_episode_mean"]

    print()
    print("=" * 78)
    print("RANDOM-50 OFFLINE EVALUATION SUMMARY")
    print("=" * 78)
    print(f"median representative episode: {median_episode}")
    print(f"median representative score:   {median_score:.4f}")
    print()
    print("macro mean over 50 episodes / micro over all valid frames")
    print(
        "stage current accuracy:       "
        f"{macro.get('stage_current_accuracy', float('nan')):.4f} / "
        f"{micro_metrics['stage_current_accuracy']:.4f}"
    )
    print(
        "stage 50-step accuracy:       "
        f"{macro.get('stage_chunk_accuracy', float('nan')):.4f} / "
        f"{micro_metrics['stage_chunk_accuracy']:.4f}"
    )
    print(
        "stage 50-step macro F1:       "
        f"{macro.get('stage_chunk_macro_f1', float('nan')):.4f} / "
        f"{micro_metrics['stage_chunk_macro_f1']:.4f}"
    )
    print(
        "action within tolerance:      "
        f"{macro.get('action_current_within_tolerance_accuracy', float('nan')):.4f} / "
        f"{micro_metrics['action_current_within_tolerance_accuracy']:.4f}"
    )
    print(
        "action sign accuracy:         "
        f"{macro.get('action_current_sign_accuracy', float('nan')):.4f} / "
        f"{micro_metrics['action_current_sign_accuracy']:.4f}"
    )
    print(
        "stage-action prior accuracy:  "
        f"{macro.get('stage_action_prior_current_accuracy', float('nan')):.4f} / "
        f"{micro_metrics['stage_action_prior_current_accuracy']:.4f}"
    )
    print(
        "low-motion satisfaction:      "
        f"{macro.get('low_motion_current_satisfaction', float('nan')):.4f} / "
        f"{micro_metrics['low_motion_current_satisfaction']:.4f}"
    )
    print()
    for action_name in ACTION_NAMES:
        print(
            f"{action_name:6s} current "
            f"MAE={micro_metrics[f'{action_name}_current_mae']:.5f} "
            f"RMSE={micro_metrics[f'{action_name}_current_rmse']:.5f} "
            f"corr={micro_metrics[f'{action_name}_current_corr']:.4f} "
            f"tol_acc="
            f"{micro_metrics[f'{action_name}_current_within_tolerance_accuracy']:.4f} "
            f"sign_acc="
            f"{micro_metrics[f'{action_name}_current_sign_accuracy']:.4f}"
        )
    print()
    print(f"all results: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SmolVLA checkpoints on selected dataset episodes and "
            "select a representative episode."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "pretrained_model directory, checkpoint directory, or run "
            "directory. Omit to auto-discover the latest final checkpoint."
        ),
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_TRAIN_ROOT,
    )
    parser.add_argument("--vlm", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--action-tolerance-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--action-active-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--low-motion-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    dataset_root = args.dataset.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    checkpoint = (
        resolve_pretrained_model(args.checkpoint)
        if args.checkpoint is not None
        else auto_discover_checkpoint(args.train_root.expanduser().resolve())
    )
    print(f"[INFO] checkpoint: {checkpoint}")
    print(f"[INFO] saved global step: {checkpoint_step(checkpoint)}")

    patch_local_vlm_paths(
        checkpoint,
        args.vlm.expanduser().resolve(),
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_output_dir = output_dir / "episodes"
    episode_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    repo_id = dataset_root.name
    metadata = LeRobotDatasetMetadata(
        repo_id=repo_id,
        root=dataset_root,
    )
    all_episode_ids = metadata_episode_ids(metadata)
    if args.num_episodes > len(all_episode_ids):
        raise ValueError(
            f"Requested {args.num_episodes} episodes, but dataset has "
            f"{len(all_episode_ids)}"
        )

    rng = random.Random(args.seed)
    selected_episodes = sorted(
        rng.sample(all_episode_ids, args.num_episodes)
    )
    lengths = episode_lengths(metadata)

    print(f"[INFO] selected episodes: {selected_episodes}")
    json_dump(
        output_dir / "selected_episodes.json",
        {
            "seed": args.seed,
            "episode_ids": selected_episodes,
        },
    )

    print("[INFO] loading policy")
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        device=str(device),
        local_files_only=True,
    )
    policy.eval()
    policy.to(device)

    if not hasattr(policy.model, "stage_head"):
        raise RuntimeError(
            "Loaded checkpoint/model has no custom stage_head"
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
    )

    chunk_size = int(policy.config.chunk_size)
    stage_key = str(policy.config.stage_source_key)
    action_dim = int(policy.config.action_feature.shape[0])
    if action_dim != 4:
        raise RuntimeError(
            f"Expected 4 excavator actions, found {action_dim}"
        )

    delta_timestamps = {
        ACTION: [
            step / metadata.fps
            for step in range(chunk_size)
        ],
        stage_key: [
            step / metadata.fps
            for step in range(chunk_size)
        ],
    }

    print("[INFO] loading selected dataset episodes")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        download_videos=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    action_mean, action_std = load_action_stats(
        checkpoint,
        action_dim,
    )
    prior = load_prior(
        policy,
        args.prior.expanduser().resolve(),
        action_dim,
    )

    per_episode_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)

    start_time = time.time()
    progress = tqdm(
        loader,
        desc="Offline inference",
        unit="batch",
    )

    for raw_batch in progress:
        episode_index = raw_batch["episode_index"].to(torch.long)
        frame_index = raw_batch["frame_index"].to(torch.long)
        absolute_index = raw_batch["index"].to(torch.long)
        timestamp = raw_batch["timestamp"].to(torch.float64)

        true_action = raw_batch[ACTION].clone().to(torch.float32)
        true_stage = raw_batch[stage_key].clone().to(torch.long)

        if true_action.ndim != 3:
            raise ValueError(
                f"Expected action [B,T,4], got {tuple(true_action.shape)}"
            )
        if true_stage.ndim != 2:
            raise ValueError(
                f"Expected stage [B,T], got {tuple(true_stage.shape)}"
            )

        action_is_pad = action_padding_mask(
            raw_batch,
            frame_index,
            episode_index,
            chunk_size,
            lengths,
        )
        stage_is_pad = stage_padding_mask(
            raw_batch,
            stage_key,
            action_is_pad,
        )

        processed_batch = preprocessor(raw_batch)
        noise = build_deterministic_noise(
            absolute_index,
            (
                chunk_size,
                int(policy.config.max_action_dim),
            ),
            device,
            args.seed,
        )

        normalized_action, stage_logits = (
            predict_chunk_and_stage_logits(
                policy,
                processed_batch,
                noise,
            )
        )
        predicted_action = unnormalize_action_chunk(
            postprocessor,
            normalized_action,
            action_mean,
            action_std,
        ).numpy()

        stage_probability = torch.softmax(
            stage_logits,
            dim=-1,
        ).detach().to(
            dtype=torch.float32,
            device="cpu",
        ).numpy()
        predicted_stage = np.argmax(
            stage_probability,
            axis=-1,
        ).astype(np.int64)

        true_action_np = true_action.cpu().numpy()
        true_stage_np = true_stage.cpu().numpy()
        action_valid_np = (~action_is_pad).cpu().numpy()
        stage_valid_np = (
            (~stage_is_pad) & (~action_is_pad)
        ).cpu().numpy()

        for batch_i in range(len(episode_index)):
            episode_id = int(episode_index[batch_i])
            per_episode_rows[episode_id].append(
                {
                    "absolute_index": int(absolute_index[batch_i]),
                    "frame_index": int(frame_index[batch_i]),
                    "timestamp": float(timestamp[batch_i]),
                    "pred_action": predicted_action[batch_i],
                    "true_action": true_action_np[batch_i],
                    "action_valid": action_valid_np[batch_i],
                    "pred_stage": predicted_stage[batch_i],
                    "true_stage": true_stage_np[batch_i],
                    "stage_valid": stage_valid_np[batch_i],
                    "stage_probability": stage_probability[batch_i],
                }
            )

    elapsed = time.time() - start_time
    print(f"[INFO] inference completed in {elapsed / 60:.2f} min")

    episode_metrics: list[dict[str, Any]] = []
    episode_arrays_by_id: dict[int, dict[str, np.ndarray]] = {}
    detailed_by_id: dict[int, dict[str, Any]] = {}

    for episode_id in selected_episodes:
        rows = per_episode_rows.get(episode_id)
        if not rows:
            raise RuntimeError(
                f"No inference rows collected for episode {episode_id}"
            )

        arrays = episode_arrays(rows)
        metrics, detailed = evaluate_arrays(
            arrays,
            prior,
            args.action_tolerance_fraction,
            args.action_active_fraction,
            args.low_motion_margin,
        )
        metrics["episode_id"] = episode_id
        episode_metrics.append(metrics)
        episode_arrays_by_id[episode_id] = arrays
        detailed_by_id[episode_id] = detailed

        np.savez_compressed(
            episode_output_dir / f"episode_{episode_id:06d}.npz",
            **arrays,
            confusion=detailed["confusion"],
            action_scale=detailed["action_scale"],
            action_tolerance=detailed["action_tolerance"],
            action_active_threshold=detailed["action_active_threshold"],
        )
        json_dump(
            episode_output_dir / f"episode_{episode_id:06d}_metrics.json",
            {
                "metrics": sanitize_metrics(metrics),
                "per_stage": [
                    {
                        key: finite_or_none(value)
                        if isinstance(value, float)
                        else value
                        for key, value in row.items()
                    }
                    for row in detailed["per_stage"]
                ],
            },
        )

    # Equal episode weighting.
    aggregate = aggregate_episode_metrics(episode_metrics)

    # Micro metrics across all valid frames/tokens.
    concatenated = {
        key: np.concatenate(
            [
                episode_arrays_by_id[episode_id][key]
                for episode_id in selected_episodes
            ],
            axis=0,
        )
        for key in episode_arrays_by_id[selected_episodes[0]]
    }
    micro_metrics, micro_detailed = evaluate_arrays(
        concatenated,
        prior,
        args.action_tolerance_fraction,
        args.action_active_fraction,
        args.low_motion_margin,
    )

    valid_scored = [
        row
        for row in episode_metrics
        if math.isfinite(float(row["representative_score"]))
    ]
    if not valid_scored:
        raise RuntimeError("No finite representative episode score")

    score_values = np.asarray(
        [row["representative_score"] for row in valid_scored],
        dtype=np.float64,
    )
    numerical_median = float(np.median(score_values))
    median_row = min(
        valid_scored,
        key=lambda row: (
            abs(row["representative_score"] - numerical_median),
            row["episode_id"],
        ),
    )
    median_episode = int(median_row["episode_id"])

    json_dump(
        output_dir / "median_episode.json",
        {
            "episode_id": median_episode,
            "representative_score": float(
                median_row["representative_score"]
            ),
            "numerical_median_score": numerical_median,
            "selection_rule": (
                "episode whose representative_score is closest to the "
                "numerical median; ties use the smaller episode_id"
            ),
            "score_components": [
                "stage_current_accuracy",
                "action_current_within_tolerance_accuracy",
                "action_current_sign_accuracy",
                "stage_action_prior_current_accuracy",
                "low_motion_current_satisfaction",
            ],
        },
    )

    write_csv(
        output_dir / "episode_metrics.csv",
        [
            {
                key: (
                    value
                    if not isinstance(value, float)
                    or math.isfinite(value)
                    else ""
                )
                for key, value in row.items()
            }
            for row in sorted(
                episode_metrics,
                key=lambda row: int(row["episode_id"]),
            )
        ],
    )

    ranked_rows = sorted(
        episode_metrics,
        key=lambda row: (
            float(row["representative_score"]),
            int(row["episode_id"]),
        ),
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["score_rank"] = rank
        row["distance_to_median_score"] = abs(
            float(row["representative_score"]) - numerical_median
        )

    write_csv(
        output_dir / "episode_metrics_ranked.csv",
        ranked_rows,
    )

    np.save(
        output_dir / "micro_stage_confusion.npy",
        micro_detailed["confusion"],
    )
    write_csv(
        output_dir / "micro_per_stage_metrics.csv",
        micro_detailed["per_stage"],
    )

    summary_payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step(checkpoint),
        "dataset": str(dataset_root),
        "seed": args.seed,
        "selected_episodes": selected_episodes,
        "num_dataset_frames_evaluated": int(
            sum(
                episode_arrays_by_id[episode_id]["frame_index"].shape[0]
                for episode_id in selected_episodes
            )
        ),
        "chunk_size": chunk_size,
        "action_names": ACTION_NAMES,
        "stage_names": STAGE_NAMES,
        "action_tolerance_fraction_of_q95_scale": (
            args.action_tolerance_fraction
        ),
        "action_active_fraction_of_q95_scale": (
            args.action_active_fraction
        ),
        "low_motion_margin": args.low_motion_margin,
        "action_scale": prior["scale"].tolist(),
        "inference_minutes": elapsed / 60,
        "aggregate": aggregate,
        "micro_all_frames": sanitize_metrics(micro_metrics),
        "median_episode": median_episode,
        "median_representative_score": float(
            median_row["representative_score"]
        ),
    }
    json_dump(output_dir / "summary.json", summary_payload)

    print_important_summary(
        aggregate,
        micro_metrics,
        median_episode,
        float(median_row["representative_score"]),
        output_dir,
    )


if __name__ == "__main__":
    main()
