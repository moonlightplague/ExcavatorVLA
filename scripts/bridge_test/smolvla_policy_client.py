#!/usr/bin/env python3
"""
Latest-checkpoint SmolVLA bridge client.

Deployment contract for the current excavator checkpoint:
  observation.state   : 27 values
  observation.effort  : 4 values
  observation.images.0/.1/.2
  action              : [swing, boom, arm, bucket]

The simulator command is the model's unnormalized 4D action itself.
There is deliberately no:
  - physical stage machine
  - swing suppression
  - boom/bucket gain
  - forced digging/lifting/dumping action
  - action clipping
  - hand-written fallback action

For simulator joints beyond the first four, the command remains zero.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    make_att_2d_masks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from excavator_common.bridge_protocol import (  # noqa: E402
    decode_camera_images,
    decode_rgb_payload as decode_rgb,
    read_json,
    write_json,
)
from excavator_common.deployment_contract import (  # noqa: E402
    OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
    build_client_contract,
    load_training_fps,
    sha256_files,
    validate_training_fps,
)


ACTION_NAMES = ("swing", "boom", "arm", "bucket")
STAGE_NAMES = (
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
)
CAMERA_IDS = ("0", "1", "2")

DEFAULT_TRAIN_ROOT = Path("/root/gpufree-data/outputs/train")
DEFAULT_RUN_PATTERN = "*lowmotion*add30ep*"
DEFAULT_VLM = Path(
    "/root/gpufree-data/checkpoints/"
    "SmolVLM2-500M-Video-Instruct"
)
DEFAULT_DATASET_META = Path(
    "/root/gpufree-data/ExcavatorVLA/"
    "excavator_auto_dataset/.dashboard_success/"
    "lerobot_v3_stage10_h30_obslabels"
)

EXPECTED_STATE_DIM = 27
EXPECTED_EFFORT_DIM = 4
EXPECTED_ACTION_DIM = 4


class MultiCameraVideoRecorder:
    """Write one policy-step frame per camera to aligned MP4 files."""

    def __init__(
        self,
        output_dir: str | Path,
        fps: float,
        codec: str = "mp4v",
        camera_ids: tuple[str, ...] = CAMERA_IDS,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for --record-video-dir. "
                "Install opencv-python in the smolvla environment."
            ) from exc

        if fps <= 0:
            raise ValueError("Video FPS must be positive")
        if len(codec) != 4:
            raise ValueError("Video codec must contain exactly four characters")

        self.cv2 = cv2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = str(codec)
        self.camera_ids = tuple(str(value) for value in camera_ids)

        self.writers: dict[str, Any] = {}
        self.paths: dict[str, Path] = {}
        self.frame_sizes: dict[str, tuple[int, int]] = {}
        self.frames_written = 0
        self.first_step: int | None = None
        self.last_step: int | None = None
        self.closed = False

    @staticmethod
    def _normalize_rgb(frame: np.ndarray, camera_id: str) -> np.ndarray:
        rgb = np.asarray(frame)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise RuntimeError(
                f"Camera {camera_id} must have shape HxWx3/HxWx4; "
                f"got {rgb.shape}"
            )
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
                if maximum <= 1.0:
                    rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return np.ascontiguousarray(rgb)

    def _open_writer(self, camera_id: str, rgb: np.ndarray) -> None:
        height, width = rgb.shape[:2]
        path = self.output_dir / f"camera_{camera_id}.mp4"
        fourcc = self.cv2.VideoWriter_fourcc(*self.codec)
        writer = self.cv2.VideoWriter(
            str(path),
            fourcc,
            self.fps,
            (int(width), int(height)),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"Failed to open video writer for {path} "
                f"with codec {self.codec!r}"
            )

        self.writers[camera_id] = writer
        self.paths[camera_id] = path
        self.frame_sizes[camera_id] = (int(width), int(height))
        print(
            f"[VIDEO] camera={camera_id} path={path} "
            f"size={width}x{height} fps={self.fps:.3f}"
        )

    def write(self, camera_rgbs: dict[str, np.ndarray], step: int) -> None:
        if self.closed:
            raise RuntimeError("Cannot write to a closed recorder")

        normalized: dict[str, np.ndarray] = {}
        for camera_id in self.camera_ids:
            if camera_id not in camera_rgbs:
                raise RuntimeError(
                    f"Missing camera {camera_id}; "
                    f"available={sorted(camera_rgbs)}"
                )
            normalized[camera_id] = self._normalize_rgb(
                camera_rgbs[camera_id],
                camera_id,
            )

        for camera_id, rgb in normalized.items():
            if camera_id not in self.writers:
                self._open_writer(camera_id, rgb)

            width, height = self.frame_sizes[camera_id]
            if rgb.shape[1] != width or rgb.shape[0] != height:
                raise RuntimeError(
                    f"Camera {camera_id} resolution changed from "
                    f"{width}x{height} to {rgb.shape[1]}x{rgb.shape[0]}"
                )

        for camera_id, rgb in normalized.items():
            bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
            self.writers[camera_id].write(bgr)

        if self.first_step is None:
            self.first_step = int(step)
        self.last_step = int(step)
        self.frames_written += 1

        if self.frames_written == 1 or self.frames_written % 300 == 0:
            print(
                f"[VIDEO] frames={self.frames_written} "
                f"last_step={step} "
                f"duration={self.frames_written / self.fps:.3f}s"
            )

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        for writer in self.writers.values():
            writer.release()

        metadata = {
            "fps": self.fps,
            "codec": self.codec,
            "frames_written_per_camera": self.frames_written,
            "first_policy_step": self.first_step,
            "last_policy_step": self.last_step,
            "encoded_duration_seconds": (
                self.frames_written / self.fps
                if self.fps > 0
                else 0.0
            ),
            "warmup_included": False,
            "camera_files": {
                key: str(value)
                for key, value in self.paths.items()
            },
            "frame_sizes": {
                key: [width, height]
                for key, (width, height) in self.frame_sizes.items()
            },
        }
        metadata_path = self.output_dir / "recording_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[VIDEO] closed frames={self.frames_written} "
            f"metadata={metadata_path}"
        )


def read_checkpoint_step(pretrained_model_dir: Path) -> int:
    state_path = (
        pretrained_model_dir.parent
        / "training_state"
        / "training_step.json"
    )
    if state_path.is_file():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            return int(payload["step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    name = pretrained_model_dir.parent.name.lstrip("0")
    return int(name) if name.isdigit() and name else -1


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Accept pretrained_model, checkpoint, checkpoints, or run directory."""
    root = Path(path).expanduser().resolve()

    if (root / "model.safetensors").is_file():
        return root

    direct = root / "pretrained_model"
    if (direct / "model.safetensors").is_file():
        return direct.resolve()

    checkpoints_dir = root
    if (root / "checkpoints").is_dir():
        checkpoints_dir = root / "checkpoints"

    if checkpoints_dir.is_dir():
        candidates: list[Path] = []
        for entry in checkpoints_dir.iterdir():
            if entry.is_symlink() or not entry.is_dir():
                continue
            candidate = entry / "pretrained_model"
            if (candidate / "model.safetensors").is_file():
                candidates.append(candidate.resolve())

        if candidates:
            return max(
                candidates,
                key=lambda item: (
                    read_checkpoint_step(item),
                    item.stat().st_mtime,
                ),
            )

    raise FileNotFoundError(
        f"Could not resolve a SmolVLA pretrained_model from {root}"
    )


def discover_latest_checkpoint(
    train_root: str | Path,
    run_pattern: str,
) -> Path:
    """
    Select the highest saved global-step checkpoint from matching final runs.

    If no directory matches run_pattern, search all run directories as a
    fallback and print a warning.
    """
    root = Path(train_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    matching_runs = [
        path for path in root.glob(run_pattern) if path.is_dir()
    ]
    if not matching_runs:
        print(
            f"[WARN] no run matched {run_pattern!r}; "
            f"falling back to all runs under {root}"
        )
        matching_runs = [
            path for path in root.iterdir() if path.is_dir()
        ]

    candidates: list[Path] = []
    for run_dir in matching_runs:
        checkpoints_dir = run_dir / "checkpoints"
        if not checkpoints_dir.is_dir():
            continue

        for checkpoint_dir in checkpoints_dir.iterdir():
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                continue
            pretrained = checkpoint_dir / "pretrained_model"
            if (pretrained / "model.safetensors").is_file():
                candidates.append(pretrained.resolve())

    if not candidates:
        raise FileNotFoundError(
            f"No saved checkpoint found below {root} "
            f"for pattern {run_pattern!r}"
        )

    selected = max(
        candidates,
        key=lambda item: (
            read_checkpoint_step(item),
            item.stat().st_mtime,
        ),
    )
    return selected


def patch_checkpoint_paths(
    checkpoint_dir: str | Path,
    vlm_dir: str | Path,
) -> None:
    """Replace Hub VLM identifiers with the local VLM path in JSON files."""
    checkpoint = Path(checkpoint_dir)
    local_vlm = str(Path(vlm_dir).expanduser().resolve())

    old_names = (
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct-mlx",
    )

    def replace_object(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: replace_object(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [replace_object(item) for item in value]
        if isinstance(value, str):
            for old_name in old_names:
                value = value.replace(old_name, local_vlm)
        return value

    for filename in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "train_config.json",
    ):
        path = checkpoint / filename
        if not path.is_file():
            continue

        try:
            original = json.loads(path.read_text(encoding="utf-8"))
            updated = replace_object(original)
            if updated != original:
                path.write_text(
                    json.dumps(updated, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"[INFO] patched local VLM path in {path}")
        except Exception as exc:
            raise RuntimeError(f"Failed to patch {path}: {exc}") from exc


def feature_shape(feature: Any) -> tuple[int, ...] | None:
    if isinstance(feature, dict):
        value = feature.get("shape")
    else:
        value = getattr(feature, "shape", None)

    if value is None:
        return None
    return tuple(int(item) for item in value)


def validate_policy_feature_contract(policy: SmolVLAPolicy) -> None:
    config = getattr(policy, "config", None)
    input_features = getattr(config, "input_features", None)
    output_features = getattr(config, "output_features", None)

    if not isinstance(input_features, dict):
        raise RuntimeError(
            "Checkpoint config does not expose input_features"
        )
    if not isinstance(output_features, dict):
        raise RuntimeError(
            "Checkpoint config does not expose output_features"
        )

    state_shape = feature_shape(
        input_features.get("observation.state")
    )
    effort_shape = feature_shape(
        input_features.get("observation.effort")
    )
    action_shape = feature_shape(output_features.get("action"))

    expected_cameras = tuple(
        f"observation.images.{camera_id}"
        for camera_id in CAMERA_IDS
    )
    missing_cameras = [
        key for key in expected_cameras if key not in input_features
    ]

    if state_shape != (EXPECTED_STATE_DIM,):
        raise RuntimeError(
            "This client is for the latest state27 checkpoint: "
            f"expected observation.state {(EXPECTED_STATE_DIM,)}, "
            f"got {state_shape}"
        )
    if effort_shape != (EXPECTED_EFFORT_DIM,):
        raise RuntimeError(
            "This client is for the latest separate-effort checkpoint: "
            f"expected observation.effort {(EXPECTED_EFFORT_DIM,)}, "
            f"got {effort_shape}"
        )
    if action_shape != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(
            f"Expected action {(EXPECTED_ACTION_DIM,)}, got {action_shape}"
        )
    if missing_cameras:
        raise RuntimeError(
            f"Checkpoint is missing camera features: {missing_cameras}"
        )

    print(
        "[MODEL CONTRACT] "
        f"state={state_shape} effort={effort_shape} "
        f"action={action_shape} cameras={list(expected_cameras)}"
    )


def load_normalization_stats(
    checkpoint_dir: str | Path,
) -> dict[str, np.ndarray]:
    checkpoint = Path(checkpoint_dir)
    files = (
        checkpoint
        / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        checkpoint
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )

    required_keys = {
        "observation.state.mean",
        "observation.state.std",
        "observation.effort.mean",
        "observation.effort.std",
        "action.mean",
        "action.std",
    }
    stats: dict[str, np.ndarray] = {}

    for path in files:
        if not path.is_file():
            continue
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key in required_keys:
                    stats[key] = handle.get_tensor(key).cpu().numpy()

    missing = sorted(required_keys - set(stats))
    if missing:
        raise RuntimeError(
            "Checkpoint normalization statistics are incomplete; "
            f"missing={missing}"
        )

    expected_shapes = {
        "observation.state.mean": (EXPECTED_STATE_DIM,),
        "observation.state.std": (EXPECTED_STATE_DIM,),
        "observation.effort.mean": (EXPECTED_EFFORT_DIM,),
        "observation.effort.std": (EXPECTED_EFFORT_DIM,),
        "action.mean": (EXPECTED_ACTION_DIM,),
        "action.std": (EXPECTED_ACTION_DIM,),
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(np.asarray(stats[key]).shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"Normalization shape mismatch for {key}: "
                f"expected {expected_shape}, got {actual_shape}"
            )

    for key in (
        "observation.state.std",
        "observation.effort.std",
        "action.std",
    ):
        stats[key] = np.where(
            np.asarray(stats[key]) < 1e-6,
            1.0,
            np.asarray(stats[key]),
        ).astype(np.float32)

    for key in (
        "observation.state.mean",
        "observation.effort.mean",
        "action.mean",
    ):
        stats[key] = np.asarray(stats[key], dtype=np.float32)

    print(
        "[NORMALIZATION] "
        f"action_mean={stats['action.mean'].round(5).tolist()} "
        f"action_std={stats['action.std'].round(5).tolist()}"
    )
    return stats


def tensor_stats(
    stats: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(value).to(
            device=device,
            dtype=torch.float32,
        )
        for key, value in stats.items()
    }


def extract_first_available(
    reply: dict[str, Any],
    keys: tuple[str, ...],
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in reply or reply[key] is None:
            continue
        value = np.asarray(reply[key], dtype=np.float32).reshape(-1)
        return value, key
    return None, None


def make_state_and_effort(
    reply: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    state, state_key = extract_first_available(
        reply,
        (
            "observation_state",
            "observation.state",
            "state",
        ),
    )
    if state is None:
        raise RuntimeError(
            "Bridge reply has no observation state; "
            f"available keys={sorted(reply)}"
        )

    effort, effort_key = extract_first_available(
        reply,
        (
            "observation_effort",
            "observation.effort",
            "joint_efforts",
            "joint_effort",
            "measured_effort",
            "effort",
        ),
    )

    # Also support a bridge that appends the four efforts to a 27D state.
    if state.size == EXPECTED_STATE_DIM + EXPECTED_EFFORT_DIM:
        appended_effort = state[
            EXPECTED_STATE_DIM:
            EXPECTED_STATE_DIM + EXPECTED_EFFORT_DIM
        ].copy()
        state = state[:EXPECTED_STATE_DIM].copy()
        if effort is None:
            effort = appended_effort
            effort_key = f"{state_key}[{EXPECTED_STATE_DIM}:]"

    if state.size != EXPECTED_STATE_DIM:
        raise RuntimeError(
            f"Expected {EXPECTED_STATE_DIM} observation.state values from "
            f"{state_key!r}, got shape={state.shape}, values={state.tolist()}"
        )

    if effort is None:
        raise RuntimeError(
            f"Expected separate {EXPECTED_EFFORT_DIM}D observation.effort; "
            f"available keys={sorted(reply)}"
        )
    if effort.size < EXPECTED_EFFORT_DIM:
        raise RuntimeError(
            f"Expected at least {EXPECTED_EFFORT_DIM} effort values from "
            f"{effort_key!r}, got shape={effort.shape}"
        )
    effort = effort[:EXPECTED_EFFORT_DIM].copy()

    if not np.all(np.isfinite(state)):
        raise RuntimeError(
            f"State contains NaN/Inf: {state.tolist()}"
        )
    if not np.all(np.isfinite(effort)):
        raise RuntimeError(
            f"Effort contains NaN/Inf: {effort.tolist()}"
        )

    state_tensor = torch.from_numpy(state).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    effort_tensor = torch.from_numpy(effort).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )
    return state_tensor, effort_tensor, str(state_key), str(effort_key)


def rgb_to_tensor(
    rgb: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise RuntimeError(
            f"Expected RGB/RGBA image, got {array.shape}"
        )
    if array.shape[2] == 4:
        array = array[:, :, :3]

    tensor = torch.from_numpy(
        np.ascontiguousarray(array)
    ).to(
        device=device,
        dtype=torch.float32,
    )
    if tensor.numel() and float(tensor.max().item()) > 1.5:
        tensor = tensor / 255.0

    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(
        tensor,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )


def load_language_tokens(
    task: str,
    vlm_dir: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    processor = AutoProcessor.from_pretrained(
        str(Path(vlm_dir).expanduser().resolve()),
        local_files_only=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(
        task,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=48,
    )
    return (
        encoded["input_ids"].to(device),
        encoded["attention_mask"].bool().to(device),
    )


def reset_policy_queue(
    policy: SmolVLAPolicy,
    reason: str,
) -> None:
    if hasattr(policy, "reset"):
        policy.reset()
        print(f"[POLICY RESET] {reason}; discarded cached action chunk")



@torch.inference_mode()
def predict_full_action_chunk_and_stage(
    policy: SmolVLAPolicy,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Predict one complete action chunk and expose the custom stage head.

    Returns:
      normalized_action_chunk: [1, chunk_size, 4]
      stage_ids:              [1, chunk_size]
      stage_confidence:       [1, chunk_size]

    Stage logits are read from the final flow-matching denoising iteration,
    matching the trained custom stage head.
    """
    policy.eval()
    prepared = policy._prepare_batch(batch)

    images, image_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    language_tokens = prepared["observation.language.tokens"]
    language_masks = prepared["observation.language.attention_mask"]

    model = policy.model
    batch_size = state.shape[0]
    device = state.device

    if not hasattr(model, "stage_head"):
        raise RuntimeError(
            "The loaded checkpoint/model does not expose the custom stage_head"
        )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        image_masks,
        language_tokens,
        language_masks,
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

    x_t = torch.randn(
        (
            batch_size,
            int(model.config.chunk_size),
            int(model.config.max_action_dim),
        ),
        dtype=torch.float32,
        device=device,
    )
    dt = -1.0 / float(model.config.num_steps)
    final_stage_logits = None

    for denoise_index in range(int(model.config.num_steps)):
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
        suffix_length = suffix_pad_masks.shape[1]
        prefix_length = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size,
            suffix_length,
            prefix_length,
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

        output_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=model.config.use_cache,
            fill_kv_cache=False,
        )

        suffix_output = output_embeds[1]
        suffix_output = suffix_output[
            :,
            -int(model.config.chunk_size):,
        ].to(dtype=torch.float32)

        velocity = model.action_out_proj(suffix_output)

        stage_head_dtype = next(
            model.stage_head.parameters()
        ).dtype
        final_stage_logits = model.stage_head(
            suffix_output.to(dtype=stage_head_dtype)
        ).to(dtype=torch.float32)

        x_t = x_t + dt * velocity

    if final_stage_logits is None:
        raise RuntimeError("The flow-matching loop produced no stage logits")

    action_dim = int(policy.config.action_feature.shape[0])
    normalized_action_chunk = x_t[:, :, :action_dim]
    stage_probabilities = torch.softmax(
        final_stage_logits,
        dim=-1,
    )
    stage_confidence, stage_ids = torch.max(
        stage_probabilities,
        dim=-1,
    )

    if normalized_action_chunk.shape[1] != len(stage_ids[0]):
        raise RuntimeError(
            "Action/stage chunk length mismatch: "
            f"action={tuple(normalized_action_chunk.shape)}, "
            f"stage={tuple(stage_ids.shape)}"
        )

    return (
        normalized_action_chunk,
        stage_ids,
        stage_confidence,
    )


def build_chunk_payload(
    *,
    replan_step: int,
    action_chunk: np.ndarray,
    stage_ids: np.ndarray,
    stage_confidence: np.ndarray,
    inference_ms: float,
    checkpoint: Path,
) -> dict[str, Any]:
    items = []
    for index in range(action_chunk.shape[0]):
        stage_id = int(stage_ids[index])
        stage_name = (
            STAGE_NAMES[stage_id]
            if 0 <= stage_id < len(STAGE_NAMES)
            else f"unknown_{stage_id}"
        )
        items.append(
            {
                "chunk_index": index,
                "future_policy_step": replan_step + index,
                "stage_id": stage_id,
                "stage_name": stage_name,
                "stage_confidence": float(stage_confidence[index]),
                "action": {
                    action_name: float(action_chunk[index, action_id])
                    for action_id, action_name in enumerate(ACTION_NAMES)
                },
            }
        )

    return {
        "type": "smolvla_action_stage_chunk",
        "replan_step": int(replan_step),
        "chunk_size": int(action_chunk.shape[0]),
        "inference_ms": float(inference_ms),
        "checkpoint": str(checkpoint),
        "items": items,
    }


def emit_chunk_payload(
    payload: dict[str, Any],
    *,
    print_full_chunk: bool,
    chunk_log_handle: Any,
) -> None:
    if print_full_chunk:
        print(
            f"[CHUNK BEGIN] replan_step={payload['replan_step']} "
            f"chunk_size={payload['chunk_size']} "
            f"inference_ms={payload['inference_ms']:.1f}",
            flush=True,
        )
        print(
            json.dumps(payload, indent=2, ensure_ascii=False),
            flush=True,
        )
        print("[CHUNK END]", flush=True)

    if chunk_log_handle is not None:
        chunk_log_handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        chunk_log_handle.flush()


def model_action_to_joint_velocities(
    action: torch.Tensor,
    num_joints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return an unmodified model command for the first four simulator joints.

    No gain, clipping, suppression, stage machine, safety override, smoothing,
    or fallback action is applied.
    """
    model_action = (
        action.detach()
        .to(dtype=torch.float32, device="cpu")
        .numpy()
        .reshape(-1)
    )
    if model_action.size < EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"Expected at least {EXPECTED_ACTION_DIM} policy outputs, "
            f"got {model_action.shape}"
        )

    model_action = model_action[:EXPECTED_ACTION_DIM].copy()
    if not np.all(np.isfinite(model_action)):
        raise RuntimeError(
            f"Model action contains NaN/Inf: {model_action.tolist()}"
        )
    if num_joints < EXPECTED_ACTION_DIM:
        raise RuntimeError(
            f"Bridge exposes only {num_joints} joints, but model outputs "
            f"{EXPECTED_ACTION_DIM}"
        )

    velocities = np.zeros(num_joints, dtype=np.float32)
    velocities[:EXPECTED_ACTION_DIM] = model_action
    return velocities, model_action


def percentile_summary(values: list[float]) -> str:
    if not values:
        return "n=0"
    array = np.asarray(values, dtype=np.float64)
    return (
        f"p50={np.percentile(array, 50):.1f} "
        f"p95={np.percentile(array, 95):.1f} "
        f"max={np.max(array):.1f} n={len(array)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the latest state27+effort4 SmolVLA checkpoint through "
            "the simulator bridge using only the model's own action."
        )
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)

    parser.add_argument(
        "--ckpt",
        default=os.environ.get("SMOLVLA_CKPT", ""),
        help=(
            "pretrained_model, checkpoint, checkpoints, or run directory. "
            "When omitted, the latest matching checkpoint is discovered."
        ),
    )
    parser.add_argument(
        "--train-root",
        default=os.environ.get(
            "SMOLVLA_TRAIN_ROOT",
            str(DEFAULT_TRAIN_ROOT),
        ),
    )
    parser.add_argument(
        "--run-pattern",
        default=os.environ.get(
            "SMOLVLA_RUN_PATTERN",
            DEFAULT_RUN_PATTERN,
        ),
        help="Run-directory glob used for automatic checkpoint discovery.",
    )
    parser.add_argument(
        "--vlm",
        default=os.environ.get("SMOLVLA_VLM", str(DEFAULT_VLM)),
    )
    parser.add_argument(
        "--dataset-meta",
        default=os.environ.get(
            "SMOLVLA_DATASET_META",
            str(DEFAULT_DATASET_META),
        ),
        help="Dataset root or meta/info.json; its FPS drives bridge cadence.",
    )
    parser.add_argument(
        "--training-fps",
        type=float,
        default=0.0,
        help="Diagnostic fallback when dataset metadata is unavailable.",
    )

    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--replan-interval",
        type=int,
        default=10,
        help=(
            "Synchronously infer a new full action/stage chunk every N "
            "deployed steps. Default: 10. The unused tail of the previous "
            "chunk is discarded."
        ),
    )
    parser.add_argument(
        "--chunk-log",
        default=(
            "/root/gpufree-data/excavator_logs/"
            "smolvla_action_stage_chunks.jsonl"
        ),
        help=(
            "JSONL path for every full action/stage chunk. "
            "Use an empty string to disable file logging."
        ),
    )
    parser.add_argument(
        "--no-print-full-chunk",
        action="store_true",
        help="Do not print the complete 50-step chunk to stdout.",
    )
    parser.add_argument(
        "--trace-log",
        default=(
            "/root/gpufree-data/excavator_logs/"
            "smolvla_policy_steps_trace.jsonl"
        ),
        help=(
            "JSONL path for every deployed policy step, including prompt, "
            "raw State27, predicted stage, and executed model action. "
            "Use an empty string to disable."
        ),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=1)

    parser.add_argument(
        "--record-video-dir",
        default="",
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=0.0,
        help="0 uses checkpoint training FPS.",
    )
    parser.add_argument(
        "--record-video-codec",
        default="mp4v",
    )

    parser.add_argument(
        "--dump-dir",
        default="",
        help="Optionally save the first model inputs for bridge debugging.",
    )
    parser.add_argument(
        "--dump-input-steps",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if args.steps < 0:
        raise SystemExit("--steps must be non-negative")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative")
    if args.replan_interval <= 0:
        raise SystemExit("--replan-interval must be positive")
    if args.print_every <= 0:
        raise SystemExit("--print-every must be positive")
    if args.record_video_fps < 0:
        raise SystemExit("--record-video-fps must be non-negative")
    if len(args.record_video_codec) != 4:
        raise SystemExit(
            "--record-video-codec must contain exactly four characters"
        )

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] device={device}")

    if args.ckpt:
        checkpoint = resolve_checkpoint_path(args.ckpt)
    else:
        checkpoint = discover_latest_checkpoint(
            args.train_root,
            args.run_pattern,
        )

    vlm_dir = Path(args.vlm).expanduser().resolve()
    if not vlm_dir.is_dir():
        raise FileNotFoundError(vlm_dir)

    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] checkpoint_global_step={read_checkpoint_step(checkpoint)}")
    print(f"[INFO] local_vlm={vlm_dir}")
    print(
        "[ACTION MODE] pure model output: "
        "[swing, boom, arm, bucket] -> joint_velocities[0:4]"
    )
    print(
        "[ACTION MODE] no auxiliary action, gain, clipping, "
        "suppression, state machine, smoothing, or fallback"
    )

    patch_checkpoint_paths(checkpoint, vlm_dir)

    print("[INFO] loading SmolVLA policy")
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint))
    policy.eval()
    policy.to(device)
    validate_policy_feature_contract(policy)
    if not hasattr(policy.model, "stage_head"):
        raise RuntimeError(
            "Latest checkpoint does not contain the trained stage_head"
        )
    print(
        "[CHUNK MODE] "
        f"model_chunk_size={int(policy.model.config.chunk_size)} "
        f"replan_interval={args.replan_interval}"
    )
    reset_policy_queue(policy, "initialization")

    stats_np = load_normalization_stats(checkpoint)
    stats = tensor_stats(stats_np, device)

    dataset_meta = str(Path(args.dataset_meta).expanduser())
    if dataset_meta:
        training_fps, dataset_meta_path = load_training_fps(dataset_meta)
        if (
            args.training_fps > 0
            and abs(training_fps - args.training_fps) > 1e-6
        ):
            raise RuntimeError(
                f"--training-fps={args.training_fps} disagrees with "
                f"{dataset_meta_path}: {training_fps}"
            )
        print(
            f"[BRIDGE CONTRACT] training_fps={training_fps} "
            f"source={dataset_meta_path}"
        )
    elif args.training_fps > 0:
        training_fps = validate_training_fps(args.training_fps)
        print(
            "[WARN] training FPS supplied explicitly: "
            f"{training_fps}"
        )
    else:
        raise RuntimeError(
            "Set --dataset-meta or --training-fps"
        )

    normalization_paths = (
        checkpoint
        / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        checkpoint
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )
    normalization_hash = sha256_files(normalization_paths)
    if not normalization_hash:
        raise RuntimeError(
            "Could not hash checkpoint normalization assets"
        )

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    handshake = build_client_contract(
        training_fps,
        normalization_hash=normalization_hash,
        observation_schema=OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
    )
    write_json(sock, handshake)
    handshake_reply = read_json(sock)
    if not handshake_reply.get("ok", False):
        raise RuntimeError(
            f"Bridge handshake rejected: {handshake_reply}"
        )
    if (
        handshake_reply.get("normalization_hash", "")
        != normalization_hash
    ):
        raise RuntimeError(
            "Bridge handshake normalization identity mismatch"
        )
    print(f"[BRIDGE CONTRACT] accepted={handshake_reply}")

    video_recorder: MultiCameraVideoRecorder | None = None
    if args.record_video_dir:
        output_fps = (
            args.record_video_fps
            if args.record_video_fps > 0
            else training_fps
        )
        video_recorder = MultiCameraVideoRecorder(
            args.record_video_dir,
            fps=output_fps,
            codec=args.record_video_codec,
        )

    dump_dir = Path(args.dump_dir).expanduser()
    if args.dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    chunk_log_handle = None
    if args.chunk_log:
        chunk_log_path = Path(args.chunk_log).expanduser().resolve()
        chunk_log_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_log_handle = chunk_log_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        print(f"[CHUNK LOG] {chunk_log_path}")

    trace_log_handle = None
    if args.trace_log:
        trace_log_path = Path(args.trace_log).expanduser().resolve()
        trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        trace_log_handle = trace_log_path.open(
            "w",
            encoding="utf-8",
            buffering=1,
        )
        print(f"[TRACE LOG] {trace_log_path}")

    cmd: dict[str, list[float]] = {
        "joint_velocities": [0.0] * EXPECTED_ACTION_DIM
    }
    task_text: str | None = None
    language_tokens: torch.Tensor | None = None
    language_mask: torch.Tensor | None = None

    current_action_chunk: np.ndarray | None = None
    current_stage_ids: np.ndarray | None = None
    current_stage_confidence: np.ndarray | None = None
    current_chunk_cursor = 0
    current_chunk_origin_step: int | None = None

    timing_samples: dict[str, list[float]] = {
        "roundtrip_ms": [],
        "client_prepare_ms": [],
        "inference_ms": [],
        "loop_ms": [],
        "server_total_ms": [],
        "physics_ms": [],
        "camera_render_copy_ms": [],
        "camera_resize_ms": [],
        "encode_ms": [],
        "bucket_load_ms": [],
    }
    simulated_seconds_total = 0.0
    deployment_wall_start = time.perf_counter()

    raw_q = np.empty(0, dtype=np.float32)

    try:
        total_loop_steps = args.warmup_steps + args.steps

        for loop_step in range(total_loop_steps):
            is_warmup = loop_step < args.warmup_steps
            policy_step = loop_step - args.warmup_steps

            loop_start = time.perf_counter()
            write_json(sock, cmd)
            reply = read_json(sock)
            reply_received = time.perf_counter()

            if reply.get("type") == "error":
                raise RuntimeError(
                    f"Bridge error: {reply.get('error')}"
                )

            roundtrip_ms = (
                reply_received - loop_start
            ) * 1000.0

            raw_q = np.asarray(
                reply.get("joint_positions", []),
                dtype=np.float32,
            ).reshape(-1)
            num_joints = (
                int(raw_q.size)
                if raw_q.size > 0
                else EXPECTED_ACTION_DIM
            )

            bridge_task = str(reply.get("task_text", "")).strip()
            if not bridge_task:
                raise RuntimeError(
                    "Bridge reply is missing task_text; "
                    f"available keys={sorted(reply)}"
                )
            if bridge_task != task_text:
                task_text = bridge_task
                language_tokens, language_mask = load_language_tokens(
                    task_text,
                    vlm_dir,
                    device,
                )
                reset_policy_queue(
                    policy,
                    f"task changed to {task_text!r}",
                )
                current_action_chunk = None
                current_stage_ids = None
                current_stage_confidence = None
                current_chunk_cursor = 0
                current_chunk_origin_step = None
                print(f"[INFO] task_text={task_text}")

            fallback_rgb = decode_rgb(reply)
            camera_rgbs = decode_camera_images(
                reply,
                np_module=np,
            )
            rgb0 = camera_rgbs.get("0", fallback_rgb)
            rgb1 = camera_rgbs.get("1", fallback_rgb)
            rgb2 = camera_rgbs.get("2", fallback_rgb)

            if video_recorder is not None and not is_warmup:
                video_recorder.write(
                    {"0": rgb0, "1": rgb1, "2": rgb2},
                    policy_step,
                )

            state_raw, effort_raw, state_key, effort_key = (
                make_state_and_effort(reply, device)
            )
            state = (
                state_raw - stats["observation.state.mean"]
            ) / stats["observation.state.std"]
            effort = (
                effort_raw - stats["observation.effort.mean"]
            ) / stats["observation.effort.std"]

            image0 = rgb_to_tensor(rgb0, device)
            image1 = rgb_to_tensor(rgb1, device)
            image2 = rgb_to_tensor(rgb2, device)

            if (
                args.dump_dir
                and not is_warmup
                and policy_step < args.dump_input_steps
            ):
                step_dir = dump_dir / f"step_{policy_step:04d}"
                step_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.asarray(rgb0)).save(
                    step_dir / "image_0.png"
                )
                Image.fromarray(np.asarray(rgb1)).save(
                    step_dir / "image_1.png"
                )
                Image.fromarray(np.asarray(rgb2)).save(
                    step_dir / "image_2.png"
                )
                np.savetxt(
                    step_dir / "state_raw.txt",
                    state_raw.squeeze(0).cpu().numpy(),
                    fmt="%.8f",
                )
                np.savetxt(
                    step_dir / "state_normalized.txt",
                    state.squeeze(0).cpu().numpy(),
                    fmt="%.8f",
                )
                np.savetxt(
                    step_dir / "effort_raw.txt",
                    effort_raw.squeeze(0).cpu().numpy(),
                    fmt="%.8f",
                )
                np.savetxt(
                    step_dir / "effort_normalized.txt",
                    effort.squeeze(0).cpu().numpy(),
                    fmt="%.8f",
                )
                (step_dir / "task.txt").write_text(
                    task_text,
                    encoding="utf-8",
                )

            batch = {
                "observation.state": state,
                "observation.effort": effort,
                "observation.images.0": image0,
                "observation.images.1": image1,
                "observation.images.2": image2,
                "task": [task_text],
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

            if is_warmup:
                logical_step = -args.warmup_steps + loop_step
                periodic_replan = False
            else:
                logical_step = policy_step
                periodic_replan = (
                    policy_step % args.replan_interval == 0
                )

            need_new_chunk = (
                current_action_chunk is None
                or current_stage_ids is None
                or current_stage_confidence is None
                or current_chunk_cursor >= len(current_action_chunk)
                or periodic_replan
            )

            inference_ms = 0.0
            if need_new_chunk:
                replan_reason = (
                    "periodic"
                    if periodic_replan and current_action_chunk is not None
                    else "empty"
                )
                reset_policy_queue(
                    policy,
                    f"{replan_reason} full-chunk inference "
                    f"at step {logical_step}",
                )

                inference_start = time.perf_counter()
                (
                    normalized_action_chunk,
                    predicted_stage_ids,
                    predicted_stage_confidence,
                ) = predict_full_action_chunk_and_stage(
                    policy,
                    batch,
                )
                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000.0

                raw_action_chunk = (
                    normalized_action_chunk
                    * stats["action.std"][None, None, :]
                    + stats["action.mean"][None, None, :]
                )

                current_action_chunk = (
                    raw_action_chunk[0]
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()
                )
                current_stage_ids = (
                    predicted_stage_ids[0]
                    .detach()
                    .to(device="cpu")
                    .numpy()
                    .astype(np.int64)
                )
                current_stage_confidence = (
                    predicted_stage_confidence[0]
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()
                )
                current_chunk_cursor = 0
                current_chunk_origin_step = int(logical_step)

                payload = build_chunk_payload(
                    replan_step=int(logical_step),
                    action_chunk=current_action_chunk,
                    stage_ids=current_stage_ids,
                    stage_confidence=current_stage_confidence,
                    inference_ms=inference_ms,
                    checkpoint=checkpoint,
                )
                payload["task_text"] = task_text
                payload["observation_state"] = (
                    state_raw.squeeze(0)
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()
                    .tolist()
                )
                payload["observation_state_schema"] = reply.get(
                    "observation_state_schema",
                    "",
                )
                payload["observation_state_names"] = reply.get(
                    "observation_state_names",
                    [],
                )
                payload["observation_effort"] = (
                    effort_raw.squeeze(0)
                    .detach()
                    .to(dtype=torch.float32, device="cpu")
                    .numpy()
                    .tolist()
                )
                emit_chunk_payload(
                    payload,
                    print_full_chunk=not args.no_print_full_chunk,
                    chunk_log_handle=chunk_log_handle,
                )

            if (
                current_action_chunk is None
                or current_stage_ids is None
                or current_stage_confidence is None
                or current_chunk_origin_step is None
            ):
                raise RuntimeError("No current action/stage chunk is available")

            executed_chunk_index = int(current_chunk_cursor)
            model_action = current_action_chunk[
                executed_chunk_index
            ].copy()
            predicted_stage_id = int(
                current_stage_ids[executed_chunk_index]
            )
            predicted_stage_confidence_value = float(
                current_stage_confidence[executed_chunk_index]
            )
            predicted_stage_name = (
                STAGE_NAMES[predicted_stage_id]
                if 0 <= predicted_stage_id < len(STAGE_NAMES)
                else f"unknown_{predicted_stage_id}"
            )

            raw_action = torch.from_numpy(
                model_action
            ).unsqueeze(0)
            velocities, model_action = (
                model_action_to_joint_velocities(
                    raw_action,
                    num_joints,
                )
            )
            current_chunk_cursor += 1

            server_timing = reply.get("bridge_timing", {})
            simulated_seconds_total += float(
                server_timing.get("simulated_seconds", 0.0)
            )

            timing_samples["roundtrip_ms"].append(roundtrip_ms)
            timing_samples["client_prepare_ms"].append(
                (inference_start - reply_received) * 1000.0
            )
            timing_samples["inference_ms"].append(inference_ms)
            for sample_name, reply_key in (
                ("server_total_ms", "total_server_ms"),
                ("physics_ms", "physics_ms"),
                ("camera_render_copy_ms", "camera_render_copy"),
                ("camera_resize_ms", "camera_resize"),
                ("encode_ms", "encode_ms"),
                ("bucket_load_ms", "bucket_load_ms"),
            ):
                timing_samples[sample_name].append(
                    float(server_timing.get(reply_key, 0.0))
                )

            if is_warmup:
                cmd = {
                    "joint_velocities": [0.0] * num_joints
                }
                print(
                    f"[WARMUP {loop_step + 1:03d}/"
                    f"{args.warmup_steps:03d}] "
                    f"chunk_origin={current_chunk_origin_step} "
                    f"chunk_index={executed_chunk_index} "
                    f"pred_stage={predicted_stage_id}:"
                    f"{predicted_stage_name} "
                    f"stage_conf={predicted_stage_confidence_value:.4f} "
                    f"discarded_model_action="
                    f"{model_action.round(5).tolist()} "
                    "commanded_zero_velocity"
                )

                if loop_step == args.warmup_steps - 1:
                    reset_policy_queue(
                        policy,
                        "warmup complete",
                    )
                    current_action_chunk = None
                    current_stage_ids = None
                    current_stage_confidence = None
                    current_chunk_cursor = 0
                    current_chunk_origin_step = None
            else:
                # This is the only deployed-action assignment.
                cmd = {
                    "joint_velocities": velocities.tolist()
                }

                if trace_log_handle is not None:
                    step_trace = {
                        "type": "smolvla_policy_step",
                        "policy_step": int(policy_step),
                        "chunk_origin_step": int(current_chunk_origin_step),
                        "chunk_index": int(executed_chunk_index),
                        "stage_id": int(predicted_stage_id),
                        "stage_name": str(predicted_stage_name),
                        "stage_confidence": float(
                            predicted_stage_confidence_value
                        ),
                        "action": {
                            name: float(model_action[index])
                            for index, name in enumerate(ACTION_NAMES)
                        },
                        "joint_velocities": velocities.tolist(),
                        "task_text": task_text,
                        "observation_state": (
                            state_raw.squeeze(0)
                            .detach()
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                            .tolist()
                        ),
                        "observation_state_schema": reply.get(
                            "observation_state_schema",
                            "",
                        ),
                        "observation_state_names": reply.get(
                            "observation_state_names",
                            [],
                        ),
                        "observation_effort": (
                            effort_raw.squeeze(0)
                            .detach()
                            .to(dtype=torch.float32, device="cpu")
                            .numpy()
                            .tolist()
                        ),
                        "joint_positions": raw_q.tolist(),
                        "bridge_timing": server_timing,
                    }
                    trace_log_handle.write(
                        json.dumps(
                            step_trace,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    trace_log_handle.flush()

                if (
                    policy_step % args.print_every == 0
                    or policy_step == args.steps - 1
                ):
                    loop_ms = (
                        time.perf_counter() - loop_start
                    ) * 1000.0
                    print(
                        f"[STEP {policy_step:05d}] "
                        f"chunk_origin={current_chunk_origin_step} "
                        f"chunk_index={executed_chunk_index} "
                        f"pred_stage={predicted_stage_id}:"
                        f"{predicted_stage_name} "
                        f"stage_conf={predicted_stage_confidence_value:.4f} "
                        f"model_action="
                        f"{model_action.round(5).tolist()} "
                        f"joint_velocities="
                        f"{velocities.round(5).tolist()} "
                        f"state_key={state_key} "
                        f"effort_key={effort_key} "
                        f"q={raw_q.round(5).tolist()} "
                        f"cameras={sorted(camera_rgbs)} "
                        f"infer_ms={inference_ms:.1f} "
                        f"server_ms="
                        f"{float(server_timing.get('total_server_ms', 0.0)):.1f} "
                        f"loop_ms={loop_ms:.1f}"
                    )

            timing_samples["loop_ms"].append(
                (time.perf_counter() - loop_start) * 1000.0
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted; sending zero velocity")

    finally:
        print("[TIMING SUMMARY] p50/p95/max milliseconds")
        for name, values in timing_samples.items():
            if values:
                print(
                    f"[TIMING SUMMARY] {name}: "
                    f"{percentile_summary(values)}"
                )

        wall_seconds = max(
            1e-9,
            time.perf_counter() - deployment_wall_start,
        )
        print(
            "[TIMING SUMMARY] "
            f"simulated_seconds={simulated_seconds_total:.3f} "
            f"wall_seconds={wall_seconds:.3f} "
            f"real_time_factor="
            f"{simulated_seconds_total / wall_seconds:.4f}"
        )

        try:
            zero_joints = (
                int(raw_q.size)
                if raw_q.size > 0
                else EXPECTED_ACTION_DIM
            )
            write_json(
                sock,
                {"joint_velocities": [0.0] * zero_joints},
            )
        except Exception:
            pass

        if chunk_log_handle is not None:
            try:
                chunk_log_handle.close()
            except Exception as exc:
                print(f"[WARN] failed to close chunk log: {exc}")

        if trace_log_handle is not None:
            try:
                trace_log_handle.close()
            except Exception as exc:
                print(f"[WARN] failed to close trace log: {exc}")

        if video_recorder is not None:
            try:
                video_recorder.close()
            except Exception as exc:
                print(
                    f"[WARN] failed to finalize videos: {exc}"
                )

        try:
            sock.close()
        except Exception:
            pass

        print("[INFO] closed")


if __name__ == "__main__":
    main()
