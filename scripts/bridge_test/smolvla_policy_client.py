#!/usr/bin/env python3
"""
Latest-checkpoint SmolVLA bridge client.

Deployment contract for the current excavator checkpoint:
  observation.state   : 27 values
  observation.effort  : 4 values
  observation.images.0/.1/.2
  action              : [swing, boom, arm, bucket]

By default, the simulator command is the model's unnormalized 4D action itself.
There is deliberately no enabled:
  - physical stage machine
  - swing suppression
  - boom/bucket gain
  - forced digging/lifting/dumping action
  - action clipping
  - hand-written fallback action

An optional monotonic excavation sequence supervisor, plus the older standalone
load-retention and unload-geometry constraints, can be enabled explicitly for
supervised deployment experiments.  They are disabled by default so model-only
evaluation remains unchanged.

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
    EFFORT_NAMES_4D,
    OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
    STATE_NAMES_27D,
    build_client_contract,
    load_training_fps,
    sha256_files,
    validate_training_fps,
)


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


def normalized_ood_violations(
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    names: tuple[str, ...],
    max_abs_sigma: float,
) -> list[tuple[int, str, float, float]]:
    """Return normalized dimensions outside the configured safe range."""
    if max_abs_sigma <= 0:
        return []

    raw = np.asarray(raw_values).reshape(-1)
    normalized = np.asarray(normalized_values).reshape(-1)
    if raw.shape != normalized.shape or raw.size != len(names):
        raise ValueError(
            "OOD guard input shape mismatch: "
            f"raw={raw.shape}, normalized={normalized.shape}, "
            f"names={len(names)}"
        )

    violations: list[tuple[int, str, float, float]] = []
    for index, (raw_value, normalized_value) in enumerate(
        zip(raw, normalized)
    ):
        normalized_float = float(normalized_value)
        if (
            not np.isfinite(normalized_float)
            or abs(normalized_float) > max_abs_sigma
        ):
            violations.append(
                (
                    index,
                    names[index],
                    float(raw_value),
                    normalized_float,
                )
            )
    return violations


def clip_normalized_state_input(
    state_raw: torch.Tensor,
    state_normalized: torch.Tensor,
    max_abs_sigma: float,
) -> tuple[torch.Tensor, list[tuple[int, str, float, float]]]:
    """Clip finite state outliers while always rejecting NaN/Inf inputs."""
    raw_np = state_raw.detach().cpu().numpy()
    normalized_np = state_normalized.detach().cpu().numpy()
    if not np.all(np.isfinite(raw_np)) or not np.all(
        np.isfinite(normalized_np)
    ):
        raise RuntimeError(
            "observation.state contains NaN or Inf before inference"
        )
    if max_abs_sigma <= 0:
        return state_normalized, []

    violations = normalized_ood_violations(
        raw_np,
        normalized_np,
        STATE_NAMES_27D,
        max_abs_sigma,
    )
    if not violations:
        return state_normalized, []
    return torch.clamp(
        state_normalized,
        min=-max_abs_sigma,
        max=max_abs_sigma,
    ), violations


def clip_normalized_effort_input(
    effort_raw: torch.Tensor,
    effort_normalized: torch.Tensor,
    max_abs_sigma: float,
) -> tuple[torch.Tensor, list[tuple[int, str, float, float]]]:
    """Clip finite contact-effort spikes while rejecting NaN/Inf inputs."""
    raw_np = effort_raw.detach().cpu().numpy()
    normalized_np = effort_normalized.detach().cpu().numpy()
    if not np.all(np.isfinite(raw_np)) or not np.all(
        np.isfinite(normalized_np)
    ):
        raise RuntimeError(
            "observation.effort contains NaN or Inf before inference"
        )
    if max_abs_sigma <= 0:
        return effort_normalized, []

    violations = normalized_ood_violations(
        raw_np,
        normalized_np,
        EFFORT_NAMES_4D,
        max_abs_sigma,
    )
    if not violations:
        return effort_normalized, []
    return torch.clamp(
        effort_normalized,
        min=-max_abs_sigma,
        max=max_abs_sigma,
    ), violations


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
    Return the selected model-derived command for the first four joints.

    No gain, clipping, suppression, stage machine, safety override, or fallback
    action is applied here.  An optional temporal ensemble is formed before
    this conversion as a convex combination of overlapping model predictions.
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


def temporal_ensemble_action(
    action_chunk_history: list[tuple[int, np.ndarray]],
    target_step: int,
    width: int,
    decay: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend overlapping predictions for one target step, newest first."""
    candidates: list[tuple[int, int, np.ndarray]] = []
    for origin_step, action_chunk in reversed(action_chunk_history):
        chunk_index = int(target_step) - int(origin_step)
        if 0 <= chunk_index < len(action_chunk):
            candidates.append(
                (
                    int(origin_step),
                    int(chunk_index),
                    np.asarray(
                        action_chunk[chunk_index],
                        dtype=np.float32,
                    ).reshape(-1)[:EXPECTED_ACTION_DIM],
                )
            )
        if len(candidates) >= int(width):
            break

    if not candidates:
        raise RuntimeError(
            f"No temporal-ensemble action prediction for step {target_step}"
        )

    ages = np.asarray(
        [target_step - origin for origin, _, _ in candidates],
        dtype=np.float64,
    )
    weights = np.exp(-float(decay) * ages)
    weights /= np.sum(weights)
    actions = np.stack(
        [candidate[2] for candidate in candidates],
        axis=0,
    ).astype(np.float64)
    ensembled = np.sum(actions * weights[:, None], axis=0).astype(
        np.float32
    )
    if not np.all(np.isfinite(ensembled)):
        raise RuntimeError(
            f"Temporal-ensemble action contains NaN/Inf: {ensembled.tolist()}"
        )

    metadata = {
        "enabled": bool(width > 1),
        "configured_width": int(width),
        "decay": float(decay),
        "num_predictions": len(candidates),
        "source_chunk_origins": [
            origin for origin, _, _ in candidates
        ],
        "source_chunk_indices": [
            chunk_index for _, chunk_index, _ in candidates
        ],
        "weights": weights.tolist(),
        "latest_action": actions[0].astype(np.float32).tolist(),
    }
    return ensembled, metadata


def state_value_by_name(
    state_values: np.ndarray,
    state_names: list[str] | tuple[str, ...],
    name: str,
) -> float:
    """Read one named State27 value and fail on a contract mismatch."""
    names = [str(value) for value in state_names]
    if name not in names:
        raise RuntimeError(
            f"Execution constraint requires State27 field {name!r}; "
            f"available fields={names}"
        )
    index = names.index(name)
    values = np.asarray(state_values, dtype=np.float32).reshape(-1)
    if index >= len(values):
        raise RuntimeError(
            f"State27 field {name!r} resolves to index {index}, but the "
            f"observation contains only {len(values)} values"
        )
    value = float(values[index])
    if not np.isfinite(value):
        raise RuntimeError(f"State27 field {name!r} is not finite: {value}")
    return value


def bounded_position_velocity(
    current: float,
    target: float,
    *,
    gain: float,
    max_abs_velocity: float,
) -> float:
    """Return a bounded proportional velocity toward one joint target."""
    command = float(gain) * (float(target) - float(current))
    return float(
        np.clip(
            command,
            -abs(float(max_abs_velocity)),
            abs(float(max_abs_velocity)),
        )
    )


def apply_excavation_sequence_supervisor(
    model_action: np.ndarray,
    *,
    state_values: np.ndarray,
    state_names: list[str] | tuple[str, ...],
    predicted_stage_id: int,
    supervisor_state: dict[str, Any],
    load_trigger: float,
    load_rise: float,
    dig_descent_height_trigger: float,
    dig_descent_stage_trigger: int,
    dig_descent_velocity: float,
    dig_bucket_half_scale_distance: float,
    dig_depth_tolerance: float,
    dig_rebound_height: float,
    dig_rebound_dwell_steps: int,
    dig_rebound_min_load: float,
    curl_target: float,
    curl_velocity: float,
    lift_boom_target: float,
    lift_load_z_target: float,
    max_lift_steps: int,
    turn_entry_boom_target: float,
    turn_entry_velocity: float,
    turn_entry_steps: int,
    turn_handoff_angle: float,
    phase_sync_bucket_closed: float,
    phase_sync_boom_lifted: float,
    unload_swing_target: float,
    unload_boom_target: float,
    unload_arm_target: float,
    unload_bucket_hold_target: float,
    swing_tolerance: float,
    unload_xy_tolerance: float,
    unload_height_margin: float,
    unload_height_velocity: float,
    position_max_steps: int,
    dump_min_steps: int,
    dump_bucket_target: float,
    dump_velocity: float,
    empty_load_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a monotonic expert-derived excavation execution sequence."""
    action = np.asarray(model_action, dtype=np.float32).reshape(-1).copy()
    if action.shape != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(
            "Excavation sequence supervisor requires a four-dimensional "
            f"action, got {action.shape}"
        )
    original_action = action.copy()

    values = {
        name: state_value_by_name(state_values, state_names, name)
        for name in (
            "base_x",
            "base_y",
            "base_yaw",
            "swing",
            "boom",
            "arm",
            "bucket",
            "bucket_load_estimate",
            "bucket_tip_z",
            "bucket_load_x",
            "bucket_load_y",
            "bucket_load_z",
            "dig_target_local_z",
            "unload_landing_local_x",
            "unload_landing_local_y",
            "unload_landing_local_z",
        )
    }
    load = values["bucket_load_estimate"]
    previous_load = supervisor_state.get("previous_load")
    baseline_load = min(
        float(supervisor_state.get("baseline_load", load)),
        load,
    )
    peak_load = max(float(supervisor_state.get("peak_load", load)), load)
    supervisor_state["previous_load"] = float(load)
    supervisor_state["baseline_load"] = float(baseline_load)
    supervisor_state["peak_load"] = float(peak_load)
    supervisor_state["step_count"] = int(
        supervisor_state.get("step_count", -1)
    ) + 1

    load_rising = bool(
        previous_load is not None
        and load > float(previous_load)
        and load >= float(load_trigger)
        and load - baseline_load >= float(load_rise)
    )
    phase_before = str(supervisor_state.get("phase", "dig"))
    phase = phase_before
    transition_reason = ""
    phase_age = int(supervisor_state["step_count"]) - int(
        supervisor_state.get("phase_enter_step", 0)
    )
    dig_depth_delta = float(
        values["bucket_tip_z"] - values["dig_target_local_z"]
    )
    dig_descent_height_triggered = bool(
        values["bucket_tip_z"] <= float(dig_descent_height_trigger)
    )
    dig_descent_stage_triggered = bool(
        int(dig_descent_stage_trigger) <= int(predicted_stage_id) <= 5
    )
    dig_descent_assist_started = False
    dig_descent_trigger_reason = ""
    if (
        phase == "dig"
        and not bool(
            supervisor_state.get("dig_descent_assist_latched", False)
        )
        and (
            dig_descent_height_triggered
            or dig_descent_stage_triggered
        )
    ):
        dig_descent_assist_started = True
        supervisor_state["dig_descent_assist_latched"] = True
        supervisor_state["dig_descent_assist_start_step"] = int(
            supervisor_state["step_count"]
        )
        dig_descent_trigger_reason = (
            "bucket_tip_height"
            if dig_descent_height_triggered
            else "model_dig_stage"
        )
        supervisor_state["dig_descent_trigger_reason"] = (
            dig_descent_trigger_reason
        )
    dig_descent_assist_latched = bool(
        supervisor_state.get("dig_descent_assist_latched", False)
    )
    dig_descent_trigger_reason = str(
        supervisor_state.get(
            "dig_descent_trigger_reason",
            dig_descent_trigger_reason,
        )
    )
    dig_descent_assist_elapsed = (
        int(supervisor_state["step_count"])
        - int(supervisor_state.get("dig_descent_assist_start_step", 0))
        if dig_descent_assist_latched
        else 0
    )
    dig_depth_target_reached = bool(
        dig_depth_delta <= float(dig_depth_tolerance)
    )
    lowest_tip_z = min(
        float(
            supervisor_state.get(
                "dig_lowest_tip_z",
                values["bucket_tip_z"],
            )
        ),
        values["bucket_tip_z"],
    )
    supervisor_state["dig_lowest_tip_z"] = float(lowest_tip_z)
    dig_rebound_distance = float(values["bucket_tip_z"] - lowest_tip_z)
    dig_material_ready = bool(
        load >= max(float(load_trigger), float(dig_rebound_min_load))
        and peak_load - baseline_load >= float(load_rise)
    )
    if phase == "dig":
        if (
            dig_rebound_distance >= float(dig_rebound_height)
            and dig_material_ready
        ):
            supervisor_state["dig_rebound_steps"] = int(
                supervisor_state.get("dig_rebound_steps", 0)
            ) + 1
        else:
            supervisor_state["dig_rebound_steps"] = 0
    dig_rebound_steps = int(
        supervisor_state.get("dig_rebound_steps", 0)
    )

    base_yaw = values["base_yaw"]
    cosine = float(np.cos(base_yaw))
    sine = float(np.sin(base_yaw))
    unload_x = (
        values["base_x"]
        + cosine * values["unload_landing_local_x"]
        - sine * values["unload_landing_local_y"]
    )
    unload_y = (
        values["base_y"]
        + sine * values["unload_landing_local_x"]
        + cosine * values["unload_landing_local_y"]
    )
    unload_xy_distance = float(
        np.hypot(
            values["bucket_load_x"] - unload_x,
            values["bucket_load_y"] - unload_y,
        )
    )
    # Legacy State27 stores target Z in world coordinates because the
    # initial-heading frame translates X/Y from the initial origin only.
    unload_height = values["unload_landing_local_z"]
    unload_height_delta = float(values["bucket_load_z"] - unload_height)
    swing_error = float(unload_swing_target) - values["swing"]
    over_truck = bool(
        unload_xy_distance <= float(unload_xy_tolerance)
        and unload_height_delta >= float(unload_height_margin)
    )

    phase_rank = {
        "dig": 0,
        "curl_secure": 1,
        "lift_loaded": 2,
        "transfer_loaded": 3,
        "position_over_truck": 4,
        "dump": 5,
        "complete": 6,
    }
    observed_phase = ""
    observed_reason = ""
    closed_bucket_threshold = min(
        float(phase_sync_bucket_closed),
        float(curl_target),
    )
    joint_target_tolerance = 1.0e-3
    curl_target_reached = bool(
        values["bucket"]
        <= float(curl_target) + joint_target_tolerance
    )
    forced_descent_incomplete = bool(
        phase == "dig"
        and dig_descent_assist_latched
        and not dig_depth_target_reached
    )
    forced_curl_incomplete = bool(
        phase == "curl_secure" and not curl_target_reached
    )
    if not (forced_descent_incomplete or forced_curl_incomplete):
        if (
            values["swing"]
            <= float(unload_swing_target) + float(swing_tolerance)
        ):
            observed_phase = "position_over_truck"
            observed_reason = "observed_unload_swing_angle"
        elif values["swing"] <= float(turn_handoff_angle):
            observed_phase = "transfer_loaded"
            observed_reason = "observed_model_turn_handoff_angle"
        elif (
            values["bucket"]
            <= closed_bucket_threshold + joint_target_tolerance
            and values["boom"] >= float(phase_sync_boom_lifted)
        ):
            observed_phase = "transfer_loaded"
            observed_reason = "observed_model_closed_bucket_and_lift"
        elif (
            values["bucket"]
            <= closed_bucket_threshold + joint_target_tolerance
        ):
            observed_phase = "lift_loaded"
            observed_reason = "observed_model_closed_bucket"

    if (
        observed_phase
        and phase_rank[observed_phase] > phase_rank[phase]
    ):
        phase = observed_phase
        transition_reason = observed_reason
    elif (
        phase == "dig"
        and dig_descent_assist_latched
        and dig_depth_target_reached
    ):
        phase = "curl_secure"
        transition_reason = "forced_dig_depth_target_reached"
    elif (
        phase == "curl_secure"
        and curl_target_reached
    ):
        phase = "lift_loaded"
        transition_reason = "bucket_curl_target_reached"
    elif (
        phase == "lift_loaded"
        and (
            (
                values["boom"] >= float(lift_boom_target)
                and values["bucket_load_z"] >= float(lift_load_z_target)
            )
            or phase_age >= int(max_lift_steps)
        )
    ):
        phase = "transfer_loaded"
        transition_reason = (
            "safe_lift_height_reached"
            if (
                values["boom"] >= float(lift_boom_target)
                and values["bucket_load_z"] >= float(lift_load_z_target)
            )
            else "maximum_lift_dwell_reached"
        )
    elif (
        phase == "transfer_loaded"
        and abs(swing_error) <= float(swing_tolerance)
    ):
        phase = "position_over_truck"
        transition_reason = "unload_swing_angle_reached"
    elif (
        phase == "position_over_truck"
        and (
            over_truck
            or (
                phase_age >= int(position_max_steps)
                and abs(swing_error) <= float(swing_tolerance)
            )
        )
    ):
        phase = "dump"
        transition_reason = (
            "bucket_above_unload_target"
            if over_truck
            else "maximum_position_dwell_reached_at_unload_angle"
        )
    elif (
        phase == "dump"
        and phase_age >= int(dump_min_steps)
        and values["bucket"] >= float(dump_bucket_target)
    ):
        phase = "complete"
        transition_reason = "forced_bucket_open_target_reached"

    if phase != phase_before:
        supervisor_state["phase"] = phase
        supervisor_state["phase_enter_step"] = int(
            supervisor_state["step_count"]
        )
    else:
        supervisor_state.setdefault("phase", phase)
        supervisor_state.setdefault("phase_enter_step", 0)

    reasons: list[str] = []
    dig_descent_assist_active = bool(
        phase == "dig"
        and dig_descent_assist_latched
        and not dig_depth_target_reached
    )
    dig_bucket_action_scale = 1.0
    turn_guide_steps_completed = int(
        supervisor_state.get("turn_guide_steps_completed", 0)
    )
    turn_guide_active = bool(
        phase in ("lift_loaded", "transfer_loaded")
        and (
            values["boom"] >= float(turn_entry_boom_target)
            or phase == "transfer_loaded"
        )
        and abs(swing_error) > float(swing_tolerance)
        and turn_guide_steps_completed < int(turn_entry_steps)
    )
    if phase == "dig" and dig_descent_assist_active:
        action[1] = min(
            float(action[1]),
            -abs(float(dig_descent_velocity)),
        )
        reasons.append("height_or_stage_triggered_forced_descent")
        if dig_depth_delta > float(dig_bucket_half_scale_distance):
            dig_bucket_action_scale = 0.5
            action[3] = float(action[3]) * dig_bucket_action_scale
            reasons.append("half_bucket_action_before_near_target")
    elif phase == "curl_secure":
        action[0] = 0.0
        action[1] = 0.0
        action[3] = min(
            float(action[3]),
            -abs(float(curl_velocity)),
        )
        reasons.append("force_curl_before_lift")
    elif phase == "lift_loaded":
        action[0] = 0.0
        action[1] = bounded_position_velocity(
            values["boom"],
            unload_boom_target,
            gain=0.8,
            max_abs_velocity=0.18,
        )
        action[2] = bounded_position_velocity(
            values["arm"],
            unload_arm_target,
            gain=0.8,
            max_abs_velocity=0.12,
        )
        action[3] = bounded_position_velocity(
            values["bucket"],
            unload_bucket_hold_target,
            gain=1.5,
            max_abs_velocity=0.8,
        )
        reasons.append("lift_load_before_transfer")
    elif phase in ("transfer_loaded", "position_over_truck"):
        action[0] = bounded_position_velocity(
            values["swing"],
            unload_swing_target,
            gain=1.0,
            max_abs_velocity=0.45,
        )
        action[1] = bounded_position_velocity(
            values["boom"],
            unload_boom_target,
            gain=0.8,
            max_abs_velocity=0.18,
        )
        action[2] = bounded_position_velocity(
            values["arm"],
            unload_arm_target,
            gain=0.8,
            max_abs_velocity=0.12,
        )
        action[3] = bounded_position_velocity(
            values["bucket"],
            unload_bucket_hold_target,
            gain=1.5,
            max_abs_velocity=0.8,
        )
        reasons.append(
            "drive_expert_unload_pose"
            if phase == "transfer_loaded"
            else "hold_closed_until_over_truck"
        )
        if (
            phase == "position_over_truck"
            and unload_height_delta < float(unload_height_margin)
        ):
            action[1] = max(
                float(action[1]),
                abs(float(unload_height_velocity)),
            )
            reasons.append("raise_boom_until_unload_height_gate")
    elif phase == "dump":
        action[0] = bounded_position_velocity(
            values["swing"],
            unload_swing_target,
            gain=1.0,
            max_abs_velocity=0.25,
        )
        action[1] = bounded_position_velocity(
            values["boom"],
            unload_boom_target,
            gain=0.8,
            max_abs_velocity=0.12,
        )
        action[2] = bounded_position_velocity(
            values["arm"],
            unload_arm_target,
            gain=0.8,
            max_abs_velocity=0.08,
        )
        action[3] = max(float(action[3]), abs(float(dump_velocity)))
        reasons.append("controlled_dump_over_truck")
    elif phase == "complete":
        action[:] = 0.0
        reasons.append("hold_after_completed_dump")

    if turn_guide_active:
        action[0] = float(
            np.copysign(
                abs(float(turn_entry_velocity)),
                swing_error,
            )
        )
        turn_guide_steps_completed += 1
        supervisor_state["turn_guide_steps_completed"] = int(
            turn_guide_steps_completed
        )
        reasons.append("fixed_duration_turn_entry_guide")

    metadata = {
        "enabled": True,
        "mode": "excavation_sequence_supervisor_v1",
        "modified": bool(not np.array_equal(action, original_action)),
        "phase_before": phase_before,
        "phase": phase,
        "phase_enter_step": int(supervisor_state["phase_enter_step"]),
        "phase_age_steps": int(supervisor_state["step_count"])
        - int(supervisor_state["phase_enter_step"]),
        "transition": bool(phase != phase_before),
        "transition_reason": transition_reason,
        "reasons": reasons,
        "load": float(load),
        "peak_load": float(peak_load),
        "load_rising": load_rising,
        "predicted_stage_id": int(predicted_stage_id),
        "dig_descent_height_triggered": dig_descent_height_triggered,
        "dig_descent_stage_triggered": dig_descent_stage_triggered,
        "dig_descent_trigger_reason": dig_descent_trigger_reason,
        "dig_descent_assist_started": dig_descent_assist_started,
        "dig_descent_assist_latched": dig_descent_assist_latched,
        "dig_descent_assist_active": dig_descent_assist_active,
        "dig_descent_assist_elapsed": int(dig_descent_assist_elapsed),
        "dig_depth_delta": float(dig_depth_delta),
        "dig_depth_target_reached": dig_depth_target_reached,
        "dig_bucket_action_scale": float(dig_bucket_action_scale),
        "dig_lowest_tip_z": float(lowest_tip_z),
        "dig_rebound_distance": float(dig_rebound_distance),
        "dig_rebound_steps": int(dig_rebound_steps),
        "dig_material_ready": dig_material_ready,
        "bucket_q": float(values["bucket"]),
        "curl_target_reached": curl_target_reached,
        "boom_q": float(values["boom"]),
        "arm_q": float(values["arm"]),
        "swing_q": float(values["swing"]),
        "swing_error": float(swing_error),
        "turn_guide_active": turn_guide_active,
        "turn_guide_steps_completed": int(turn_guide_steps_completed),
        "bucket_load_z": float(values["bucket_load_z"]),
        "unload_xy_distance": float(unload_xy_distance),
        "unload_height_delta": float(unload_height_delta),
        "over_truck": over_truck,
        "position_timeout_reached": bool(
            phase_before == "position_over_truck"
            and phase_age >= int(position_max_steps)
        ),
        "dump_bucket_target_reached": bool(
            values["bucket"] >= float(dump_bucket_target)
        ),
        "empty_load_completion_disabled": True,
        "observed_phase_promotion": bool(
            transition_reason.startswith("observed_")
        ),
        "targets": {
            "curl_q": float(curl_target),
            "dig_descent_height_trigger": float(
                dig_descent_height_trigger
            ),
            "dig_descent_stage_trigger": int(dig_descent_stage_trigger),
            "dig_descent_velocity": float(dig_descent_velocity),
            "dig_bucket_half_scale_distance": float(
                dig_bucket_half_scale_distance
            ),
            "dig_depth_tolerance": float(dig_depth_tolerance),
            "dig_rebound_height": float(dig_rebound_height),
            "dig_rebound_dwell_steps": int(dig_rebound_dwell_steps),
            "dig_rebound_min_load": float(dig_rebound_min_load),
            "lift_boom_q": float(lift_boom_target),
            "lift_load_z": float(lift_load_z_target),
            "max_lift_steps": int(max_lift_steps),
            "turn_entry_boom_q": float(turn_entry_boom_target),
            "turn_entry_velocity": float(turn_entry_velocity),
            "turn_entry_steps": int(turn_entry_steps),
            "turn_handoff_angle": float(turn_handoff_angle),
            "phase_sync_bucket_closed": float(phase_sync_bucket_closed),
            "phase_sync_boom_lifted": float(phase_sync_boom_lifted),
            "unload_swing_q": float(unload_swing_target),
            "unload_boom_q": float(unload_boom_target),
            "unload_arm_q": float(unload_arm_target),
            "unload_bucket_hold_q": float(unload_bucket_hold_target),
            "unload_xy_tolerance": float(unload_xy_tolerance),
            "unload_height_margin": float(unload_height_margin),
            "unload_height_velocity": float(unload_height_velocity),
            "position_max_steps": int(position_max_steps),
            "dump_min_steps": int(dump_min_steps),
            "dump_bucket_q": float(dump_bucket_target),
            "dump_velocity": float(dump_velocity),
        },
    }
    return action.astype(np.float32), metadata


def apply_optional_execution_constraints(
    model_action: np.ndarray,
    *,
    state_values: np.ndarray,
    state_names: list[str] | tuple[str, ...],
    predicted_stage_id: int,
    constraint_state: dict[str, Any],
    enable_load_retention: bool,
    enable_unload_geometry_gate: bool,
    retention_min_load: float,
    retention_min_rise: float,
    retention_bucket_target: float,
    retention_curl_velocity: float,
    unload_swing_target: float,
    unload_swing_tolerance: float,
    unload_swing_velocity: float,
    unload_xy_tolerance: float,
    unload_min_height: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optionally supervise material retention and unload alignment.

    With both feature flags false this function is an exact identity apart
    from copying the input array.  This keeps model-only evaluation auditable.
    """
    action = np.asarray(model_action, dtype=np.float32).reshape(-1).copy()
    if action.shape != (EXPECTED_ACTION_DIM,):
        raise RuntimeError(
            "Execution constraints require a four-dimensional action, got "
            f"{action.shape}"
        )

    enabled = bool(enable_load_retention or enable_unload_geometry_gate)
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "load_retention_enabled": bool(enable_load_retention),
        "unload_geometry_gate_enabled": bool(enable_unload_geometry_gate),
        "modified": False,
        "reasons": [],
    }
    if not enabled:
        return action, metadata

    load = state_value_by_name(
        state_values,
        state_names,
        "bucket_load_estimate",
    )
    bucket_q = state_value_by_name(state_values, state_names, "bucket")
    swing_q = state_value_by_name(state_values, state_names, "swing")
    previous_load = constraint_state.get("previous_load")
    baseline_load = float(constraint_state.get("baseline_load", load))
    baseline_load = min(baseline_load, load)
    peak_load = max(float(constraint_state.get("peak_load", load)), load)
    constraint_state["previous_load"] = float(load)
    constraint_state["baseline_load"] = float(baseline_load)
    constraint_state["peak_load"] = float(peak_load)

    load_rising = bool(
        previous_load is not None
        and load > float(previous_load)
        and load >= float(retention_min_load)
        and load - baseline_load >= float(retention_min_rise)
    )
    if enabled and load_rising:
        constraint_state["retention_latched"] = True

    retention_latched = bool(
        constraint_state.get("retention_latched", False)
    )
    retention_secured = bool(
        constraint_state.get("retention_secured", False)
    )
    if retention_latched and bucket_q <= float(retention_bucket_target):
        retention_secured = True
        constraint_state["retention_secured"] = True

    original_action = action.copy()
    reasons: list[str] = []

    if enable_load_retention and retention_latched:
        if not retention_secured:
            # Negative bucket velocity curls this excavator's bucket closed.
            action[0] = 0.0
            action[1] = min(float(action[1]), 0.0)
            action[3] = min(
                float(action[3]),
                -abs(float(retention_curl_velocity)),
            )
            reasons.append("curl_before_lift_or_swing")
        elif int(predicted_stage_id) < 9 and float(action[3]) > 0.0:
            action[3] = 0.0
            reasons.append("hold_bucket_closed_before_unload_stage")

    unload_gate_active = bool(
        enable_unload_geometry_gate
        and retention_latched
        and retention_secured
        and int(predicted_stage_id) >= 8
    )
    unload_aligned = False
    unload_xy_distance = None
    unload_height_margin = None
    swing_aligned = False
    if unload_gate_active:
        swing_aligned = bool(
            swing_q
            <= float(unload_swing_target) + float(unload_swing_tolerance)
        )
        if not swing_aligned:
            action[0] = min(
                float(action[0]),
                -abs(float(unload_swing_velocity)),
            )
            reasons.append("continue_swing_to_unload")

        base_x = state_value_by_name(state_values, state_names, "base_x")
        base_y = state_value_by_name(state_values, state_names, "base_y")
        base_yaw = state_value_by_name(
            state_values,
            state_names,
            "base_yaw",
        )
        load_x = state_value_by_name(
            state_values,
            state_names,
            "bucket_load_x",
        )
        load_y = state_value_by_name(
            state_values,
            state_names,
            "bucket_load_y",
        )
        load_z = state_value_by_name(
            state_values,
            state_names,
            "bucket_load_z",
        )
        unload_local_x = state_value_by_name(
            state_values,
            state_names,
            "unload_landing_local_x",
        )
        unload_local_y = state_value_by_name(
            state_values,
            state_names,
            "unload_landing_local_y",
        )
        unload_z = state_value_by_name(
            state_values,
            state_names,
            "unload_landing_local_z",
        )
        cosine = float(np.cos(base_yaw))
        sine = float(np.sin(base_yaw))
        unload_x = (
            base_x + cosine * unload_local_x - sine * unload_local_y
        )
        unload_y = (
            base_y + sine * unload_local_x + cosine * unload_local_y
        )
        unload_xy_distance = float(
            np.hypot(load_x - unload_x, load_y - unload_y)
        )
        unload_height_margin = float(load_z - unload_z)
        bucket_over_truck = bool(
            unload_xy_distance <= float(unload_xy_tolerance)
            and unload_height_margin >= float(unload_min_height)
        )
        unload_aligned = bool(swing_aligned and bucket_over_truck)
        if not unload_aligned and float(action[3]) > 0.0:
            action[3] = 0.0
            reasons.append("block_bucket_open_until_over_truck")

    metadata.update(
        {
            "modified": bool(not np.array_equal(action, original_action)),
            "reasons": reasons,
            "load": float(load),
            "peak_load": float(peak_load),
            "load_rising": load_rising,
            "retention_latched": retention_latched,
            "retention_secured": retention_secured,
            "bucket_q": float(bucket_q),
            "unload_gate_active": unload_gate_active,
            "swing_q": float(swing_q),
            "swing_aligned": swing_aligned,
            "unload_xy_distance": unload_xy_distance,
            "unload_height_margin": unload_height_margin,
            "unload_aligned": unload_aligned,
        }
    )
    return action, metadata

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
        "--temporal-ensemble-width",
        type=int,
        default=1,
        help=(
            "Blend predictions for the current step from this many newest "
            "overlapping chunks. Default: 1 (disabled)."
        ),
    )
    parser.add_argument(
        "--temporal-ensemble-decay",
        type=float,
        default=0.35,
        help=(
            "Exponential age decay for overlapping action predictions; "
            "larger values favor the newest chunk. Default: 0.35."
        ),
    )
    parser.add_argument(
        "--enable-excavation-sequence-supervisor",
        action="store_true",
        help=(
            "Opt in to the monotonic dig, curl, lift, transfer, position, "
            "and dump execution sequence. Disabled by default."
        ),
    )
    parser.add_argument("--supervisor-load-trigger", type=float, default=50.0)
    parser.add_argument("--supervisor-load-rise", type=float, default=40.0)
    parser.add_argument(
        "--supervisor-dig-descent-height-trigger",
        type=float,
        default=1.50,
    )
    parser.add_argument(
        "--supervisor-dig-descent-stage-trigger",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--supervisor-dig-descent-velocity",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--supervisor-dig-bucket-half-scale-distance",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--supervisor-dig-depth-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--supervisor-dig-rebound-height",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--supervisor-dig-rebound-dwell-steps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--supervisor-dig-rebound-min-load",
        type=float,
        default=500.0,
    )
    parser.add_argument("--supervisor-curl-target", type=float, default=-2.04)
    parser.add_argument("--supervisor-curl-velocity", type=float, default=0.85)
    parser.add_argument(
        "--supervisor-lift-boom-target",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--supervisor-lift-load-z-target",
        type=float,
        default=1.90,
    )
    parser.add_argument(
        "--supervisor-max-lift-steps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--supervisor-turn-entry-boom-target",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--supervisor-turn-entry-velocity",
        type=float,
        default=0.22,
    )
    parser.add_argument(
        "--supervisor-turn-entry-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--supervisor-turn-handoff-angle",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--supervisor-phase-sync-bucket-closed",
        type=float,
        default=-1.90,
    )
    parser.add_argument(
        "--supervisor-phase-sync-boom-lifted",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--supervisor-unload-swing-target",
        type=float,
        default=-1.615,
    )
    parser.add_argument(
        "--supervisor-unload-boom-target",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--supervisor-unload-arm-target",
        type=float,
        default=-1.017,
    )
    parser.add_argument(
        "--supervisor-unload-bucket-hold-target",
        type=float,
        default=-2.04,
    )
    parser.add_argument(
        "--supervisor-swing-tolerance",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--supervisor-unload-xy-tolerance",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--supervisor-unload-height-margin",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--supervisor-unload-height-velocity",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--supervisor-position-max-steps",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--supervisor-dump-min-steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--supervisor-dump-bucket-target",
        type=float,
        default=-0.80,
    )
    parser.add_argument(
        "--supervisor-dump-velocity",
        type=float,
        default=1.20,
    )
    parser.add_argument(
        "--supervisor-empty-load-threshold",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--enable-load-retention-constraint",
        action="store_true",
    )
    parser.add_argument(
        "--retention-min-load",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--retention-min-rise",
        type=float,
        default=40.0,
    )
    parser.add_argument(
        "--retention-bucket-target",
        type=float,
        default=-1.95,
    )
    parser.add_argument(
        "--retention-curl-velocity",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--enable-unload-geometry-gate",
        action="store_true",
    )
    parser.add_argument(
        "--unload-swing-target",
        type=float,
        default=-1.60,
    )
    parser.add_argument(
        "--unload-swing-tolerance",
        type=float,
        default=0.12,
    )
    parser.add_argument(
        "--unload-swing-velocity",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--unload-xy-tolerance",
        type=float,
        default=1.25,
    )
    parser.add_argument(
        "--unload-min-height",
        type=float,
        default=0.0,
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
        "--episode-status-json",
        default="",
        help="Write a machine-readable final episode success record.",
    )
    parser.add_argument(
        "--simulation-launch-unix-ns",
        type=int,
        default=0,
        help="Simulation launch timestamp used for completion timing.",
    )
    parser.add_argument(
        "--truck-success-min-particles",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--truck-success-no-growth-steps",
        "--truck-success-dwell-steps",
        dest="truck_success_no_growth_steps",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--bucket-open-threshold",
        type=float,
        default=-0.80,
    )
    parser.add_argument(
        "--early-stop-on-success",
        action="store_true",
    )
    parser.add_argument(
        "--max-input-abs-sigma",
        type=float,
        default=8.0,
        help=(
            "Clip normalized state inputs to this absolute value. "
            "Set to 0 or a negative value to disable clipping."
        ),
    )
    parser.add_argument(
        "--max-effort-abs-sigma",
        type=float,
        default=8.0,
        help=(
            "Clip normalized effort inputs to this absolute value. "
            "Set to 0 or a negative value to disable clipping."
        ),
    )

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
        "--record-overview-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Record the simulator's fixed overview viewport as a separate "
            "native-resolution video without adding it to model inputs."
        ),
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
    if args.temporal_ensemble_width <= 0:
        raise SystemExit("--temporal-ensemble-width must be positive")
    if (
        not np.isfinite(args.temporal_ensemble_decay)
        or args.temporal_ensemble_decay < 0
    ):
        raise SystemExit(
            "--temporal-ensemble-decay must be finite and non-negative"
        )
    if args.print_every <= 0:
        raise SystemExit("--print-every must be positive")
    if args.truck_success_min_particles <= 0:
        raise SystemExit("--truck-success-min-particles must be positive")
    if args.truck_success_no_growth_steps <= 0:
        raise SystemExit(
            "--truck-success-no-growth-steps must be positive"
        )
    if not np.isfinite(args.bucket_open_threshold):
        raise SystemExit("--bucket-open-threshold must be finite")
    if args.simulation_launch_unix_ns < 0:
        raise SystemExit("--simulation-launch-unix-ns must be non-negative")
    if args.early_stop_on_success and not args.episode_status_json:
        raise SystemExit(
            "--early-stop-on-success requires --episode-status-json"
        )
    if args.record_video_fps < 0:
        raise SystemExit("--record-video-fps must be non-negative")
    if len(args.record_video_codec) != 4:
        raise SystemExit(
            "--record-video-codec must contain exactly four characters"
        )
    if not np.isfinite(args.max_input_abs_sigma):
        raise SystemExit("--max-input-abs-sigma must be finite")
    if not np.isfinite(args.max_effort_abs_sigma):
        raise SystemExit("--max-effort-abs-sigma must be finite")
    finite_constraint_values = {
        "--supervisor-load-trigger": args.supervisor_load_trigger,
        "--supervisor-load-rise": args.supervisor_load_rise,
        "--supervisor-dig-descent-height-trigger": (
            args.supervisor_dig_descent_height_trigger
        ),
        "--supervisor-dig-descent-velocity": (
            args.supervisor_dig_descent_velocity
        ),
        "--supervisor-dig-bucket-half-scale-distance": (
            args.supervisor_dig_bucket_half_scale_distance
        ),
        "--supervisor-dig-depth-tolerance": (
            args.supervisor_dig_depth_tolerance
        ),
        "--supervisor-dig-rebound-height": (
            args.supervisor_dig_rebound_height
        ),
        "--supervisor-dig-rebound-min-load": (
            args.supervisor_dig_rebound_min_load
        ),
        "--supervisor-curl-target": args.supervisor_curl_target,
        "--supervisor-curl-velocity": args.supervisor_curl_velocity,
        "--supervisor-lift-boom-target": args.supervisor_lift_boom_target,
        "--supervisor-lift-load-z-target": (
            args.supervisor_lift_load_z_target
        ),
        "--supervisor-turn-entry-boom-target": (
            args.supervisor_turn_entry_boom_target
        ),
        "--supervisor-turn-entry-velocity": (
            args.supervisor_turn_entry_velocity
        ),
        "--supervisor-turn-handoff-angle": (
            args.supervisor_turn_handoff_angle
        ),
        "--supervisor-phase-sync-bucket-closed": (
            args.supervisor_phase_sync_bucket_closed
        ),
        "--supervisor-phase-sync-boom-lifted": (
            args.supervisor_phase_sync_boom_lifted
        ),
        "--supervisor-unload-swing-target": (
            args.supervisor_unload_swing_target
        ),
        "--supervisor-unload-boom-target": (
            args.supervisor_unload_boom_target
        ),
        "--supervisor-unload-arm-target": args.supervisor_unload_arm_target,
        "--supervisor-unload-bucket-hold-target": (
            args.supervisor_unload_bucket_hold_target
        ),
        "--supervisor-swing-tolerance": args.supervisor_swing_tolerance,
        "--supervisor-unload-xy-tolerance": (
            args.supervisor_unload_xy_tolerance
        ),
        "--supervisor-unload-height-margin": (
            args.supervisor_unload_height_margin
        ),
        "--supervisor-unload-height-velocity": (
            args.supervisor_unload_height_velocity
        ),
        "--supervisor-dump-bucket-target": (
            args.supervisor_dump_bucket_target
        ),
        "--supervisor-dump-velocity": args.supervisor_dump_velocity,
        "--supervisor-empty-load-threshold": (
            args.supervisor_empty_load_threshold
        ),
        "--retention-min-load": args.retention_min_load,
        "--retention-min-rise": args.retention_min_rise,
        "--retention-bucket-target": args.retention_bucket_target,
        "--retention-curl-velocity": args.retention_curl_velocity,
        "--unload-swing-target": args.unload_swing_target,
        "--unload-swing-tolerance": args.unload_swing_tolerance,
        "--unload-swing-velocity": args.unload_swing_velocity,
        "--unload-xy-tolerance": args.unload_xy_tolerance,
        "--unload-min-height": args.unload_min_height,
    }
    for option, value in finite_constraint_values.items():
        if not np.isfinite(value):
            raise SystemExit(f"{option} must be finite")
    for option, value in (
        ("--supervisor-load-trigger", args.supervisor_load_trigger),
        ("--supervisor-load-rise", args.supervisor_load_rise),
        (
            "--supervisor-dig-descent-velocity",
            args.supervisor_dig_descent_velocity,
        ),
        (
            "--supervisor-dig-bucket-half-scale-distance",
            args.supervisor_dig_bucket_half_scale_distance,
        ),
        (
            "--supervisor-dig-depth-tolerance",
            args.supervisor_dig_depth_tolerance,
        ),
        (
            "--supervisor-dig-rebound-height",
            args.supervisor_dig_rebound_height,
        ),
        (
            "--supervisor-dig-rebound-min-load",
            args.supervisor_dig_rebound_min_load,
        ),
        ("--supervisor-curl-velocity", args.supervisor_curl_velocity),
        (
            "--supervisor-lift-load-z-target",
            args.supervisor_lift_load_z_target,
        ),
        (
            "--supervisor-swing-tolerance",
            args.supervisor_swing_tolerance,
        ),
        (
            "--supervisor-unload-xy-tolerance",
            args.supervisor_unload_xy_tolerance,
        ),
        (
            "--supervisor-unload-height-margin",
            args.supervisor_unload_height_margin,
        ),
        (
            "--supervisor-unload-height-velocity",
            args.supervisor_unload_height_velocity,
        ),
        ("--supervisor-dump-velocity", args.supervisor_dump_velocity),
        (
            "--supervisor-turn-entry-velocity",
            args.supervisor_turn_entry_velocity,
        ),
        (
            "--supervisor-empty-load-threshold",
            args.supervisor_empty_load_threshold,
        ),
        ("--retention-min-load", args.retention_min_load),
        ("--retention-min-rise", args.retention_min_rise),
        ("--retention-curl-velocity", args.retention_curl_velocity),
        ("--unload-swing-tolerance", args.unload_swing_tolerance),
        ("--unload-swing-velocity", args.unload_swing_velocity),
        ("--unload-xy-tolerance", args.unload_xy_tolerance),
    ):
        if value < 0:
            raise SystemExit(f"{option} must be non-negative")
    if args.supervisor_dig_rebound_dwell_steps <= 0:
        raise SystemExit(
            "--supervisor-dig-rebound-dwell-steps must be positive"
        )
    if not 0 <= args.supervisor_dig_descent_stage_trigger <= 5:
        raise SystemExit(
            "--supervisor-dig-descent-stage-trigger must be in [0, 5]"
        )
    if args.supervisor_max_lift_steps <= 0:
        raise SystemExit("--supervisor-max-lift-steps must be positive")
    if args.supervisor_turn_entry_steps <= 0:
        raise SystemExit("--supervisor-turn-entry-steps must be positive")
    if args.supervisor_position_max_steps <= 0:
        raise SystemExit(
            "--supervisor-position-max-steps must be positive"
        )
    if args.supervisor_dump_min_steps <= 0:
        raise SystemExit("--supervisor-dump-min-steps must be positive")

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
    constraints_enabled = bool(
        args.enable_excavation_sequence_supervisor
        or args.enable_load_retention_constraint
        or args.enable_unload_geometry_gate
    )
    if constraints_enabled:
        print(
            "[ACTION MODE] supervised model output: execution assistance "
            "is enabled"
        )
    else:
        print(
            "[ACTION MODE] pure model output: "
            "[swing, boom, arm, bucket] -> joint_velocities[0:4]"
        )
        print(
            "[ACTION MODE] no auxiliary action, gain, action clipping, "
            "suppression, state machine, execution constraint, or fallback"
        )
    print(
        "[EXECUTION CONSTRAINTS] "
        "sequence_supervisor="
        f"{args.enable_excavation_sequence_supervisor} "
        f"load_retention={args.enable_load_retention_constraint} "
        f"unload_geometry_gate={args.enable_unload_geometry_gate}"
    )
    if args.max_input_abs_sigma > 0:
        print(
            "[STATE INPUT CLIP] enabled: "
            f"max_abs_sigma={args.max_input_abs_sigma:g}"
        )
    else:
        print("[STATE INPUT CLIP] disabled explicitly")
    if args.max_effort_abs_sigma > 0:
        print(
            "[EFFORT INPUT CLIP] enabled: "
            f"max_abs_sigma={args.max_effort_abs_sigma:g}"
        )
    else:
        print("[EFFORT INPUT CLIP] disabled explicitly")

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
        f"replan_interval={args.replan_interval} "
        f"temporal_ensemble_width={args.temporal_ensemble_width} "
        f"temporal_ensemble_decay={args.temporal_ensemble_decay:g}"
    )
    if args.temporal_ensemble_width > int(policy.model.config.chunk_size):
        raise RuntimeError(
            "--temporal-ensemble-width cannot exceed model chunk_size: "
            f"width={args.temporal_ensemble_width}, "
            f"chunk_size={int(policy.model.config.chunk_size)}"
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
            camera_ids=(
                CAMERA_IDS + ("overview",)
                if args.record_overview_camera
                else CAMERA_IDS
            ),
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

    record_overview_camera = bool(
        video_recorder is not None and args.record_overview_camera
    )
    cmd: dict[str, Any] = {
        "joint_velocities": [0.0] * EXPECTED_ACTION_DIM,
        "record_overview_camera": record_overview_camera,
    }
    task_text: str | None = None
    language_tokens: torch.Tensor | None = None
    language_mask: torch.Tensor | None = None

    current_action_chunk: np.ndarray | None = None
    current_stage_ids: np.ndarray | None = None
    current_stage_confidence: np.ndarray | None = None
    current_chunk_cursor = 0
    current_chunk_origin_step: int | None = None
    action_chunk_history: list[tuple[int, np.ndarray]] = []

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
        "truck_bed_load_ms": [],
    }
    simulated_seconds_total = 0.0
    policy_execution_started_unix_ns = time.time_ns()
    deployment_wall_start = time.perf_counter()
    effort_clip_event_count = 0
    state_clip_event_count = 0
    execution_constraint_state: dict[str, Any] = {}

    raw_q = np.empty(0, dtype=np.float32)
    final_reply: dict[str, Any] = {}
    truck_initial_count: int | None = None
    final_truck_raw_count = 0
    final_truck_deposited_count = 0
    final_bucket_q: float | None = None
    previous_truck_deposited_count: int | None = None
    success_no_growth_observations = 0
    first_particle_detected_unix_ns: int | None = None
    first_success_unix_ns: int | None = None
    task_completion_unix_ns: int | None = None
    policy_steps_inferred = 0
    policy_steps_executed = 0
    termination_reason = "max_steps"
    early_terminated = False
    episode_error: str | None = None

    def elapsed_seconds(
        event_unix_ns: int | None,
        start_unix_ns: int,
    ) -> float | None:
        if event_unix_ns is None or start_unix_ns <= 0:
            return None
        return max(0.0, (event_unix_ns - start_unix_ns) / 1.0e9)

    def update_episode_success(
        observation_reply: dict[str, Any],
        joint_positions: np.ndarray,
        executed_steps: int,
    ) -> bool:
        nonlocal truck_initial_count
        nonlocal final_truck_raw_count
        nonlocal final_truck_deposited_count
        nonlocal final_bucket_q
        nonlocal previous_truck_deposited_count
        nonlocal success_no_growth_observations
        nonlocal first_particle_detected_unix_ns
        nonlocal first_success_unix_ns
        nonlocal task_completion_unix_ns
        nonlocal policy_steps_executed

        policy_steps_executed = max(
            policy_steps_executed,
            int(executed_steps),
        )
        metric_required = bool(
            args.episode_status_json or args.early_stop_on_success
        )
        metrics = observation_reply.get("truck_bed_load_metrics")
        if not isinstance(metrics, dict):
            if metric_required:
                raise RuntimeError(
                    "Bridge reply is missing truck_bed_load_metrics"
                )
            return False
        if (
            metric_required
            and (
                not bool(metrics.get("available", False))
                or not bool(metrics.get("source_tracking_valid", False))
            )
        ):
            raise RuntimeError(
                "Truck-bed success metric is unavailable or not source "
                f"tracked: {metrics}"
            )
        if (
            metric_required
            and metrics.get("method")
            != "world_aabb_excluding_bucket_v2"
        ):
            raise RuntimeError(
                "Truck-bed success metric version mismatch: "
                "expected='world_aabb_excluding_bucket_v2', "
                f"actual={metrics.get('method')!r}. Upload the matching "
                "run_simulation.py before collecting success-rate results."
            )

        raw_count = int(metrics.get("count", 0))
        if truck_initial_count is None:
            truck_initial_count = raw_count
        deposited_count = max(0, raw_count - truck_initial_count)
        final_truck_raw_count = raw_count
        final_truck_deposited_count = deposited_count
        final_bucket_q = (
            float(joint_positions[3])
            if joint_positions.size >= 4
            else None
        )

        currently_successful = bool(
            deposited_count >= args.truck_success_min_particles
        )
        if currently_successful:
            if first_particle_detected_unix_ns is None:
                first_particle_detected_unix_ns = time.time_ns()
            if (
                previous_truck_deposited_count is not None
                and previous_truck_deposited_count
                >= args.truck_success_min_particles
                and deposited_count
                <= previous_truck_deposited_count
            ):
                success_no_growth_observations += 1
            else:
                success_no_growth_observations = 0
        else:
            success_no_growth_observations = 0
        previous_truck_deposited_count = deposited_count

        bucket_open = bool(
            final_bucket_q is not None
            and final_bucket_q >= args.bucket_open_threshold
        )
        stable_success = bool(
            currently_successful
            and success_no_growth_observations
            >= args.truck_success_no_growth_steps
        )
        if stable_success and first_success_unix_ns is None:
            first_success_unix_ns = time.time_ns()
        if stable_success and bucket_open:
            if task_completion_unix_ns is None:
                task_completion_unix_ns = time.time_ns()
            return True
        return False

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
            if not is_warmup:
                final_reply = reply
                task_complete = update_episode_success(
                    reply,
                    raw_q,
                    executed_steps=max(0, policy_step),
                )
                if args.early_stop_on_success and task_complete:
                    early_terminated = True
                    termination_reason = (
                        "truck_success_and_bucket_open"
                    )
                    print(
                        "[EPISODE SUCCESS] "
                        f"policy_steps_executed={policy_steps_executed} "
                        "truck_particles="
                        f"{final_truck_deposited_count} "
                        "no_growth_steps="
                        f"{success_no_growth_observations} "
                        f"bucket_q={final_bucket_q:.6f} "
                        "early_stop=true"
                    )
                    break

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
                action_chunk_history.clear()
                execution_constraint_state.clear()
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
                video_frames = {"0": rgb0, "1": rgb1, "2": rgb2}
                recording_camera_payloads = reply.get(
                    "recording_cameras",
                    {},
                )
                if args.record_overview_camera:
                    if not isinstance(recording_camera_payloads, dict):
                        raise RuntimeError(
                            "Bridge reply has invalid recording_cameras"
                        )
                    overview_payload = recording_camera_payloads.get(
                        "overview"
                    )
                    if not isinstance(overview_payload, dict):
                        raise RuntimeError(
                            "Bridge reply is missing the requested "
                            "overview recording camera"
                        )
                    video_frames["overview"] = decode_rgb(
                        overview_payload
                    )
                video_recorder.write(
                    video_frames,
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
            state, state_clip_violations = clip_normalized_state_input(
                state_raw,
                state,
                args.max_input_abs_sigma,
            )
            if state_clip_violations:
                state_clip_event_count += 1
                if (
                    state_clip_event_count <= 5
                    or state_clip_event_count % 25 == 0
                ):
                    details = ", ".join(
                        f"{name}={normalized:.3f}sigma"
                        for _, name, _, normalized in state_clip_violations
                    )
                    print(
                        "[STATE INPUT CLIP] "
                        f"policy_step={policy_step} "
                        f"event={state_clip_event_count} "
                        f"limit={args.max_input_abs_sigma:g}sigma "
                        f"clipped={details}"
                    )
            effort, effort_clip_violations = clip_normalized_effort_input(
                effort_raw,
                effort,
                args.max_effort_abs_sigma,
            )
            if effort_clip_violations:
                effort_clip_event_count += 1
                if (
                    effort_clip_event_count <= 5
                    or effort_clip_event_count % 25 == 0
                ):
                    details = ", ".join(
                        f"{name}={normalized:.3f}sigma"
                        for _, name, _, normalized
                        in effort_clip_violations
                    )
                    print(
                        "[EFFORT INPUT CLIP] "
                        f"policy_step={policy_step} "
                        f"event={effort_clip_event_count} "
                        f"limit={args.max_effort_abs_sigma:g}sigma "
                        f"clipped={details}"
                    )

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
                action_chunk_history.append(
                    (
                        current_chunk_origin_step,
                        current_action_chunk.copy(),
                    )
                )
                if len(action_chunk_history) > args.temporal_ensemble_width:
                    del action_chunk_history[
                        : -args.temporal_ensemble_width
                    ]

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
            latest_model_action = current_action_chunk[
                executed_chunk_index
            ].copy()
            model_action, temporal_ensemble = temporal_ensemble_action(
                action_chunk_history,
                target_step=int(logical_step),
                width=args.temporal_ensemble_width,
                decay=args.temporal_ensemble_decay,
            )
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

            state_values_np = (
                state_raw.squeeze(0)
                .detach()
                .to(dtype=torch.float32, device="cpu")
                .numpy()
            )
            if args.enable_excavation_sequence_supervisor:
                executed_action, execution_constraints = (
                    apply_excavation_sequence_supervisor(
                        model_action,
                        state_values=state_values_np,
                        state_names=reply.get(
                            "observation_state_names",
                            [],
                        ),
                        predicted_stage_id=predicted_stage_id,
                        supervisor_state=execution_constraint_state,
                        load_trigger=args.supervisor_load_trigger,
                        load_rise=args.supervisor_load_rise,
                        dig_descent_height_trigger=(
                            args.supervisor_dig_descent_height_trigger
                        ),
                        dig_descent_stage_trigger=(
                            args.supervisor_dig_descent_stage_trigger
                        ),
                        dig_descent_velocity=(
                            args.supervisor_dig_descent_velocity
                        ),
                        dig_bucket_half_scale_distance=(
                            args.supervisor_dig_bucket_half_scale_distance
                        ),
                        dig_depth_tolerance=(
                            args.supervisor_dig_depth_tolerance
                        ),
                        dig_rebound_height=(
                            args.supervisor_dig_rebound_height
                        ),
                        dig_rebound_dwell_steps=(
                            args.supervisor_dig_rebound_dwell_steps
                        ),
                        dig_rebound_min_load=(
                            args.supervisor_dig_rebound_min_load
                        ),
                        curl_target=args.supervisor_curl_target,
                        curl_velocity=args.supervisor_curl_velocity,
                        lift_boom_target=args.supervisor_lift_boom_target,
                        lift_load_z_target=(
                            args.supervisor_lift_load_z_target
                        ),
                        max_lift_steps=args.supervisor_max_lift_steps,
                        turn_entry_boom_target=(
                            args.supervisor_turn_entry_boom_target
                        ),
                        turn_entry_velocity=(
                            args.supervisor_turn_entry_velocity
                        ),
                        turn_entry_steps=args.supervisor_turn_entry_steps,
                        turn_handoff_angle=(
                            args.supervisor_turn_handoff_angle
                        ),
                        phase_sync_bucket_closed=(
                            args.supervisor_phase_sync_bucket_closed
                        ),
                        phase_sync_boom_lifted=(
                            args.supervisor_phase_sync_boom_lifted
                        ),
                        unload_swing_target=(
                            args.supervisor_unload_swing_target
                        ),
                        unload_boom_target=(
                            args.supervisor_unload_boom_target
                        ),
                        unload_arm_target=args.supervisor_unload_arm_target,
                        unload_bucket_hold_target=(
                            args.supervisor_unload_bucket_hold_target
                        ),
                        swing_tolerance=args.supervisor_swing_tolerance,
                        unload_xy_tolerance=(
                            args.supervisor_unload_xy_tolerance
                        ),
                        unload_height_margin=(
                            args.supervisor_unload_height_margin
                        ),
                        unload_height_velocity=(
                            args.supervisor_unload_height_velocity
                        ),
                        position_max_steps=(
                            args.supervisor_position_max_steps
                        ),
                        dump_min_steps=args.supervisor_dump_min_steps,
                        dump_bucket_target=(
                            args.supervisor_dump_bucket_target
                        ),
                        dump_velocity=args.supervisor_dump_velocity,
                        empty_load_threshold=(
                            args.supervisor_empty_load_threshold
                        ),
                    )
                )
                if execution_constraints["transition"]:
                    print(
                        "[SEQUENCE SUPERVISOR] "
                        f"step={logical_step} "
                        f"transition={execution_constraints['phase_before']}"
                        f"->{execution_constraints['phase']} "
                        f"reason={execution_constraints['transition_reason']} "
                        f"load={execution_constraints['load']:.1f} "
                        f"q=[{execution_constraints['swing_q']:.3f}, "
                        f"{execution_constraints['boom_q']:.3f}, "
                        f"{execution_constraints['arm_q']:.3f}, "
                        f"{execution_constraints['bucket_q']:.3f}]"
                    )
            else:
                executed_action, execution_constraints = (
                    apply_optional_execution_constraints(
                        model_action,
                        state_values=state_values_np,
                        state_names=reply.get(
                            "observation_state_names",
                            [],
                        ),
                        predicted_stage_id=predicted_stage_id,
                        constraint_state=execution_constraint_state,
                        enable_load_retention=(
                            args.enable_load_retention_constraint
                        ),
                        enable_unload_geometry_gate=(
                            args.enable_unload_geometry_gate
                        ),
                        retention_min_load=args.retention_min_load,
                        retention_min_rise=args.retention_min_rise,
                        retention_bucket_target=args.retention_bucket_target,
                        retention_curl_velocity=args.retention_curl_velocity,
                        unload_swing_target=args.unload_swing_target,
                        unload_swing_tolerance=args.unload_swing_tolerance,
                        unload_swing_velocity=args.unload_swing_velocity,
                        unload_xy_tolerance=args.unload_xy_tolerance,
                        unload_min_height=args.unload_min_height,
                    )
                )
            raw_action = torch.from_numpy(
                executed_action
            ).unsqueeze(0)
            velocities, executed_action = (
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
                ("truck_bed_load_ms", "truck_bed_load_ms"),
            ):
                timing_samples[sample_name].append(
                    float(server_timing.get(reply_key, 0.0))
                )

            if is_warmup:
                cmd = {
                    "joint_velocities": [0.0] * num_joints,
                    "record_overview_camera": record_overview_camera,
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
                    action_chunk_history.clear()
                    execution_constraint_state.clear()
            else:
                # This is the only deployed-action assignment.
                cmd = {
                    "joint_velocities": velocities.tolist(),
                    "record_overview_camera": record_overview_camera,
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
                        "executed_action": {
                            name: float(executed_action[index])
                            for index, name in enumerate(ACTION_NAMES)
                        },
                        "execution_constraints": execution_constraints,
                        "input_clipping": {
                            "state": [
                                {
                                    "index": int(index),
                                    "name": str(name),
                                    "raw": float(raw),
                                    "normalized_before_clip": float(
                                        normalized
                                    ),
                                    "normalized_after_clip": float(
                                        np.clip(
                                            normalized,
                                            -args.max_input_abs_sigma,
                                            args.max_input_abs_sigma,
                                        )
                                    ),
                                }
                                for index, name, raw, normalized
                                in state_clip_violations
                            ],
                            "effort": [
                                {
                                    "index": int(index),
                                    "name": str(name),
                                    "raw": float(raw),
                                    "normalized_before_clip": float(
                                        normalized
                                    ),
                                    "normalized_after_clip": float(
                                        np.clip(
                                            normalized,
                                            -args.max_effort_abs_sigma,
                                            args.max_effort_abs_sigma,
                                        )
                                    ),
                                }
                                for index, name, raw, normalized
                                in effort_clip_violations
                            ],
                        },
                        "latest_action": {
                            name: float(latest_model_action[index])
                            for index, name in enumerate(ACTION_NAMES)
                        },
                        "temporal_ensemble": temporal_ensemble,
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
                        "truck_bed_load_metrics": reply.get(
                            "truck_bed_load_metrics",
                            {},
                        ),
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
                    supervisor_log = ""
                    if (
                        execution_constraints.get("mode")
                        == "excavation_sequence_supervisor_v1"
                    ):
                        supervisor_log = (
                            "supervisor_phase="
                            f"{execution_constraints['phase']} "
                            "phase_age="
                            f"{execution_constraints['phase_age_steps']} "
                            "depth_delta="
                            f"{execution_constraints['dig_depth_delta']:.3f} "
                            "rebound="
                            f"{execution_constraints['dig_rebound_distance']:.3f} "
                            "rebound_steps="
                            f"{execution_constraints['dig_rebound_steps']} "
                            "material_ready="
                            f"{execution_constraints['dig_material_ready']} "
                            "descent_assist="
                            f"{execution_constraints['dig_descent_assist_active']} "
                            "descent_started="
                            f"{execution_constraints['dig_descent_assist_started']} "
                            "descent_trigger="
                            f"{execution_constraints['dig_descent_trigger_reason']} "
                            "descent_elapsed="
                            f"{execution_constraints['dig_descent_assist_elapsed']} "
                            "depth_target_reached="
                            f"{execution_constraints['dig_depth_target_reached']} "
                            "bucket_action_scale="
                            f"{execution_constraints['dig_bucket_action_scale']:.2f} "
                            "turn_guide="
                            f"{execution_constraints['turn_guide_active']} "
                            "turn_guide_steps="
                            f"{execution_constraints['turn_guide_steps_completed']} "
                            "swing_error="
                            f"{execution_constraints['swing_error']:.3f} "
                            "unload_xy_distance="
                            f"{execution_constraints['unload_xy_distance']:.3f} "
                            "unload_height_delta="
                            f"{execution_constraints['unload_height_delta']:.3f} "
                            "position_timeout="
                            f"{execution_constraints['position_timeout_reached']} "
                            "dump_target_reached="
                            f"{execution_constraints['dump_bucket_target_reached']} "
                            "constraint_reasons="
                            f"{execution_constraints['reasons']} "
                        )
                    print(
                        f"[STEP {policy_step:05d}] "
                        f"chunk_origin={current_chunk_origin_step} "
                        f"chunk_index={executed_chunk_index} "
                        f"pred_stage={predicted_stage_id}:"
                        f"{predicted_stage_name} "
                        f"stage_conf={predicted_stage_confidence_value:.4f} "
                        f"model_action="
                        f"{model_action.round(5).tolist()} "
                        f"executed_action="
                        f"{executed_action.round(5).tolist()} "
                        f"latest_action="
                        f"{latest_model_action.round(5).tolist()} "
                        f"ensemble_n="
                        f"{temporal_ensemble['num_predictions']} "
                        f"joint_velocities="
                        f"{velocities.round(5).tolist()} "
                        f"state_key={state_key} "
                        f"effort_key={effort_key} "
                        f"q={raw_q.round(5).tolist()} "
                        f"{supervisor_log}"
                        f"cameras={sorted(camera_rgbs)} "
                        f"infer_ms={inference_ms:.1f} "
                        f"server_ms="
                        f"{float(server_timing.get('total_server_ms', 0.0)):.1f} "
                        f"loop_ms={loop_ms:.1f}"
                    )
                policy_steps_inferred = max(
                    policy_steps_inferred,
                    policy_step + 1,
                )

            timing_samples["loop_ms"].append(
                (time.perf_counter() - loop_start) * 1000.0
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        # The regular loop infers the final action after receiving its last
        # observation. A statistical episode must send that action once and
        # read the resulting terminal state so --steps means exactly that
        # many deployed model actions.
        if (
            args.episode_status_json
            and not early_terminated
            and args.steps > 0
            and policy_steps_inferred >= args.steps
        ):
            write_json(sock, cmd)
            terminal_reply = read_json(sock)
            if terminal_reply.get("type") == "error":
                raise RuntimeError(
                    "Bridge error while reading terminal episode state: "
                    f"{terminal_reply.get('error')}"
                )
            final_reply = terminal_reply
            raw_q = np.asarray(
                terminal_reply.get("joint_positions", []),
                dtype=np.float32,
            ).reshape(-1)
            task_complete = update_episode_success(
                terminal_reply,
                raw_q,
                executed_steps=args.steps,
            )
            terminal_timing = terminal_reply.get("bridge_timing", {})
            simulated_seconds_total += float(
                terminal_timing.get("simulated_seconds", 0.0)
            )
            if task_complete:
                termination_reason = (
                    "truck_success_and_bucket_open_at_step_limit"
                )

    except KeyboardInterrupt:
        termination_reason = "interrupted"
        episode_error = "KeyboardInterrupt"
        print("\n[INFO] interrupted; sending zero velocity")

    except Exception as exc:
        termination_reason = "error"
        episode_error = f"{type(exc).__name__}: {exc}"
        raise

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

        print(
            "[INPUT SUMMARY] "
            f"state_clip_events={state_clip_event_count} "
            f"effort_clip_events={effort_clip_event_count}"
        )

        if args.episode_status_json:
            rollout_ended_unix_ns = time.time_ns()
            final_success = bool(
                final_truck_deposited_count
                >= args.truck_success_min_particles
                and success_no_growth_observations
                >= args.truck_success_no_growth_steps
            )
            final_bucket_open = bool(
                final_bucket_q is not None
                and final_bucket_q >= args.bucket_open_threshold
            )
            status_payload = {
                "schema_version": "excavator_success_rate_episode_v1",
                "status": (
                    "error" if episode_error is not None else "completed"
                ),
                "error": episode_error,
                "success": final_success,
                "task_completed_success_and_bucket_open": bool(
                    task_completion_unix_ns is not None
                ),
                "early_terminated": bool(early_terminated),
                "termination_reason": termination_reason,
                "max_policy_steps": int(args.steps),
                "policy_steps_inferred": int(policy_steps_inferred),
                "policy_steps_executed": int(policy_steps_executed),
                "truck_success_min_particles": int(
                    args.truck_success_min_particles
                ),
                "truck_success_no_growth_steps": int(
                    args.truck_success_no_growth_steps
                ),
                "truck_initial_particle_count": (
                    int(truck_initial_count)
                    if truck_initial_count is not None
                    else None
                ),
                "final_truck_particle_count": int(
                    final_truck_raw_count
                ),
                "final_newly_deposited_particle_count": int(
                    final_truck_deposited_count
                ),
                "success_no_growth_observations_at_end": int(
                    success_no_growth_observations
                ),
                "bucket_open_threshold_rad": float(
                    args.bucket_open_threshold
                ),
                "final_bucket_position_rad": final_bucket_q,
                "final_bucket_open": final_bucket_open,
                "policy_execution_started_unix_ns": int(
                    policy_execution_started_unix_ns
                ),
                "simulation_launch_unix_ns": (
                    int(args.simulation_launch_unix_ns)
                    if args.simulation_launch_unix_ns > 0
                    else None
                ),
                "first_success_unix_ns": first_success_unix_ns,
                "first_particle_detected_unix_ns": (
                    first_particle_detected_unix_ns
                ),
                "task_completion_unix_ns": task_completion_unix_ns,
                "rollout_ended_unix_ns": int(rollout_ended_unix_ns),
                "success_time_from_policy_start_seconds": elapsed_seconds(
                    first_success_unix_ns,
                    policy_execution_started_unix_ns,
                ),
                "success_time_from_simulation_launch_seconds": (
                    elapsed_seconds(
                        first_success_unix_ns,
                        args.simulation_launch_unix_ns,
                    )
                    if args.simulation_launch_unix_ns > 0
                    else None
                ),
                "task_completion_time_from_policy_start_seconds": (
                    elapsed_seconds(
                        task_completion_unix_ns,
                        policy_execution_started_unix_ns,
                    )
                ),
                "task_completion_time_from_simulation_launch_seconds": (
                    elapsed_seconds(
                        task_completion_unix_ns,
                        args.simulation_launch_unix_ns,
                    )
                    if args.simulation_launch_unix_ns > 0
                    else None
                ),
                "rollout_time_from_policy_start_seconds": elapsed_seconds(
                    rollout_ended_unix_ns,
                    policy_execution_started_unix_ns,
                ),
                "rollout_time_from_simulation_launch_seconds": (
                    elapsed_seconds(
                        rollout_ended_unix_ns,
                        args.simulation_launch_unix_ns,
                    )
                    if args.simulation_launch_unix_ns > 0
                    else None
                ),
                "final_truck_bed_load_metrics": final_reply.get(
                    "truck_bed_load_metrics",
                    {},
                ),
            }
            status_path = (
                Path(args.episode_status_json).expanduser().resolve()
            )
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_status_path = status_path.with_name(
                status_path.name + ".tmp"
            )
            temporary_status_path.write_text(
                json.dumps(
                    status_payload,
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_status_path, status_path)
            print(
                "[EPISODE STATUS] "
                f"path={status_path} "
                f"success={final_success} "
                "task_completed="
                f"{status_payload['task_completed_success_and_bucket_open']} "
                f"steps={policy_steps_executed} "
                f"truck_particles={final_truck_deposited_count} "
                f"bucket_open={final_bucket_open}"
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
