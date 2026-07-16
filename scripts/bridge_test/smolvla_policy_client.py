#!/usr/bin/env python3
# PHYSICAL_STAGE_MACHINE_V10
# Deep dig target + smooth loaded transport + 3 strong-open cues in 60 frames + repeatable extra-open + post-unload return-to-pile
import argparse
import json
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (
    decode_camera_images,
    decode_rgb_payload as decode_rgb,
    read_json,
    write_json,
)
from excavator_common.deployment_contract import (
    build_client_contract,
    load_training_fps,
    sha256_files,
    validate_training_fps,
)


class MultiCameraVideoRecorder:
    """Write one RGB frame per policy step to one MP4 file per camera.

    The video timestamp is simulation-step based rather than wall-clock based:
    every recorded policy step contributes exactly one frame. Therefore, with
    fps=30, 4800 policy steps produce a 160-second video when the run completes.
    Warmup frames are intentionally excluded.
    """

    def __init__(self, output_dir, fps=30.0, codec="mp4v", camera_ids=("0", "1", "2")):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for --record-video-dir. "
                "Install opencv-python in the smolvla environment."
            ) from exc

        if fps <= 0:
            raise ValueError("Video fps must be positive.")
        if len(codec) != 4:
            raise ValueError("Video codec must contain exactly four characters.")

        self.cv2 = cv2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = str(codec)
        self.camera_ids = tuple(str(camera_id) for camera_id in camera_ids)
        self.writers = {}
        self.paths = {}
        self.frame_sizes = {}
        self.frames_written = 0
        self.first_step = None
        self.last_step = None
        self.closed = False

    @staticmethod
    def _normalize_rgb(frame, camera_id):
        rgb = np.asarray(frame)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise RuntimeError(
                f"Camera {camera_id} frame must have shape HxWx3 or HxWx4; "
                f"got {rgb.shape}."
            )
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            if np.issubdtype(rgb.dtype, np.floating):
                max_value = float(np.nanmax(rgb)) if rgb.size else 0.0
                if max_value <= 1.0:
                    rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)

    def _open_writer(self, camera_id, rgb):
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
                f"Failed to open video writer for {path} using codec "
                f"{self.codec!r}."
            )
        self.writers[camera_id] = writer
        self.paths[camera_id] = path
        self.frame_sizes[camera_id] = (int(width), int(height))
        print(
            f"[VIDEO] camera={camera_id} path={path} "
            f"size={width}x{height} fps={self.fps:.3f} codec={self.codec}"
        )

    def write(self, camera_rgbs, step):
        if self.closed:
            raise RuntimeError("Cannot write to a closed video recorder.")

        normalized = {}
        for camera_id in self.camera_ids:
            if camera_id not in camera_rgbs:
                raise RuntimeError(
                    f"Missing camera {camera_id} while recording. "
                    f"Available cameras: {sorted(camera_rgbs.keys())}"
                )
            normalized[camera_id] = self._normalize_rgb(
                camera_rgbs[camera_id], camera_id
            )

        # Initialize all writers before writing any camera for this step so the
        # three files always remain frame-aligned.
        for camera_id, rgb in normalized.items():
            if camera_id not in self.writers:
                self._open_writer(camera_id, rgb)
            width, height = self.frame_sizes[camera_id]
            if rgb.shape[1] != width or rgb.shape[0] != height:
                raise RuntimeError(
                    f"Camera {camera_id} resolution changed from "
                    f"{width}x{height} to {rgb.shape[1]}x{rgb.shape[0]}."
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
                f"[VIDEO] frames_written={self.frames_written} "
                f"last_policy_step={step} "
                f"encoded_duration={self.frames_written / self.fps:.3f}s"
            )

    def close(self):
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
                self.frames_written / self.fps if self.fps > 0 else 0.0
            ),
            "warmup_included": False,
            "camera_files": {
                camera_id: str(path)
                for camera_id, path in self.paths.items()
            },
            "frame_sizes": {
                camera_id: [width, height]
                for camera_id, (width, height) in self.frame_sizes.items()
            },
        }
        metadata_path = self.output_dir / "recording_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[VIDEO] closed: frames_per_camera={self.frames_written}, "
            f"duration={metadata['encoded_duration_seconds']:.3f}s, "
            f"metadata={metadata_path}"
        )


def rgb_to_tensor(rgb, device):
    """
    Input:
      rgb: H x W x 3, uint8, range 0..255

    Output:
      tensor: 1 x 3 x 256 x 256, float32, range 0..1
    """
    x = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    x = x / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x


def make_state(reply, device):
    """
    Final effort18 checkpoint input order:

      0:14  = original observation_state
      14:18 = measured effort

    The measured-effort order must be:
      [swing, boom, arm, bucket]
    """
    if "observation_state" not in reply:
        raise RuntimeError(
            "Bridge reply is missing observation_state. "
            f"Available reply keys: {sorted(reply.keys())}"
        )

    base_state = np.asarray(
        reply["observation_state"],
        dtype=np.float32,
    ).reshape(-1)

    # Also support a future server that already sends the full 18D state.
    if base_state.shape[0] == 18:
        state18 = base_state

    elif base_state.shape[0] == 14:
        effort_raw = None
        effort_key = None

        # Support the likely bridge field names.
        for key in (
            "observation_effort",
            "observation.effort",
            "joint_efforts",
            "joint_effort",
            "measured_effort",
            "effort",
        ):
            if key in reply and reply[key] is not None:
                effort_raw = reply[key]
                effort_key = key
                break

        if effort_raw is None:
            raise RuntimeError(
                "The final checkpoint requires an 18D state, but "
                "the bridge only sent the 14D observation_state and "
                "no measured-effort field was found. "
                f"Available reply keys: {sorted(reply.keys())}"
            )

        effort = np.asarray(
            effort_raw,
            dtype=np.float32,
        ).reshape(-1)

        if effort.shape[0] < 4:
            raise RuntimeError(
                f"Expected at least 4 measured-effort values from "
                f"reply[{effort_key!r}], got shape {effort.shape}: "
                f"{effort.tolist()}"
            )

        state18 = np.concatenate(
            [
                base_state,
                effort[:4],
            ],
            axis=0,
        )

    else:
        raise RuntimeError(
            "Expected observation_state to contain either "
            f"14 or 18 values, got {base_state.shape}: "
            f"{base_state.tolist()}"
        )

    if state18.shape != (18,):
        raise RuntimeError(
            f"Expected final state shape (18,), "
            f"got {state18.shape}"
        )

    if not np.all(np.isfinite(state18)):
        raise RuntimeError(
            f"18D state contains NaN or Inf: "
            f"{state18.tolist()}"
        )

    return (
        torch.from_numpy(state18)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )


def _feature_shape(feature):
    if isinstance(feature, dict):
        value = feature.get("shape")
    else:
        value = getattr(feature, "shape", None)
    if value is None:
        return None
    return tuple(int(item) for item in value)


def validate_policy_feature_contract(policy):
    config = getattr(policy, "config", None)
    input_features = getattr(config, "input_features", None)
    output_features = getattr(config, "output_features", None)
    if not isinstance(input_features, dict) or not isinstance(output_features, dict):
        raise RuntimeError("Checkpoint config does not expose input_features/output_features")
    state_shape = _feature_shape(input_features.get("observation.state"))
    action_shape = _feature_shape(output_features.get("action"))
    if state_shape != (18,):
        raise RuntimeError(f"Checkpoint observation.state shape must be (18,), got {state_shape}")
    if action_shape != (4,):
        raise RuntimeError(f"Checkpoint action shape must be (4,), got {action_shape}")
    camera_keys = tuple(f"observation.images.{index}" for index in range(3))
    missing_cameras = [key for key in camera_keys if key not in input_features]
    if missing_cameras:
        raise RuntimeError(f"Checkpoint is missing camera features: {missing_cameras}")
    print(
        "[BRIDGE CONTRACT] checkpoint features: "
        f"state_shape={state_shape} action_shape={action_shape} cameras={list(camera_keys)}"
    )


def load_normalization_stats(ckpt_dir):
    preprocessor_path = Path(ckpt_dir) / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    postprocessor_path = Path(ckpt_dir) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"

    state_mean = None
    state_std = None
    action_mean = None
    action_std = None

    for path in [preprocessor_path, postprocessor_path]:
        if not path.exists():
            continue
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key == "observation.state.mean":
                    state_mean = tensor.numpy()
                elif key == "observation.state.std":
                    state_std = tensor.numpy()
                elif key == "action.mean":
                    action_mean = tensor.numpy()
                elif key == "action.std":
                    action_std = tensor.numpy()

    if state_mean is not None and state_std is not None:
        state_std = np.where(state_std < 1e-6, 1.0, state_std)
        print(f"[INFO] state_mean[:4]={state_mean[:4].round(4).tolist()} state_std[:4]={state_std[:4].round(4).tolist()}")
    if action_mean is not None and action_std is not None:
        action_std = np.where(action_std < 1e-6, 1.0, action_std)
        print(f"[INFO] action_mean={action_mean.round(4).tolist()} action_std={action_std.round(4).tolist()}")

    return state_mean, state_std, action_mean, action_std


def action_to_joint_velocities(
    action,
    num_joints,
    swing_min_gain=0.25,
    boom_activity_ref=0.05,
    arm_activity_ref=0.06,
    bucket_activity_ref=0.12,
    arm_activity_weight=0.50,
    activity_deadzone=0.15,
):
    """
    Action order:

      0 = swing
      1 = boom
      2 = arm
      3 = bucket

    Current simulator direction:

      boom < 0:
        lower the working equipment

      boom > 0:
        raise the working equipment

      bucket < 0:
        curl / close bucket

      bucket > 0:
        open / dump bucket

    Processing:

      swing:
        Dynamically suppressed during active
        boom, arm, or bucket movement.

      boom:
        Only downward motion is amplified.
        Maximum gain: 1.25.

      arm:
        Unchanged.

      bucket:
        Curl is amplified only while boom is
        descending or not moving upward.
        Maximum gain: 1.15.

    No max-velocity clipping.
    No global action multiplier.
    """

    a = (
        action
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    if a.ndim == 2:
        a = a[0]

    a = np.asarray(
        a,
        dtype=np.float32,
    ).reshape(-1)

    if a.size < 4:
        raise ValueError(
            "Expected at least four "
            f"action values, got {a.shape}"
        )

    # Keep the original policy output unchanged
    # while calculating every gain.
    a = a[:4].copy()

    if not np.all(
        np.isfinite(a)
    ):
        raise RuntimeError(
            "Policy action contains "
            f"NaN or Inf: {a.tolist()}"
        )


    # ========================================================
    # A. Dynamic swing suppression
    # ========================================================

    boom_activity = (
        abs(float(a[1]))
        / max(
            float(boom_activity_ref),
            1e-6,
        )
    )

    arm_activity = (
        float(arm_activity_weight)
        * abs(float(a[2]))
        / max(
            float(arm_activity_ref),
            1e-6,
        )
    )

    bucket_activity = (
        abs(float(a[3]))
        / max(
            float(bucket_activity_ref),
            1e-6,
        )
    )

    raw_work_activity = max(
        boom_activity,
        arm_activity,
        bucket_activity,
    )

    deadzone = np.clip(
        float(activity_deadzone),
        0.0,
        0.95,
    )

    work_activity = np.clip(
        (
            raw_work_activity
            - deadzone
        )
        / (
            1.0
            - deadzone
        ),
        0.0,
        1.0,
    )

    minimum_swing_gain = np.clip(
        float(swing_min_gain),
        0.0,
        1.0,
    )

    swing_gain = (
        1.0
        - (
            1.0
            - minimum_swing_gain
        )
        * work_activity
    )


    # ========================================================
    # B. Boom downward-motion boost
    #
    # boom < 0:
    #     descending
    #     gain = 1.00 ~ 1.25
    #
    # boom >= 0:
    #     rising or stationary
    #     gain = 1.00
    # ========================================================

    if float(a[1]) < 0.0:

        downward_speed = abs(
            float(a[1])
        )

        boom_boost_activity = np.clip(
            (
                downward_speed
                - 0.005
            )
            / (
                0.05
                - 0.005
            ),
            0.0,
            1.0,
        )

        boom_gain = (
            1.0
            + 0.25
            * boom_boost_activity
        )

        boom_gain = min(
            float(boom_gain),
            1.25,
        )

    else:

        boom_boost_activity = 0.0
        boom_gain = 1.0


    # ========================================================
    # C. Bucket curl boost
    #
    # Only enable extra curl when:
    #
    #   bucket < 0
    #       and
    #   boom <= 0
    #
    # This avoids accelerating bucket curl while
    # the boom is already lifting upward.
    #
    # Maximum bucket gain is 1.15, smaller than
    # the maximum boom gain of 1.25.
    # ========================================================

    allow_bucket_boost = (
        float(a[3]) < 0.0
        and float(a[1]) <= 0.0
    )

    if allow_bucket_boost:

        bucket_curl_speed = abs(
            float(a[3])
        )

        bucket_boost_activity = np.clip(
            (
                bucket_curl_speed
                - 0.01
            )
            / (
                0.12
                - 0.01
            ),
            0.0,
            1.0,
        )

        bucket_gain = (
            1.0
            + 0.15
            * bucket_boost_activity
        )

        bucket_gain = min(
            float(bucket_gain),
            1.15,
        )

    else:

        bucket_boost_activity = 0.0
        bucket_gain = 1.0


    # ========================================================
    # D. Final simulator command
    # ========================================================

    vel = np.zeros(
        num_joints,
        dtype=np.float32,
    )

    n = min(
        num_joints,
        4,
    )

    # Start from the original model output.
    vel[:n] = a[:n]

    # Swing:
    # dynamic suppression.
    if n > 0:
        vel[0] = (
            a[0]
            * swing_gain
        )

    # Boom:
    # boost downward motion only.
    if n > 1:
        vel[1] = (
            a[1]
            * boom_gain
        )

    # Arm:
    # unchanged.
    if n > 2:
        vel[2] = a[2]

    # Bucket:
    # boost curl only while not lifting.
    if n > 3:
        vel[3] = (
            a[3]
            * bucket_gain
        )

    if not np.all(
        np.isfinite(vel)
    ):
        raise RuntimeError(
            "Processed velocity contains "
            f"NaN or Inf: {vel.tolist()}"
        )

    return (
        vel,
        float(swing_gain),
        float(work_activity),
        float(boom_gain),
        float(boom_boost_activity),
        float(bucket_gain),
        float(bucket_boost_activity),
    )



def patch_checkpoint_paths(ckpt_dir, vlm_dir):
    """
    Make the checkpoint fully local/offline.

    Some SmolVLA checkpoint files may still contain:
      HuggingFaceTB/SmolVLM2-500M-Video-Instruct

    Replace that with the local VLM path in config/preprocessor files.
    """
    ckpt_dir = Path(ckpt_dir)
    vlm_dir = str(Path(vlm_dir))

    files = [
        ckpt_dir / "config.json",
        ckpt_dir / "policy_preprocessor.json",
        ckpt_dir / "train_config.json",
    ]

    old = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def replace_obj(obj):
        if isinstance(obj, dict):
            return {k: replace_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_obj(x) for x in obj]
        if isinstance(obj, str):
            return obj.replace(old, vlm_dir)
        return obj

    for p in files:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            new_data = replace_obj(data)
            if new_data != data:
                p.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
                print(f"[INFO] patched local model path in {p}")
        except Exception as e:
            print(f"[WARN] failed to patch {p}: {e}")


def load_language_tokens(task, vlm_dir, device):
    """
    Create language token tensors manually.

    We also pass raw "task" string in the batch, but these token fields help if
    the policy expects pre-tokenized language inputs.
    """
    processor = AutoProcessor.from_pretrained(vlm_dir, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    lang = tokenizer(
        task,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=48,
    )

    language_tokens = lang["input_ids"].to(device)
    language_mask = lang["attention_mask"].bool().to(device)

    return language_tokens, language_mask




def extract_measured_effort(reply):
    """Return raw measured effort in [swing, boom, arm, bucket] order."""
    for key in (
        "observation_effort",
        "observation.effort",
        "joint_efforts",
        "joint_effort",
        "measured_effort",
        "effort",
    ):
        value = reply.get(key)
        if value is None:
            continue

        effort = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        if effort.size >= 4:
            return effort[:4].copy(), key

    return np.zeros(4, dtype=np.float32), None


def ema(previous, value, alpha):
    """Numerically simple exponential moving average."""
    value = float(value)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    if previous is None:
        return value

    return (
        (1.0 - alpha) * float(previous)
        + alpha * value
    )


def reset_policy_queue(policy, reason):
    if hasattr(policy, "reset"):
        policy.reset()
        print(
            f"[POLICY RESET] {reason}; "
            "old action chunk discarded"
        )


def stage_suppress_swing(
    vel,
    model_action_np,
    swing_gain,
    stage_gain,
):
    """
    Apply a strict swing cap relative to the raw model swing action.

    The regular dynamic swing suppression is retained; the stage cap can only
    make the swing smaller, never larger.
    """
    if vel.size <= 0 or model_action_np.size <= 0:
        return float(swing_gain)

    effective_gain = min(
        float(swing_gain),
        float(np.clip(stage_gain, 0.0, 1.0)),
    )

    vel[0] = (
        float(model_action_np[0])
        * effective_gain
    )

    return effective_gain


def main():
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Bridge/model.
    # --------------------------------------------------------
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ckpt", default=os.environ.get("SMOLVLA_CKPT", ""))
    parser.add_argument("--vlm", default=os.environ.get("SMOLVLA_VLM", ""))
    parser.add_argument(
        "--dataset-meta",
        default=os.environ.get("SMOLVLA_DATASET_META", ""),
        help="Old training dataset root or meta/info.json; its fps drives bridge cadence.",
    )
    parser.add_argument(
        "--training-fps",
        type=float,
        default=0.0,
        help="Explicit diagnostic fallback when dataset metadata is unavailable.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--replan-interval", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--record-video-dir",
        default="",
        help=(
            "Directory for camera_0.mp4, camera_1.mp4, and camera_2.mp4. "
            "One frame is written for every non-warmup policy step."
        ),
    )
    parser.add_argument(
        "--record-video-fps",
        type=float,
        default=0.0,
        help="Encoded playback FPS; 0 uses the checkpoint training dataset FPS.",
    )
    parser.add_argument(
        "--record-video-codec",
        default="mp4v",
        help="FourCC codec used by OpenCV VideoWriter; default: mp4v.",
    )

    # --------------------------------------------------------
    # Existing model-action shaping.
    # --------------------------------------------------------
    parser.add_argument("--swing-min-gain", type=float, default=0.25)
    parser.add_argument("--boom-activity-ref", type=float, default=0.05)
    parser.add_argument("--arm-activity-ref", type=float, default=0.06)
    parser.add_argument("--bucket-activity-ref", type=float, default=0.12)
    parser.add_argument("--arm-activity-weight", type=float, default=0.50)
    parser.add_argument("--activity-deadzone", type=float, default=0.15)
    parser.add_argument("--manipulation-swing-gain", type=float, default=0.15)
    parser.add_argument(
        "--dump-swing-gain",
        type=float,
        default=0.35,
        help=(
            "Maximum retained swing gain during dump_descent, dump_ready, "
            "dumping, and extra_open."
        ),
    )

    # --------------------------------------------------------
    # Real-height trend detector.
    # --------------------------------------------------------
    parser.add_argument(
        "--height-trend-window",
        type=int,
        default=50,
        help="Compare current real tip_z with tip_z N frames ago.",
    )
    parser.add_argument(
        "--height-trend-min-drop",
        type=float,
        default=0.08,
        help="Real tip drop over the window required to count as descent.",
    )
    parser.add_argument(
        "--height-trend-confirm",
        type=int,
        default=3,
        help="Number of consecutive window decisions required.",
    )
    parser.add_argument(
        "--transition-rise-threshold",
        type=float,
        default=0.01,
        help="After real descent is seen, model boom >= this means it wants to rise.",
    )

    # --------------------------------------------------------
    # Digging minimum target.
    #
    # Semantics:
    #   --dig-target-tip-z is a minimum depth requirement, not a floor.
    #   If the bucket is still above it when the model tries to rise/curl,
    #   force descent until it is reached. Once reached, model descent is
    #   unrestricted, including during loading.
    # --------------------------------------------------------
    parser.add_argument(
        "--dig-target-tip-z",
        type=float,
        default=None,
        help="Minimum digging tip_z that must be reached; lower z is deeper.",
    )
    parser.add_argument(
        "--extra-descent-distance",
        type=float,
        default=0.25,
        help="Fallback relative drop when --dig-target-tip-z is omitted.",
    )
    parser.add_argument("--dig-descent-speed", type=float, default=0.05)
    parser.add_argument("--dig-descent-max-steps", type=int, default=2000)
    parser.add_argument("--target-z-tolerance", type=float, default=0.01)

    # --------------------------------------------------------
    # Training-normalized multi-joint effort.
    #
    # The checkpoint's state mean/std for dimensions 14:18 are used to
    # normalize [swing, boom, arm, bucket] efforts. Scores below are thus in
    # training-standard-deviation units rather than simulator raw units.
    # --------------------------------------------------------
    parser.add_argument("--effort-ema-alpha", type=float, default=0.20)
    parser.add_argument("--effort-baseline-alpha", type=float, default=0.02)
    parser.add_argument(
        "--load-effort-contact-score",
        type=float,
        default=2.5,
        help="Multi-joint normalized effort score for sand contact.",
    )
    parser.add_argument(
        "--load-effort-sufficient-score",
        type=float,
        default=4.0,
        help="Multi-joint normalized effort score considered enough load.",
    )
    parser.add_argument("--load-contact-confirm", type=int, default=5)
    parser.add_argument("--load-sufficient-confirm", type=int, default=5)
    parser.add_argument(
        "--load-curl-start-confirm",
        type=int,
        default=3,
        help="Clear model-curl frames required to enter loading after depth is reached.",
    )

    # --------------------------------------------------------
    # Bucket manipulation and physical joint limits.
    # --------------------------------------------------------
    parser.add_argument("--bucket-motion-threshold", type=float, default=0.03)
    parser.add_argument("--bucket-stop-threshold", type=float, default=0.01)
    parser.add_argument(
        "--bucket-stop-confirm",
        type=int,
        default=6,
        help=(
            "Weighted consecutive score required to trigger each extra-open. "
            "After the bucket has opened at least once: a near-zero bucket "
            "command contributes 1, a closing command below -bucket-stop-"
            "threshold contributes 2, and any positive non-near-zero command "
            "breaks the sequence and resets the score."
        ),
    )
    parser.add_argument("--extra-curl-speed", type=float, default=0.04)
    parser.add_argument("--extra-curl-angle", type=float, default=0.06)
    parser.add_argument("--extra-curl-max-steps", type=int, default=40)
    parser.add_argument(
        "--bucket-curl-limit",
        type=float,
        default=-2.04,
        help="Stop negative bucket commands at/under this real joint angle.",
    )
    parser.add_argument(
        "--bucket-open-limit",
        type=float,
        default=1.52,
        help="Stop positive bucket commands at/over this real joint angle.",
    )

    # --------------------------------------------------------
    # Forced lift after loading.
    # --------------------------------------------------------
    parser.add_argument("--post-load-min-tip-z", type=float, default=2.10)
    parser.add_argument("--post-load-lift-speed", type=float, default=0.05)
    parser.add_argument("--post-load-lift-max-steps", type=int, default=2000)

    # --------------------------------------------------------
    # Loaded transport / unloading height ceiling.
    #
    # This is a hard maximum height, not a target that forces descent during
    # ordinary transport. Below the ceiling, the model keeps full authority.
    # Above the ceiling, boom is driven down until the tip returns below it.
    # --------------------------------------------------------
    parser.add_argument(
        "--loaded-max-tip-z",
        type=float,
        default=5.50,
        help="Hard maximum bucket-tip height during loaded transport/unloading.",
    )
    parser.add_argument(
        "--loaded-height-recovery-speed",
        type=float,
        default=0.04,
        help="Downward boom speed used only to recover from exceeding the loaded height ceiling.",
    )

    # --------------------------------------------------------
    # Loaded transport and dump descent.
    #
    # dump-target-tip-z is a minimum descent requirement.
    # dump-min-safe-tip-z is a hard lower safety bound. Because lower z is
    # deeper, dump-min-safe-tip-z must be numerically smaller than target.
    # After target is reached, model descent remains allowed until the hard
    # safety bound is reached.
    # --------------------------------------------------------
    parser.add_argument(
        "--dump-target-tip-z",
        type=float,
        default=None,
        help="Required dump approach height; lower z is deeper.",
    )
    parser.add_argument(
        "--dump-min-safe-tip-z",
        type=float,
        default=None,
        help="Hard lowest safe dump tip_z; downward commands are blocked below it.",
    )
    parser.add_argument("--dump-descent-speed", type=float, default=0.04)
    parser.add_argument("--dump-descent-max-steps", type=int, default=2000)
    parser.add_argument("--dump-safety-recovery-speed", type=float, default=0.05)
    parser.add_argument(
        "--dump-model-descent-threshold",
        type=float,
        default=0.03,
        help="Clear negative model-boom request required before forced dump descent is allowed.",
    )
    parser.add_argument(
        "--dump-model-descent-confirm",
        type=int,
        default=3,
        help="Number of clear model-descent frames required to enter dump_descent.",
    )
    parser.add_argument(
        "--transport-descent-ignore-threshold",
        type=float,
        default=0.03,
        help=(
            "During loaded_transport, model boom commands at or below the "
            "negative of this threshold are treated as obvious downward "
            "commands and physically ignored. Height-ceiling recovery remains "
            "active as a separate safety correction."
        ),
    )
    parser.add_argument(
        "--dump-open-confirm",
        type=int,
        default=3,
        help=(
            "Number of strong bucket-opening frames required inside the rolling "
            "dump-open window after the minimum-safe unload height is reached. "
            "Non-opening/closing frames append zero rather than decrementing or "
            "clearing evidence; evidence expires only when it leaves the window."
        ),
    )
    parser.add_argument(
        "--dump-open-window",
        type=int,
        default=60,
        help="Rolling frame window used to count bucket-opening evidence.",
    )
    parser.add_argument(
        "--dump-open-strong-threshold",
        type=float,
        default=0.15,
        help=(
            "A model bucket command at or above this value contributes one "
            "strong-opening evidence frame. V10 does not enter dumping from a "
            "single strong frame; the rolling-window count must reach "
            "--dump-open-confirm after the minimum-safe height is reached."
        ),
    )
    parser.add_argument(
        "--dump-entry-effort-score",
        type=float,
        default=1.5,
        help="All-four-joint normalized effort change retained for dump-entry diagnostics; V10 entry itself is controlled by strong-opening evidence.",
    )
    parser.add_argument(
        "--dump-entry-effort-confirm",
        type=int,
        default=2,
        help="Retained for CLI compatibility and diagnostics; V10 does not use this as an independent dump-entry path.",
    )

    # --------------------------------------------------------
    # Unloading effort and repeatable weighted extra opening.
    # --------------------------------------------------------
    parser.add_argument(
        "--unload-effort-change-score",
        type=float,
        default=1.5,
        help="Normalized multi-joint effort change from dump-start reference.",
    )
    parser.add_argument("--unload-confirm", type=int, default=3)
    parser.add_argument("--extra-open-speed", type=float, default=0.04)
    parser.add_argument("--extra-open-angle", type=float, default=0.06)
    parser.add_argument("--extra-open-max-steps", type=int, default=40)

    # --------------------------------------------------------
    # Post-unload decision, optional lift, and reverse return to the pile.
    # --------------------------------------------------------
    parser.add_argument(
        "--post-unload-rise-threshold",
        type=float,
        default=0.03,
        help="Positive model-boom command considered a clear request to raise.",
    )
    parser.add_argument(
        "--post-unload-rise-confirm",
        type=int,
        default=3,
        help="Consecutive clear model-raise frames required before forced lift.",
    )
    parser.add_argument(
        "--post-unload-near-ceiling-margin",
        type=float,
        default=0.30,
        help=(
            "If tip_z is within this distance below --loaded-max-tip-z when "
            "unloading finishes, skip the lift and return toward the pile."
        ),
    )
    parser.add_argument("--post-dump-min-tip-z", type=float, default=5.20)
    parser.add_argument("--post-dump-lift-speed", type=float, default=0.05)
    parser.add_argument("--post-dump-lift-max-steps", type=int, default=2000)
    parser.add_argument(
        "--return-swing-speed",
        type=float,
        default=0.05,
        help="Absolute swing speed used to return toward the initial pile heading.",
    )
    parser.add_argument(
        "--return-swing-tolerance",
        type=float,
        default=0.03,
        help="Swing-angle tolerance for completing the return-to-pile turn.",
    )

    parser.add_argument("--dump-dir", default="")

    args = parser.parse_args()

    print(
        "[INFO] V10 logic: "
        f"strong_open_evidence={args.dump_open_confirm}/{args.dump_open_window} "
        f"at threshold={args.dump_open_strong_threshold:.3f}, "
        f"repeatable_extra_open_score={args.bucket_stop_confirm} "
        f"(near_zero=+1, closing=+2), "
        f"transport_down_ignore={args.transport_descent_ignore_threshold:.3f}, "
        "dumping_closing_commands_are_counted_but_not_executed, "
        f"unload_effort={args.unload_effort_change_score:.3f}/"
        f"{args.unload_confirm}, "
        f"dump_swing_gain={args.dump_swing_gain:.3f}, "
        f"post_unload_rise={args.post_unload_rise_confirm}x"
        f"{args.post_unload_rise_threshold:.3f}, "
        f"direct_return_margin={args.post_unload_near_ceiling_margin:.3f}"
    )

    if args.height_trend_window < 2:
        raise SystemExit("--height-trend-window must be >= 2")
    if args.record_video_fps < 0:
        raise SystemExit("--record-video-fps must be >= 0 (0 uses training FPS)")
    if len(args.record_video_codec) != 4:
        raise SystemExit("--record-video-codec must contain exactly four characters")
    if (
        args.dump_target_tip_z is not None
        and args.dump_min_safe_tip_z is not None
        and float(args.dump_min_safe_tip_z) >= float(args.dump_target_tip_z)
    ):
        raise SystemExit(
            "Because lower z is deeper, --dump-min-safe-tip-z must be "
            "numerically smaller than --dump-target-tip-z."
        )
    if float(args.post_load_min_tip_z) > float(args.loaded_max_tip_z):
        raise SystemExit(
            "--post-load-min-tip-z must not exceed --loaded-max-tip-z."
        )
    if float(args.post_dump_min_tip_z) > float(args.loaded_max_tip_z):
        raise SystemExit(
            "--post-dump-min-tip-z must not exceed --loaded-max-tip-z."
        )
    if float(args.post_unload_near_ceiling_margin) < 0.0:
        raise SystemExit("--post-unload-near-ceiling-margin must be >= 0")
    if (
        args.dump_target_tip_z is not None
        and float(args.dump_target_tip_z) > float(args.loaded_max_tip_z)
    ):
        print(
            "[WARN] --dump-target-tip-z is above --loaded-max-tip-z. "
            "The loaded height ceiling will take priority."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    if not args.ckpt:
        raise SystemExit("--ckpt is required, or set SMOLVLA_CKPT.")
    if not args.vlm:
        raise SystemExit("--vlm is required, or set SMOLVLA_VLM.")

    print("[INFO] checkpoint:", args.ckpt)
    print("[INFO] local VLM:", args.vlm)
    patch_checkpoint_paths(args.ckpt, args.vlm)

    print("[INFO] loading SmolVLA policy...")
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(device)
    validate_policy_feature_contract(policy)

    print("[INFO] loading normalization stats...")
    state_mean = state_std = action_mean = action_std = None
    try:
        state_mean, state_std, action_mean, action_std = load_normalization_stats(args.ckpt)
        state_mean_t = (
            torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
            if state_mean is not None else None
        )
        state_std_t = (
            torch.from_numpy(state_std).to(device=device, dtype=torch.float32)
            if state_std is not None else None
        )
        action_mean_t = (
            torch.from_numpy(action_mean).to(device=device, dtype=torch.float32)
            if action_mean is not None else None
        )
        action_std_t = (
            torch.from_numpy(action_std).to(device=device, dtype=torch.float32)
            if action_std is not None else None
        )
        has_norm = all(
            x is not None
            for x in (state_mean_t, state_std_t, action_mean_t, action_std_t)
        )
    except Exception as exc:
        print(f"[WARN] failed to load normalization stats: {exc}")
        has_norm = False
        state_mean_t = state_std_t = action_mean_t = action_std_t = None

    if not has_norm:
        raise RuntimeError(
            "The 18D deployment requires checkpoint state/action normalization statistics."
        )
    if tuple(np.asarray(state_mean).shape) != (18,) or tuple(np.asarray(state_std).shape) != (18,):
        raise RuntimeError(
            f"Expected 18D state normalization, got mean={np.asarray(state_mean).shape} "
            f"std={np.asarray(state_std).shape}"
        )
    if tuple(np.asarray(action_mean).shape) != (4,) or tuple(np.asarray(action_std).shape) != (4,):
        raise RuntimeError(
            f"Expected 4D action normalization, got mean={np.asarray(action_mean).shape} "
            f"std={np.asarray(action_std).shape}"
        )

    if state_mean is not None and state_std is not None and len(state_mean) >= 18:
        effort_train_mean = np.asarray(state_mean[14:18], dtype=np.float32)
        effort_train_std = np.maximum(
            np.asarray(state_std[14:18], dtype=np.float32),
            1e-6,
        )
        print(
            "[INFO] training effort mean/std: "
            f"mean={effort_train_mean.round(6).tolist()} "
            f"std={effort_train_std.round(6).tolist()}"
        )
    else:
        effort_train_mean = np.zeros(4, dtype=np.float32)
        effort_train_std = np.ones(4, dtype=np.float32)
        print(
            "[WARN] effort normalization stats unavailable; "
            "effort scores will use raw bridge units"
        )

    print("[INFO] task text will be read from the simulator bridge")
    language_tokens = None
    language_mask = None
    reset_policy_queue(policy, "initialization")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    if args.dataset_meta:
        training_fps, dataset_meta_path = load_training_fps(args.dataset_meta)
        if args.training_fps > 0.0 and abs(training_fps - args.training_fps) > 1.0e-6:
            raise RuntimeError(
                f"--training-fps={args.training_fps} disagrees with {dataset_meta_path}: {training_fps}"
            )
        print(f"[BRIDGE CONTRACT] training_fps={training_fps} source={dataset_meta_path}")
    elif args.training_fps > 0.0:
        training_fps = validate_training_fps(args.training_fps)
        print(
            "[WARN] training FPS supplied explicitly rather than read from meta/info.json: "
            f"{training_fps}"
        )
    else:
        raise RuntimeError(
            "Set --dataset-meta/SMOLVLA_DATASET_META to the old training dataset meta/info.json."
        )

    normalization_paths = [
        Path(args.ckpt) / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        Path(args.ckpt) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    normalization_hash = sha256_files(normalization_paths)
    if not normalization_hash:
        raise RuntimeError("Could not hash checkpoint normalization assets")
    handshake = build_client_contract(training_fps, normalization_hash=normalization_hash)
    write_json(sock, handshake)
    handshake_reply = read_json(sock)
    if not handshake_reply.get("ok", False):
        raise RuntimeError(f"Bridge handshake rejected: {handshake_reply}")
    if handshake_reply.get("normalization_hash", "") != normalization_hash:
        raise RuntimeError("Bridge handshake normalization identity mismatch")
    print("[BRIDGE CONTRACT] accepted:", handshake_reply)

    cmd = {"joint_velocities": [0.0, 0.0, 0.0, 0.0]}
    dump_dir = args.dump_dir
    dump_frames = []
    dump_limit = 10
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    video_recorder = None
    if args.record_video_dir:
        record_video_fps = (
            float(args.record_video_fps)
            if float(args.record_video_fps) > 0.0
            else float(training_fps)
        )
        video_recorder = MultiCameraVideoRecorder(
            output_dir=args.record_video_dir,
            fps=record_video_fps,
            codec=args.record_video_codec,
            camera_ids=("0", "1", "2"),
        )
        print(
            "[VIDEO] recording enabled: one frame per non-warmup policy step; "
            f"output_dir={args.record_video_dir}, fps={record_video_fps:.3f}"
        )

    # ========================================================
    # State machine.
    # ========================================================
    stage = "digging"
    has_loaded = False

    tip_history = deque(maxlen=max(2, args.height_trend_window + 1))
    height_trend_delta = 0.0
    real_descent_now = False
    descent_confirm_count = 0
    dig_descent_seen = False
    dump_descent_seen = False

    # Dig minimum-target enforcement.
    dig_target_active = False
    dig_target_value = None
    dig_target_start_tip_z = None
    dig_target_steps = 0
    dig_target_reached = False
    dig_target_timeout_warned = False

    # Effort features in checkpoint training-normalized coordinates.
    effort_norm_ema = None
    empty_effort_baseline = None
    load_effort_score = 0.0
    load_contact_count = 0
    load_sufficient_count = 0
    load_contact_confirmed = False
    load_sufficient = False

    # Loading.
    load_curl_start_count = 0
    bucket_curl_seen = False
    bucket_curl_stop_count = 0
    extra_curl_active = False
    extra_curl_used = False
    extra_curl_start_q = None
    extra_curl_steps = 0
    extra_curl_progress = 0.0
    load_complete_reason = ""

    # Forced post-load lift.
    post_load_lift_steps = 0
    post_load_lift_timeout_warned = False

    # Dump descent/dumping.
    dump_descent_steps = 0
    dump_target_reached = False
    dump_target_timeout_warned = False
    dump_model_descent_count = 0
    dump_open_history = deque(maxlen=max(1, int(args.dump_open_window)))
    dump_open_count = 0
    dump_entry_effort_count = 0
    dump_entry_effort_score = 0.0
    loaded_effort_reference = None
    dump_effort_reference = None
    dump_entry_reason = ""
    loaded_height_limited = False
    unload_effort_score = 0.0
    unload_effort_count = 0
    unload_effort_confirmed = False
    bucket_open_seen = False
    bucket_open_stop_count = 0

    # Extra opening.
    extra_open_active = False
    extra_open_used = False
    extra_open_count = 0
    extra_open_start_q = None
    extra_open_steps = 0
    extra_open_progress = 0.0
    dump_complete_reason = ""

    # Post-unload decision, optional lift, and return to the initial pile heading.
    post_unload_rise_count = 0
    post_unload_decision_steps = 0
    post_unload_near_ceiling = False
    post_dump_lift_steps = 0
    post_dump_lift_timeout_warned = False
    home_swing_q = None
    return_swing_error = 0.0
    return_to_pile_steps = 0
    timing_samples = {
        "roundtrip_ms": [],
        "client_prepare_ms": [],
        "inference_ms": [],
        "server_total_ms": [],
        "physics_ms": [],
        "camera_render_copy_ms": [],
        "camera_resize_ms": [],
        "encode_ms": [],
        "bucket_load_ms": [],
    }
    simulated_seconds_total = 0.0
    deployment_wall_start = time.perf_counter()

    def reset_height_trend(current_tip_z):
        nonlocal tip_history, descent_confirm_count
        tip_history.clear()
        tip_history.append(float(current_tip_z))
        descent_confirm_count = 0

    def current_bucket_q(raw_q):
        return float(raw_q[3]) if raw_q.size > 3 else float("nan")

    try:
        task_text = None
        total_steps = args.warmup_steps + args.steps

        for loop_step in range(total_steps):
            is_warmup = loop_step < args.warmup_steps
            step = loop_step - args.warmup_steps

            policy_loop_start = time.perf_counter()
            write_json(sock, cmd)
            reply = read_json(sock)
            reply_received = time.perf_counter()
            roundtrip_ms = (reply_received - policy_loop_start) * 1000.0
            if reply.get("type") == "error":
                raise RuntimeError(f"Bridge error: {reply.get('error')}")

            raw_observation_state = np.asarray(
                reply["observation_state"], dtype=np.float32
            ).reshape(-1)
            if raw_observation_state.size < 14:
                raise RuntimeError(
                    "Expected at least 14 raw observation-state values, "
                    f"got {raw_observation_state.shape}"
                )

            raw_tip_z = float(raw_observation_state[10])
            raw_load_particles = max(0.0, float(raw_observation_state[7]))
            raw_q = np.asarray(
                reply.get("joint_positions", []), dtype=np.float32
            ).reshape(-1)
            bucket_q = current_bucket_q(raw_q)
            if home_swing_q is None and raw_q.size > 0:
                home_swing_q = float(raw_q[0])
                print(f"[RETURN] captured initial pile heading swing_q={home_swing_q:.5f}")
            bucket_at_curl_limit = (
                np.isfinite(bucket_q)
                and bucket_q <= float(args.bucket_curl_limit)
            )
            bucket_at_open_limit = (
                np.isfinite(bucket_q)
                and bucket_q >= float(args.bucket_open_limit)
            )

            measured_effort, effort_key = extract_measured_effort(reply)
            measured_effort = np.asarray(measured_effort[:4], dtype=np.float32)
            effort_norm = (
                measured_effort - effort_train_mean
            ) / effort_train_std
            if effort_norm_ema is None:
                effort_norm_ema = effort_norm.copy()
            else:
                a = float(np.clip(args.effort_ema_alpha, 0.0, 1.0))
                effort_norm_ema = a * effort_norm + (1.0 - a) * effort_norm_ema

            # Empty baseline is allowed to adapt only before real digging
            # descent and before loading starts.
            if empty_effort_baseline is None:
                empty_effort_baseline = effort_norm_ema.copy()
            elif (
                is_warmup
                or (
                    stage == "digging"
                    and not dig_descent_seen
                    and not has_loaded
                )
            ):
                a = float(np.clip(args.effort_baseline_alpha, 0.0, 1.0))
                empty_effort_baseline = (
                    a * effort_norm_ema
                    + (1.0 - a) * empty_effort_baseline
                )

            load_effort_delta_vec = np.abs(
                effort_norm_ema[1:4] - empty_effort_baseline[1:4]
            )
            load_effort_score = float(np.max(load_effort_delta_vec))

            bridge_task = str(reply.get("task_text", "")).strip()
            if not bridge_task:
                raise RuntimeError(
                    "Bridge reply is missing task_text. "
                    f"Available keys: {sorted(reply.keys())}"
                )
            if bridge_task != task_text:
                task_text = bridge_task
                print(f"[INFO] task_text from bridge: {task_text}")
                language_tokens, language_mask = load_language_tokens(
                    task_text, args.vlm, device
                )

            rgb = decode_rgb(reply)
            camera_rgbs = decode_camera_images(reply, np_module=np)
            rgb0 = camera_rgbs.get("0", rgb)
            rgb1 = camera_rgbs.get("1", rgb)
            rgb2 = camera_rgbs.get("2", rgb)

            # Record exactly one frame from each camera for every policy step.
            # Warmup is excluded so --steps 4800 yields exactly 4800 frames per
            # output video, i.e. 160 seconds at 30 FPS.
            if video_recorder is not None and not is_warmup:
                video_recorder.write(
                    {"0": rgb0, "1": rgb1, "2": rgb2},
                    step=step,
                )

            image0 = rgb_to_tensor(rgb0, device)
            image1 = rgb_to_tensor(rgb1, device)
            image2 = rgb_to_tensor(rgb2, device)

            state = make_state(reply, device)
            if has_norm:
                state = (state - state_mean_t) / state_std_t

            if dump_dir and not is_warmup and step < dump_limit:
                step_dir = os.path.join(dump_dir, f"step_{step:04d}")
                os.makedirs(step_dir, exist_ok=True)
                Image.fromarray(rgb0).save(os.path.join(step_dir, "image_0.png"))
                Image.fromarray(rgb1).save(os.path.join(step_dir, "image_1.png"))
                Image.fromarray(rgb2).save(os.path.join(step_dir, "image_2.png"))
                np.savetxt(
                    os.path.join(step_dir, "state.txt"),
                    state.squeeze(0).cpu().numpy(),
                    fmt="%.6f",
                )
                with open(
                    os.path.join(step_dir, "task.txt"), "w", encoding="utf-8"
                ) as handle:
                    handle.write(task_text)
                np.savetxt(
                    os.path.join(step_dir, "observation_state.txt"),
                    raw_observation_state,
                    fmt="%.6f",
                )
                dump_frames.append(step)
                if step == dump_limit - 1:
                    print(f"[DUMP] saved {len(dump_frames)} frames to {dump_dir}")
                    break

            batch = {
                "observation.state": state,
                "observation.images.0": image0,
                "observation.images.1": image1,
                "observation.images.2": image2,
                "task": [task_text],
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

            if (
                args.replan_interval > 0
                and not is_warmup
                and step % args.replan_interval == 0
            ):
                reset_policy_queue(policy, f"periodic replan at step {step}")

            inference_start = time.perf_counter()
            with torch.no_grad():
                action = policy.select_action(batch)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            server_timing = reply.get("bridge_timing", {})
            simulated_seconds_total += float(server_timing.get("simulated_seconds", 0.0))
            timing_samples["roundtrip_ms"].append(float(roundtrip_ms))
            timing_samples["client_prepare_ms"].append(
                float((inference_start - reply_received) * 1000.0)
            )
            timing_samples["inference_ms"].append(float(inference_ms))
            for sample_key, reply_key in (
                ("server_total_ms", "total_server_ms"),
                ("physics_ms", "physics_ms"),
                ("camera_render_copy_ms", "camera_render_copy"),
                ("camera_resize_ms", "camera_resize"),
                ("encode_ms", "encode_ms"),
                ("bucket_load_ms", "bucket_load_ms"),
            ):
                timing_samples[sample_key].append(float(server_timing.get(reply_key, 0.0)))
            if has_norm:
                action = action * action_std_t + action_mean_t

            if is_warmup:
                num_joints = len(raw_q) if len(raw_q) > 0 else 4
                cmd = {
                    "joint_velocities": [0.0] * num_joints,
                }
                discarded = (
                    action.detach().float().cpu().numpy().round(4).tolist()
                )
                print(
                    f"[WARMUP {loop_step + 1:03d}/{args.warmup_steps:03d}] "
                    f"discarded_action={discarded} "
                    f"tip_z={raw_tip_z:.5f} "
                    f"effort_norm={effort_norm.round(3).tolist()} "
                    "commanded_zero_velocity"
                )
                if loop_step == args.warmup_steps - 1:
                    reset_policy_queue(policy, "warmup complete")
                    reset_height_trend(raw_tip_z)
                time.sleep(args.sleep)
                continue

            # ------------------------------------------------
            # Real tip-height trend over the requested window.
            # ------------------------------------------------
            tip_history.append(raw_tip_z)
            if len(tip_history) >= args.height_trend_window + 1:
                height_trend_delta = float(raw_tip_z - tip_history[0])
                real_descent_now = (
                    height_trend_delta
                    <= -abs(float(args.height_trend_min_drop))
                )
            else:
                height_trend_delta = 0.0
                real_descent_now = False

            # ------------------------------------------------
            # Base model-derived command.
            # ------------------------------------------------
            num_joints = len(raw_q) if len(raw_q) > 0 else 4
            (
                vel,
                swing_gain,
                work_activity,
                boom_gain,
                boom_boost_activity,
                bucket_gain,
                bucket_boost_activity,
            ) = action_to_joint_velocities(
                action,
                num_joints=num_joints,
                swing_min_gain=args.swing_min_gain,
                boom_activity_ref=args.boom_activity_ref,
                arm_activity_ref=args.arm_activity_ref,
                bucket_activity_ref=args.bucket_activity_ref,
                arm_activity_weight=args.arm_activity_weight,
                activity_deadzone=args.activity_deadzone,
            )
            model_action_np = (
                action.detach().float().cpu().numpy().reshape(-1)
            )
            if model_action_np.size < 4:
                raise RuntimeError(
                    f"Expected four model actions, got {model_action_np.shape}"
                )
            model_boom = float(model_action_np[1])
            model_bucket = float(model_action_np[3])
            clear_model_curl = (
                model_bucket <= -abs(float(args.bucket_motion_threshold))
            )
            clear_model_open = (
                model_bucket >= abs(float(args.bucket_motion_threshold))
            )
            model_bucket_stopped = (
                abs(model_bucket) <= abs(float(args.bucket_stop_threshold))
            )

            # ------------------------------------------------
            # Dig descent is based on actual 50-frame tip-height change.
            # ------------------------------------------------
            if stage == "digging" and not dig_target_active:
                if real_descent_now:
                    descent_confirm_count += 1
                else:
                    descent_confirm_count = max(0, descent_confirm_count - 1)

                if (
                    not dig_descent_seen
                    and descent_confirm_count
                    >= max(1, int(args.height_trend_confirm))
                ):
                    dig_descent_seen = True
                    print(
                        "[DIG] real descent confirmed: "
                        f"window={args.height_trend_window}, "
                        f"tip_delta={height_trend_delta:.5f} m"
                    )

            dig_low_enough = (
                args.dig_target_tip_z is None
                or raw_tip_z
                <= float(args.dig_target_tip_z)
                + abs(float(args.target_z_tolerance))
            )

            # Start minimum-depth enforcement only when the model tries to
            # leave descent (rise) or starts curling. Reaching the target does
            # NOT create a lower bound; later model descent is unrestricted.
            if (
                stage == "digging"
                and dig_descent_seen
                and not dig_target_active
                and (
                    model_boom >= abs(float(args.transition_rise_threshold))
                    or clear_model_curl
                )
            ):
                if args.dig_target_tip_z is None:
                    dig_target_value = (
                        raw_tip_z - max(0.0, float(args.extra_descent_distance))
                    )
                else:
                    dig_target_value = float(args.dig_target_tip_z)

                if raw_tip_z > dig_target_value + abs(float(args.target_z_tolerance)):
                    dig_target_active = True
                    dig_target_start_tip_z = raw_tip_z
                    dig_target_steps = 0
                    dig_target_timeout_warned = False
                    stage = "dig_target_descent"
                    print(
                        "[DIG TARGET] enforcing minimum digging depth: "
                        f"start_tip_z={raw_tip_z:.5f}, "
                        f"target_tip_z={dig_target_value:.5f}"
                    )
                else:
                    dig_target_reached = True
                    print(
                        "[DIG TARGET] minimum depth already reached; "
                        "no downward restriction will be applied"
                    )

            # Enforce the minimum only while it has not been reached.
            if dig_target_active:
                stage = "dig_target_descent"
                dig_target_steps += 1
                reached = (
                    raw_tip_z
                    <= float(dig_target_value)
                    + abs(float(args.target_z_tolerance))
                )
                if (
                    not reached
                    and not dig_target_timeout_warned
                    and dig_target_steps
                    >= max(1, int(args.dig_descent_max_steps))
                ):
                    dig_target_timeout_warned = True
                    print(
                        "[WARN] dig target still not reached after "
                        f"{dig_target_steps} steps; continuing because the "
                        "configured dig height is a mandatory minimum depth"
                    )

                if reached:
                    dig_target_active = False
                    dig_target_reached = True
                    if vel.size > 1:
                        vel[1] = 0.0
                    reset_policy_queue(policy, "dig minimum target phase finished")
                    stage = "loading" if load_contact_confirmed else "digging"
                    print(
                        "[DIG TARGET] finished: "
                        f"reason=target, "
                        f"tip_z={raw_tip_z:.5f}, "
                        f"target_tip_z={dig_target_value:.5f}, "
                        f"steps={dig_target_steps}. "
                        "Further model descent is unrestricted."
                    )
                    reset_height_trend(raw_tip_z)
                else:
                    if vel.size > 1:
                        vel[1] = -abs(float(args.dig_descent_speed))

            # ------------------------------------------------
            # Enter loading from real depth + sustained model curl.
            # This does not depend on simulator particle count.
            # ------------------------------------------------
            if (
                stage == "digging"
                and dig_descent_seen
                and dig_low_enough
            ):
                if clear_model_curl:
                    load_curl_start_count += 1
                else:
                    load_curl_start_count = max(0, load_curl_start_count - 1)

                if load_curl_start_count >= max(1, args.load_curl_start_confirm):
                    stage = "loading"
                    bucket_curl_seen = True
                    print(
                        "[STAGE] digging -> loading: "
                        f"tip_z={raw_tip_z:.5f}, "
                        f"load_effort_score={load_effort_score:.3f}"
                    )

            # ------------------------------------------------
            # Loading effort evidence and one-time extra curl.
            # ------------------------------------------------
            if stage in {"loading", "extra_curl"}:
                contact_now = (
                    load_effort_score >= float(args.load_effort_contact_score)
                )
                sufficient_now = (
                    load_effort_score >= float(args.load_effort_sufficient_score)
                )

                load_contact_count = (
                    load_contact_count + 1
                    if contact_now
                    else max(0, load_contact_count - 1)
                )
                load_sufficient_count = (
                    load_sufficient_count + 1
                    if sufficient_now
                    else max(0, load_sufficient_count - 1)
                )

                if (
                    not load_contact_confirmed
                    and load_contact_count >= max(1, args.load_contact_confirm)
                ):
                    load_contact_confirmed = True
                    print(
                        "[LOAD] effort contact confirmed: "
                        f"score={load_effort_score:.3f}, "
                        f"threshold={args.load_effort_contact_score:.3f}"
                    )

                if (
                    not load_sufficient
                    and load_sufficient_count
                    >= max(1, args.load_sufficient_confirm)
                ):
                    load_sufficient = True
                    print(
                        "[LOAD] sufficient effort load confirmed: "
                        f"score={load_effort_score:.3f}, "
                        f"threshold={args.load_effort_sufficient_score:.3f}"
                    )

                if clear_model_curl:
                    bucket_curl_seen = True
                    bucket_curl_stop_count = 0
                elif bucket_curl_seen and model_bucket_stopped:
                    bucket_curl_stop_count += 1
                else:
                    bucket_curl_stop_count = max(0, bucket_curl_stop_count - 1)

                if bucket_at_curl_limit:
                    load_complete_reason = "bucket_curl_limit"
                elif load_sufficient and not extra_curl_active:
                    load_complete_reason = "sufficient_effort"
                elif (
                    not extra_curl_active
                    and not extra_curl_used
                    and bucket_curl_seen
                    and bucket_curl_stop_count
                    >= max(1, args.bucket_stop_confirm)
                    and not load_sufficient
                ):
                    extra_curl_active = True
                    extra_curl_used = True
                    extra_curl_start_q = bucket_q if np.isfinite(bucket_q) else None
                    extra_curl_steps = 0
                    extra_curl_progress = 0.0
                    stage = "extra_curl"
                    print(
                        "[EXTRA CURL] model curl stopped but effort load is "
                        "below sufficient threshold; starting a repeatable correction"
                    )

            if extra_curl_active:
                stage = "extra_curl"
                if extra_curl_start_q is not None and np.isfinite(bucket_q):
                    extra_curl_progress = max(
                        0.0, float(extra_curl_start_q) - float(bucket_q)
                    )

                curl_limit_finished = bucket_at_curl_limit
                curl_angle_finished = (
                    extra_curl_progress >= max(0.0, float(args.extra_curl_angle))
                )
                curl_timeout_finished = (
                    extra_curl_steps >= max(1, int(args.extra_curl_max_steps))
                )
                curl_effort_finished = load_sufficient

                if (
                    curl_limit_finished
                    or curl_angle_finished
                    or curl_timeout_finished
                    or curl_effort_finished
                ):
                    extra_curl_active = False
                    if vel.size > 3:
                        vel[3] = 0.0
                    if curl_limit_finished:
                        load_complete_reason = "extra_curl_joint_limit"
                    elif curl_effort_finished:
                        load_complete_reason = "extra_curl_sufficient_effort"
                    elif curl_angle_finished:
                        load_complete_reason = "extra_curl_angle"
                    else:
                        load_complete_reason = "extra_curl_max_steps"
                else:
                    if vel.size > 3:
                        vel[3] = -abs(float(args.extra_curl_speed))
                    extra_curl_steps += 1

            # Physical limit always wins over model/extra negative curl.
            if bucket_at_curl_limit and vel.size > 3 and vel[3] < 0.0:
                vel[3] = 0.0

            # Loading complete -> mandatory lift.
            if load_complete_reason and stage in {"loading", "extra_curl"}:
                stage = "lift_after_load"
                has_loaded = True
                post_load_lift_steps = 0
                post_load_lift_timeout_warned = False
                if vel.size > 3 and vel[3] < 0.0:
                    vel[3] = 0.0
                reset_policy_queue(policy, f"loading complete: {load_complete_reason}")
                reset_height_trend(raw_tip_z)
                print(
                    "[STAGE] loading -> lift_after_load: "
                    f"reason={load_complete_reason}, "
                    f"tip_z={raw_tip_z:.5f}, "
                    f"bucket_q={bucket_q:.5f}, "
                    f"load_effort_score={load_effort_score:.3f}"
                )
                load_complete_reason = ""

            # ------------------------------------------------
            # Mandatory lift after loading.
            # ------------------------------------------------
            if stage == "lift_after_load":
                post_load_lift_steps += 1
                lift_reached = (
                    raw_tip_z
                    >= float(args.post_load_min_tip_z)
                    - abs(float(args.target_z_tolerance))
                )
                if (
                    not lift_reached
                    and not post_load_lift_timeout_warned
                    and post_load_lift_steps
                    >= max(1, int(args.post_load_lift_max_steps))
                ):
                    post_load_lift_timeout_warned = True
                    print(
                        "[WARN] post-load minimum height still not reached "
                        "after the configured warning step count; continuing lift"
                    )

                if lift_reached:
                    if vel.size > 1:
                        vel[1] = 0.0
                    stage = "loaded_transport"
                    reset_policy_queue(policy, "post-load lift finished")
                    reset_height_trend(raw_tip_z)
                    dump_descent_seen = False
                    dump_model_descent_count = 0
                    dump_open_history.clear()
                    dump_open_count = 0
                    dump_entry_effort_count = 0
                    dump_entry_effort_score = 0.0
                    loaded_effort_reference = effort_norm_ema.copy()
                    dump_effort_reference = None
                    dump_entry_reason = ""
                    print(
                        "[STAGE] lift_after_load -> loaded_transport: "
                        f"reason=height, tip_z={raw_tip_z:.5f}, "
                        f"loaded_effort_reference="
                        f"{loaded_effort_reference.round(3).tolist()}"
                    )
                else:
                    if vel.size > 1:
                        vel[1] = abs(float(args.post_load_lift_speed))
                    # Hold the loaded bucket during the mandatory lift.
                    if vel.size > 3:
                        vel[3] = 0.0

            # ------------------------------------------------
            # V9 rolling strong-opening evidence.
            #
            # Evidence is counted only after the bucket tip is at/above the
            # configured minimum-safe unload height. Each frame with
            # model_bucket >= --dump-open-strong-threshold contributes one.
            # Other frames append zero: they do not decrement or clear evidence.
            # An event expires only after it leaves the rolling window. If the
            # tip drops below the safe height, the window is cleared so evidence
            # collected too low cannot be carried upward into a dump trigger.
            # ------------------------------------------------
            dump_open_height_ok = (
                args.dump_min_safe_tip_z is None
                or raw_tip_z
                >= float(args.dump_min_safe_tip_z)
                - abs(float(args.target_z_tolerance))
            )
            strong_model_open = (
                model_bucket
                >= abs(float(args.dump_open_strong_threshold))
            )

            if stage in {"loaded_transport", "dump_descent", "dump_ready"}:
                if dump_open_height_ok:
                    previous_dump_open_count = int(dump_open_count)
                    if strong_model_open and previous_dump_open_count == 0:
                        # Capture a pre-opening effort reference before the first
                        # strong opening cue in the current rolling window.
                        dump_effort_reference = effort_norm_ema.copy()
                    dump_open_history.append(1 if strong_model_open else 0)
                    dump_open_count = int(sum(dump_open_history))
                else:
                    dump_open_history.clear()
                    dump_open_count = 0
                    dump_entry_effort_count = 0

            # Below the configured safe unloading height, positive bucket
            # commands are physically blocked during loaded transport.
            if (
                stage == "loaded_transport"
                and not dump_open_height_ok
                and vel.size > 3
                and vel[3] > 0.0
            ):
                vel[3] = 0.0

            # ------------------------------------------------
            # Loaded transport.
            #
            # V10 transport rules:
            #   1) Do NOT suppress a clear bucket-opening request. Opening is
            #      itself a dump-entry cue.
            #   2) Do NOT force descent merely because the machine is loaded.
            #      Forced target descent is allowed only after a clear model
            #      boom-down request (or actual descent while boom is negative).
            #   3) Four-joint effort change can reinforce an opening cue.
            # ------------------------------------------------
            if stage == "loaded_transport":
                if loaded_effort_reference is None:
                    loaded_effort_reference = effort_norm_ema.copy()

                dump_entry_effort_score = float(
                    np.max(
                        np.abs(
                            effort_norm_ema[0:4]
                            - loaded_effort_reference[0:4]
                        )
                    )
                )

                # V10 transport smoothing: obvious model boom-down commands
                # are ignored during loaded transport. They remain visible in
                # the log and in dump_model_descent_count for diagnostics, but
                # they cannot trigger dump_descent. The separate 5.5 m ceiling
                # recovery below is still allowed to command a safe descent.
                clear_dump_descent_request = (
                    model_boom
                    <= -abs(float(args.transport_descent_ignore_threshold))
                )
                if clear_dump_descent_request:
                    dump_model_descent_count += 1
                    if vel.size > 1 and vel[1] < 0.0:
                        vel[1] = 0.0
                else:
                    dump_model_descent_count = 0

                if real_descent_now:
                    descent_confirm_count += 1
                else:
                    descent_confirm_count = 0

                # V10 dump entry is exclusively the requested rule:
                # at/above the safe height, at least N strong opening frames
                # inside the rolling window. A single strong frame and the
                # effort diagnostic cannot enter dumping on their own.
                open_entry_confirmed = (
                    dump_open_height_ok
                    and dump_open_count
                    >= max(1, int(args.dump_open_confirm))
                )

                if open_entry_confirmed:
                    stage = "dumping"
                    bucket_open_seen = True
                    bucket_open_stop_count = 0
                    unload_effort_count = 0
                    unload_effort_confirmed = False
                    dump_entry_reason = "strong_bucket_open_window"
                    if dump_effort_reference is None:
                        dump_effort_reference = effort_norm_ema.copy()
                    print(
                        "[STAGE] loaded_transport -> dumping: "
                        f"reason={dump_entry_reason}, "
                        f"tip_z={raw_tip_z:.5f}, "
                        f"model_bucket={model_bucket:+.4f}, "
                        f"strong_open_count={dump_open_count}/"
                        f"{int(args.dump_open_window)}, "
                        f"required={int(args.dump_open_confirm)}, "
                        f"height_ok={int(dump_open_height_ok)}, "
                        f"entry_effort_score={dump_entry_effort_score:.3f}, "
                        f"dump_effort_reference="
                        f"{dump_effort_reference.round(3).tolist()}"
                    )
                else:
                    # V10 intentionally has no loaded_transport -> dump_descent
                    # transition. Transport ignores obvious boom-down requests;
                    # unloading starts from the rolling strong-open evidence.
                    pass

            # ------------------------------------------------
            # Dump descent.
            #
            # Before the required target height is reached, V10 injects the
            # configured downward boom speed and suppresses opening. As soon
            # as the target is reached, the state advances to dump_ready.
            # dump_ready no longer injects descent: the model may hold, rise,
            # or descend farther, while the hard lower safety guard remains
            # active.
            # ------------------------------------------------
            if stage == "dump_descent":
                dump_descent_steps += 1

                # The opening rule remains active during the optional descent
                # path. Three strong cues in the safe-height rolling window can
                # start dumping immediately; dump_descent is never mandatory.
                open_entry_confirmed = (
                    dump_open_height_ok
                    and dump_open_count
                    >= max(1, int(args.dump_open_confirm))
                )
                if open_entry_confirmed:
                    stage = "dumping"
                    bucket_open_seen = True
                    bucket_open_stop_count = 0
                    unload_effort_count = 0
                    unload_effort_confirmed = False
                    dump_entry_reason = "strong_bucket_open_window"
                    if dump_effort_reference is None:
                        dump_effort_reference = effort_norm_ema.copy()
                    print(
                        "[STAGE] dump_descent -> dumping: "
                        f"reason={dump_entry_reason}, "
                        f"tip_z={raw_tip_z:.5f}, "
                        f"strong_open_count={dump_open_count}/"
                        f"{int(args.dump_open_window)}, "
                        f"required={int(args.dump_open_confirm)}, "
                        f"model_bucket={model_bucket:+.4f}"
                    )
                else:
                    dump_target_reached = (
                        args.dump_target_tip_z is None
                        or raw_tip_z
                        <= float(args.dump_target_tip_z)
                        + abs(float(args.target_z_tolerance))
                    )

                    if not dump_target_reached:
                        if vel.size > 1:
                            vel[1] = -abs(float(args.dump_descent_speed))
                        # Keep the bucket closed until either the rolling strong
                        # opening rule confirms dumping or the target is reached.
                        if vel.size > 3 and vel[3] > 0.0:
                            vel[3] = 0.0
                    else:
                        stage = "dump_ready"
                        # Preserve strong-opening evidence collected at safe
                        # height across the target-height boundary.
                        dump_entry_effort_count = 0
                        dump_entry_reason = "dump_target_reached"
                        print(
                            "[STAGE] dump_descent -> dump_ready: "
                            f"tip_z={raw_tip_z:.5f}, "
                            f"target={args.dump_target_tip_z}, "
                            f"safe_min={args.dump_min_safe_tip_z}. "
                            "Forced descent is now disabled; model height action "
                            "is preserved until the hard safety floor."
                        )

                    if (
                        not dump_target_reached
                        and not dump_target_timeout_warned
                        and dump_descent_steps
                        >= max(1, int(args.dump_descent_max_steps))
                    ):
                        dump_target_timeout_warned = True
                        print(
                            "[WARN] dump target still not reached after the "
                            "configured warning step count; continuing descent "
                            "while keeping the hard minimum-safe-height guard active"
                        )

            # ------------------------------------------------
            # Dump ready.
            #
            # The required approach height has been reached. There is no
            # forced downward command here. Dump entry still uses exactly the
            # V10 rule: N strong opening frames in the rolling safe-height window.
            # Effort remains diagnostic here and is not an independent entry.
            # ------------------------------------------------
            if stage == "dump_ready":
                dump_target_reached = True

                if loaded_effort_reference is None:
                    loaded_effort_reference = effort_norm_ema.copy()

                dump_entry_effort_score = float(
                    np.max(
                        np.abs(
                            effort_norm_ema[0:4]
                            - loaded_effort_reference[0:4]
                        )
                    )
                )

                open_entry_confirmed = (
                    dump_open_height_ok
                    and dump_open_count
                    >= max(1, int(args.dump_open_confirm))
                )

                if open_entry_confirmed:
                    stage = "dumping"
                    bucket_open_seen = True
                    bucket_open_stop_count = 0
                    unload_effort_count = 0
                    unload_effort_confirmed = False
                    dump_entry_reason = "strong_bucket_open_window"
                    if dump_effort_reference is None:
                        dump_effort_reference = effort_norm_ema.copy()
                    print(
                        "[STAGE] dump_ready -> dumping: "
                        f"reason={dump_entry_reason}, "
                        f"tip_z={raw_tip_z:.5f}, "
                        f"strong_open_count={dump_open_count}/"
                        f"{int(args.dump_open_window)}, "
                        f"required={int(args.dump_open_confirm)}, "
                        f"model_bucket={model_bucket:+.4f}, "
                        f"height_ok={int(dump_open_height_ok)}, "
                        f"entry_effort_score={dump_entry_effort_score:.3f}, "
                        f"dump_effort_reference="
                        f"{dump_effort_reference.round(3).tolist()}"
                    )

            # ------------------------------------------------
            # Dumping effort change and repeatable weighted extra opening.
            # ------------------------------------------------
            if stage in {"dumping", "extra_open"}:
                if dump_effort_reference is None:
                    dump_effort_reference = effort_norm_ema.copy()

                unload_effort_score = float(
                    np.max(
                        np.abs(
                            effort_norm_ema[0:4]
                            - dump_effort_reference[0:4]
                        )
                    )
                )

                # Weighted consecutive evidence that the model has stopped
                # opening or has started closing again after an actual open phase.
                # This score is updated only while the model is in normal dumping;
                # the controller's own extra-open motion must not count as model
                # evidence.
                if stage == "dumping":
                    if clear_model_open:
                        bucket_open_seen = True
                        bucket_open_stop_count = 0
                    elif bucket_open_seen and model_bucket_stopped:
                        # Near zero: weak evidence that opening has stalled.
                        bucket_open_stop_count += 1
                    elif (
                        bucket_open_seen
                        and model_bucket
                        < -abs(float(args.bucket_stop_threshold))
                    ):
                        # Clear closing command: stronger evidence that the model
                        # is undoing the unload motion.
                        bucket_open_stop_count += 2
                    elif bucket_open_seen:
                        # A positive, non-near-zero bucket command means the model
                        # is still opening. "Consecutive" evidence is broken.
                        bucket_open_stop_count = 0
                    else:
                        bucket_open_stop_count = 0

                    # V10: a clear closing request is useful evidence for
                    # extra-open, but it must not physically re-curl the bucket
                    # during the unload phase. Near-zero commands remain zero and
                    # positive commands are preserved.
                    if (
                        model_bucket < -abs(float(args.bucket_stop_threshold))
                        and vel.size > 3
                        and vel[3] < 0.0
                    ):
                        vel[3] = 0.0

                # Require an actual open phase and subsequent stall/closing
                # evidence before effort change confirms unloading.
                stable_after_open = (
                    bucket_open_stop_count > 0
                    or bucket_at_open_limit
                    or extra_open_active
                )
                unload_now = (
                    bucket_open_seen
                    and stable_after_open
                    and unload_effort_score
                    >= float(args.unload_effort_change_score)
                )
                unload_effort_count = (
                    unload_effort_count + 1
                    if unload_now
                    else max(0, unload_effort_count - 1)
                )
                unload_effort_confirmed = (
                    unload_effort_count >= max(1, args.unload_confirm)
                )

                if bucket_at_open_limit:
                    dump_complete_reason = "bucket_open_limit"
                elif unload_effort_confirmed and not extra_open_active:
                    dump_complete_reason = "unload_effort_change"
                elif (
                    stage == "dumping"
                    and not extra_open_active
                    and bucket_open_seen
                    and bucket_open_stop_count
                    >= max(1, args.bucket_stop_confirm)
                    and not unload_effort_confirmed
                ):
                    extra_open_active = True
                    extra_open_used = True
                    extra_open_count += 1
                    extra_open_start_q = bucket_q if np.isfinite(bucket_q) else None
                    extra_open_steps = 0
                    extra_open_progress = 0.0
                    bucket_open_stop_count = 0
                    stage = "extra_open"
                    print(
                        "[EXTRA OPEN] weighted consecutive stall/closing score "
                        f"reached {int(args.bucket_stop_confirm)}; starting "
                        f"repeatable correction #{extra_open_count}"
                    )

            if extra_open_active:
                stage = "extra_open"
                if extra_open_start_q is not None and np.isfinite(bucket_q):
                    extra_open_progress = max(
                        0.0, float(bucket_q) - float(extra_open_start_q)
                    )

                open_limit_finished = bucket_at_open_limit
                open_angle_finished = (
                    extra_open_progress >= max(0.0, float(args.extra_open_angle))
                )
                open_timeout_finished = (
                    extra_open_steps >= max(1, int(args.extra_open_max_steps))
                )
                open_effort_finished = unload_effort_confirmed

                if (
                    open_limit_finished
                    or open_angle_finished
                    or open_timeout_finished
                    or open_effort_finished
                ):
                    extra_open_active = False
                    if vel.size > 3:
                        vel[3] = 0.0

                    if open_limit_finished:
                        # The physical open limit is a definitive dump completion.
                        dump_complete_reason = "extra_open_joint_limit"
                    elif open_effort_finished:
                        # Effort confirms that the payload has been released.
                        dump_complete_reason = "extra_open_effort_change"
                    else:
                        # Reaching the correction angle or step budget no longer
                        # ends dumping. Return to normal dumping and allow another
                        # weighted trigger later. This makes extra-open repeatable.
                        finished_reason = (
                            "extra_open_angle"
                            if open_angle_finished
                            else "extra_open_max_steps"
                        )
                        stage = "dumping"
                        bucket_open_stop_count = 0
                        extra_open_start_q = None
                        extra_open_steps = 0
                        print(
                            "[EXTRA OPEN] correction finished without unload "
                            f"confirmation: reason={finished_reason}, "
                            f"count={extra_open_count}; returning to dumping "
                            "and re-arming the weighted trigger"
                        )
                else:
                    if vel.size > 3:
                        vel[3] = abs(float(args.extra_open_speed))
                    extra_open_steps += 1

            # Physical open limit always wins.
            if bucket_at_open_limit and vel.size > 3 and vel[3] > 0.0:
                vel[3] = 0.0

            # ------------------------------------------------
            # Hard maximum height while carrying or unloading.
            #
            # This is the only case where V10 forces a descent without a model
            # descent request: it is a ceiling-recovery safety correction.
            # Below the ceiling, no descent is injected here.
            # ------------------------------------------------
            loaded_height_limited = False
            if stage in {
                "loaded_transport",
                "dump_descent",
                "dump_ready",
                "dumping",
                "extra_open",
            }:
                max_loaded_z = float(args.loaded_max_tip_z)
                tol = abs(float(args.target_z_tolerance))
                if raw_tip_z > max_loaded_z + tol:
                    loaded_height_limited = True
                    if vel.size > 1:
                        vel[1] = -abs(
                            float(args.loaded_height_recovery_speed)
                        )
                    # Hold arm so the height correction is predictable.
                    if vel.size > 2:
                        vel[2] = 0.0
                    # Do not continue opening above the transport/unload
                    # ceiling. The stage remains dumping, but physical opening
                    # resumes only after the tip has returned below 5.5 m.
                    if vel.size > 3 and vel[3] > 0.0:
                        vel[3] = 0.0
                elif raw_tip_z >= max_loaded_z - tol:
                    if vel.size > 1 and vel[1] > 0.0:
                        loaded_height_limited = True
                        vel[1] = 0.0

            # ------------------------------------------------
            # Hard dump lower-height safety in every loaded dump stage.
            # It does not prevent descent after the target; it only prevents
            # crossing the explicit safety floor.
            # ------------------------------------------------
            if (
                stage in {"dump_descent", "dump_ready", "dumping", "extra_open"}
                and args.dump_min_safe_tip_z is not None
            ):
                safe_z = float(args.dump_min_safe_tip_z)
                tol = abs(float(args.target_z_tolerance))
                if raw_tip_z < safe_z - tol:
                    if vel.size > 1:
                        vel[1] = abs(float(args.dump_safety_recovery_speed))
                    if vel.size > 2:
                        vel[2] = 0.0
                elif raw_tip_z <= safe_z + tol:
                    if vel.size > 1 and vel[1] < 0.0:
                        vel[1] = 0.0
                    # Arm motion can also lower the real bucket tip, so hold
                    # it at the hard safety boundary.
                    if vel.size > 2:
                        vel[2] = 0.0

            # Dump complete -> post-unload decision.
            # The next stage is selected by two independent checks:
            #   1) already close to the 5.5 m ceiling -> return directly;
            #   2) otherwise, consecutive model boom-up requests -> forced lift.
            if dump_complete_reason and stage in {"dumping", "extra_open"}:
                stage = "post_unload_decision"
                has_loaded = False
                post_unload_rise_count = 0
                post_unload_decision_steps = 0
                post_unload_near_ceiling = False
                post_dump_lift_steps = 0
                post_dump_lift_timeout_warned = False
                if vel.size > 3:
                    vel[3] = 0.0
                reset_policy_queue(policy, f"dump complete: {dump_complete_reason}")
                reset_height_trend(raw_tip_z)
                print(
                    "[STAGE] dumping -> post_unload_decision: "
                    f"reason={dump_complete_reason}, "
                    f"tip_z={raw_tip_z:.5f}, "
                    f"bucket_q={bucket_q:.5f}, "
                    f"unload_effort_score={unload_effort_score:.3f}"
                )
                dump_complete_reason = ""

            # ------------------------------------------------
            # Post-unload dual decision.
            # ------------------------------------------------
            if stage == "post_unload_decision":
                post_unload_decision_steps += 1
                direct_return_z = (
                    float(args.loaded_max_tip_z)
                    - max(0.0, float(args.post_unload_near_ceiling_margin))
                )
                post_unload_near_ceiling = (
                    raw_tip_z
                    >= direct_return_z - abs(float(args.target_z_tolerance))
                )

                clear_model_raise = (
                    model_boom
                    >= abs(float(args.post_unload_rise_threshold))
                )
                if clear_model_raise:
                    post_unload_rise_count += 1
                else:
                    post_unload_rise_count = 0

                # Hold heading and bucket while deciding. The model may raise,
                # but it may not lower or re-curl the unloaded bucket.
                if vel.size > 0:
                    vel[0] = 0.0
                if vel.size > 1 and vel[1] < 0.0:
                    vel[1] = 0.0
                if vel.size > 3:
                    vel[3] = 0.0

                if post_unload_near_ceiling:
                    stage = "return_to_pile"
                    return_to_pile_steps = 0
                    reset_policy_queue(policy, "post-unload direct return")
                    print(
                        "[STAGE] post_unload_decision -> return_to_pile: "
                        f"reason=near_height_ceiling, tip_z={raw_tip_z:.5f}, "
                        f"direct_return_z={direct_return_z:.5f}"
                    )
                elif (
                    post_unload_rise_count
                    >= max(1, int(args.post_unload_rise_confirm))
                ):
                    stage = "lift_after_dump"
                    post_dump_lift_steps = 0
                    post_dump_lift_timeout_warned = False
                    reset_policy_queue(policy, "post-unload model raise confirmed")
                    print(
                        "[STAGE] post_unload_decision -> lift_after_dump: "
                        f"reason=model_raise, count={post_unload_rise_count}, "
                        f"model_boom={model_boom:+.4f}, tip_z={raw_tip_z:.5f}"
                    )

            # ------------------------------------------------
            # Mandatory lift after dumping when the model requests a raise.
            # ------------------------------------------------
            if stage == "lift_after_dump":
                post_dump_lift_steps += 1
                lift_reached = (
                    raw_tip_z
                    >= float(args.post_dump_min_tip_z)
                    - abs(float(args.target_z_tolerance))
                )
                if (
                    not lift_reached
                    and not post_dump_lift_timeout_warned
                    and post_dump_lift_steps
                    >= max(1, int(args.post_dump_lift_max_steps))
                ):
                    post_dump_lift_timeout_warned = True
                    print(
                        "[WARN] post-dump minimum height still not reached "
                        "after the configured warning step count; continuing lift"
                    )

                if lift_reached:
                    if vel.size > 1:
                        vel[1] = 0.0
                    stage = "return_to_pile"
                    return_to_pile_steps = 0
                    reset_policy_queue(policy, "post-dump lift finished")
                    reset_height_trend(raw_tip_z)
                    print(
                        "[STAGE] lift_after_dump -> return_to_pile: "
                        f"reason=height, tip_z={raw_tip_z:.5f}, "
                        f"target={float(args.post_dump_min_tip_z):.5f}"
                    )
                else:
                    if vel.size > 1:
                        vel[1] = abs(float(args.post_dump_lift_speed))
                    # Do not continue opening during forced lift.
                    if vel.size > 3 and vel[3] > 0.0:
                        vel[3] = 0.0

            # ------------------------------------------------
            # Reverse turn back toward the initial sand-pile heading.
            # ------------------------------------------------
            if stage == "return_to_pile":
                return_to_pile_steps += 1
                current_swing_q = (
                    float(raw_q[0]) if raw_q.size > 0 else float("nan")
                )
                if home_swing_q is None or not np.isfinite(current_swing_q):
                    return_swing_error = 0.0
                    if vel.size > 0:
                        vel[0] = 0.0
                else:
                    # Shortest signed angular error in [-pi, pi].
                    return_swing_error = float(
                        (home_swing_q - current_swing_q + np.pi)
                        % (2.0 * np.pi)
                        - np.pi
                    )

                # Hold boom/arm/bucket while the upper structure reverses.
                vel[:] = 0.0
                return_reached = (
                    home_swing_q is not None
                    and np.isfinite(current_swing_q)
                    and abs(return_swing_error)
                    <= abs(float(args.return_swing_tolerance))
                )
                if return_reached:
                    stage = "post_dump_complete"
                    reset_policy_queue(policy, "return-to-pile heading reached")
                    print(
                        "[STAGE] return_to_pile -> post_dump_complete: "
                        f"current_swing_q={current_swing_q:.5f}, "
                        f"home_swing_q={home_swing_q:.5f}, "
                        f"error={return_swing_error:+.5f}, "
                        f"steps={return_to_pile_steps}"
                    )
                elif vel.size > 0 and home_swing_q is not None:
                    vel[0] = (
                        abs(float(args.return_swing_speed))
                        if return_swing_error > 0.0
                        else -abs(float(args.return_swing_speed))
                    )

            # ------------------------------------------------
            # Stage-specific swing suppression.
            #
            # Dig/loading/forced-lift stages retain at most the manipulation
            # gain (default 15%). Dump approach/readiness/opening stages retain
            # at most the dedicated dump gain (default 35%).
            # ------------------------------------------------
            effective_swing_gain = float(swing_gain)
            if (
                stage in {
                    "dig_target_descent",
                    "loading",
                    "extra_curl",
                    "lift_after_load",
                    "lift_after_dump",
                    "post_unload_decision",
                }
                or (stage == "loaded_transport" and loaded_height_limited)
            ):
                effective_swing_gain = stage_suppress_swing(
                    vel,
                    model_action_np,
                    swing_gain,
                    args.manipulation_swing_gain,
                )
            elif stage in {
                "dump_descent",
                "dump_ready",
                "dumping",
                "extra_open",
            }:
                effective_swing_gain = stage_suppress_swing(
                    vel,
                    model_action_np,
                    swing_gain,
                    args.dump_swing_gain,
                )

            cmd = {
                "joint_velocities": vel.tolist(),
            }

            print(
                f"[STEP {step:04d}] "
                f"stage={stage} "
                f"action={model_action_np.round(4).tolist()} "
                f"vel={vel.round(4).tolist()} "
                f"tip_z={raw_tip_z:.5f} "
                f"height_delta_{args.height_trend_window}={height_trend_delta:+.5f} "
                f"real_descent={int(real_descent_now)} "
                f"dig_descent_seen={int(dig_descent_seen)} "
                f"dig_target={dig_target_value if dig_target_value is not None else float('nan'):.5f} "
                f"dig_target_active={int(dig_target_active)} "
                f"dig_target_reached={int(dig_target_reached)} "
                f"q={raw_q.round(5).tolist()} "
                f"bucket_q={bucket_q:.5f} "
                f"curl_limit={int(bucket_at_curl_limit)} "
                f"open_limit={int(bucket_at_open_limit)} "
                f"model_boom={model_boom:+.4f} "
                f"model_bucket={model_bucket:+.4f} "
                f"swing_gain={effective_swing_gain:.3f} "
                f"work={work_activity:.3f} "
                f"boom_gain={boom_gain:.3f} "
                f"bucket_gain={bucket_gain:.3f} "
                f"effort_norm={effort_norm_ema.round(3).tolist()} "
                f"load_effort_score={load_effort_score:.3f} "
                f"load_contact={int(load_contact_confirmed)} "
                f"load_sufficient={int(load_sufficient)} "
                f"extra_curl={int(extra_curl_active)} "
                f"extra_curl_used={int(extra_curl_used)} "
                f"extra_curl_progress={extra_curl_progress:.4f} "
                f"has_loaded={int(has_loaded)} "
                f"loaded_max_tip_z={float(args.loaded_max_tip_z):.3f} "
                f"loaded_height_limited={int(loaded_height_limited)} "
                f"dump_model_descent_count={dump_model_descent_count} "
                f"dump_real_descent_count={descent_confirm_count} "
                f"dump_strong_open_count={dump_open_count} "
                f"dump_open_window={int(args.dump_open_window)} "
                f"strong_open={int(strong_model_open)} "
                f"dump_open_height_ok={int(dump_open_height_ok)} "
                f"dump_entry_effort_score={dump_entry_effort_score:.3f} "
                f"dump_entry_reason={dump_entry_reason or 'none'} "
                f"dump_target_reached={int(dump_target_reached)} "
                f"unload_effort_score={unload_effort_score:.3f} "
                f"unload_confirmed={int(unload_effort_confirmed)} "
                f"post_unload_rise_count={post_unload_rise_count} "
                f"post_unload_near_ceiling={int(post_unload_near_ceiling)} "
                f"home_swing_q={home_swing_q if home_swing_q is not None else float('nan'):.5f} "
                f"return_swing_error={return_swing_error:+.5f} "
                f"extra_open_trigger_score={bucket_open_stop_count} "
                f"extra_open_trigger_required={int(args.bucket_stop_confirm)} "
                f"extra_open={int(extra_open_active)} "
                f"extra_open_used={int(extra_open_used)} "
                f"extra_open_count={extra_open_count} "
                f"extra_open_progress={extra_open_progress:.4f} "
                f"load_raw_debug={raw_load_particles:.1f} "
                f"effort_key={effort_key} "
                f"rgb_mean={float(rgb.mean()):.2f} "
                f"cameras={sorted(camera_rgbs.keys())} "
                f"infer_ms={inference_ms:.1f} "
                f"server_ms={float(reply.get('bridge_timing', {}).get('total_server_ms', 0.0)):.1f} "
                f"loop_ms={(time.perf_counter() - policy_loop_start) * 1000.0:.1f}"
            )

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted, sending zero velocity")

    finally:
        print("[TIMING SUMMARY] p50/p95/max milliseconds")
        for timing_name, timing_values in timing_samples.items():
            if not timing_values:
                continue
            values = np.asarray(timing_values, dtype=np.float64)
            print(
                f"[TIMING SUMMARY] {timing_name}: "
                f"p50={np.percentile(values, 50):.1f} "
                f"p95={np.percentile(values, 95):.1f} "
                f"max={np.max(values):.1f} n={len(values)}"
            )
        deployment_wall_seconds = max(1.0e-9, time.perf_counter() - deployment_wall_start)
        print(
            "[TIMING SUMMARY] "
            f"simulated_seconds={simulated_seconds_total:.3f} "
            f"wall_seconds={deployment_wall_seconds:.3f} "
            f"real_time_factor={simulated_seconds_total / deployment_wall_seconds:.4f}"
        )
        try:
            num_joints = (
                len(raw_q)
                if "raw_q" in locals() and len(raw_q) > 0
                else 4
            )
            zero = {
                "joint_velocities": [0.0] * num_joints,
            }
            write_json(sock, zero)
        except Exception:
            pass

        if video_recorder is not None:
            try:
                video_recorder.close()
            except Exception as exc:
                print(f"[WARN] failed to finalize camera videos: {exc}")

        sock.close()
        print("[INFO] closed")

        if dump_dir:
            import tarfile
            tarball = os.path.join(
                os.path.dirname(dump_dir) or ".",
                "model_input_dump.tar.gz",
            )
            with tarfile.open(tarball, "w:gz") as tar:
                tar.add(dump_dir, arcname=os.path.basename(dump_dir))
            print(f"[DUMP] tarball: {tarball}")


if __name__ == "__main__":
    main()
