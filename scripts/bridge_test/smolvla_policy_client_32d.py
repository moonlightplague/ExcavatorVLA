#!/usr/bin/env python3
"""Strict 28D state + 4D effort SmolVLA deployment client."""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (  # noqa: E402
    decode_camera_images,
    read_json,
    write_json,
)
from excavator_common.deployment_contract import (  # noqa: E402
    OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
    build_client_contract,
    load_training_fps,
    sha256_files,
    validate_training_fps,
)
from excavator_common import vla_observation_contract  # noqa: E402


def feature_shape(feature):
    if isinstance(feature, dict):
        value = feature.get("shape")
    else:
        value = getattr(feature, "shape", None)
    return None if value is None else tuple(int(item) for item in value)


def validate_policy_contract(policy):
    config = getattr(policy, "config", None)
    inputs = getattr(config, "input_features", None)
    outputs = getattr(config, "output_features", None)
    if not isinstance(inputs, dict) or not isinstance(outputs, dict):
        raise RuntimeError(
            "Checkpoint config does not expose input_features/output_features"
        )
    expected = {
        "observation.state": (28,),
        "observation.effort": (4,),
        "observation.images.0": (3, 256, 256),
        "observation.images.1": (3, 256, 256),
        "observation.images.2": (3, 256, 256),
    }
    for key, shape in expected.items():
        actual = feature_shape(inputs.get(key))
        if actual != shape:
            raise RuntimeError(
                f"Checkpoint feature {key!r} must have shape {shape}, got {actual}"
            )
    action_shape = feature_shape(outputs.get("action"))
    if action_shape != (4,):
        raise RuntimeError(
            f"Checkpoint action must have shape (4,), got {action_shape}"
        )


def load_stats(checkpoint_dir):
    values = {}
    paths = [
        Path(checkpoint_dir)
        / "policy_preprocessor_step_5_normalizer_processor.safetensors",
        Path(checkpoint_dir)
        / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ]
    for path in paths:
        if not path.is_file():
            continue
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                if key in {
                    "observation.state.mean",
                    "observation.state.std",
                    "observation.effort.mean",
                    "observation.effort.std",
                    "action.mean",
                    "action.std",
                }:
                    values[key] = handle.get_tensor(key).numpy().astype(np.float32)
    required = {
        "observation.state.mean": (28,),
        "observation.state.std": (28,),
        "observation.effort.mean": (4,),
        "observation.effort.std": (4,),
        "action.mean": (4,),
        "action.std": (4,),
    }
    for key, shape in required.items():
        value = values.get(key)
        if value is None or tuple(value.shape) != shape:
            raise RuntimeError(
                f"Checkpoint normalization {key!r} must have shape {shape}, "
                f"got {None if value is None else value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Checkpoint normalization {key!r} is non-finite")
    values["observation.state.std"] = np.maximum(
        values["observation.state.std"],
        1.0e-6,
    )
    values["observation.effort.std"] = np.maximum(
        values["observation.effort.std"],
        1.0e-6,
    )
    values["action.std"] = np.maximum(values["action.std"], 1.0e-6)
    return values, paths


def load_context(path):
    context_path = Path(path).expanduser()
    value = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Observation context must be a JSON object: {context_path}")
    return value


def require_vector(reply, key, size):
    value = np.asarray(reply.get(key), dtype=np.float32).reshape(-1)
    if value.shape != (int(size),) or not np.all(np.isfinite(value)):
        raise RuntimeError(
            f"Bridge {key!r} must contain {size} finite values, got "
            f"shape={value.shape}, value={value.tolist()}"
        )
    return value


def require_cameras(reply):
    cameras = decode_camera_images(reply, np_module=np)
    required = ("0", "1", "2")
    missing = [name for name in required if name not in cameras]
    if missing:
        raise RuntimeError(
            f"Bridge is missing cameras {missing}; available={sorted(cameras)}"
        )
    result = {}
    for name in required:
        rgb = np.asarray(cameras[name], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Camera {name} has invalid RGB shape {rgb.shape}")
        if not np.any(rgb):
            raise RuntimeError(f"Camera {name} returned an all-zero frame")
        result[name] = rgb
    return result


def rgb_tensor(rgb, device):
    value = torch.from_numpy(np.ascontiguousarray(rgb)).to(
        device=device,
        dtype=torch.float32,
    )
    value = value.permute(2, 0, 1).unsqueeze(0) / 255.0
    if tuple(value.shape[-2:]) != (256, 256):
        value = F.interpolate(
            value,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
    return value


def language_tokens(task, processor, device):
    tokenizer = getattr(processor, "tokenizer", processor)
    value = tokenizer(
        task,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return (
        value["input_ids"].to(device),
        value["attention_mask"].bool().to(device),
    )


def tensor(value, device):
    return torch.from_numpy(np.asarray(value, dtype=np.float32)).unsqueeze(0).to(
        device=device,
        dtype=torch.float32,
    )


def patch_checkpoint_vlm_path(checkpoint_dir, vlm_dir):
    old = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            return value.replace(old, str(Path(vlm_dir).resolve()))
        return value

    for filename in ("config.json", "policy_preprocessor.json", "train_config.json"):
        path = Path(checkpoint_dir) / filename
        if not path.is_file():
            continue
        original = json.loads(path.read_text(encoding="utf-8"))
        updated = replace(original)
        if updated != original:
            path.write_text(json.dumps(updated, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ckpt", default=os.environ.get("SMOLVLA_CKPT", ""))
    parser.add_argument("--vlm", default=os.environ.get("SMOLVLA_VLM", ""))
    parser.add_argument("--dataset-meta", default="")
    parser.add_argument("--training-fps", type=float, default=0.0)
    parser.add_argument("--observation-context", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--max-abs-velocity",
        type=float,
        default=0.0,
        help="Optional symmetric rad/s clip; 0 preserves checkpoint output.",
    )
    args = parser.parse_args()
    if not args.ckpt or not args.vlm:
        raise SystemExit("--ckpt and --vlm are required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_checkpoint_vlm_path(args.ckpt, args.vlm)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(device)
    validate_policy_contract(policy)
    stats, normalization_paths = load_stats(args.ckpt)
    processor = AutoProcessor.from_pretrained(args.vlm, local_files_only=True)

    if args.dataset_meta:
        training_fps, metadata_path = load_training_fps(args.dataset_meta)
        print(f"[BRIDGE CONTRACT] training_fps={training_fps} source={metadata_path}")
    elif args.training_fps > 0.0:
        training_fps = validate_training_fps(args.training_fps)
    else:
        raise RuntimeError("--dataset-meta or --training-fps is required")

    context = load_context(args.observation_context)
    normalization_hash = sha256_files(normalization_paths)
    if not normalization_hash:
        raise RuntimeError("Could not hash checkpoint normalization assets")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    handshake = build_client_contract(
        training_fps,
        normalization_hash=normalization_hash,
        observation_schema=OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
        observation_context=context,
    )
    write_json(sock, handshake)
    reply = read_json(sock)
    if not reply.get("ok", False):
        raise RuntimeError(f"Bridge handshake rejected: {reply}")
    if reply.get("normalization_hash", "") != normalization_hash:
        raise RuntimeError("Bridge normalization identity mismatch")
    print("[BRIDGE CONTRACT] accepted:", reply)

    state_mean = tensor(stats["observation.state.mean"], device)
    state_std = tensor(stats["observation.state.std"], device)
    effort_mean = tensor(stats["observation.effort.mean"], device)
    effort_std = tensor(stats["observation.effort.std"], device)
    action_mean = tensor(stats["action.mean"], device)
    action_std = tensor(stats["action.std"], device)
    active_task = ""
    tokens = mask = None
    command = {
        "joint_velocities": [0.0, 0.0, 0.0, 0.0],
        "observation_context": context,
    }

    try:
        for step in range(max(1, int(args.steps))):
            write_json(sock, command)
            reply = read_json(sock)
            if reply.get("type") == "error":
                raise RuntimeError(f"Bridge rejected command: {reply}")
            if not bool(reply.get("observation_32d_ready", False)):
                raise RuntimeError(
                    "Bridge cannot build the live 28D+4D input: "
                    f"{reply.get('observation_32d_error', 'unknown error')}"
                )
            state = require_vector(reply, "observation_state_28d", 28)
            effort = require_vector(reply, "observation_effort", 4)
            cameras = require_cameras(reply)
            task = str(reply.get("task_text") or "").strip()
            if not task:
                raise RuntimeError("Bridge task_text is empty")
            if task != active_task:
                active_task = task
                tokens, mask = language_tokens(task, processor, device)
            state_input = (tensor(state, device) - state_mean) / state_std
            effort_input = (tensor(effort, device) - effort_mean) / effort_std
            batch = {
                "observation.state": state_input,
                "observation.effort": effort_input,
                "observation.images.0": rgb_tensor(cameras["0"], device),
                "observation.images.1": rgb_tensor(cameras["1"], device),
                "observation.images.2": rgb_tensor(cameras["2"], device),
                "task": [task],
                "observation.language.tokens": tokens,
                "observation.language.attention_mask": mask,
            }
            with torch.no_grad():
                normalized_action = policy.select_action(batch)
            action = normalized_action * action_std + action_mean
            velocity = action.detach().float().cpu().numpy().reshape(-1)[:4]
            if velocity.shape != (4,) or not np.all(np.isfinite(velocity)):
                raise RuntimeError(f"Policy action is invalid: {velocity.tolist()}")
            if args.max_abs_velocity > 0.0:
                velocity = np.clip(
                    velocity,
                    -float(args.max_abs_velocity),
                    float(args.max_abs_velocity),
                )
            context = load_context(args.observation_context)
            command = {
                "joint_velocities": velocity.tolist(),
                "observation_context": context,
            }
            if step == 0 or step % 25 == 0:
                print(
                    f"[STEP {step:05d}] phase={context.get('phase_name')} "
                    f"action_rad_s={velocity.round(5).tolist()} "
                    f"bucket_load={state[7]:.1f}"
                )
    finally:
        try:
            write_json(
                sock,
                {
                    "joint_velocities": [0.0, 0.0, 0.0, 0.0],
                    "observation_context": load_context(
                        args.observation_context
                    ),
                },
            )
        except Exception:
            pass
        sock.close()


if __name__ == "__main__":
    main()
