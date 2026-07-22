#!/usr/bin/env python3
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
from excavator_common import vla_observation_contract


DEPLOYMENT_ACTION_LIMITS_RAD_S = np.asarray(
    vla_observation_contract.ACTION_LIMITS_RAD_S_4D,
    dtype=np.float32,
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


def require_camera_rgbs(camera_rgbs):
    required = ("0", "1", "2")
    missing = [name for name in required if name not in camera_rgbs]
    if missing:
        raise RuntimeError(
            f"Bridge reply is missing required cameras {missing}; "
            f"available={sorted(camera_rgbs)}"
        )
    validated = {}
    for name in required:
        rgb = np.asarray(camera_rgbs[name], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] < 2 or rgb.shape[1] < 2:
            raise RuntimeError(f"Camera {name} has invalid RGB shape: {rgb.shape}")
        if not np.any(rgb):
            raise RuntimeError(f"Camera {name} returned an all-zero RGB frame")
        validated[name] = rgb
    return validated


def make_state(reply, device, expected_state_dim):
    if int(expected_state_dim) == 28:
        if not bool(reply.get("observation_32d_ready", False)):
            raise RuntimeError(
                "Bridge cannot provide the checkpoint's 28D state: "
                f"{reply.get('observation_32d_error', 'unknown error')}"
            )
        state = np.asarray(
            reply.get("observation_state_28d"),
            dtype=np.float32,
        ).reshape(-1)
    else:
        state = np.asarray(reply.get("observation_state"), dtype=np.float32).reshape(-1)
        if int(expected_state_dim) == 18 and state.shape == (14,):
            effort = np.asarray(
                reply.get("observation_effort"),
                dtype=np.float32,
            ).reshape(-1)
            if effort.shape != (4,):
                raise RuntimeError(
                    "Legacy 18D checkpoint requires bridge 14D state + 4D effort; "
                    f"got state={state.shape}, effort={effort.shape}"
                )
            state = np.concatenate([state, effort], axis=0)
    if state.shape != (int(expected_state_dim),):
        raise RuntimeError(
            f"Expected observation.state shape ({expected_state_dim},), got "
            f"{state.shape}: {state.tolist()}"
        )
    if not np.all(np.isfinite(state)):
        raise RuntimeError(f"observation.state contains NaN or Inf: {state.tolist()}")
    return torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32)


def make_effort(reply, device, expected_effort_dim):
    if int(expected_effort_dim) <= 0:
        return None
    effort = np.asarray(reply.get("observation_effort"), dtype=np.float32).reshape(-1)
    if effort.shape != (int(expected_effort_dim),):
        raise RuntimeError(
            f"Expected observation.effort shape ({expected_effort_dim},), got "
            f"{effort.shape}: {effort.tolist()}"
        )
    if not np.all(np.isfinite(effort)):
        raise RuntimeError(f"observation.effort contains NaN or Inf: {effort.tolist()}")
    return torch.from_numpy(effort).unsqueeze(0).to(device=device, dtype=torch.float32)


def _feature_shape_from_object(value, feature_name):
    if isinstance(value, dict):
        if feature_name in value:
            spec = value[feature_name]
            if isinstance(spec, dict):
                shape = spec.get("shape")
                if isinstance(shape, (list, tuple)) and shape:
                    return int(shape[0])
        for child in value.values():
            result = _feature_shape_from_object(child, feature_name)
            if result is not None:
                return result
    elif isinstance(value, list):
        for child in value:
            result = _feature_shape_from_object(child, feature_name)
            if result is not None:
                return result
    return None


def checkpoint_feature_dims(checkpoint_dir, state_dim_override=0, effort_dim_override=-1):
    state_dim = int(state_dim_override or 0)
    effort_dim = int(effort_dim_override)
    for filename in ("config.json", "train_config.json", "policy_preprocessor.json"):
        path = Path(checkpoint_dir) / filename
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state_dim <= 0:
            found = _feature_shape_from_object(data, "observation.state")
            if found is not None:
                state_dim = int(found)
        if effort_dim < 0:
            found = _feature_shape_from_object(data, "observation.effort")
            if found is not None:
                effort_dim = int(found)

    if state_dim not in (14, 18, 28):
        raise RuntimeError(
            "Cannot establish checkpoint observation.state dimension. "
            "Use --state-dim 14, 18, or 28 and verify it against training metadata."
        )
    if effort_dim < 0:
        effort_dim = 0
    if effort_dim not in (0, 4):
        raise RuntimeError(f"Unsupported observation.effort dimension: {effort_dim}")
    return state_dim, effort_dim


def validate_bridge_contract(reply, state_dim, effort_dim):
    contract = reply.get("observation_contract")
    if not isinstance(contract, dict):
        raise RuntimeError("Bridge reply is missing observation_contract metadata")
    if int(state_dim) != 28:
        return
    state_spec = contract.get("observation.state")
    effort_spec = contract.get("observation.effort")
    action_spec = contract.get("action")
    expected_state_names = list(vla_observation_contract.STATE_NAMES_28D)
    expected_effort_names = list(vla_observation_contract.EFFORT_NAMES_4D)
    expected_action_names = list(vla_observation_contract.ACTION_NAMES_4D)
    if not isinstance(state_spec, dict) or state_spec.get("names") != expected_state_names:
        raise RuntimeError("Bridge observation.state names/order do not match the 28D contract")
    if int(effort_dim) > 0 and (
        not isinstance(effort_spec, dict)
        or effort_spec.get("names") != expected_effort_names
    ):
        raise RuntimeError("Bridge observation.effort names/order do not match the 4D contract")
    if not isinstance(action_spec, dict) or action_spec.get("names") != expected_action_names:
        raise RuntimeError("Bridge action names/order do not match the canonical 4D contract")
    if str(action_spec.get("unit", "")).strip().lower() != "rad/s":
        raise RuntimeError(f"Bridge action unit must be rad/s, got {action_spec.get('unit')!r}")
    if int(effort_dim) not in (0, 4):
        raise RuntimeError(f"Bridge effort dimension must be 0 or 4, got {effort_dim}")


def load_observation_context(path):
    if not str(path or "").strip():
        return None
    context_path = Path(path).expanduser()
    data = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Observation context must be a JSON object: {context_path}")
    return data


def action_to_joint_velocities(action, num_joints, max_vel=0.0):
    """
    Model action order:
      action[0] = swing
      action[1] = boom
      action[2] = arm
      action[3] = bucket

    Bridge joint_velocities order is the same canonical order.
    """
    a = action.detach().float().cpu().numpy()

    if a.ndim == 2:
        a = a[0]

    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size < 4:
        raise RuntimeError(f"Model action must contain 4 joint velocities, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise RuntimeError(f"Model action contains NaN or Inf: {a.tolist()}")
    limits = DEPLOYMENT_ACTION_LIMITS_RAD_S.copy()
    if float(max_vel or 0.0) > 0.0:
        limits = np.minimum(limits, float(max_vel))
    a = np.clip(a[:4], -limits, limits)

    vel = np.zeros(num_joints, dtype=np.float32)

    n = min(num_joints, 4)
    vel[:n] = a[:n]

    return vel

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)

    parser.add_argument(
        "--ckpt",
        default=os.environ.get("SMOLVLA_CKPT", ""),
    )
    parser.add_argument(
        "--vlm",
        default=os.environ.get("SMOLVLA_VLM", ""),
    )

    parser.add_argument(
        "--task",
        default="Dig soil from the marked area and dump it into the target container.",
    )

    parser.add_argument(
        "--max-vel",
        type=float,
        default=0.0,
        help=(
            "Optional additional scalar velocity cap in rad/s. By default the "
            "model's physical rad/s output is preserved within the dataset limits."
        ),
    )
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--state-dim", type=int, default=0)
    parser.add_argument("--effort-dim", type=int, default=-1)
    parser.add_argument(
        "--observation-context",
        default="",
        help="JSON file updated by the deployment task/phase supervisor.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    if not args.ckpt:
        raise SystemExit("--ckpt is required, or set SMOLVLA_CKPT.")
    if not args.vlm:
        raise SystemExit("--vlm is required, or set SMOLVLA_VLM.")

    print("[INFO] checkpoint:", args.ckpt)
    print("[INFO] local VLM:", args.vlm)

    patch_checkpoint_paths(args.ckpt, args.vlm)
    state_dim, effort_dim = checkpoint_feature_dims(
        args.ckpt,
        state_dim_override=args.state_dim,
        effort_dim_override=args.effort_dim,
    )
    print(
        f"[INFO] checkpoint observation contract: state_dim={state_dim} "
        f"effort_dim={effort_dim}"
    )

    print("[INFO] loading SmolVLA policy...")
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(device)

    print("[INFO] loading language tokenizer...")
    language_tokens, language_mask = load_language_tokens(args.task, args.vlm, device)

    if hasattr(policy, "reset"):
        policy.reset()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    cmd = {
        "ticks": args.ticks,
    }
    initial_context = load_observation_context(args.observation_context)
    if initial_context is not None:
        cmd["reset_observation_context"] = True
        cmd["observation_context"] = initial_context

    try:
        for step in range(args.steps):
            write_json(sock, cmd)
            reply = read_json(sock)
            if step == 0:
                validate_bridge_contract(reply, state_dim, effort_dim)

            rgb = decode_rgb(reply)
            camera_rgbs = require_camera_rgbs(
                decode_camera_images(reply, np_module=np)
            )
            rgb0 = camera_rgbs["0"]
            rgb1 = camera_rgbs["1"]
            rgb2 = camera_rgbs["2"]
            image0 = rgb_to_tensor(rgb0, device)
            image1 = rgb_to_tensor(rgb1, device)
            image2 = rgb_to_tensor(rgb2, device)
            state = make_state(reply, device, state_dim)
            effort = make_effort(reply, device, effort_dim)
            bridge_task = str(reply.get("task_text", "") or "").strip()
            if bridge_task and bridge_task != args.task:
                args.task = bridge_task
                language_tokens, language_mask = load_language_tokens(
                    args.task,
                    args.vlm,
                    device,
                )

            batch = {
                # Match training dataset keys
                "observation.state": state,
                "observation.images.0": image0,
                "observation.images.1": image1,
                "observation.images.2": image2,

                # Raw task string used by LeRobot preprocessor-style pipelines
                "task": [args.task],

                # Also provide tokenized language for direct policy inference compatibility
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }
            if effort is not None:
                batch["observation.effort"] = effort

            with torch.no_grad():
                action = policy.select_action(batch)

            q = reply.get("joint_positions", [])
            num_joints = len(q) if len(q) > 0 else 4

            vel = action_to_joint_velocities(
                action,
                num_joints=num_joints,
                max_vel=args.max_vel,
            )

            cmd = {
                "joint_velocities": vel.tolist(),
                "ticks": args.ticks,
            }
            context = load_observation_context(args.observation_context)
            if context is not None:
                cmd["observation_context"] = context

            print(
                f"[STEP {step:04d}] "
                f"action={action.detach().cpu().numpy().round(4).tolist()} "
                f"vel={vel.round(4).tolist()} "
                f"state={state.detach().cpu().numpy().round(3).tolist()} "
                f"rgb_mean={float(rgb.mean()):.2f} "
                f"cameras={sorted(camera_rgbs.keys())}"
            )

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted, sending zero velocity")

    finally:
        try:
            q = reply.get("joint_positions", []) if "reply" in locals() else []
            num_joints = len(q) if len(q) > 0 else 4
            zero = {
                "joint_velocities": [0.0] * num_joints,
                "ticks": 2,
            }
            write_json(sock, zero)
        except Exception:
            pass

        sock.close()
        print("[INFO] closed")


if __name__ == "__main__":
    main()
