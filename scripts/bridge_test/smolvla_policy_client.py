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
    New LeRobot dataset uses 14D observation.state:

      base_x
      base_y
      base_yaw
      swing
      boom
      arm
      bucket
      bucket_load_estimate
      bucket_tip_x
      bucket_tip_y
      bucket_tip_z
      bucket_load_x
      bucket_load_y
      bucket_load_z

    Bridge may not provide exactly these fields yet, so this function tries:
      1. reply["observation_state"]
      2. reply["state"]
      3. reply["joint_positions"], padded to 14D
    """
    if "observation_state" in reply:
        arr = np.asarray(reply["observation_state"], dtype=np.float32)
    elif "state" in reply:
        arr = np.asarray(reply["state"], dtype=np.float32)
    else:
        arr = np.asarray(reply.get("joint_positions", []), dtype=np.float32)

    state = np.zeros(14, dtype=np.float32)
    n = min(14, len(arr))
    state[:n] = arr[:n]

    return torch.from_numpy(state).unsqueeze(0).to(device)


def action_to_joint_velocities(action, num_joints, max_vel=1.0):
    """
    Model action order:
      action[0] = swing
      action[1] = boom
      action[2] = arm
      action[3] = bucket

    Bridge joint_velocities order:
      vel[0] = bucket
      vel[1] = arm
      vel[2] = boom
      vel[3] = swing
    """
    a = action.detach().float().cpu().numpy()

    if a.ndim == 2:
        a = a[0]

    a = np.asarray(a, dtype=np.float32)
    a = np.tanh(a) * max_vel

    vel = np.zeros(num_joints, dtype=np.float32)

    if num_joints >= 4:
        vel[0] = a[3]  # bucket
        vel[1] = a[2]  # arm
        vel[2] = a[1]  # boom
        vel[3] = a[0]  # swing
    else:
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

    parser.add_argument("--max-vel", type=float, default=0.02)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=0.01)

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

    try:
        for step in range(args.steps):
            write_json(sock, cmd)
            reply = read_json(sock)

            rgb = decode_rgb(reply)
            camera_rgbs = decode_camera_images(reply, np_module=np)
            rgb0 = camera_rgbs.get("0", rgb)
            rgb1 = camera_rgbs.get("1", rgb)
            image0 = rgb_to_tensor(rgb0, device)
            image1 = rgb_to_tensor(rgb1, device)
            state = make_state(reply, device)

            batch = {
                # Match training dataset keys
                "observation.state": state,
                "observation.images.0": image0,
                "observation.images.1": image1,
                # "observation.images.2": image,

                # Raw task string used by LeRobot preprocessor-style pipelines
                "task": [args.task],

                # Also provide tokenized language for direct policy inference compatibility
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

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
