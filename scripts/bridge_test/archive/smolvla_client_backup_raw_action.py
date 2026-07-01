#!/usr/bin/env python3
import argparse
import os
import socket
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def find_project_root(start):
    current = os.path.abspath(start)
    while True:
        if (
            os.path.exists(os.path.join(current, "excavator_config.json"))
            and os.path.isdir(os.path.join(current, "excavator_common"))
        ):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        current = parent


PROJECT_ROOT = find_project_root(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (
    decode_rgb_payload as decode_rgb,
    read_json,
    write_json,
)


def rgb_to_tensor(rgb, device):
    """
    Input rgb: H x W x 3, uint8, range 0..255.
    Output: 1 x 3 x 256 x 256, float32, range 0..1.
    """
    x = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    x = x / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)  # 1 x 3 x H x W
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x


def make_state(reply, device):
    """
    SmolVLA expects state shape (1, 6).
    Here we use first 6 joint positions.
    If fewer than 6 joints, pad with zeros.
    """
    q = np.asarray(reply.get("joint_positions", []), dtype=np.float32)

    state = np.zeros(6, dtype=np.float32)
    n = min(6, len(q))
    state[:n] = q[:n]

    return torch.from_numpy(state).unsqueeze(0).to(device)


def action_to_joint_velocities(action, num_joints, max_vel=0.10):
    """
    Convert SmolVLA 6D action to excavator joint velocity command.

    First version:
    - use tanh for safety
    - scale by max_vel
    - fill first 6 robot joints
    """
    a = action.detach().float().cpu().numpy()

    if a.ndim == 2:
        a = a[0]

    a = np.asarray(a, dtype=np.float32)
    a = np.tanh(a) * max_vel

    vel = np.zeros(num_joints, dtype=np.float32)
    n = min(num_joints, len(a))
    vel[:n] = a[:n]

    return vel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ckpt", default=os.environ.get("SMOLVLA_CKPT", ""))
    parser.add_argument("--vlm", default=os.environ.get("SMOLVLA_VLM", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))
    parser.add_argument("--task", default="dig the soil and load it into the truck")
    parser.add_argument("--max-vel", type=float, default=0.10)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    if not args.ckpt:
        raise SystemExit("--ckpt is required, or set SMOLVLA_CKPT.")

    print("[INFO] loading SmolVLA from:", args.ckpt)
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(device)

    print("[INFO] loading tokenizer/processor:", args.vlm)
    processor = AutoProcessor.from_pretrained(args.vlm, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)

    lang = tokenizer(
        args.task,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    language_tokens = lang["input_ids"].to(device)
    language_mask = lang["attention_mask"].bool().to(device)

    if hasattr(policy, "reset"):
        policy.reset()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    # Initial command only steps the simulation and asks for observation.
    # Do not send joint_velocities before knowing num_joints.
    cmd = {
        "ticks": args.ticks,
    }

    try:
        for step in range(args.steps):
            write_json(sock, cmd)
            reply = read_json(sock)

            rgb = decode_rgb(reply)
            image = rgb_to_tensor(rgb, device)
            state = make_state(reply, device)

            batch = {
                "observation.state": state,
                "observation.images.camera1": image,
                "observation.images.camera2": image,
                "observation.images.camera3": image,
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

            with torch.no_grad():
                action = policy.select_action(batch)

            q = reply.get("joint_positions", [])
            num_joints = len(q) if len(q) > 0 else 6

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
                f"action={action.detach().cpu().numpy().round(3).tolist()} "
                f"vel={vel.round(3).tolist()} "
                f"rgb_mean={float(rgb.mean()):.2f}"
            )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted, sending zero velocity")

    finally:
        try:
            if "num_joints" in locals() and num_joints > 0:
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
