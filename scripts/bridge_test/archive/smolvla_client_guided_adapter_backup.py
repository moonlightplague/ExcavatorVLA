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
    Input:
        rgb: H x W x 3, uint8, range 0..255

    Output:
        1 x 3 x 256 x 256, float32, range 0..1
    """
    x = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    x = x / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
    return x


def make_state(reply, device):
    """
    SmolVLA checkpoint expects observation.state shape = (1, 6).

    Your excavator currently has 4 joints, so:
    - put 4 joint positions into first 4 dims
    - pad remaining dims with zeros
    """
    q = np.asarray(reply.get("joint_positions", []), dtype=np.float32)

    state = np.zeros(6, dtype=np.float32)
    n = min(6, len(q))
    state[:n] = q[:n]

    return torch.from_numpy(state).unsqueeze(0).to(device)


def extract_action_np(action):
    a = action.detach().float().cpu().numpy()

    if a.ndim == 2:
        a = a[0]

    return np.asarray(a, dtype=np.float32)


def raw_action_to_joint_velocities(action, num_joints, max_vel):
    """
    Pure zero-shot mapping:
        SmolVLA 6D action -> first num_joints joint velocities

    This usually causes only small random shaking for excavator,
    because pretrained SmolVLA was not trained on this excavator.
    """
    a = extract_action_np(action)
    a = np.tanh(a)

    vel = np.zeros(num_joints, dtype=np.float32)
    n = min(num_joints, len(a))
    vel[:n] = max_vel * a[:n]

    return vel


def model_guided_digging_velocities(action, step, num_joints, max_vel, cycle_len=120):
    """
    Model-guided excavator action adapter.

    SmolVLA is still used every step, but its raw action is treated as a
    modulation signal. The digging motion prior gives the general task phase:

        lower -> scoop -> lift -> swing -> dump -> return

    Assumed joint order:
        joint 0 = swing
        joint 1 = boom
        joint 2 = arm / stick
        joint 3 = bucket

    If the robot moves in the wrong direction, flip signs in base_vel below.
    """
    a = extract_action_np(action)
    a = np.tanh(a).astype(np.float32)

    vel = np.zeros(num_joints, dtype=np.float32)
    base_vel = np.zeros(num_joints, dtype=np.float32)

    phase = step % cycle_len

    # Only define first 4 joints because your robot has 4 joints.
    # Values here are normalized direction commands, later multiplied by max_vel.
    if phase < 25:
        phase_name = "lower"
        # Lower bucket toward soil.
        # [swing, boom, arm, bucket]
        base4 = np.array([0.00, -0.25, 0.18, 0.08], dtype=np.float32)

    elif phase < 50:
        phase_name = "scoop"
        # Pull arm and curl bucket.
        base4 = np.array([0.00, -0.05, -0.18, -0.55], dtype=np.float32)

    elif phase < 75:
        phase_name = "lift"
        # Lift boom and keep bucket curled.
        base4 = np.array([0.00, 0.35, -0.05, -0.12], dtype=np.float32)

    elif phase < 95:
        phase_name = "swing"
        # Swing toward truck / side.
        base4 = np.array([0.35, 0.05, 0.00, 0.00], dtype=np.float32)

    elif phase < 110:
        phase_name = "dump"
        # Dump bucket.
        base4 = np.array([0.00, 0.00, 0.00, 0.60], dtype=np.float32)

    else:
        phase_name = "return"
        # Move gently back.
        base4 = np.array([-0.25, -0.10, 0.08, -0.15], dtype=np.float32)

    n = min(num_joints, 4)
    base_vel[:n] = base4[:n]

    # Use SmolVLA action magnitude as modulation.
    # This keeps the model in the loop but prevents raw random mapping.
    model_gain = np.ones(num_joints, dtype=np.float32)
    m = min(num_joints, len(a))
    model_gain[:m] = 0.6 + 0.6 * np.clip(np.abs(a[:m]), 0.0, 1.0)

    vel = max_vel * base_vel * model_gain

    return vel, phase_name


def position_target_from_velocity(reply, vel, ticks, scale=3.0):
    """
    Optional position-control mode.

    Some Isaac articulations respond more clearly to joint_positions than
    joint_velocities. This integrates velocity into a next position target.
    """
    q = np.asarray(reply.get("joint_positions", []), dtype=np.float32)

    if len(q) == 0:
        return None

    n = min(len(q), len(vel))
    q_target = q.copy()

    dt = ticks / 60.0
    q_target[:n] = q[:n] + vel[:n] * dt * scale

    return q_target


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)

    parser.add_argument("--ckpt", default=os.environ.get("SMOLVLA_CKPT", ""))
    parser.add_argument("--vlm", default=os.environ.get("SMOLVLA_VLM", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"))

    parser.add_argument("--task", default="dig the soil and load it into the truck")

    parser.add_argument("--max-vel", type=float, default=1.0)
    parser.add_argument("--ticks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=240)

    parser.add_argument(
        "--mode",
        default="guided",
        choices=["raw", "guided", "guided_position"],
        help=(
            "raw: directly map SmolVLA action to joint velocity; "
            "guided: model-guided digging velocity; "
            "guided_position: model-guided digging converted to joint position targets."
        ),
    )

    parser.add_argument(
        "--cycle-len",
        type=int,
        default=120,
        help="Number of inference steps per digging cycle.",
    )

    parser.add_argument(
        "--position-scale",
        type=float,
        default=3.0,
        help="Scale used only in guided_position mode.",
    )

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
    print(f"[INFO] mode: {args.mode}")
    print(f"[INFO] task: {args.task}")

    # Important:
    # Initial command should not include joint_velocities because
    # we do not know robot joint count before first reply.
    cmd = {
        "ticks": args.ticks,
    }

    num_joints = None

    try:
        for step in range(args.steps):
            write_json(sock, cmd)
            reply = read_json(sock)

            if "error" in reply:
                print("[SERVER ERROR]", reply["error"])

            rgb = decode_rgb(reply)
            image = rgb_to_tensor(rgb, device)
            state = make_state(reply, device)

            q_print = np.asarray(reply.get("joint_positions", []), dtype=np.float32)
            qd_print = np.asarray(reply.get("joint_velocities", []), dtype=np.float32)

            if num_joints is None:
                if "num_joints" in reply:
                    num_joints = int(reply["num_joints"])
                else:
                    num_joints = len(q_print)

                print("[INFO] detected num_joints:", num_joints)

            batch = {
                "observation.state": state,

                # SmolVLA base checkpoint expects 3 camera inputs.
                # For now, use the same Isaac viewport image for all 3 cameras.
                "observation.images.camera1": image,
                "observation.images.camera2": image,
                "observation.images.camera3": image,

                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

            with torch.no_grad():
                action = policy.select_action(batch)

            if args.mode == "raw":
                vel = raw_action_to_joint_velocities(
                    action=action,
                    num_joints=num_joints,
                    max_vel=args.max_vel,
                )
                phase_name = "raw"

                cmd = {
                    "joint_velocities": vel.tolist(),
                    "ticks": args.ticks,
                }

            elif args.mode == "guided":
                vel, phase_name = model_guided_digging_velocities(
                    action=action,
                    step=step,
                    num_joints=num_joints,
                    max_vel=args.max_vel,
                    cycle_len=args.cycle_len,
                )

                cmd = {
                    "joint_velocities": vel.tolist(),
                    "ticks": args.ticks,
                }

            elif args.mode == "guided_position":
                vel, phase_name = model_guided_digging_velocities(
                    action=action,
                    step=step,
                    num_joints=num_joints,
                    max_vel=args.max_vel,
                    cycle_len=args.cycle_len,
                )

                q_target = position_target_from_velocity(
                    reply=reply,
                    vel=vel,
                    ticks=args.ticks,
                    scale=args.position_scale,
                )

                if q_target is None:
                    cmd = {
                        "joint_velocities": vel.tolist(),
                        "ticks": args.ticks,
                    }
                else:
                    cmd = {
                        "joint_positions": q_target.tolist(),
                        "ticks": args.ticks,
                    }

            else:
                raise ValueError(f"Unknown mode: {args.mode}")

            action_np = extract_action_np(action)

            print(
                f"[STEP {step:04d}] "
                f"phase={phase_name} "
                f"q={q_print.round(3).tolist()} "
                f"qd={qd_print.round(3).tolist()} "
                f"action={action_np.round(3).tolist()} "
                f"vel={vel.round(3).tolist()} "
                f"rgb_mean={float(rgb.mean()):.2f}"
            )

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] interrupted")

    finally:
        try:
            if num_joints is not None and num_joints > 0:
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
