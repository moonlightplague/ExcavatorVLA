#!/usr/bin/env python3
"""
Minimal protocol-v2 SmolVLA policy client.

Every loop iteration:
1. Send current velocity (last model output)
2. Receive latest observation from bridge
3. Run model inference while the deterministic bridge is paused
4. Update velocity from model output
5. Repeat

Cadence comes from the old dataset metadata. The exact 18D state and 4D action
normalization statistics are applied once by this client.
"""

import argparse
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from excavator_common.bridge_protocol import (
    decode_camera_images,
    decode_rgb_payload as decode_rgb,
)
from excavator_common.deployment_contract import (
    build_client_contract,
    load_training_fps,
    sha256_files,
    validate_training_fps,
)
from smolvla_policy_client import (
    load_normalization_stats,
    make_state,
    validate_policy_feature_contract,
)


def rgb_to_tensor(rgb, device):
    x = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0)
    return F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)


def load_language_tokens(task, vlm_dir, device):
    processor = AutoProcessor.from_pretrained(vlm_dir, local_files_only=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    lang = tokenizer(task, return_tensors="pt", padding=True, truncation=True, max_length=48)
    return lang["input_ids"].to(device), lang["attention_mask"].bool().to(device)


def patch_checkpoint_paths(ckpt_dir, vlm_dir):
    ckpt_dir = Path(ckpt_dir)
    vlm_dir = str(Path(vlm_dir))
    old = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def replace_obj(obj):
        if isinstance(obj, dict):
            return {k: replace_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_obj(x) for x in obj]
        if isinstance(obj, str):
            return obj.replace(old, vlm_dir)
        return obj

    for p in [ckpt_dir / "config.json", ckpt_dir / "policy_preprocessor.json", ckpt_dir / "train_config.json"]:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
            new_data = replace_obj(data)
            if new_data != data:
                p.write_text(json.dumps(new_data, indent=2), encoding="utf-8")
                print(f"[INFO] patched {p}")
        except Exception as e:
            print(f"[WARN] failed to patch {p}: {e}")


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_json(sock):
    header = recv_exact(sock, 4)
    n = struct.unpack("!I", header)[0]
    return json.loads(recv_exact(sock, n).decode("utf-8"))


def write_json(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ckpt", default=os.environ.get("SMOLVLA_CKPT", ""))
    parser.add_argument("--vlm", default=os.environ.get("SMOLVLA_VLM", ""))
    parser.add_argument("--task", default="Dig soil from the marked area and dump it into the target container.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--dataset-meta",
        default=os.environ.get("SMOLVLA_DATASET_META", ""),
        help="Old training dataset root or meta/info.json.",
    )
    parser.add_argument("--training-fps", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    if not args.ckpt:
        raise SystemExit("--ckpt required")
    if not args.vlm:
        raise SystemExit("--vlm required")

    patch_checkpoint_paths(args.ckpt, args.vlm)

    print("[INFO] loading policy...")
    policy = SmolVLAPolicy.from_pretrained(args.ckpt)
    policy.eval()
    policy.to(device)
    validate_policy_feature_contract(policy)
    if hasattr(policy, "reset"):
        policy.reset()

    print("[INFO] loading tokenizer...")
    language_tokens, language_mask = load_language_tokens(args.task, args.vlm, device)

    state_mean, state_std, action_mean, action_std = load_normalization_stats(args.ckpt)
    if tuple(np.asarray(state_mean).shape) != (18,) or tuple(np.asarray(state_std).shape) != (18,):
        raise RuntimeError("Persistent client requires exact 18D state normalization stats")
    if tuple(np.asarray(action_mean).shape) != (4,) or tuple(np.asarray(action_std).shape) != (4,):
        raise RuntimeError("Persistent client requires exact 4D action normalization stats")
    state_mean_t = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
    state_std_t = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)
    action_mean_t = torch.from_numpy(action_mean).to(device=device, dtype=torch.float32)
    action_std_t = torch.from_numpy(action_std).to(device=device, dtype=torch.float32)

    if args.dataset_meta:
        training_fps, dataset_meta_path = load_training_fps(args.dataset_meta)
        if args.training_fps > 0.0 and abs(training_fps - args.training_fps) > 1.0e-6:
            raise RuntimeError("Explicit training FPS disagrees with dataset metadata")
        print(f"[BRIDGE CONTRACT] training_fps={training_fps} source={dataset_meta_path}")
    elif args.training_fps > 0.0:
        training_fps = validate_training_fps(args.training_fps)
        print(f"[WARN] using explicit training_fps={training_fps}")
    else:
        raise RuntimeError("Set --dataset-meta/SMOLVLA_DATASET_META")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"[INFO] connected to {args.host}:{args.port}")

    norm_hash = sha256_files(
        [
            Path(args.ckpt) / "policy_preprocessor_step_5_normalizer_processor.safetensors",
            Path(args.ckpt) / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        ]
    )
    if not norm_hash:
        raise RuntimeError("Could not hash checkpoint normalization assets")
    write_json(sock, build_client_contract(training_fps, normalization_hash=norm_hash))
    handshake_reply = read_json(sock)
    if not handshake_reply.get("ok", False):
        raise RuntimeError(f"Bridge handshake rejected: {handshake_reply}")
    if handshake_reply.get("normalization_hash", "") != norm_hash:
        raise RuntimeError("Bridge handshake normalization identity mismatch")
    print("[BRIDGE CONTRACT] accepted:", handshake_reply)

    vel = np.zeros(4, dtype=np.float32)
    task_text = args.task

    try:
        for step in range(args.steps):
            t_send = time.perf_counter()

            write_json(sock, {"joint_velocities": vel.tolist()})
            reply = read_json(sock)
            if reply.get("type") == "error":
                raise RuntimeError(f"Bridge error: {reply.get('error')}")

            if step == 0 and reply.get("task_text"):
                task_text = str(reply["task_text"])
                if task_text != args.task:
                    print(f"[INFO] task_text: {task_text}")
                    language_tokens, language_mask = load_language_tokens(task_text, args.vlm, device)

            rgb = decode_rgb(reply)
            camera_rgbs = decode_camera_images(reply, np_module=np)
            rgb0 = camera_rgbs.get("0", rgb)
            rgb1 = camera_rgbs.get("1", rgb)
            rgb2 = camera_rgbs.get("2", rgb)
            image0 = rgb_to_tensor(rgb0, device)
            image1 = rgb_to_tensor(rgb1, device)
            image2 = rgb_to_tensor(rgb2, device)

            obs_state = reply.get("observation_state")
            state = make_state(reply, device)
            state = (state - state_mean_t) / state_std_t

            batch = {
                "observation.state": state,
                "observation.images.0": image0,
                "observation.images.1": image1,
                "observation.images.2": image2,
                "task": [task_text],
                "observation.language.tokens": language_tokens,
                "observation.language.attention_mask": language_mask,
            }

            t_infer_start = time.perf_counter()
            with torch.no_grad():
                action = policy.select_action(batch)
            action = action * action_std_t + action_mean_t
            t_infer_end = time.perf_counter()

            vel_np = action.detach().float().cpu().numpy()
            if vel_np.ndim == 2:
                vel_np = vel_np[0]
            vel_np = vel_np.reshape(-1)[:4]
            vel = vel_np.astype(np.float32)

            print(
                f"[STEP {step:04d}] "
                f"vel={vel.round(4).tolist()} "
                f"infer_ms={(t_infer_end - t_infer_start) * 1000:.0f} "
                f"total_ms={(time.perf_counter() - t_send) * 1000:.0f} "
                f"state={obs_state[3:7]} "
                f"rgb_mean={float(rgb.mean()):.1f}"
            )

    except KeyboardInterrupt:
        print("\n[INFO] interrupted")
    finally:
        try:
            write_json(sock, {"joint_velocities": [0.0, 0.0, 0.0, 0.0]})
            time.sleep(0.1)
        except Exception:
            pass
        sock.close()
        print("[INFO] closed")


if __name__ == "__main__":
    main()
