#!/usr/bin/env python3
"""Replay one offline-evaluation episode into Isaac Sim with chunk=1.

The evaluator stores ``pred_action`` as [frame, prediction_horizon, action].
This client selects only prediction_horizon index 0 from every frame and sends
that four-dimensional action directly to the simulator for one control step.
It performs no model inference, temporal ensembling, stage supervision, or
execution assistance.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from excavator_common.bridge_protocol import (  # noqa: E402
    decode_camera_images,
    read_json,
    write_json,
)
from excavator_common.deployment_contract import (  # noqa: E402
    OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
    build_client_contract,
    load_training_fps,
)


ACTION_NAMES = ("swing", "boom", "arm", "bucket")
EXPECTED_ACTION_DIM = len(ACTION_NAMES)


class MultiCameraVideoRecorder:
    """Write aligned RGB frames from the bridge to one MP4 per camera."""

    def __init__(self, output_dir: Path, fps: float, codec: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required when --record-video-dir is used"
            ) from exc

        if fps <= 0:
            raise ValueError("Video FPS must be positive")
        if len(codec) != 4:
            raise ValueError("Video codec must contain four characters")

        self.cv2 = cv2
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.codec = str(codec)
        self.writers: dict[str, Any] = {}
        self.sizes: dict[str, tuple[int, int]] = {}
        self.paths: dict[str, Path] = {}
        self.frames = 0

    def write(self, cameras: dict[str, np.ndarray]) -> None:
        for camera_id, frame in sorted(cameras.items()):
            rgb = np.asarray(frame)
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise RuntimeError(
                    f"Camera {camera_id} has invalid shape {rgb.shape}"
                )
            if rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            rgb = np.ascontiguousarray(rgb)

            height, width = rgb.shape[:2]
            if camera_id not in self.writers:
                path = self.output_dir / f"camera_{camera_id}.mp4"
                writer = self.cv2.VideoWriter(
                    str(path),
                    self.cv2.VideoWriter_fourcc(*self.codec),
                    self.fps,
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Failed to open video writer: {path}")
                self.writers[camera_id] = writer
                self.sizes[camera_id] = (int(width), int(height))
                self.paths[camera_id] = path
                print(f"[VIDEO] camera={camera_id} path={path}")
            elif self.sizes[camera_id] != (int(width), int(height)):
                raise RuntimeError(
                    f"Camera {camera_id} changed resolution: "
                    f"{self.sizes[camera_id]} -> {(width, height)}"
                )

            bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
            self.writers[camera_id].write(bgr)
        self.frames += 1

    def close(self) -> None:
        for writer in self.writers.values():
            writer.release()
        metadata = {
            "frames": int(self.frames),
            "fps": float(self.fps),
            "codec": self.codec,
            "videos": {
                camera_id: str(path)
                for camera_id, path in sorted(self.paths.items())
            },
        }
        (self.output_dir / "recording_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[VIDEO] closed frames={self.frames}")


def load_chunk1_actions(
    episode_npz: Path,
    action_key: str,
    start_frame: int,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load one current-step prediction per episode frame."""
    with np.load(episode_npz, allow_pickle=False) as payload:
        if action_key not in payload.files:
            raise RuntimeError(
                f"{episode_npz} has no {action_key!r}; keys={payload.files}"
            )
        stored_actions = np.asarray(payload[action_key], dtype=np.float32)
        if stored_actions.ndim == 3:
            if stored_actions.shape[1] < 1:
                raise RuntimeError("Stored action horizon is empty")
            actions = stored_actions[:, 0, :]
        elif stored_actions.ndim == 2:
            actions = stored_actions
        else:
            raise RuntimeError(
                f"Expected action array [F,H,A] or [F,A], got "
                f"{stored_actions.shape}"
            )

        if actions.shape[1:] != (EXPECTED_ACTION_DIM,):
            raise RuntimeError(
                f"Expected four actions {ACTION_NAMES}, got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            bad = np.argwhere(~np.isfinite(actions))[:20].tolist()
            raise RuntimeError(f"Action array contains NaN/Inf at {bad}")

        frame_indices = (
            np.asarray(payload["frame_index"], dtype=np.int64).reshape(-1)
            if "frame_index" in payload.files
            else np.arange(actions.shape[0], dtype=np.int64)
        )
        if frame_indices.shape != (actions.shape[0],):
            raise RuntimeError(
                "frame_index length does not match the action array: "
                f"{frame_indices.shape} versus {actions.shape}"
            )

        stages = None
        if "pred_stage" in payload.files:
            stored_stages = np.asarray(payload["pred_stage"])
            if stored_stages.ndim == 2 and stored_stages.shape[1] >= 1:
                stages = stored_stages[:, 0].astype(np.int64)
            elif stored_stages.ndim == 1:
                stages = stored_stages.astype(np.int64)

        if "action_valid" in payload.files:
            stored_valid = np.asarray(payload["action_valid"], dtype=bool)
            current_valid = (
                stored_valid[:, 0]
                if stored_valid.ndim == 2
                else stored_valid.reshape(-1)
            )
            if current_valid.shape != (actions.shape[0],):
                raise RuntimeError("action_valid has an incompatible shape")
            invalid_frames = frame_indices[~current_valid]
            if invalid_frames.size:
                raise RuntimeError(
                    "Current-step prediction is invalid at frames "
                    f"{invalid_frames[:20].tolist()}"
                )

    if start_frame < 0 or start_frame >= actions.shape[0]:
        raise RuntimeError(
            f"--start-frame must be in [0, {actions.shape[0] - 1}]"
        )
    stop = actions.shape[0]
    if max_steps > 0:
        stop = min(stop, start_frame + max_steps)
    selection = slice(start_frame, stop)
    selected_actions = np.ascontiguousarray(actions[selection])
    selected_frames = np.ascontiguousarray(frame_indices[selection])
    selected_stages = (
        np.ascontiguousarray(stages[selection])
        if stages is not None
        else None
    )
    if selected_frames.size > 1 and not np.all(
        np.diff(selected_frames) == 1
    ):
        raise RuntimeError(
            f"Selected frame indices are not consecutive: {selected_frames}"
        )
    return selected_actions, selected_frames, selected_stages


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay pred_action[:, 0, :] from one offline episode NPZ "
            "directly into the Isaac Sim bridge."
        )
    )
    parser.add_argument("--episode-npz", type=Path, required=True)
    parser.add_argument("--dataset-meta", type=Path, required=True)
    parser.add_argument("--action-key", default="pred_action")
    parser.add_argument("--execution-chunk-size", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--trace-log", type=Path, default=None)
    parser.add_argument("--record-video-dir", type=Path, default=None)
    parser.add_argument("--record-video-codec", default="mp4v")
    parser.add_argument("--print-every", type=int, default=1)
    args = parser.parse_args()

    if args.execution_chunk_size != 1:
        raise SystemExit(
            "This replay client is strict chunk=1; "
            "--execution-chunk-size must equal 1"
        )
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    if args.print_every <= 0:
        raise SystemExit("--print-every must be positive")

    episode_npz = args.episode_npz.expanduser().resolve()
    if not episode_npz.is_file():
        raise FileNotFoundError(episode_npz)
    training_fps, dataset_meta = load_training_fps(
        args.dataset_meta.expanduser().resolve()
    )
    actions, frame_indices, stages = load_chunk1_actions(
        episode_npz,
        args.action_key,
        args.start_frame,
        args.max_steps,
    )
    print(f"[SOURCE] episode_npz={episode_npz}")
    print(f"[SOURCE] action_key={args.action_key}")
    print(
        f"[SOURCE] selection={args.action_key}[:, 0, :] strict chunk=1"
    )
    print(
        f"[SOURCE] steps={len(actions)} frames="
        f"{int(frame_indices[0])}..{int(frame_indices[-1])}"
    )
    print(f"[SOURCE] dataset_meta={dataset_meta} fps={training_fps:g}")

    trace_handle = None
    recorder = None
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((args.host, args.port))
        handshake = build_client_contract(
            training_fps,
            observation_schema=OBSERVATION_SCHEMA_27D_PLUS_EFFORT,
        )
        write_json(sock, handshake)
        reply = read_json(sock)
        if not reply.get("ok", False):
            raise RuntimeError(f"Bridge handshake rejected: {reply}")
        print(f"[BRIDGE CONTRACT] accepted={reply}")

        if args.trace_log is not None:
            trace_path = args.trace_log.expanduser().resolve()
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_path.open(
                "w",
                encoding="utf-8",
                buffering=1,
            )
            print(f"[TRACE LOG] {trace_path}")
        if args.record_video_dir is not None:
            recorder = MultiCameraVideoRecorder(
                args.record_video_dir.expanduser().resolve(),
                training_fps,
                args.record_video_codec,
            )

        wall_start = time.perf_counter()
        simulated_seconds = 0.0
        for replay_step, (frame_index, action) in enumerate(
            zip(frame_indices, actions, strict=True)
        ):
            step_start = time.perf_counter()
            write_json(sock, {"joint_velocities": action.tolist()})
            reply = read_json(sock)
            if reply.get("type") == "error":
                raise RuntimeError(
                    f"Bridge rejected replay step {replay_step}: "
                    f"{reply.get('error')}"
                )
            timing = reply.get("bridge_timing") or {}
            simulated_seconds += float(timing.get("simulated_seconds", 0.0))
            stage = (
                int(stages[replay_step]) if stages is not None else None
            )
            q = np.asarray(reply.get("joint_positions", []), dtype=np.float32)
            load = float(
                (reply.get("bucket_load_metrics") or {}).get("count", 0.0)
            )

            if recorder is not None:
                recorder.write(decode_camera_images(reply, np_module=np))
            if trace_handle is not None:
                trace_handle.write(
                    json.dumps(
                        {
                            "type": "offline_episode_chunk1_replay_step",
                            "replay_step": int(replay_step),
                            "source_frame_index": int(frame_index),
                            "source_prediction_offset": 0,
                            "execution_chunk_size": 1,
                            "source_predicted_stage": stage,
                            "action": {
                                name: float(action[index])
                                for index, name in enumerate(ACTION_NAMES)
                            },
                            "joint_positions": q.astype(float).tolist(),
                            "bucket_load": load,
                            "bridge_timing": timing,
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )

            if (
                replay_step % args.print_every == 0
                or replay_step == len(actions) - 1
            ):
                print(
                    f"[REPLAY {replay_step:05d}] "
                    f"source_frame={int(frame_index)} "
                    f"stage={stage} action={action.round(5).tolist()} "
                    f"q={q.round(5).tolist()} load={load:.1f} "
                    f"loop_ms={(time.perf_counter() - step_start) * 1000:.1f}"
                )

        wall_seconds = time.perf_counter() - wall_start
        print(
            f"[DONE] replayed_steps={len(actions)} "
            f"simulated_seconds={simulated_seconds:.3f} "
            f"wall_seconds={wall_seconds:.3f}"
        )
    finally:
        if trace_handle is not None:
            trace_handle.close()
        if recorder is not None:
            recorder.close()
        sock.close()


if __name__ == "__main__":
    main()
