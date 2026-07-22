#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from lerobot.datasets.lerobot_dataset import LeRobotDataset


STAGE_NAMES = [
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
]
ACTION_NAMES = ["swing", "boom", "arm", "bucket"]

DEFAULT_DATASET = Path(
    "/root/gpufree-data/ExcavatorVLA/"
    "excavator_auto_dataset/.dashboard_success/"
    "lerobot_v3_stage10_h30_obslabels"
)
DEFAULT_EVAL_DIR = Path(
    "/root/gpufree-data/excavator_final_offline_eval"
)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def stage_metrics_from_confusion(
    confusion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(confusion).astype(np.float64)
    precision = safe_divide(tp, confusion.sum(axis=0))
    recall = safe_divide(tp, confusion.sum(axis=1))
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def tensor_image_to_pil(value: Any) -> Image.Image:
    if hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))

    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.shape[-1] == 4:
        return Image.fromarray(array, mode="RGBA").convert("RGB")
    return Image.fromarray(array, mode="RGB")


def make_camera_contact_sheet(
    dataset_root: Path,
    episode_id: int,
    arrays: dict[str, np.ndarray],
    output_path: Path,
    num_frames: int = 12,
) -> None:
    dataset = LeRobotDataset(
        repo_id=dataset_root.name,
        root=dataset_root,
        episodes=[episode_id],
        download_videos=True,
    )
    camera_keys = list(dataset.meta.camera_keys)
    if not camera_keys:
        print("[WARN] no camera keys; skipping contact sheet")
        return

    count = len(dataset)
    selected = np.unique(
        np.linspace(
            0,
            max(0, count - 1),
            min(num_frames, count),
            dtype=int,
        )
    )

    target_height = 220
    label_height = 58
    row_images = []

    for local_index in selected:
        item = dataset[int(local_index)]
        camera_images = []

        for camera_key in camera_keys:
            image = tensor_image_to_pil(item[camera_key])
            ratio = target_height / image.height
            image = image.resize(
                (
                    max(1, int(round(image.width * ratio))),
                    target_height,
                ),
                Image.Resampling.BILINEAR,
            )
            camera_images.append(image)

        row_width = sum(image.width for image in camera_images)
        row = Image.new(
            "RGB",
            (row_width, target_height + label_height),
            "white",
        )
        x_offset = 0
        for camera_key, image in zip(
            camera_keys,
            camera_images,
            strict=True,
        ):
            row.paste(image, (x_offset, label_height))
            draw = ImageDraw.Draw(row)
            draw.text(
                (x_offset + 4, 2),
                camera_key,
                fill="black",
            )
            x_offset += image.width

        true_stage = int(arrays["true_stage"][local_index, 0])
        pred_stage = int(arrays["pred_stage"][local_index, 0])
        true_action = arrays["true_action"][local_index, 0]
        pred_action = arrays["pred_action"][local_index, 0]

        header = (
            f"frame={int(arrays['frame_index'][local_index])}  "
            f"true_stage={true_stage}:{STAGE_NAMES[true_stage]}  "
            f"pred_stage={pred_stage}:{STAGE_NAMES[pred_stage]}\n"
            f"true_action={np.round(true_action, 3).tolist()}  "
            f"pred_action={np.round(pred_action, 3).tolist()}"
        )
        ImageDraw.Draw(row).text(
            (4, 18),
            header,
            fill="black",
        )
        row_images.append(row)

    sheet_width = max(image.width for image in row_images)
    sheet_height = sum(image.height for image in row_images)
    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white",
    )

    y_offset = 0
    for row in row_images:
        sheet.paste(row, (0, y_offset))
        y_offset += row.height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_frame_csv(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "frame_index",
        "timestamp",
        "true_stage_id",
        "true_stage_name",
        "pred_stage_id",
        "pred_stage_name",
        "stage_confidence",
    ]
    for action_name in ACTION_NAMES:
        fields.extend(
            [
                f"true_{action_name}",
                f"pred_{action_name}",
                f"abs_error_{action_name}",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        confidence = arrays["stage_probability"][:, 0, :].max(axis=-1)
        for i in range(len(arrays["frame_index"])):
            true_stage = int(arrays["true_stage"][i, 0])
            pred_stage = int(arrays["pred_stage"][i, 0])
            row = {
                "frame_index": int(arrays["frame_index"][i]),
                "timestamp": float(arrays["timestamp"][i]),
                "true_stage_id": true_stage,
                "true_stage_name": STAGE_NAMES[true_stage],
                "pred_stage_id": pred_stage,
                "pred_stage_name": STAGE_NAMES[pred_stage],
                "stage_confidence": float(confidence[i]),
            }
            for action_id, action_name in enumerate(ACTION_NAMES):
                true_value = float(
                    arrays["true_action"][i, 0, action_id]
                )
                pred_value = float(
                    arrays["pred_action"][i, 0, action_id]
                )
                row[f"true_{action_name}"] = true_value
                row[f"pred_{action_name}"] = pred_value
                row[f"abs_error_{action_name}"] = abs(
                    pred_value - true_value
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create detailed figures for the median representative episode "
            "selected by evaluate_smolvla_episodes.py."
        )
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--contact-sheet-frames",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    eval_dir = args.eval_dir.expanduser().resolve()
    dataset_root = args.dataset.expanduser().resolve()

    median_payload = json.loads(
        (eval_dir / "median_episode.json").read_text(
            encoding="utf-8"
        )
    )
    episode_id = int(median_payload["episode_id"])

    episode_file = (
        eval_dir / "episodes" / f"episode_{episode_id:06d}.npz"
    )
    if not episode_file.is_file():
        raise FileNotFoundError(episode_file)

    loaded = np.load(episode_file)
    arrays = {key: loaded[key] for key in loaded.files}

    figure_dir = eval_dir / f"median_episode_{episode_id:06d}_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    frame = arrays["frame_index"].astype(np.int64)
    time_s = arrays["timestamp"].astype(np.float64)
    true_stage = arrays["true_stage"][:, 0].astype(np.int64)
    pred_stage = arrays["pred_stage"][:, 0].astype(np.int64)
    stage_probability = arrays["stage_probability"][:, 0, :]
    stage_confidence = stage_probability.max(axis=-1)
    stage_correct = true_stage == pred_stage

    true_action = arrays["true_action"][:, 0, :]
    pred_action = arrays["pred_action"][:, 0, :]
    action_error = pred_action - true_action
    action_abs_error = np.abs(action_error)
    action_scale = arrays["action_scale"]
    action_tolerance = arrays["action_tolerance"]

    # 01: one stage prediction and one true stage for every dataset frame.
    plt.figure(figsize=(15, 5))
    plt.step(frame, true_stage, where="post", label="true stage")
    plt.step(frame, pred_stage, where="post", label="predicted stage")
    plt.yticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    plt.xlabel("episode frame")
    plt.ylabel("stage")
    plt.title(
        f"Episode {episode_id}: per-frame true and predicted stage"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "01_stage_true_vs_pred.png")

    # 02: stage confidence and correctness at every frame.
    plt.figure(figsize=(15, 4))
    plt.plot(frame, stage_confidence, label="maximum stage probability")
    plt.scatter(
        frame[~stage_correct],
        stage_confidence[~stage_correct],
        marker="x",
        label="incorrect prediction",
    )
    plt.ylim(-0.02, 1.02)
    plt.xlabel("episode frame")
    plt.ylabel("probability")
    plt.title(
        f"Episode {episode_id}: stage confidence and errors"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "02_stage_confidence.png")

    # 03-06: true and predicted raw motion command for each action.
    for action_id, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(15, 4))
        plt.plot(
            frame,
            true_action[:, action_id],
            label=f"true {action_name}",
        )
        plt.plot(
            frame,
            pred_action[:, action_id],
            label=f"predicted {action_name}",
        )
        plt.xlabel("episode frame")
        plt.ylabel("raw action")
        plt.title(
            f"Episode {episode_id}: {action_name} true vs predicted"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_figure(
            figure_dir
            / f"{3 + action_id:02d}_action_{action_name}_true_vs_pred.png"
        )

    # 07-10: signed and absolute error for each action.
    for action_id, action_name in enumerate(ACTION_NAMES):
        plt.figure(figsize=(15, 4))
        plt.plot(
            frame,
            action_error[:, action_id],
            label="signed error",
        )
        plt.plot(
            frame,
            action_abs_error[:, action_id],
            label="absolute error",
        )
        plt.axhline(
            action_tolerance[action_id],
            linestyle="--",
            label="10% q95 tolerance",
        )
        plt.xlabel("episode frame")
        plt.ylabel("action error")
        plt.title(
            f"Episode {episode_id}: {action_name} prediction error"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_figure(
            figure_dir
            / f"{7 + action_id:02d}_action_{action_name}_error.png"
        )

    # 11: combined action vector error.
    action_l2 = np.linalg.norm(action_error, axis=-1)
    normalized_l2 = np.linalg.norm(
        action_error / action_scale[None, :],
        axis=-1,
    )
    plt.figure(figsize=(15, 4))
    plt.plot(frame, action_l2, label="raw L2 action error")
    plt.plot(
        frame,
        normalized_l2,
        label="q95-normalized L2 error",
    )
    plt.xlabel("episode frame")
    plt.ylabel("error")
    plt.title(f"Episode {episode_id}: complete action-vector error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "11_action_vector_error.png")

    # 12: confusion matrix for this episode.
    confusion = arrays["confusion"].astype(np.int64)
    plt.figure(figsize=(9, 8))
    plt.imshow(confusion)
    plt.xticks(
        range(len(STAGE_NAMES)),
        STAGE_NAMES,
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(STAGE_NAMES)), STAGE_NAMES)
    plt.xlabel("predicted stage")
    plt.ylabel("true stage")
    plt.title(f"Episode {episode_id}: stage confusion matrix")
    plt.colorbar(label="number of valid stage tokens")
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            plt.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
            )
    save_figure(figure_dir / "12_stage_confusion_matrix.png")

    # 13: per-stage precision, recall, and F1.
    precision, recall, f1 = stage_metrics_from_confusion(confusion)
    x = np.arange(len(STAGE_NAMES))
    width = 0.26
    plt.figure(figsize=(15, 5))
    plt.bar(x - width, precision, width, label="precision")
    plt.bar(x, recall, width, label="recall")
    plt.bar(x + width, f1, width, label="F1")
    plt.xticks(x, STAGE_NAMES, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title(f"Episode {episode_id}: per-stage metrics")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "13_per_stage_metrics.png")

    # 14: stage accuracy by prediction horizon 0..49.
    stage_valid = arrays["stage_valid"].astype(bool)
    stage_correct_chunk = (
        arrays["pred_stage"].astype(np.int64)
        == arrays["true_stage"].astype(np.int64)
    )
    stage_correct_by_horizon = safe_divide(
        np.sum(stage_correct_chunk & stage_valid, axis=0),
        np.sum(stage_valid, axis=0),
    )
    horizon = np.arange(stage_correct_by_horizon.size)
    plt.figure(figsize=(12, 4))
    plt.plot(horizon, stage_correct_by_horizon, marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("future horizon step")
    plt.ylabel("stage accuracy")
    plt.title(
        f"Episode {episode_id}: 50-step stage accuracy by horizon"
    )
    plt.grid(True, alpha=0.3)
    save_figure(figure_dir / "14_stage_accuracy_by_horizon.png")

    # 15-18: action MAE by prediction horizon for each command.
    action_valid = arrays["action_valid"].astype(bool)
    for action_id, action_name in enumerate(ACTION_NAMES):
        chunk_abs_error = np.abs(
            arrays["pred_action"][:, :, action_id]
            - arrays["true_action"][:, :, action_id]
        )
        mae_by_horizon = safe_divide(
            np.sum(chunk_abs_error * action_valid, axis=0),
            np.sum(action_valid, axis=0),
        )

        plt.figure(figsize=(12, 4))
        plt.plot(horizon, mae_by_horizon, marker="o")
        plt.xlabel("future horizon step")
        plt.ylabel("raw MAE")
        plt.title(
            f"Episode {episode_id}: {action_name} MAE by horizon"
        )
        plt.grid(True, alpha=0.3)
        save_figure(
            figure_dir
            / f"{15 + action_id:02d}_{action_name}_mae_by_horizon.png"
        )

    # 19-22: action sign correctness at every current frame.
    active_threshold = arrays["action_active_threshold"]
    for action_id, action_name in enumerate(ACTION_NAMES):
        active = (
            np.abs(true_action[:, action_id])
            > active_threshold[action_id]
        )
        correct = (
            np.sign(pred_action[:, action_id])
            == np.sign(true_action[:, action_id])
        )
        timeline = np.full(frame.shape, np.nan, dtype=np.float64)
        timeline[active] = correct[active].astype(np.float64)

        plt.figure(figsize=(15, 3.5))
        plt.scatter(
            frame[active],
            timeline[active],
            label="active true action",
        )
        plt.yticks([0, 1], ["wrong sign", "correct sign"])
        plt.ylim(-0.2, 1.2)
        plt.xlabel("episode frame")
        plt.title(
            f"Episode {episode_id}: {action_name} direction correctness"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_figure(
            figure_dir
            / f"{19 + action_id:02d}_{action_name}_sign_correctness.png"
        )

    # 23: explicit low-motion safety rule for stages 1..7.
    low_motion_stage = (true_stage >= 1) & (true_stage <= 7)
    normalized_abs_swing = (
        np.abs(pred_action[:, 0]) / action_scale[0]
    )
    plt.figure(figsize=(15, 4))
    plt.plot(
        frame,
        normalized_abs_swing,
        label="predicted normalized |swing|",
    )
    plt.axhline(
        0.10,
        linestyle="--",
        label="low-motion margin",
    )
    plt.scatter(
        frame[low_motion_stage],
        normalized_abs_swing[low_motion_stage],
        marker=".",
        label="frames with true stage 1..7",
    )
    plt.xlabel("episode frame")
    plt.ylabel("|swing| / global q95 scale")
    plt.title(
        f"Episode {episode_id}: stage 1..7 swing low-motion rule"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "23_low_motion_swing_rule.png")

    # 24: stage transition comparison.
    true_transition = np.zeros_like(true_stage, dtype=np.int64)
    pred_transition = np.zeros_like(pred_stage, dtype=np.int64)
    true_transition[1:] = true_stage[1:] != true_stage[:-1]
    pred_transition[1:] = pred_stage[1:] != pred_stage[:-1]
    plt.figure(figsize=(15, 3.5))
    plt.step(
        frame,
        true_transition,
        where="post",
        label="true stage transition",
    )
    plt.step(
        frame,
        pred_transition,
        where="post",
        label="predicted stage transition",
    )
    plt.yticks([0, 1], ["no transition", "transition"])
    plt.ylim(-0.2, 1.2)
    plt.xlabel("episode frame")
    plt.title(f"Episode {episode_id}: stage transition timing")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_figure(figure_dir / "24_stage_transition_timing.png")

    # 25: detailed camera montage at evenly spaced frames.
    make_camera_contact_sheet(
        dataset_root,
        episode_id,
        arrays,
        figure_dir / "25_camera_contact_sheet.png",
        args.contact_sheet_frames,
    )

    write_frame_csv(
        figure_dir / "median_episode_per_frame_outputs.csv",
        arrays,
    )

    report = [
        f"# Median episode {episode_id}",
        "",
        (
            "The episode was chosen as the sampled episode whose composite "
            "representative score was closest to the median score among the "
            "the evaluated episode set."
        ),
        "",
        "## Generated outputs",
        "",
        "- Per-frame true and predicted stage timeline",
        "- Stage confidence and error markers",
        "- Separate true/predicted curves for swing, boom, arm, and bucket",
        "- Separate signed/absolute action-error curves",
        "- Complete action-vector error",
        "- Stage confusion matrix and per-stage precision/recall/F1",
        "- Stage accuracy by 50-step prediction horizon",
        "- Per-action MAE by 50-step prediction horizon",
        "- Per-action sign-correctness timelines",
        "- Stage 1–7 swing low-motion verification",
        "- True and predicted stage-transition timing",
        "- Multi-camera contact sheet with stage/action labels",
        "- CSV containing the first-step prediction for every episode frame",
        "",
        f"Figure directory: `{figure_dir}`",
    ]
    (figure_dir / "README.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] median episode: {episode_id}")
    print(f"[OK] detailed figures: {figure_dir}")
    print(
        "[OK] per-frame CSV: "
        f"{figure_dir / 'median_episode_per_frame_outputs.csv'}"
    )


if __name__ == "__main__":
    main()
