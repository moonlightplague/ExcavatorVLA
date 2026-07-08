import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


ACTION_NAMES = ["swing", "boom", "arm", "bucket"]


def add_batch_dim(value):
    if torch.is_tensor(value):
        return value.unsqueeze(0)
    return value


def move_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def extract_action(output):
    if isinstance(output, dict):
        output = output["action"]

    output = output.detach().cpu()

    if output.ndim > 1:
        output = output[0]

    return output.numpy().astype(np.float64)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default=(
            "/root/gpufree-data/ExcavatorVLA/"
            "excavator_auto_dataset/.dashboard_success/lerobot_v3_gop4"
        ),
    )
    parser.add_argument(
        "--repo-id",
        default="runtingz/excavator_smolvla_3cam_gop4",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/root/gpufree-data/outputs/train/"
            "excavator_smolvla_3cam_gop4_lr5e5_b128_epoch20/"
            "checkpoints/015500/pretrained_model"
        ),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default="/tmp/smolvla_100_frame_comparison.csv",
    )

    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(args.device)

    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
    )

    if args.start_index < 0 or args.start_index >= len(dataset):
        raise IndexError(
            f"start-index={args.start_index} is outside dataset length {len(dataset)}"
        )

    first_sample = dataset[args.start_index]
    target_episode = int(first_sample["episode_index"])

    valid_indices = []

    for index in range(args.start_index, len(dataset)):
        sample = dataset[index]

        if int(sample["episode_index"]) != target_episode:
            break

        valid_indices.append(index)

        if len(valid_indices) == args.num_frames:
            break

    if len(valid_indices) < args.num_frames:
        raise RuntimeError(
            f"Only {len(valid_indices)} consecutive frames remain in "
            f"episode {target_episode}. Choose an earlier start index."
        )

    checkpoint = Path(args.checkpoint)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading policy:", checkpoint)

    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
        local_files_only=True,
    )
    policy.eval()
    policy.to(device)
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        dataset_stats=dataset.meta.stats,
    )

    rows = []
    predicted_actions = []
    gt_actions = []

    print("Episode:", target_episode)
    print("Task:", first_sample["task"])
    print("Dataset indices:", valid_indices[0], "to", valid_indices[-1])
    print("Number of frames:", len(valid_indices))

    for sequence_index, dataset_index in enumerate(valid_indices):
        sample = dataset[dataset_index]

        batch = {
            "observation.images.camera1": add_batch_dim(
                sample["observation.images.0"]
            ),
            "observation.images.camera2": add_batch_dim(
                sample["observation.images.1"]
            ),
            "observation.images.camera3": add_batch_dim(
                sample["observation.images.2"]
            ),
            "observation.state": add_batch_dim(
                sample["observation.state"]
            ),
            "task": [sample["task"]],
        }

        batch = move_to_device(batch, device)

        with torch.inference_mode():
            processed_batch = preprocessor(batch)
            predicted = policy.select_action(processed_batch)
            predicted = postprocessor(predicted)

        predicted = extract_action(predicted)
        ground_truth = (
            sample["action"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        predicted_actions.append(predicted)
        gt_actions.append(ground_truth)

        abs_error = np.abs(predicted - ground_truth)

        row = {
            "sequence_index": sequence_index,
            "dataset_index": dataset_index,
            "episode_index": int(sample["episode_index"]),
            "frame_index": int(sample["frame_index"]),
            "timestamp": float(sample["timestamp"]),
        }

        for action_index in range(len(ground_truth)):
            name = (
                ACTION_NAMES[action_index]
                if action_index < len(ACTION_NAMES)
                else f"action_{action_index}"
            )

            row[f"pred_{name}"] = float(predicted[action_index])
            row[f"gt_{name}"] = float(ground_truth[action_index])
            row[f"abs_error_{name}"] = float(abs_error[action_index])

        rows.append(row)

        print(
            f"[{sequence_index + 1:03d}/{args.num_frames}] "
            f"dataset={dataset_index} "
            f"frame={int(sample['frame_index'])} "
            f"pred={np.round(predicted, 4).tolist()} "
            f"gt={np.round(ground_truth, 4).tolist()}"
        )

    predicted_actions = np.stack(predicted_actions)
    gt_actions = np.stack(gt_actions)

    error = predicted_actions - gt_actions
    abs_error = np.abs(error)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nOverall metrics")
    print("MAE:", float(abs_error.mean()))
    print("MSE:", float(np.mean(error ** 2)))
    print("RMSE:", float(np.sqrt(np.mean(error ** 2))))
    print("Max absolute error:", float(abs_error.max()))

    print("\nPer-action metrics")

    for action_index in range(gt_actions.shape[1]):
        name = (
            ACTION_NAMES[action_index]
            if action_index < len(ACTION_NAMES)
            else f"action_{action_index}"
        )

        action_error = error[:, action_index]

        print(
            f"{name:>8}: "
            f"MAE={np.mean(np.abs(action_error)):.6f}, "
            f"RMSE={np.sqrt(np.mean(action_error ** 2)):.6f}, "
            f"pred_mean={np.mean(predicted_actions[:, action_index]):.6f}, "
            f"gt_mean={np.mean(gt_actions[:, action_index]):.6f}"
        )

    print("\nSaved CSV:", output_path)


if __name__ == "__main__":
    main()
