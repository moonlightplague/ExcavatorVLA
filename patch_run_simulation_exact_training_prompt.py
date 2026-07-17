#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_TARGET = Path(
    "/root/isaacsim/ExcavatorVLA/run_simulation.py"
)

HELPER_CODE = '\n    # -----------------------------------------------------------------\n    # Deployment prompt must match one of the exact training tasks.\n    # Directions are measured relative to the INITIAL base pose.\n    # -----------------------------------------------------------------\n    TRAINING_TASK_BY_DIRECTIONS = {\n        ("left", "front-right"): (\n            0,\n            "Excavate one scoop of sand from the sand pile to the left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the front-right of the excavator\'s initial base pose.",\n        ),\n        ("behind", "front-left"): (\n            1,\n            "Excavate one scoop of sand from the sand pile behind the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the front-left of the excavator\'s initial base pose.",\n        ),\n        ("front", "right"): (\n            2,\n            "Excavate one scoop of sand from the sand pile in front of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed to the right of the excavator\'s initial base pose.",\n        ),\n        ("front-left", "right"): (\n            3,\n            "Excavate one scoop of sand from the sand pile at the front-left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed to the right of the excavator\'s initial base pose.",\n        ),\n        ("front", "rear-right"): (\n            4,\n            "Excavate one scoop of sand from the sand pile in front of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-right of the excavator\'s initial base pose.",\n        ),\n        ("behind", "left"): (\n            5,\n            "Excavate one scoop of sand from the sand pile behind the excavator\'s initial base pose, then carry and dump the collected material into the truck bed to the left of the excavator\'s initial base pose.",\n        ),\n        ("right", "rear-left"): (\n            6,\n            "Excavate one scoop of sand from the sand pile to the right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-left of the excavator\'s initial base pose.",\n        ),\n        ("rear-right", "left"): (\n            7,\n            "Excavate one scoop of sand from the sand pile at the rear-right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed to the left of the excavator\'s initial base pose.",\n        ),\n        ("front", "behind"): (\n            8,\n            "Excavate one scoop of sand from the sand pile in front of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed behind the excavator\'s initial base pose.",\n        ),\n        ("front-right", "rear-right"): (\n            9,\n            "Excavate one scoop of sand from the sand pile at the front-right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-right of the excavator\'s initial base pose.",\n        ),\n        ("rear-left", "front"): (\n            10,\n            "Excavate one scoop of sand from the sand pile at the rear-left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed in front of the excavator\'s initial base pose.",\n        ),\n        ("left", "front"): (\n            11,\n            "Excavate one scoop of sand from the sand pile to the left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed in front of the excavator\'s initial base pose.",\n        ),\n        ("front-left", "front-right"): (\n            12,\n            "Excavate one scoop of sand from the sand pile at the front-left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the front-right of the excavator\'s initial base pose.",\n        ),\n        ("front-right", "rear-left"): (\n            13,\n            "Excavate one scoop of sand from the sand pile at the front-right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-left of the excavator\'s initial base pose.",\n        ),\n        ("front-left", "rear-right"): (\n            14,\n            "Excavate one scoop of sand from the sand pile at the front-left of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-right of the excavator\'s initial base pose.",\n        ),\n        ("right", "behind"): (\n            15,\n            "Excavate one scoop of sand from the sand pile to the right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed behind the excavator\'s initial base pose.",\n        ),\n        ("front-right", "behind"): (\n            16,\n            "Excavate one scoop of sand from the sand pile at the front-right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed behind the excavator\'s initial base pose.",\n        ),\n        ("rear-right", "rear-left"): (\n            17,\n            "Excavate one scoop of sand from the sand pile at the rear-right of the excavator\'s initial base pose, then carry and dump the collected material into the truck bed at the rear-left of the excavator\'s initial base pose.",\n        ),\n    }\n\n    def initial_base_direction_label(point_world):\n        point = np.asarray(point_world, dtype=np.float64).reshape(-1)\n        initial_position = np.asarray(\n            ROBOT_INITIAL_POS,\n            dtype=np.float64,\n        ).reshape(-1)\n\n        if point.size < 2 or initial_position.size < 2:\n            raise RuntimeError(\n                f"Invalid prompt target point: {point_world}"\n            )\n\n        initial_yaw = quaternion_wxyz_to_yaw(\n            ROBOT_INITIAL_ORI\n        )\n        dx = float(point[0] - initial_position[0])\n        dy = float(point[1] - initial_position[1])\n\n        if abs(dx) + abs(dy) < 1.0e-8:\n            raise RuntimeError(\n                "Prompt target is at the initial base origin; "\n                "no directional training task can represent it."\n            )\n\n        relative_angle = wrap_angle_radians(\n            math.atan2(dy, dx) - initial_yaw\n        )\n        sector = int(\n            math.floor(\n                (\n                    math.degrees(relative_angle)\n                    + 22.5\n                )\n                % 360.0\n                / 45.0\n            )\n        )\n\n        labels = (\n            "front",\n            "front-left",\n            "left",\n            "rear-left",\n            "behind",\n            "rear-right",\n            "right",\n            "front-right",\n        )\n        return labels[sector % len(labels)]\n\n    def select_exact_training_prompt(\n        dig_target_world,\n        unload_target_world,\n    ):\n        sand_direction = initial_base_direction_label(\n            dig_target_world\n        )\n        unload_direction = initial_base_direction_label(\n            unload_target_world\n        )\n        direction_pair = (\n            sand_direction,\n            unload_direction,\n        )\n\n        match = TRAINING_TASK_BY_DIRECTIONS.get(\n            direction_pair\n        )\n        if match is None:\n            raise RuntimeError(\n                "Current scene direction pair was not present in the "\n                "18 training tasks. Refusing to send an unseen prompt. "\n                f"current_pair={direction_pair}, "\n                f"dig_target_world={np.asarray(dig_target_world).tolist()}, "\n                f"unload_target_world={np.asarray(unload_target_world).tolist()}, "\n                f"initial_base_position={np.asarray(ROBOT_INITIAL_POS).tolist()}, "\n                f"available_pairs={sorted(TRAINING_TASK_BY_DIRECTIONS.keys())}"\n            )\n\n        task_index, task_text = match\n        return (\n            int(task_index),\n            str(task_text),\n            sand_direction,\n            unload_direction,\n        )\n\n    (\n        deployment_task_index,\n        deployment_task_text,\n        deployment_sand_direction,\n        deployment_unload_direction,\n    ) = select_exact_training_prompt(\n        dig_target_world,\n        unload_landing_world,\n    )\n\n    print(\n        "[PROMPT] Exact training task selected:",\n        f"task_index={deployment_task_index}",\n        f"sand_direction={deployment_sand_direction}",\n        f"unload_direction={deployment_unload_direction}",\n        flush=True,\n    )\n    print(\n        "[PROMPT] task_text:",\n        deployment_task_text,\n        flush=True,\n    )\n\n'
REPLACEMENT_BLOCK = '            # Use one exact prompt from meta/tasks.parquet.\n            task_text = deployment_task_text\n\n'


def replace_exactly_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one match, found {count}"
        )
    return text.replace(old, new, 1)


def patch_file(target):
    target = target.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)

    text = target.read_text(encoding="utf-8")

    if "TRAINING_TASK_BY_DIRECTIONS" in text:
        raise RuntimeError(
            "Exact-training-prompt patch already appears to be present"
        )
    if "dig_target_world" not in text or "unload_landing_world" not in text:
        raise RuntimeError(
            "State27 target helpers are missing. Apply the State27 patch first."
        )

    backup = target.with_suffix(
        target.suffix + ".before_exact_training_prompt"
    )
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[OK] backup: {backup}")

    anchor = "    # TCP Bridge Server functions.\n"
    text = replace_exactly_once(
        text,
        anchor,
        HELPER_CODE + anchor,
        "insert exact prompt selector",
    )

    pattern = re.compile(
        r'            task_text = "Dig soil from the marked area and dump it into the target container\."\n'
        r'            try:\n'
        r'.*?'
        r'            except Exception:\n'
        r'                pass\n\n',
        flags=re.DOTALL,
    )
    text, count = pattern.subn(
        REPLACEMENT_BLOCK,
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(
            "Could not uniquely replace the old dynamic prompt block; "
            f"matches={count}"
        )

    target.write_text(text, encoding="utf-8")
    print(f"[OK] patched: {target}")
    print(
        "[INFO] Prompt now uses one exact training task and is fixed "
        "relative to the initial base pose"
    )
    print(
        "[INFO] An unseen direction pair now raises an error "
        "instead of creating an out-of-distribution prompt"
    )


def restore_file(target):
    target = target.expanduser().resolve()
    backup = target.with_suffix(
        target.suffix + ".before_exact_training_prompt"
    )
    if not backup.is_file():
        raise FileNotFoundError(backup)
    shutil.copy2(backup, target)
    print(f"[OK] restored: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        default=DEFAULT_TARGET,
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    if args.restore:
        restore_file(args.target)
    else:
        patch_file(args.target)


if __name__ == "__main__":
    main()
