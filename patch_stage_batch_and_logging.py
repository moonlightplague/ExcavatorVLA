#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_MODELING = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/policies/smolvla/modeling_smolvla.py"
)
DEFAULT_TRAIN = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/scripts/lerobot_train.py"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count == 0 and new in text:
        print(f"[INFO] Already patched: {label}")
        return text
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one source match, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling", type=Path, default=DEFAULT_MODELING)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    modeling = args.modeling.resolve()
    train = args.train.resolve()

    modeling_backup = modeling.with_suffix(
        modeling.suffix + ".before_stage_batch_fix"
    )
    train_backup = train.with_suffix(
        train.suffix + ".before_stage_metric_logging"
    )

    if args.restore:
        if not modeling_backup.exists() or not train_backup.exists():
            raise FileNotFoundError("Backup files are missing")
        shutil.copy2(modeling_backup, modeling)
        shutil.copy2(train_backup, train)
        print("[OK] Restored both files")
        return

    if not modeling_backup.exists():
        shutil.copy2(modeling, modeling_backup)
        print("[OK] Backup:", modeling_backup)
    if not train_backup.exists():
        shutil.copy2(train, train_backup)
        print("[OK] Backup:", train_backup)

    modeling_text = modeling.read_text(encoding="utf-8")
    modeling_text = replace_once(
        modeling_text,
        '        actions_is_pad = batch.get("actions_id_pad")\n',
        '        actions_is_pad = batch.get("action_is_pad", batch.get("actions_id_pad"))\n',
        "accept actual action_is_pad key",
    )
    modeling.write_text(modeling_text, encoding="utf-8")

    train_text = train.read_text(encoding="utf-8")
    train_text = replace_once(
        train_text,
        """        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
""",
        """        if is_log_step:
            logging.info(train_tracker)
            if output_dict:
                logging.info(
                    "policy_metrics: %s",
                    {
                        key: (
                            round(value, 8)
                            if isinstance(value, float)
                            else value
                        )
                        for key, value in output_dict.items()
                    },
                )
            if wandb_logger:
""",
        "write policy loss metrics to local log",
    )
    train.write_text(train_text, encoding="utf-8")

    print("[OK] Patched:", modeling)
    print("[OK] Patched:", train)


if __name__ == "__main__":
    main()
