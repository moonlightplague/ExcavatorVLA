#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SMOLVLA_DIR = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/policies/smolvla"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def patch_configuration(text: str) -> str:
    if "stage_action_low_motion_weight" in text:
        raise RuntimeError("Low-motion configuration already exists")

    config_anchor = '''    stage_action_loss_ema_beta: float = 0.99
    stage_action_scale_min: float = 0.02
    stage_action_scale_max: float = 20.0
'''

    config_replacement = '''    stage_action_loss_ema_beta: float = 0.99
    stage_action_scale_min: float = 0.02
    stage_action_scale_max: float = 20.0

    # Additional low-motion rule inside the stage-action auxiliary objective.
    #
    # Dataset action order:
    #   0 swing, 1 boom, 2 arm, 3 bucket
    #
    # Strong dataset evidence shows that swing should remain near zero for:
    #   1 approach_contact
    #   2 insert
    #   3 pull_mid
    #   4 pull_exit
    #   5 curl
    #   6 secure_load
    #   7 lift_carry
    #
    # The low-motion component is combined with the existing direction
    # component before the existing stage-action EMA normalization.
    stage_action_low_motion_weight: float = 0.5
    stage_action_low_motion_margin: float = 0.10
    stage_action_low_motion_swing_scale: float = 2.1780849
'''

    text = replace_once(
        text,
        config_anchor,
        config_replacement,
        "add low-motion configuration",
    )

    validation_anchor = '''        if not 0.0 < self.stage_action_scale_min <= self.stage_action_scale_max:
            raise ValueError(
                "Invalid stage-action scale range: "
                f"{self.stage_action_scale_min}.."
                f"{self.stage_action_scale_max}."
            )
'''

    validation_replacement = '''        if not 0.0 < self.stage_action_scale_min <= self.stage_action_scale_max:
            raise ValueError(
                "Invalid stage-action scale range: "
                f"{self.stage_action_scale_min}.."
                f"{self.stage_action_scale_max}."
            )
        if self.stage_action_low_motion_weight < 0:
            raise ValueError(
                "`stage_action_low_motion_weight` must be non-negative, got "
                f"{self.stage_action_low_motion_weight}."
            )
        if self.stage_action_low_motion_margin < 0:
            raise ValueError(
                "`stage_action_low_motion_margin` must be non-negative, got "
                f"{self.stage_action_low_motion_margin}."
            )
        if self.stage_action_low_motion_swing_scale <= 0:
            raise ValueError(
                "`stage_action_low_motion_swing_scale` must be positive, got "
                f"{self.stage_action_low_motion_swing_scale}."
            )
'''

    return replace_once(
        text,
        validation_anchor,
        validation_replacement,
        "add low-motion validation",
    )


def patch_modeling(text: str) -> str:
    if "stage_action_low_motion_raw_loss" in text:
        raise RuntimeError("Low-motion loss already exists")

    init_anchor = """        self.register_buffer(
            "stage_action_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.reset()
"""

    init_replacement = """        self.register_buffer(
            "stage_action_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )
        # Not persisted in checkpoints: on the first batch of each process,
        # bootstrap the EMA from the new combined objective. This avoids
        # reusing a direction-only EMA when resuming an older checkpoint.
        self.register_buffer(
            "stage_action_combined_ema_bootstrapped",
            torch.tensor(False, dtype=torch.bool),
            persistent=False,
        )

        self.reset()
"""

    text = replace_once(
        text,
        init_anchor,
        init_replacement,
        "add combined-objective EMA bootstrap marker",
    )

    old_loss_block = '''        effective_prior_weight = (
            prior_weight
            * prior_mask.to(prior_weight.dtype)
        )
        prior_weight_sum = effective_prior_weight.sum()

        if bool((prior_weight_sum > 0).item()):
            stage_action_raw_loss = (
                direction_violation
                * effective_prior_weight
            ).sum() / prior_weight_sum

            per_sample_prior_weight_sum = (
                effective_prior_weight.sum(dim=(1, 2))
            )
            per_sample_stage_action_raw_loss = (
                (
                    direction_violation
                    * effective_prior_weight
                ).sum(dim=(1, 2))
                / per_sample_prior_weight_sum.clamp_min(1e-8)
            )
        else:
            stage_action_raw_loss = (
                predicted_clean_action_raw.sum() * 0.0
            )
            per_sample_stage_action_raw_loss = torch.zeros_like(
                per_sample_action_loss
            )
'''

    new_loss_block = '''        effective_prior_weight = (
            prior_weight
            * prior_mask.to(prior_weight.dtype)
        )
        prior_weight_sum = effective_prior_weight.sum()

        # Existing sign/direction component.
        if bool((prior_weight_sum > 0).item()):
            stage_action_direction_raw_loss = (
                direction_violation
                * effective_prior_weight
            ).sum() / prior_weight_sum

            per_sample_prior_weight_sum = (
                effective_prior_weight.sum(dim=(1, 2))
            )
            per_sample_stage_action_direction_raw_loss = (
                (
                    direction_violation
                    * effective_prior_weight
                ).sum(dim=(1, 2))
                / per_sample_prior_weight_sum.clamp_min(1e-8)
            )
        else:
            stage_action_direction_raw_loss = (
                predicted_clean_action_raw.sum() * 0.0
            )
            per_sample_stage_action_direction_raw_loss = (
                torch.zeros_like(per_sample_action_loss)
            )

        # Dataset-validated low-motion component.
        #
        # Action order is [swing, boom, arm, bucket], so index 0 is swing.
        # Only stages 1..7 are constrained. Stages 0, 8, and 9 need genuine
        # swing motion and are deliberately excluded.
        predicted_swing_raw = predicted_clean_action_raw[:, :, 0]
        normalized_abs_swing = (
            predicted_swing_raw.abs()
            / self.config.stage_action_low_motion_swing_scale
        )

        low_motion_mask = (
            stage_valid
            & (stage_target >= 1)
            & (stage_target <= 7)
        )
        low_motion_excess = F.relu(
            normalized_abs_swing
            - self.config.stage_action_low_motion_margin
        )
        low_motion_violation = low_motion_excess.square()

        low_motion_count = low_motion_mask.sum()
        if bool((low_motion_count > 0).item()):
            stage_action_low_motion_raw_loss = (
                low_motion_violation
                * low_motion_mask.to(low_motion_violation.dtype)
            ).sum() / low_motion_count

            per_sample_low_motion_count = low_motion_mask.sum(dim=1)
            per_sample_stage_action_low_motion_raw_loss = (
                (
                    low_motion_violation
                    * low_motion_mask.to(low_motion_violation.dtype)
                ).sum(dim=1)
                / per_sample_low_motion_count.clamp_min(1)
            )
        else:
            stage_action_low_motion_raw_loss = (
                predicted_swing_raw.sum() * 0.0
            )
            per_sample_stage_action_low_motion_raw_loss = (
                torch.zeros_like(per_sample_action_loss)
            )

        # Keep the original sign loss and add only a conservative fraction
        # of the low-motion loss. The existing outer stage_action_loss_weight
        # and EMA normalization still control the total auxiliary magnitude.
        stage_action_raw_loss = (
            stage_action_direction_raw_loss
            + self.config.stage_action_low_motion_weight
            * stage_action_low_motion_raw_loss
        )
        per_sample_stage_action_raw_loss = (
            per_sample_stage_action_direction_raw_loss
            + self.config.stage_action_low_motion_weight
            * per_sample_stage_action_low_motion_raw_loss
        )
'''

    text = replace_once(
        text,
        old_loss_block,
        new_loss_block,
        "replace direction-only raw loss with combined loss",
    )

    old_ema_block = """        with torch.no_grad():
            if bool((prior_weight_sum > 0).item()):
                beta = self.config.stage_action_loss_ema_beta
                if not bool(
                    self.stage_action_ema_initialized.item()
                ):
                    self.stage_action_raw_loss_ema.copy_(
                        stage_action_raw_loss.detach().float()
                    )
                    self.stage_action_ema_initialized.fill_(True)
                else:
                    self.stage_action_raw_loss_ema.mul_(beta).add_(
                        stage_action_raw_loss.detach().float(),
                        alpha=1.0 - beta,
                    )

            if bool(self.stage_action_ema_initialized.item()):
                stage_action_scale = (
                    self.action_loss_ema
                    / self.stage_action_raw_loss_ema.clamp_min(1e-8)
                ).clamp(
                    min=self.config.stage_action_scale_min,
                    max=self.config.stage_action_scale_max,
                )
            else:
                stage_action_scale = action_loss.new_tensor(1.0)
"""

    new_ema_block = """        with torch.no_grad():
            stage_action_has_signal = (
                (prior_weight_sum > 0)
                | (low_motion_count > 0)
            )
            if bool(stage_action_has_signal.item()):
                beta = self.config.stage_action_loss_ema_beta

                # Always bootstrap once from the combined direction +
                # low-motion objective. This is important when resuming a
                # checkpoint whose saved EMA was direction-only.
                if not bool(
                    self.stage_action_combined_ema_bootstrapped.item()
                ):
                    self.stage_action_raw_loss_ema.copy_(
                        stage_action_raw_loss.detach().float()
                    )
                    self.stage_action_ema_initialized.fill_(True)
                    self.stage_action_combined_ema_bootstrapped.fill_(True)
                else:
                    self.stage_action_raw_loss_ema.mul_(beta).add_(
                        stage_action_raw_loss.detach().float(),
                        alpha=1.0 - beta,
                    )

            if bool(self.stage_action_ema_initialized.item()):
                stage_action_scale = (
                    self.action_loss_ema
                    / self.stage_action_raw_loss_ema.clamp_min(1e-8)
                ).clamp(
                    min=self.config.stage_action_scale_min,
                    max=self.config.stage_action_scale_max,
                )
            else:
                stage_action_scale = action_loss.new_tensor(1.0)
"""

    text = replace_once(
        text,
        old_ema_block,
        new_ema_block,
        "bootstrap EMA for combined stage-action objective",
    )

    old_logging_block = '''        loss_dict["stage_action_raw_loss"] = (
            stage_action_raw_loss.detach().item()
        )
        loss_dict["stage_action_scale"] = (
            stage_action_scale.detach().item()
        )
        loss_dict["normalized_stage_action_loss"] = (
            normalized_stage_action_loss.detach().item()
        )
        loss_dict["weighted_stage_action_loss"] = (
            weighted_stage_action_loss.detach().item()
        )
        loss_dict["stage_action_loss_weight"] = (
            self.config.stage_action_loss_weight
        )
        loss_dict["stage_action_prior_coverage"] = (
            prior_mask.float().mean().detach().item()
        )
        loss_dict["stage_action_violation_rate"] = (
            (
                prior_mask
                & (signed_raw_action < 0)
            ).sum().float()
            / prior_mask.sum().clamp_min(1)
        ).detach().item()
        loss_dict["predicted_clean_action_abs_mean"] = (
            predicted_clean_action_raw.abs().mean().detach().item()
        )
'''

    new_logging_block = '''        loss_dict["stage_action_direction_raw_loss"] = (
            stage_action_direction_raw_loss.detach().item()
        )
        loss_dict["stage_action_low_motion_raw_loss"] = (
            stage_action_low_motion_raw_loss.detach().item()
        )
        loss_dict["stage_action_low_motion_weighted_raw_loss"] = (
            (
                self.config.stage_action_low_motion_weight
                * stage_action_low_motion_raw_loss
            ).detach().item()
        )
        loss_dict["stage_action_raw_loss"] = (
            stage_action_raw_loss.detach().item()
        )
        loss_dict["stage_action_scale"] = (
            stage_action_scale.detach().item()
        )
        loss_dict["normalized_stage_action_loss"] = (
            normalized_stage_action_loss.detach().item()
        )
        loss_dict["weighted_stage_action_loss"] = (
            weighted_stage_action_loss.detach().item()
        )
        loss_dict["stage_action_loss_weight"] = (
            self.config.stage_action_loss_weight
        )
        loss_dict["stage_action_low_motion_weight"] = (
            self.config.stage_action_low_motion_weight
        )
        loss_dict["stage_action_low_motion_margin"] = (
            self.config.stage_action_low_motion_margin
        )
        loss_dict["stage_action_prior_coverage"] = (
            prior_mask.float().mean().detach().item()
        )
        loss_dict["stage_action_violation_rate"] = (
            (
                prior_mask
                & (signed_raw_action < 0)
            ).sum().float()
            / prior_mask.sum().clamp_min(1)
        ).detach().item()
        loss_dict["stage_action_low_motion_coverage"] = (
            low_motion_mask.float().mean().detach().item()
        )
        loss_dict["stage_action_low_motion_violation_rate"] = (
            (
                low_motion_mask
                & (
                    normalized_abs_swing
                    > self.config.stage_action_low_motion_margin
                )
            ).sum().float()
            / low_motion_mask.sum().clamp_min(1)
        ).detach().item()
        loss_dict["stage_action_low_motion_abs_swing_mean"] = (
            (
                normalized_abs_swing
                * low_motion_mask.to(normalized_abs_swing.dtype)
            ).sum()
            / low_motion_mask.sum().clamp_min(1)
        ).detach().item()
        loss_dict["stage_action_low_motion_excess_mean"] = (
            (
                low_motion_excess
                * low_motion_mask.to(low_motion_excess.dtype)
            ).sum()
            / low_motion_mask.sum().clamp_min(1)
        ).detach().item()
        loss_dict["predicted_clean_action_abs_mean"] = (
            predicted_clean_action_raw.abs().mean().detach().item()
        )
'''

    return replace_once(
        text,
        old_logging_block,
        new_logging_block,
        "add low-motion logging",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smolvla-dir",
        type=Path,
        default=DEFAULT_SMOLVLA_DIR,
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    root = args.smolvla_dir.expanduser().resolve()
    config_path = root / "configuration_smolvla.py"
    modeling_path = root / "modeling_smolvla.py"

    backup_suffix = ".before_stage_action_low_motion"
    config_backup = config_path.with_suffix(
        config_path.suffix + backup_suffix
    )
    modeling_backup = modeling_path.with_suffix(
        modeling_path.suffix + backup_suffix
    )

    if args.restore:
        if not config_backup.exists() or not modeling_backup.exists():
            raise FileNotFoundError(
                "Low-motion backup files are missing"
            )
        shutil.copy2(config_backup, config_path)
        shutil.copy2(modeling_backup, modeling_path)
        print("[OK] Restored configuration and modeling files")
        return

    for source, backup in (
        (config_path, config_backup),
        (modeling_path, modeling_backup),
    ):
        if not source.exists():
            raise FileNotFoundError(source)
        if not backup.exists():
            shutil.copy2(source, backup)
            print("[OK] Backup:", backup)

    config_text = config_path.read_text(encoding="utf-8")
    modeling_text = modeling_path.read_text(encoding="utf-8")

    if "stage_action_prior_direction" not in modeling_text:
        raise RuntimeError(
            "The existing stage-action direction patch is not present. "
            "Apply patch_smolvla_stage_action_direction_loss.py first."
        )

    patched_config = patch_configuration(config_text)
    patched_modeling = patch_modeling(modeling_text)

    config_path.write_text(patched_config, encoding="utf-8")
    modeling_path.write_text(patched_modeling, encoding="utf-8")

    print("[OK] Patched:", config_path)
    print("[OK] Patched:", modeling_path)
    print("[INFO] Existing direction loss is preserved.")
    print("[INFO] Added swing low-motion loss for ground-truth stages 1..7.")
    print("[INFO] Inference/select_action behavior is unchanged.")
    print("[INFO] Restart or resume training for the patch to take effect.")


if __name__ == "__main__":
    main()
