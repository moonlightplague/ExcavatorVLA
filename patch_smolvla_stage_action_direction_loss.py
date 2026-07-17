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
    anchor = '''    stage_loss_ema_beta: float = 0.99
    stage_scale_min: float = 0.02
    stage_scale_max: float = 20.0
'''

    replacement = '''    stage_loss_ema_beta: float = 0.99
    stage_scale_min: float = 0.02
    stage_scale_max: float = 20.0

    # Stage-conditioned action-direction auxiliary loss. This uses the
    # ground-truth 50-step stage sequence during fine-tuning only.
    stage_action_prior_path: str | None = None
    stage_action_loss_weight: float = 0.05
    stage_action_min_prior_weight: float = 0.8
    stage_action_loss_ema_beta: float = 0.99
    stage_action_scale_min: float = 0.02
    stage_action_scale_max: float = 20.0
'''

    text = replace_once(
        text,
        anchor,
        replacement,
        "add stage-action configuration",
    )

    validation_anchor = '''        if not 0.0 < self.stage_scale_min <= self.stage_scale_max:
            raise ValueError(
                "Invalid stage scale range: "
                f"{self.stage_scale_min}..{self.stage_scale_max}."
            )
'''

    validation_replacement = '''        if not 0.0 < self.stage_scale_min <= self.stage_scale_max:
            raise ValueError(
                "Invalid stage scale range: "
                f"{self.stage_scale_min}..{self.stage_scale_max}."
            )
        if self.stage_action_loss_weight < 0:
            raise ValueError(
                "`stage_action_loss_weight` must be non-negative, got "
                f"{self.stage_action_loss_weight}."
            )
        if not 0.0 <= self.stage_action_min_prior_weight <= 1.0:
            raise ValueError(
                "`stage_action_min_prior_weight` must be in [0, 1], got "
                f"{self.stage_action_min_prior_weight}."
            )
        if not 0.0 <= self.stage_action_loss_ema_beta < 1.0:
            raise ValueError(
                "`stage_action_loss_ema_beta` must be in [0, 1), got "
                f"{self.stage_action_loss_ema_beta}."
            )
        if not 0.0 < self.stage_action_scale_min <= self.stage_action_scale_max:
            raise ValueError(
                "Invalid stage-action scale range: "
                f"{self.stage_action_scale_min}.."
                f"{self.stage_action_scale_max}."
            )
'''

    return replace_once(
        text,
        validation_anchor,
        validation_replacement,
        "add stage-action validation",
    )


def patch_modeling(text: str) -> str:
    if "import json\n" not in text:
        text = replace_once(
            text,
            "import math\n",
            "import json\nimport math\nfrom pathlib import Path\n",
            "import json and Path",
        )
    elif "from pathlib import Path\n" not in text:
        text = replace_once(
            text,
            "import json\n",
            "import json\nfrom pathlib import Path\n",
            "import Path",
        )

    init_anchor = '''        self.register_buffer(
            "loss_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.reset()
'''

    init_replacement = '''        self.register_buffer(
            "loss_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        action_dim = int(self.config.action_feature.shape[0])
        prior_shape = (self.config.num_stages, action_dim)

        prior_direction = torch.zeros(prior_shape, dtype=torch.float32)
        prior_weight = torch.zeros(prior_shape, dtype=torch.float32)
        prior_strong_mask = torch.zeros(prior_shape, dtype=torch.bool)
        prior_action_scale = torch.ones(action_dim, dtype=torch.float32)
        prior_action_mean = torch.zeros(action_dim, dtype=torch.float32)
        prior_action_std = torch.ones(action_dim, dtype=torch.float32)

        prior_path_value = self.config.stage_action_prior_path
        if prior_path_value:
            prior_path = Path(prior_path_value).expanduser()
            if prior_path.exists():
                with prior_path.open("r", encoding="utf-8") as f:
                    prior_payload = json.load(f)

                prior_direction = torch.tensor(
                    prior_payload["direction"],
                    dtype=torch.float32,
                )
                prior_weight = torch.tensor(
                    prior_payload["weight"],
                    dtype=torch.float32,
                )
                prior_reliable = torch.tensor(
                    prior_payload["reliable"],
                    dtype=torch.bool,
                )
                prior_action_scale = torch.tensor(
                    prior_payload["action_scale"],
                    dtype=torch.float32,
                )
                prior_action_mean = torch.tensor(
                    prior_payload["action_mean"],
                    dtype=torch.float32,
                )
                prior_action_std = torch.tensor(
                    prior_payload["action_std"],
                    dtype=torch.float32,
                )

                if tuple(prior_direction.shape) != prior_shape:
                    raise ValueError(
                        "Stage-action prior direction shape must be "
                        f"{prior_shape}, got {tuple(prior_direction.shape)}."
                    )
                if tuple(prior_weight.shape) != prior_shape:
                    raise ValueError(
                        "Stage-action prior weight shape must be "
                        f"{prior_shape}, got {tuple(prior_weight.shape)}."
                    )

                prior_strong_mask = (
                    prior_reliable
                    & (
                        prior_weight
                        >= self.config.stage_action_min_prior_weight
                    )
                    & (prior_direction != 0)
                )

        self.register_buffer(
            "stage_action_prior_direction",
            prior_direction,
        )
        self.register_buffer(
            "stage_action_prior_weight",
            prior_weight,
        )
        self.register_buffer(
            "stage_action_prior_strong_mask",
            prior_strong_mask,
        )
        self.register_buffer(
            "stage_action_prior_action_scale",
            prior_action_scale.clamp_min(1e-8),
        )
        self.register_buffer(
            "stage_action_prior_action_mean",
            prior_action_mean,
        )
        self.register_buffer(
            "stage_action_prior_action_std",
            prior_action_std.clamp_min(1e-8),
        )
        self.register_buffer(
            "stage_action_raw_loss_ema",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "stage_action_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.reset()
'''

    text = replace_once(
        text,
        init_anchor,
        init_replacement,
        "register stage-action prior buffers",
    )

    inner_return_anchor = '''        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, stage_logits
'''

    inner_return_replacement = '''        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        # x_t = t * noise + (1 - t) * clean_action
        # v_t approximates noise - clean_action, therefore:
        # predicted_clean_action = x_t - t * v_t
        predicted_clean_actions = (
            x_t - time_expanded * v_t
        )
        return losses, stage_logits, predicted_clean_actions
'''

    text = replace_once(
        text,
        inner_return_anchor,
        inner_return_replacement,
        "return predicted clean action",
    )

    policy_call_anchor = '''        losses, stage_logits = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
        )
'''

    policy_call_replacement = '''        (
            losses,
            stage_logits,
            predicted_clean_actions,
        ) = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
        )
'''

    text = replace_once(
        text,
        policy_call_anchor,
        policy_call_replacement,
        "unpack predicted clean action",
    )

    total_anchor = '''        normalized_stage_loss = (
            stage_raw_loss
            * stage_scale.detach().to(stage_raw_loss.dtype)
        )
        weighted_stage_loss = (
            self.config.stage_loss_weight
            * normalized_stage_loss
        )
        total_loss = action_loss + weighted_stage_loss

        per_sample_stage_loss = (
            token_stage_ce.sum(dim=1)
            / stage_valid.sum(dim=1).clamp_min(1)
        )
        per_sample_loss = (
            per_sample_action_loss
            + self.config.stage_loss_weight
            * stage_scale.detach().to(per_sample_stage_loss.dtype)
            * per_sample_stage_loss
        )
'''

    total_replacement = '''        normalized_stage_loss = (
            stage_raw_loss
            * stage_scale.detach().to(stage_raw_loss.dtype)
        )
        weighted_stage_loss = (
            self.config.stage_loss_weight
            * normalized_stage_loss
        )

        # Stage-conditioned direction-only loss.
        predicted_clean_action_normalized = (
            predicted_clean_actions[:, :, :action_dim]
        )
        predicted_clean_action_raw = (
            predicted_clean_action_normalized
            * self.stage_action_prior_action_std
            + self.stage_action_prior_action_mean
        )

        prior_direction = self.stage_action_prior_direction[
            stage_target
        ]
        prior_weight = self.stage_action_prior_weight[
            stage_target
        ]
        prior_mask = self.stage_action_prior_strong_mask[
            stage_target
        ] & stage_valid.unsqueeze(-1)

        signed_raw_action = (
            prior_direction
            * predicted_clean_action_raw
            / self.stage_action_prior_action_scale
        )
        direction_violation = F.relu(
            -signed_raw_action
        ).square()

        effective_prior_weight = (
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

        with torch.no_grad():
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

        normalized_stage_action_loss = (
            stage_action_raw_loss
            * stage_action_scale.detach().to(
                stage_action_raw_loss.dtype
            )
        )
        weighted_stage_action_loss = (
            self.config.stage_action_loss_weight
            * normalized_stage_action_loss
        )

        total_loss = (
            action_loss
            + weighted_stage_loss
            + weighted_stage_action_loss
        )

        per_sample_stage_loss = (
            token_stage_ce.sum(dim=1)
            / stage_valid.sum(dim=1).clamp_min(1)
        )
        per_sample_loss = (
            per_sample_action_loss
            + self.config.stage_loss_weight
            * stage_scale.detach().to(per_sample_stage_loss.dtype)
            * per_sample_stage_loss
            + self.config.stage_action_loss_weight
            * stage_action_scale.detach().to(
                per_sample_stage_action_raw_loss.dtype
            )
            * per_sample_stage_action_raw_loss
        )
'''

    text = replace_once(
        text,
        total_anchor,
        total_replacement,
        "add stage-action direction loss",
    )

    logging_anchor = '''        loss_dict["stage_loss_weight"] = (
            self.config.stage_loss_weight
        )
        loss_dict["loss"] = total_loss.detach().item()
'''

    logging_replacement = '''        loss_dict["stage_loss_weight"] = (
            self.config.stage_loss_weight
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
        loss_dict["loss"] = total_loss.detach().item()
'''

    return replace_once(
        text,
        logging_anchor,
        logging_replacement,
        "log stage-action loss",
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

    backup_suffix = ".before_stage_action_direction"
    config_backup = config_path.with_suffix(
        config_path.suffix + backup_suffix
    )
    modeling_backup = modeling_path.with_suffix(
        modeling_path.suffix + backup_suffix
    )

    if args.restore:
        if not config_backup.exists() or not modeling_backup.exists():
            raise FileNotFoundError(
                "Stage-action direction backup files are missing"
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

    patched_config = patch_configuration(
        config_path.read_text(encoding="utf-8")
    )
    patched_modeling = patch_modeling(
        modeling_path.read_text(encoding="utf-8")
    )

    config_path.write_text(patched_config, encoding="utf-8")
    modeling_path.write_text(patched_modeling, encoding="utf-8")

    print("[OK] Patched:", config_path)
    print("[OK] Patched:", modeling_path)
    print("[INFO] Direction loss uses ground-truth stage labels.")
    print("[INFO] Inference/select_action behavior is unchanged.")


if __name__ == "__main__":
    main()
