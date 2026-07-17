#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


DEFAULT_DIR = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/policies/smolvla"
)


def replace_regex_once(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return updated


def patch_config(text: str) -> str:
    text, count = re.subn(
        r'stage_target_key: str = "[^"]+"',
        'stage_target_key: str = "observation.stage_sequence_50"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update stage_target_key")

    text, count = re.subn(
        r'stage_valid_key: str = "[^"]+"',
        'stage_valid_key: str = "observation.stage_sequence_valid_50"',
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not update stage_valid_key")

    return text


def patch_modeling(text: str) -> str:
    head_pattern = r"""        # Predict the future execution stage from the final valid prefix token\.
        # The prefix contains image, language, and padded robot-state context\.
        prefix_hidden_size = self\.vlm_with_expert\.config\.text_config\.hidden_size
        self\.stage_head = nn\.Sequential\(
            nn\.LayerNorm\(prefix_hidden_size\),
            nn\.Linear\(prefix_hidden_size, self\.config\.num_stages\),
        \)
"""
    head_replacement = """        # Predict one stage for every action token in the 50-step chunk.
        # suffix_out has shape [B, chunk_size, expert_hidden_size].
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size
        self.stage_head = nn.Sequential(
            nn.LayerNorm(expert_hidden_size),
            nn.Linear(expert_hidden_size, self.config.num_stages),
        )
"""
    text = replace_regex_once(
        text,
        head_pattern,
        head_replacement,
        "replace stage head",
    )

    helper_pattern = r"""    def _stage_logits_from_prefix\(
        self,
        prefix_out: Tensor,
        prefix_pad_masks: Tensor,
    \) -> Tensor:
        \"\"\"Classify stage from the final non-padding prefix token\.\"\"\"
        last_token_index = \(
            prefix_pad_masks\.to\(dtype=torch\.long\)\.sum\(dim=1\)\.clamp_min\(1\) - 1
        \)
        batch_index = torch\.arange\(
            prefix_out\.shape\[0\],
            device=prefix_out\.device,
        \)
        pooled_prefix = prefix_out\[batch_index, last_token_index\]
        head_dtype = self\.stage_head\[1\]\.weight\.dtype
        pooled_prefix = pooled_prefix\.to\(dtype=head_dtype\)
        return self\.stage_head\(pooled_prefix\)

"""
    text = replace_regex_once(
        text,
        helper_pattern,
        "",
        "remove scalar stage pooling helper",
    )

    forward_pattern = r"""        \(prefix_out, suffix_out\), _ = self\.vlm_with_expert\.forward\(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=\[prefix_embs, suffix_embs\],
            use_cache=False,
            fill_kv_cache=False,
        \)
        stage_logits = self\._stage_logits_from_prefix\(
            prefix_out,
            prefix_pad_masks,
        \)
        suffix_out = suffix_out\[:, -self\.config\.chunk_size :\]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out\.to\(dtype=torch\.float32\)
        v_t = self\.action_out_proj\(suffix_out\)
        losses = F\.mse_loss\(u_t, v_t, reduction="none"\)
        return losses, stage_logits
"""
    forward_replacement = """        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)

        head_dtype = self.stage_head[1].weight.dtype
        stage_logits = self.stage_head(
            suffix_out.to(dtype=head_dtype)
        ).to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, stage_logits
"""
    text = replace_regex_once(
        text,
        forward_pattern,
        forward_replacement,
        "make stage logits a 50-step sequence",
    )

    loss_pattern = r"""        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses \* in_episode_bound\.unsqueeze\(-1\)
            loss_dict\["losses_after_in_ep_bound"\] = losses\.clone\(\)\.mean\(\)\.item\(\)

        # Remove padding
        losses = losses\[:, :, : self\.config\.max_action_dim\]
        loss_dict\["losses_after_rm_padding"\] = losses\.clone\(\)\.mean\(\)\.item\(\)

        # Action flow-matching loss per sample\.
.*?        return total_loss, loss_dict
"""
    loss_replacement = """        # Keep only the real excavator action dimensions.
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        loss_dict["losses_after_rm_padding"] = (
            losses.detach().mean().item()
        )

        if actions_is_pad is None:
            action_valid = torch.ones(
                losses.shape[:2],
                dtype=torch.bool,
                device=losses.device,
            )
        else:
            action_valid = ~actions_is_pad.to(
                device=losses.device,
                dtype=torch.bool,
            )

        action_mask = action_valid.unsqueeze(-1).to(losses.dtype)
        action_numerator = (losses * action_mask).sum()
        action_denominator = (
            action_valid.sum().clamp_min(1) * action_dim
        )
        action_loss = action_numerator / action_denominator

        per_sample_action_denominator = (
            action_valid.sum(dim=1).clamp_min(1) * action_dim
        )
        per_sample_action_loss = (
            (losses * action_mask).sum(dim=(1, 2))
            / per_sample_action_denominator
        )

        if self.config.stage_target_key not in batch:
            raise KeyError(
                f"Missing stage target field: {self.config.stage_target_key}. "
                f"Available batch keys: {sorted(batch.keys())}"
            )

        stage_target = batch[self.config.stage_target_key].to(
            device=stage_logits.device,
            dtype=torch.long,
        )
        stage_valid = batch[self.config.stage_valid_key].to(
            device=stage_logits.device,
            dtype=torch.bool,
        )

        expected_stage_shape = (
            stage_logits.shape[0],
            self.config.chunk_size,
        )
        if tuple(stage_target.shape) != expected_stage_shape:
            raise ValueError(
                "Stage target must have shape "
                f"{expected_stage_shape}, got {tuple(stage_target.shape)}"
            )
        if tuple(stage_valid.shape) != expected_stage_shape:
            raise ValueError(
                "Stage validity must have shape "
                f"{expected_stage_shape}, got {tuple(stage_valid.shape)}"
            )

        stage_valid = stage_valid & action_valid
        valid_flat = stage_valid.reshape(-1)
        target_flat = stage_target.reshape(-1)
        logits_flat = stage_logits.reshape(
            -1,
            self.config.num_stages,
        )

        token_stage_loss = torch.zeros(
            stage_target.shape,
            dtype=stage_logits.dtype,
            device=stage_logits.device,
        )

        if valid_flat.any():
            valid_stage_losses = F.cross_entropy(
                logits_flat[valid_flat],
                target_flat[valid_flat],
                reduction="none",
            )
            token_stage_loss.reshape(-1)[valid_flat] = valid_stage_losses
            stage_loss = valid_stage_losses.mean()

            with torch.no_grad():
                stage_prediction = logits_flat[valid_flat].argmax(dim=-1)
                stage_accuracy = (
                    stage_prediction == target_flat[valid_flat]
                ).float().mean()
        else:
            stage_loss = stage_logits.sum() * 0.0
            stage_accuracy = stage_logits.new_tensor(0.0)

        per_sample_stage_denominator = (
            stage_valid.sum(dim=1).clamp_min(1)
        )
        per_sample_stage_loss = (
            token_stage_loss.sum(dim=1)
            / per_sample_stage_denominator
        )

        per_sample_loss = (
            per_sample_action_loss
            + self.config.stage_loss_weight * per_sample_stage_loss
        )
        total_loss = (
            action_loss
            + self.config.stage_loss_weight * stage_loss
        )

        loss_dict["action_dim"] = action_dim
        loss_dict["action_loss"] = action_loss.detach().item()
        loss_dict["stage_loss"] = stage_loss.detach().item()
        loss_dict["weighted_stage_loss"] = (
            self.config.stage_loss_weight * stage_loss.detach().item()
        )
        loss_dict["stage_accuracy"] = stage_accuracy.detach().item()
        loss_dict["stage_valid_fraction"] = (
            stage_valid.float().mean().detach().item()
        )
        loss_dict["stage_loss_weight"] = self.config.stage_loss_weight
        loss_dict["loss"] = total_loss.detach().item()

        if reduction == "none":
            return per_sample_loss, loss_dict

        if reduction != "mean":
            raise ValueError(
                f"Unsupported reduction: {reduction!r}"
            )

        return total_loss, loss_dict
"""
    text = replace_regex_once(
        text,
        loss_pattern,
        loss_replacement,
        "replace scalar stage and padded action loss",
        flags=re.DOTALL,
    )

    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    root = args.smolvla_dir.expanduser().resolve()
    config_path = root / "configuration_smolvla.py"
    modeling_path = root / "modeling_smolvla.py"

    config_backup = config_path.with_suffix(
        config_path.suffix + ".before_stage_seq50"
    )
    modeling_backup = modeling_path.with_suffix(
        modeling_path.suffix + ".before_stage_seq50"
    )

    if args.restore:
        if not config_backup.exists() or not modeling_backup.exists():
            raise FileNotFoundError("Sequence-stage backup files are missing")
        shutil.copy2(config_backup, config_path)
        shutil.copy2(modeling_backup, modeling_path)
        print("[OK] Restored scalar-stage version")
        return

    if not config_backup.exists():
        shutil.copy2(config_path, config_backup)
        print("[OK] Backup:", config_backup)
    if not modeling_backup.exists():
        shutil.copy2(modeling_path, modeling_backup)
        print("[OK] Backup:", modeling_backup)

    config_text = patch_config(config_path.read_text(encoding="utf-8"))
    modeling_text = patch_modeling(modeling_path.read_text(encoding="utf-8"))

    config_path.write_text(config_text, encoding="utf-8")
    modeling_path.write_text(modeling_text, encoding="utf-8")

    print("[OK] Updated:", config_path)
    print("[OK] Updated:", modeling_path)
    print("[INFO] Stage logits now have shape [B, 50, 10].")
    print("[INFO] Action loss now uses only the real action dimensions.")


if __name__ == "__main__":
    main()
