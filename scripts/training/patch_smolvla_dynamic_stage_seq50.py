#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


SMOLVLA_DIR = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/policies/smolvla"
)
FACTORY_PATH = Path(
    "/opt/conda/envs/smolvla/lib/python3.10/site-packages/"
    "lerobot/datasets/factory.py"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return text.replace(old, new, 1)


def replace_regex_once(
    text: str,
    pattern: str,
    replacement: str,
    label: str,
    flags: int = 0,
) -> str:
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=flags,
    )
    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}"
        )
    return updated


def patch_configuration(text: str) -> str:
    pattern = r'''    # Auxiliary stage-classification task\.
    # These dataset fields are supervision labels, not policy input features\.
    num_stages: int = 10
    stage_loss_weight: float = [^\n]+
    stage_target_key: str = "[^"]+"
    stage_valid_key: str = "[^"]+"
'''

    replacement = '''    # Auxiliary 50-step stage-sequence task.
    # The stored dataset keeps one scalar stage ID per frame. The dataset
    # loader dynamically queries the next `chunk_size` timestamps.
    num_stages: int = 10
    stage_source_key: str = "observation.stage_current_id"

    # Total objective:
    # action_loss + stage_loss_weight * normalized_stage_loss
    stage_loss_weight: float = 1.0

    # Penalize isolated predictions inside a constant ground-truth segment.
    stage_exception_weight: float = 0.25

    # EMA normalization keeps the stage contribution near the action-loss
    # scale while preventing extreme rescaling.
    stage_loss_ema_beta: float = 0.99
    stage_scale_min: float = 0.02
    stage_scale_max: float = 20.0
'''

    text = replace_regex_once(
        text,
        pattern,
        replacement,
        "replace scalar stage configuration",
    )

    validation_anchor = '''        if self.num_stages <= 1:
            raise ValueError(
                f"`num_stages` must be greater than 1, got {self.num_stages}."
            )
        if self.stage_loss_weight < 0:
            raise ValueError(
                f"`stage_loss_weight` must be non-negative, got {self.stage_loss_weight}."
            )
'''

    validation_replacement = '''        if self.num_stages <= 1:
            raise ValueError(
                f"`num_stages` must be greater than 1, got {self.num_stages}."
            )
        if self.stage_loss_weight < 0:
            raise ValueError(
                f"`stage_loss_weight` must be non-negative, got {self.stage_loss_weight}."
            )
        if self.stage_exception_weight < 0:
            raise ValueError(
                "`stage_exception_weight` must be non-negative, got "
                f"{self.stage_exception_weight}."
            )
        if not 0.0 <= self.stage_loss_ema_beta < 1.0:
            raise ValueError(
                "`stage_loss_ema_beta` must be in [0, 1), got "
                f"{self.stage_loss_ema_beta}."
            )
        if not 0.0 < self.stage_scale_min <= self.stage_scale_max:
            raise ValueError(
                "Invalid stage scale range: "
                f"{self.stage_scale_min}..{self.stage_scale_max}."
            )
'''

    text = replace_once(
        text,
        validation_anchor,
        validation_replacement,
        "extend stage validation",
    )

    action_property = '''    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))
'''

    action_and_stage_property = '''    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def stage_delta_indices(self) -> list:
        return list(range(self.chunk_size))
'''

    return replace_once(
        text,
        action_property,
        action_and_stage_property,
        "add stage_delta_indices",
    )


def patch_factory(text: str) -> str:
    old = '''        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
'''

    new = '''        if key == ACTION and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]

        # The stage label is stored once per frame, but training requests the
        # next 50 labels so they align one-to-one with the action chunk.
        if (
            key == getattr(cfg, "stage_source_key", None)
            and getattr(cfg, "stage_delta_indices", None) is not None
        ):
            delta_timestamps[key] = [
                i / ds_meta.fps for i in cfg.stage_delta_indices
            ]
        elif key.startswith(OBS_PREFIX) and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]
'''

    return replace_once(
        text,
        old,
        new,
        "add dynamic stage delta timestamps",
    )


def patch_modeling(text: str) -> str:
    old_policy_init = '''        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()
'''

    new_policy_init = '''        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)

        self.register_buffer(
            "action_loss_ema",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "stage_raw_loss_ema",
            torch.tensor(0.0, dtype=torch.float32),
        )
        self.register_buffer(
            "loss_ema_initialized",
            torch.tensor(False, dtype=torch.bool),
        )

        self.reset()
'''

    text = replace_once(
        text,
        old_policy_init,
        new_policy_init,
        "register loss-normalization buffers",
    )

    old_stage_head = '''        # Predict the future execution stage from the final valid prefix token.
        # The prefix contains image, language, and padded robot-state context.
        prefix_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.stage_head = nn.Sequential(
            nn.LayerNorm(prefix_hidden_size),
            nn.Linear(prefix_hidden_size, self.config.num_stages),
        )
'''

    new_stage_head = '''        # Predict one stage for every action token in the chunk.
        expert_hidden_size = self.vlm_with_expert.expert_hidden_size
        self.stage_head = nn.Sequential(
            nn.LayerNorm(expert_hidden_size),
            nn.Linear(expert_hidden_size, self.config.num_stages),
        )
'''

    text = replace_once(
        text,
        old_stage_head,
        new_stage_head,
        "replace scalar stage head",
    )

    helper_pattern = r'''    def _stage_logits_from_prefix\(
        self,
        prefix_out: Tensor,
        prefix_pad_masks: Tensor,
    \) -> Tensor:
        ["]{3}Classify stage from the final non-padding prefix token\.["]{3}
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

'''

    text = replace_regex_once(
        text,
        helper_pattern,
        "",
        "remove prefix stage pooling helper",
    )

    old_inner_forward = '''        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        stage_logits = self._stage_logits_from_prefix(
            prefix_out,
            prefix_pad_masks,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, stage_logits
'''

    new_inner_forward = '''        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        stage_head_dtype = self.stage_head[1].weight.dtype
        stage_logits = self.stage_head(
            suffix_out.to(dtype=stage_head_dtype)
        ).to(dtype=torch.float32)

        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, stage_logits
'''

    text = replace_once(
        text,
        old_inner_forward,
        new_inner_forward,
        "produce 50-step stage logits",
    )

    loss_pattern = r'''        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses \* in_episode_bound\.unsqueeze\(-1\)
            loss_dict\["losses_after_in_ep_bound"\] = losses\.clone\(\)\.mean\(\)\.item\(\)

        # Remove padding
        losses = losses\[:, :, : self\.config\.max_action_dim\]
        loss_dict\["losses_after_rm_padding"\] = losses\.clone\(\)\.mean\(\)\.item\(\)

        # Action flow-matching loss per sample\.
.*?        return total_loss, loss_dict
'''

    new_loss_block = '''        # Action loss: only real excavator action dimensions.
        action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :action_dim]
        loss_dict["losses_after_rm_action_dim_padding"] = (
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
        action_loss = (
            (losses * action_mask).sum()
            / (action_valid.sum().clamp_min(1) * action_dim)
        )
        per_sample_action_loss = (
            (losses * action_mask).sum(dim=(1, 2))
            / (action_valid.sum(dim=1).clamp_min(1) * action_dim)
        )

        # Dynamically sampled 50-step stage target.
        stage_key = self.config.stage_source_key
        stage_pad_key = f"{stage_key}_is_pad"

        if stage_key not in batch:
            raise KeyError(
                f"Missing stage field: {stage_key}. "
                f"Available batch keys: {sorted(batch.keys())}"
            )

        stage_target = batch[stage_key].to(
            device=stage_logits.device,
            dtype=torch.long,
        )
        expected_shape = (
            stage_logits.shape[0],
            self.config.chunk_size,
        )
        if tuple(stage_target.shape) != expected_shape:
            raise ValueError(
                f"Expected stage target {expected_shape}, "
                f"got {tuple(stage_target.shape)}."
            )

        stage_is_pad = batch.get(stage_pad_key)
        if stage_is_pad is None:
            stage_valid = action_valid
        else:
            stage_valid = (
                ~stage_is_pad.to(
                    device=stage_logits.device,
                    dtype=torch.bool,
                )
            ) & action_valid

        valid_flat = stage_valid.reshape(-1)
        target_flat = stage_target.reshape(-1)
        logits_flat = stage_logits.reshape(
            -1,
            self.config.num_stages,
        )

        token_stage_ce = torch.zeros(
            stage_target.shape,
            dtype=stage_logits.dtype,
            device=stage_logits.device,
        )

        if valid_flat.any():
            valid_stage_ce = F.cross_entropy(
                logits_flat[valid_flat],
                target_flat[valid_flat],
                reduction="none",
            )
            token_stage_ce.reshape(-1)[valid_flat] = valid_stage_ce
            stage_ce_loss = valid_stage_ce.mean()

            with torch.no_grad():
                stage_prediction = logits_flat[valid_flat].argmax(dim=-1)
                stage_accuracy = (
                    stage_prediction == target_flat[valid_flat]
                ).float().mean()
        else:
            stage_ce_loss = stage_logits.sum() * 0.0
            stage_accuracy = stage_logits.new_tensor(0.0)

        # Extra penalty for an isolated prediction inside a constant
        # ground-truth stage segment.
        stage_probability = torch.softmax(stage_logits, dim=-1)
        same_triplet = (
            stage_valid[:, :-2]
            & stage_valid[:, 1:-1]
            & stage_valid[:, 2:]
            & (stage_target[:, :-2] == stage_target[:, 1:-1])
            & (stage_target[:, 1:-1] == stage_target[:, 2:])
        )

        neighbor_probability = 0.5 * (
            stage_probability[:, :-2]
            + stage_probability[:, 2:]
        )
        center_probability = stage_probability[:, 1:-1]
        exception_per_token = (
            center_probability
            - neighbor_probability.detach()
        ).square().sum(dim=-1)

        if same_triplet.any():
            stage_exception_loss = (
                exception_per_token[same_triplet].mean()
            )
        else:
            stage_exception_loss = stage_logits.sum() * 0.0

        stage_raw_loss = (
            stage_ce_loss
            + self.config.stage_exception_weight
            * stage_exception_loss
        )

        # EMA scale normalization: stage contribution tracks action-loss scale.
        with torch.no_grad():
            beta = self.config.stage_loss_ema_beta
            if not bool(self.loss_ema_initialized.item()):
                self.action_loss_ema.copy_(
                    action_loss.detach().float()
                )
                self.stage_raw_loss_ema.copy_(
                    stage_raw_loss.detach().float()
                )
                self.loss_ema_initialized.fill_(True)
            else:
                self.action_loss_ema.mul_(beta).add_(
                    action_loss.detach().float(),
                    alpha=1.0 - beta,
                )
                self.stage_raw_loss_ema.mul_(beta).add_(
                    stage_raw_loss.detach().float(),
                    alpha=1.0 - beta,
                )

            stage_scale = (
                self.action_loss_ema
                / self.stage_raw_loss_ema.clamp_min(1e-8)
            ).clamp(
                min=self.config.stage_scale_min,
                max=self.config.stage_scale_max,
            )

        normalized_stage_loss = (
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

        with torch.no_grad():
            predicted_stage = stage_logits.argmax(dim=-1)
            isolated_exception = (
                same_triplet
                & (predicted_stage[:, 1:-1] != predicted_stage[:, :-2])
                & (predicted_stage[:, 1:-1] != predicted_stage[:, 2:])
            )
            stage_exception_rate = (
                isolated_exception.sum().float()
                / same_triplet.sum().clamp_min(1)
            )

        loss_dict["action_dim"] = action_dim
        loss_dict["action_loss"] = action_loss.detach().item()
        loss_dict["stage_ce_loss"] = stage_ce_loss.detach().item()
        loss_dict["stage_exception_loss"] = (
            stage_exception_loss.detach().item()
        )
        loss_dict["stage_exception_weight"] = (
            self.config.stage_exception_weight
        )
        loss_dict["stage_raw_loss"] = stage_raw_loss.detach().item()
        loss_dict["stage_scale"] = stage_scale.detach().item()
        loss_dict["normalized_stage_loss"] = (
            normalized_stage_loss.detach().item()
        )
        loss_dict["weighted_stage_loss"] = (
            weighted_stage_loss.detach().item()
        )
        loss_dict["stage_accuracy"] = stage_accuracy.detach().item()
        loss_dict["stage_valid_fraction"] = (
            stage_valid.float().mean().detach().item()
        )
        loss_dict["stage_exception_rate"] = (
            stage_exception_rate.detach().item()
        )
        loss_dict["stage_loss_weight"] = (
            self.config.stage_loss_weight
        )
        loss_dict["loss"] = total_loss.detach().item()

        if reduction == "none":
            return per_sample_loss, loss_dict

        if reduction != "mean":
            raise ValueError(
                f"Unsupported reduction: {reduction!r}"
            )

        return total_loss, loss_dict
'''

    return replace_regex_once(
        text,
        loss_pattern,
        new_loss_block,
        "replace action and stage loss",
        flags=re.DOTALL,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smolvla-dir",
        type=Path,
        default=SMOLVLA_DIR,
    )
    parser.add_argument(
        "--factory",
        type=Path,
        default=FACTORY_PATH,
    )
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    smolvla_dir = args.smolvla_dir.expanduser().resolve()
    factory_path = args.factory.expanduser().resolve()
    config_path = smolvla_dir / "configuration_smolvla.py"
    modeling_path = smolvla_dir / "modeling_smolvla.py"

    paths = (config_path, modeling_path, factory_path)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

    suffix = ".before_dynamic_stage_seq50"
    backups = {
        path: path.with_suffix(path.suffix + suffix)
        for path in paths
    }

    if args.restore:
        for path, source in backups.items():
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, path)
            print(f"[OK] Restored: {path}")
        return

    for path, backup in backups.items():
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"[OK] Backup created: {backup}")
        else:
            print(f"[INFO] Backup already exists: {backup}")

    # Compute all edits first so a failed match cannot leave a partial patch.
    patched_config = patch_configuration(
        config_path.read_text(encoding="utf-8")
    )
    patched_modeling = patch_modeling(
        modeling_path.read_text(encoding="utf-8")
    )
    patched_factory = patch_factory(
        factory_path.read_text(encoding="utf-8")
    )

    config_path.write_text(patched_config, encoding="utf-8")
    modeling_path.write_text(patched_modeling, encoding="utf-8")
    factory_path.write_text(patched_factory, encoding="utf-8")

    print(f"[OK] Patched: {config_path}")
    print(f"[OK] Patched: {modeling_path}")
    print(f"[OK] Patched: {factory_path}")
    print("[INFO] Dataset files were not modified.")
    print("[INFO] Stage target is dynamically sampled as [B, 50].")
    print("[INFO] Stage logits have shape [B, 50, 10].")
    print("[INFO] Action loss uses only real action dimensions.")
    print("[INFO] Stage loss uses CE + exception penalty + EMA scaling.")


if __name__ == "__main__":
    main()
