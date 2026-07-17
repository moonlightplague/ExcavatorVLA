#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_ROOT = Path(
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


def patch_config(text: str) -> str:
    if "stage_target_key: str = " in text:
        print("[INFO] configuration_smolvla.py already contains stage fields")
        return text

    old = """    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping:"""

    new = """    chunk_size: int = 50
    n_action_steps: int = 50

    # Auxiliary stage-classification task.
    # These dataset fields are supervision labels, not policy input features.
    num_stages: int = 10
    stage_loss_weight: float = 0.5
    stage_target_key: str = "stage_target_30"
    stage_valid_key: str = "stage_valid_30"

    normalization_mapping:"""

    text = replace_once(
        text,
        old,
        new,
        "insert stage config fields",
    )

    old_validation = """        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:"""

    new_validation = """        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.num_stages <= 1:
            raise ValueError(
                f"`num_stages` must be greater than 1, got {self.num_stages}."
            )
        if self.stage_loss_weight < 0:
            raise ValueError(
                f"`stage_loss_weight` must be non-negative, got {self.stage_loss_weight}."
            )
        if self.use_delta_joint_actions_aloha:"""

    return replace_once(
        text,
        old_validation,
        new_validation,
        "insert stage config validation",
    )


def patch_modeling(text: str) -> str:
    if "def _stage_logits_from_prefix(" in text:
        print("[INFO] modeling_smolvla.py already contains stage head")
        return text

    old_head = """        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear("""

    new_head = """        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        # Predict the future execution stage from the final valid prefix token.
        # The prefix contains image, language, and padded robot-state context.
        prefix_hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.stage_head = nn.Sequential(
            nn.LayerNorm(prefix_hidden_size),
            nn.Linear(prefix_hidden_size, self.config.num_stages),
        )

        self.action_time_mlp_in = nn.Linear("""

    text = replace_once(
        text,
        old_head,
        new_head,
        "insert stage head",
    )

    old_helper_anchor = """    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled
"""

    new_helper_anchor = """    def _stage_logits_from_prefix(
        self,
        prefix_out: Tensor,
        prefix_pad_masks: Tensor,
    ) -> Tensor:
        \"\"\"Classify stage from the final non-padding prefix token.\"\"\"
        last_token_index = (
            prefix_pad_masks.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        )
        batch_index = torch.arange(
            prefix_out.shape[0],
            device=prefix_out.device,
        )
        pooled_prefix = prefix_out[batch_index, last_token_index]
        head_dtype = self.stage_head[1].weight.dtype
        pooled_prefix = pooled_prefix.to(dtype=head_dtype)
        return self.stage_head(pooled_prefix)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled
"""

    text = replace_once(
        text,
        old_helper_anchor,
        new_helper_anchor,
        "insert stage pooling helper",
    )

    old_forward_call = """        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
"""

    new_forward_call = """        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
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
"""

    text = replace_once(
        text,
        old_forward_call,
        new_forward_call,
        "retain prefix output during training",
    )

    old_return = """        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    def sample_actions("""
    new_return = """        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, stage_logits

    def sample_actions("""
    text = replace_once(
        text,
        old_return,
        new_return,
        "return stage logits from VLAFlowMatching.forward",
    )

    old_policy_call = """        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone().mean().item()
"""

    new_policy_call = """        loss_dict = {}
        losses, stage_logits = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
        )
        loss_dict["losses_after_forward"] = losses.detach().mean().item()
"""

    text = replace_once(
        text,
        old_policy_call,
        new_policy_call,
        "unpack stage logits in policy forward",
    )

    old_reduction = """        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=(1, 2))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict
"""

    new_reduction = """        # Action flow-matching loss per sample.
        per_sample_action_loss = losses.mean(dim=(1, 2))

        if self.config.stage_target_key not in batch:
            raise KeyError(
                f"Missing stage target field: {self.config.stage_target_key}. "
                f"Available batch keys: {sorted(batch.keys())}"
            )

        stage_target = batch[self.config.stage_target_key].to(
            device=stage_logits.device,
            dtype=torch.long,
        ).reshape(-1)

        if stage_target.shape[0] != stage_logits.shape[0]:
            raise ValueError(
                "Stage target batch size does not match stage logits: "
                f"{stage_target.shape[0]} vs {stage_logits.shape[0]}"
            )

        stage_valid_value = batch.get(self.config.stage_valid_key)
        if stage_valid_value is None:
            stage_valid = torch.ones_like(
                stage_target,
                dtype=torch.bool,
                device=stage_logits.device,
            )
        else:
            stage_valid = stage_valid_value.to(
                device=stage_logits.device,
                dtype=torch.bool,
            ).reshape(-1)

        if stage_valid.shape[0] != stage_logits.shape[0]:
            raise ValueError(
                "Stage-valid batch size does not match stage logits: "
                f"{stage_valid.shape[0]} vs {stage_logits.shape[0]}"
            )

        per_sample_stage_loss = torch.zeros_like(
            per_sample_action_loss
        )

        if stage_valid.any():
            valid_stage_losses = F.cross_entropy(
                stage_logits[stage_valid],
                stage_target[stage_valid],
                reduction="none",
            )
            per_sample_stage_loss[stage_valid] = valid_stage_losses
            stage_loss = valid_stage_losses.mean()

            with torch.no_grad():
                stage_prediction = stage_logits[stage_valid].argmax(dim=-1)
                stage_accuracy = (
                    stage_prediction == stage_target[stage_valid]
                ).float().mean()
        else:
            # Keep a differentiable zero connected to the stage head.
            stage_loss = stage_logits.sum() * 0.0
            stage_accuracy = stage_logits.new_tensor(0.0)

        per_sample_loss = (
            per_sample_action_loss
            + self.config.stage_loss_weight * per_sample_stage_loss
        )

        action_loss = per_sample_action_loss.mean()
        total_loss = (
            action_loss
            + self.config.stage_loss_weight * stage_loss
        )

        loss_dict["action_loss"] = action_loss.detach().item()
        loss_dict["stage_loss"] = stage_loss.detach().item()
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

    return replace_once(
        text,
        old_reduction,
        new_reduction,
        "replace policy loss reduction",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smolvla-dir",
        type=Path,
        default=DEFAULT_ROOT,
    )
    parser.add_argument(
        "--restore",
        action="store_true",
    )
    args = parser.parse_args()

    root = args.smolvla_dir.expanduser().resolve()
    config_path = root / "configuration_smolvla.py"
    modeling_path = root / "modeling_smolvla.py"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not modeling_path.exists():
        raise FileNotFoundError(modeling_path)

    config_backup = config_path.with_suffix(
        config_path.suffix + ".before_stage"
    )
    modeling_backup = modeling_path.with_suffix(
        modeling_path.suffix + ".before_stage"
    )

    if args.restore:
        if not config_backup.exists() or not modeling_backup.exists():
            raise FileNotFoundError(
                "Backup files are missing; cannot restore"
            )
        shutil.copy2(config_backup, config_path)
        shutil.copy2(modeling_backup, modeling_path)
        print(f"[OK] Restored {config_path}")
        print(f"[OK] Restored {modeling_path}")
        return

    if not config_backup.exists():
        shutil.copy2(config_path, config_backup)
        print(f"[OK] Backup created: {config_backup}")
    else:
        print(f"[INFO] Backup already exists: {config_backup}")

    if not modeling_backup.exists():
        shutil.copy2(modeling_path, modeling_backup)
        print(f"[OK] Backup created: {modeling_backup}")
    else:
        print(f"[INFO] Backup already exists: {modeling_backup}")

    config_text = config_path.read_text(encoding="utf-8")
    modeling_text = modeling_path.read_text(encoding="utf-8")

    patched_config = patch_config(config_text)
    patched_modeling = patch_modeling(modeling_text)

    config_path.write_text(patched_config, encoding="utf-8")
    modeling_path.write_text(patched_modeling, encoding="utf-8")

    print(f"[OK] Patched {config_path}")
    print(f"[OK] Patched {modeling_path}")
    print("[INFO] Run py_compile and the one-batch smoke test next.")


if __name__ == "__main__":
    main()
