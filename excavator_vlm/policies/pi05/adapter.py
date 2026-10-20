"""Unchanged PI05 architecture with timestamp-masked, weighted flow matching."""
import torch

from ...data.contract import CAMERAS


def supervision_weights(batch):
    padding, supervision = batch['action_is_pad'], batch['action_loss_weight']
    if supervision.ndim == 3 and supervision.shape[-1] == 1:
        supervision = supervision.squeeze(-1)
    if padding.dtype != torch.bool or padding.ndim != 2 or supervision.shape != padding.shape:
        raise ValueError('Expected boolean padding and timestamp-matched weights [B, chunk]')
    if not torch.isfinite(supervision).all() or (supervision < 0).any():
        raise ValueError('Supervision weights must be finite and nonnegative')
    return (~padding).float() * supervision.float()


def flow_loss_terms(element_losses, batch):
    if element_losses.ndim != 3 or element_losses.shape[-1] < 4:
        raise ValueError('Expected elementwise flow losses [B, chunk, >=4]')
    weights = supervision_weights(batch)
    if weights.shape != element_losses.shape[:2]:
        raise ValueError('Loss and supervision timestamps differ')
    numerator = (element_losses[..., :4].float() * weights.unsqueeze(-1)).sum()
    return numerator, weights.sum() * 4


def training_batch(preprocessor, batch):
    if 'task' not in batch:
        raise ValueError('PI05 requires the recorded language task')
    processed = preprocessor({key: batch[key] for key in ('observation.state', *CAMERAS, 'action', 'task')})
    for key in ('action_is_pad', 'action_loss_weight'):
        processed[key] = batch[key].to(processed['action'].device)
    return processed


def policy_class():
    # Keep CPU-only configuration/loss tests independent of PI05's optional stack.
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    class ExcavatorPI05Policy(PI05Policy):
        def __init__(self, config):
            super().__init__(config)
            # The causal-language-model output head is unused in PI05 action flow.
            # Keep its checkpoint weights, but do not save/train an unused head.
            for name, parameter in self.named_parameters():
                if '.gemma_expert.lm_head.' in name:
                    parameter.requires_grad_(False)

        def forward(self, batch, denominator=None, world_size=1):
            images, image_masks = self._preprocess_images(batch)
            losses = self.model(images, image_masks, batch['observation.language.tokens'],
                                batch['observation.language.attention_mask'], self.prepare_action(batch))
            numerator, local_denominator = flow_loss_terms(losses, batch)
            divisor = local_denominator if denominator is None else denominator
            loss = numerator * world_size / divisor.detach().clamp_min(1)
            return loss, {'flow_numerator': numerator.detach(),
                          'valid_dimensions': local_denominator.detach()}

    return ExcavatorPI05Policy
