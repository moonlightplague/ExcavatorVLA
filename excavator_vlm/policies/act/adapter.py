"""ACT training loss with future supervision and correct episode padding."""
import torch
import torch.distributed as dist
from lerobot.policies.act.modeling_act import ACTPolicy


def training_batch(preprocessor, batch):
    # LeRobot 0.4.4's batch converter drops arbitrary action_* labels. Carry
    # supervision explicitly across preprocessing without normalizing it.
    processed = preprocessor({key: value for key, value in batch.items()
                              if key in ('observation.state', 'action', 'action_is_pad')
                              or key.startswith('observation.images.')})
    processed['action_loss_weight'] = batch['action_loss_weight'].to(processed['action'].device)
    return processed


def weighted_l1(predicted, target, padding, supervision):
    if predicted.shape != target.shape or predicted.ndim != 3 or predicted.shape[-1] != 4:
        raise ValueError('Expected matching [B, chunk, 4] predictions and targets')
    if padding.shape != target.shape[:2] or padding.dtype != torch.bool:
        raise ValueError('action_is_pad must be bool [B, chunk]')
    if supervision.ndim == 3 and supervision.shape[-1] == 1:
        supervision = supervision.squeeze(-1)
    if supervision.shape != padding.shape:
        raise ValueError('action_loss_weight must be chunked at the same timestamps as action')
    if not torch.isfinite(supervision).all() or (supervision < 0).any():
        raise ValueError('Supervision weights must be finite and nonnegative')
    weights = (~padding).float() * supervision.float()
    numerator = ((predicted.float() - target.float()).abs() * weights.unsqueeze(-1)).sum()
    denominator = weights.sum() * target.shape[-1]
    # DDP averages gradients across ranks. Normalize against the global valid
    # weight count so unequal episode padding does not change rank weighting.
    global_denominator = denominator.detach().clone()
    world = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_denominator)
        world = dist.get_world_size()
    safe_denominator = torch.where(global_denominator > 0, global_denominator,
                                   torch.ones_like(global_denominator))
    loss = numerator * world / safe_denominator
    return loss, weights


class ExcavatorACTPolicy(ACTPolicy):
    """Same architecture/checkpoint keys as ACTPolicy; only the training loss differs."""

    def forward(self, batch):
        if self.config.image_features:
            batch = {**batch, 'observation.images': [batch[key] for key in self.config.image_features]}
        predicted, (mu, logvar) = self.model(batch)
        l1, weights = weighted_l1(predicted, batch['action'], batch['action_is_pad'],
                                  batch['action_loss_weight'])
        kl = l1.new_zeros(())
        if self.config.use_vae:
            per_sample_kl = (-.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1)
            active = (weights.sum(-1) > 0).float()
            count = active.sum().detach().clone()
            world = 1
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(count)
                world = dist.get_world_size()
            kl = (per_sample_kl * active).sum() * world / count.clamp_min(1)
        loss = l1 + self.config.kl_weight * kl
        return loss, {'l1_loss': l1.detach().item(), 'kld_loss': kl.detach().item(),
                      'total_loss': loss.detach().item(),
                      'valid_action_fraction': (~batch['action_is_pad']).float().mean().item(),
                      'supervised_action_fraction': (weights > 0).float().mean().item()}
