"""FP32 trainable storage with a frozen backbone and matching inference autocast."""
from contextlib import nullcontext

import torch

PRECISION_CONTRACT = 'fp32_trainable_bf16_autocast_v1'


def configure_trainable_precision(policy):
    """Call BEFORE loading FP32 base tensors and BEFORE building the optimizer."""
    parameters = [p for p in policy.parameters() if p.requires_grad]
    if not parameters or any(p.grad is not None for p in parameters):
        raise ValueError('Configure trainable precision before backward/optimizer initialization')
    for parameter in parameters:
        parameter.data = parameter.data.to(dtype=torch.float32)
    policy._excavator_precision_contract = PRECISION_CONTRACT


def inference_precision(policy):
    if (getattr(policy, '_excavator_precision_contract', None) == PRECISION_CONTRACT
            and next(policy.parameters()).device.type == 'cuda'
            and policy.config.dtype == 'bfloat16'):
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def precision_summary(policy):
    counts = {'trainable': {}, 'frozen': {}}
    for parameter in policy.parameters():
        group = counts['trainable' if parameter.requires_grad else 'frozen']
        key = str(parameter.dtype)
        group[key] = group.get(key, 0) + parameter.numel()
    return {'contract': getattr(policy, '_excavator_precision_contract', None), **counts}


def assert_fp32_training(policy, optimizer=None):
    if any(p.dtype != torch.float32 for p in policy.parameters() if p.requires_grad):
        raise ValueError('All trainable parameters must retain FP32 storage')
    if optimizer is not None:
        for state in optimizer.state.values():
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                if key in state and state[key].dtype != torch.float32:
                    raise ValueError(f'AdamW {key} must retain FP32 storage')


def validate_resume_precision(manifest):
    if manifest.get('precision_contract') != PRECISION_CONTRACT:
        raise ValueError('Exact resume requires the FP32 trainable precision contract. '
                         'A legacy BF16 run cannot be exactly resumed as FP32; start a new run. '
                         'Legacy checkpoints remain loadable for evaluation.')
