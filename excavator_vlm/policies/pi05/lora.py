"""Scoped, unmerged Gemma LoRA with immutable base keys and FP32 adapters.

The native Linear interface (including .weight) is retained because upstream
PI05's combined attention path reads these properties directly. No PEFT model
wrapper, upstream source edit, or second copy of the backbone is needed.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

EXPERT_ONLY = 'expert_only'
VLM_LORA = 'vlm_lora_expert_full'
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


def finetuning_spec(mode=EXPERT_ONLY, lora=None):
    if mode not in (EXPERT_ONLY, VLM_LORA):
        raise ValueError(f'Unsupported PI05 training mode: {mode}')
    if mode == EXPERT_ONLY:
        if lora is not None:
            raise ValueError('expert_only must not specify LoRA parameters')
        return {'mode': mode, 'lora': None}
    defaults = {'r': 16, 'alpha': 32., 'dropout': 0., 'targets': list(TARGETS), 'init_seed': 0}
    if not isinstance(lora, dict) or set(lora) - set(defaults):
        raise ValueError('LoRA requires a mapping with r/alpha/dropout/targets/init_seed only')
    values = {**defaults, **copy.deepcopy(lora)}
    if type(values['r']) is not int or not 1 <= values['r'] <= 256:
        raise ValueError('LoRA r must be an integer in [1,256]')
    if not isinstance(values['alpha'], (int, float)) or not math.isfinite(values['alpha']) or values['alpha'] <= 0:
        raise ValueError('LoRA alpha must be finite and positive')
    if not isinstance(values['dropout'], (int, float)) or not 0 <= values['dropout'] < 1:
        raise ValueError('LoRA dropout must be in [0,1)')
    if (not isinstance(values['targets'], list) or not values['targets']
            or len(set(values['targets'])) != len(values['targets'])
            or not set(values['targets']).issubset(TARGETS)):
        raise ValueError('LoRA targets must be unique Gemma attention projection names')
    if type(values['init_seed']) is not int or not 0 <= values['init_seed'] < 2**63:
        raise ValueError('LoRA init_seed must be a nonnegative signed 64-bit integer')
    values['alpha'], values['dropout'] = float(values['alpha']), float(values['dropout'])
    return {'mode': mode, 'lora': values}


class LoRALinear(nn.Module):
    def __init__(self, base, spec, generator):
        super().__init__()
        if not isinstance(base, nn.Linear) or base.weight.requires_grad:
            raise ValueError('LoRA may wrap only a frozen VLM Linear')
        self.weight, self.bias = base.weight, base.bias
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scaling = spec['alpha'] / spec['r']
        self.dropout = nn.Dropout(spec['dropout'])
        options = {'device': base.weight.device, 'dtype': torch.float32}
        self.lora_A = nn.Parameter(torch.empty(spec['r'], self.in_features, **options))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, spec['r'], **options))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs):
        result = F.linear(inputs, self.weight, self.bias)
        delta = F.linear(F.linear(self.dropout(inputs).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return result + (delta * self.scaling).to(result.dtype)


def setup_finetuning(policy, spec):
    """Inject AFTER strict base loading, BEFORE delta restore and optimizer init."""
    spec = finetuning_spec(spec['mode'], spec.get('lora'))
    if hasattr(policy, '_excavator_finetuning'):
        raise ValueError('PI05 fine-tuning was already configured')
    names = []
    if spec['mode'] == VLM_LORA:
        language = policy.model.paligemma_with_expert.paligemma.language_model
        root = next(name for name, module in policy.named_modules() if module is language)
        candidates = [(name, module) for name, module in language.named_modules()
                      if name.rsplit('.', 1)[-1] in spec['lora']['targets']]
        if not candidates or any(not isinstance(module, nn.Linear) or module.weight.requires_grad
                                 for _, module in candidates):
            raise ValueError('Expected frozen Gemma attention Linear targets only')
        if any('.self_attn.' not in f'.{name}' for name, _ in candidates):
            raise ValueError('LoRA target outside Gemma self-attention')
        generators = {}
        for name, module in candidates:
            device = module.weight.device
            if str(device) not in generators:
                generators[str(device)] = torch.Generator(device=device).manual_seed(spec['lora']['init_seed'])
            parent_name, attribute = name.rsplit('.', 1)
            setattr(language.get_submodule(parent_name), attribute,
                    LoRALinear(module, spec['lora'], generators[str(device)]))
            names.append(f'{root}.{name}')
    # The dedicated generators above do not perturb training noise/data RNG.
    policy._excavator_finetuning = {**spec, 'target_modules': sorted(names)}
    return finetuning_metadata(policy)


def finetuning_metadata(policy):
    return copy.deepcopy(getattr(policy, '_excavator_finetuning',
                                 {**finetuning_spec(), 'target_modules': []}))


def setup_from_manifest(policy, manifest):
    saved = manifest.get('finetuning', {**finetuning_spec(), 'target_modules': []})
    actual = setup_finetuning(policy, finetuning_spec(saved['mode'], saved.get('lora')))
    if actual != saved:
        raise ValueError('Resolved LoRA targets differ from the checkpoint manifest')


def assert_finetuning_matches(policy, manifest):
    saved = manifest.get('finetuning', {**finetuning_spec(), 'target_modules': []})
    if finetuning_metadata(policy) != saved:
        raise ValueError('Checkpoint training mode/LoRA configuration differs from the current policy')


def set_lora_training(policy, mode):
    # Upstream freezes PaliGemma in eval mode. Train only the adapter dropout;
    # never enable vision augmentation/base dropout or unfreeze base weights.
    if finetuning_metadata(policy)['mode'] == VLM_LORA:
        for module in policy.modules():
            if isinstance(module, LoRALinear):
                module.train(mode)


def optimizer_groups(policy):
    named = [(name, p) for name, p in policy.named_parameters() if p.requires_grad]
    if finetuning_metadata(policy)['mode'] == EXPERT_ONLY:
        return [p for _, p in named]  # Preserve old one-group exact-resume format.
    lora = [p for name, p in named if name.endswith(('.lora_A', '.lora_B'))]
    expert = [p for name, p in named if not name.endswith(('.lora_A', '.lora_B'))]
    if not lora or not expert:
        raise ValueError('Expected both full expert and LoRA optimizer parameter groups')
    return [{'params': expert, 'name': 'action_expert'}, {'params': lora, 'name': 'vlm_lora'}]


def update_snapshot(policy):
    """Small first-update audit: all adapters plus samples from frozen/expert weights."""
    snapshot = {}
    expert_sampled, frozen_sampled = False, False
    for name, parameter in policy.named_parameters():
        adapter = name.endswith(('.lora_A', '.lora_B'))
        expert = parameter.requires_grad and '.gemma_expert.' in name and not expert_sampled
        frozen = not parameter.requires_grad and not frozen_sampled
        if adapter or expert or frozen:
            count = parameter.numel() if adapter else min(4096, parameter.numel())
            snapshot[name] = parameter.detach().reshape(-1)[:count].cpu().clone()
            expert_sampled |= expert
            frozen_sampled |= frozen
    return snapshot


def check_first_update(policy, snapshot):
    groups = {}
    for name, parameter in policy.named_parameters():
        if name not in snapshot:
            continue
        before = snapshot[name]
        after = parameter.detach().reshape(-1)[:before.numel()].cpu()
        if not torch.isfinite(after).all():
            raise FloatingPointError(f'Nonfinite first update: {name}')
        kind = ('lora_B' if name.endswith('.lora_B') else 'lora_A' if name.endswith('.lora_A')
                else 'expert_sample' if parameter.requires_grad else 'frozen_sample')
        row = groups.setdefault(kind, {'elements': 0, 'changed': 0, 'nonzero_gradient': 0})
        row['elements'] += before.numel()
        row['changed'] += int((before != after).sum())
        if parameter.grad is not None:
            row['nonzero_gradient'] += int(torch.count_nonzero(parameter.grad.detach().reshape(-1)[:before.numel()]))
    if groups.get('frozen_sample', {}).get('changed', 0):
        raise AssertionError('Frozen sampled base weights changed during an optimizer step')
    if finetuning_metadata(policy)['mode'] == VLM_LORA:
        if not groups.get('lora_B', {}).get('changed', 0) or not groups['lora_B']['nonzero_gradient']:
            raise AssertionError('LoRA B received no effective gradient/update')
    return {'step': 1, 'groups': groups,
            'note': 'LoRA A can have zero gradient on the first step because B is zero-initialized; '
                    'frozen/expert checks cover recorded samples, not all base elements'}
