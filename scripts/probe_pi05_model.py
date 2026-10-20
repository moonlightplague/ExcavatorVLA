"""Numerical-only real-weight probe with synthetic token IDs, NOT a training benchmark."""
import argparse
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/pi05/smoke.yaml')
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must not already exist')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from excavator_vlm.common.runtime import dump
    from excavator_vlm.data.contract import CAMERAS
    from excavator_vlm.policies.pi05.adapter import policy_class
    from excavator_vlm.policies.pi05.checkpoints import load_base
    from excavator_vlm.policies.pi05.config import build_policy_config, resolve_settings
    from excavator_vlm.policies.pi05.inference import check_environment

    check_environment()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable')
    settings = resolve_settings(args.config)
    config = build_policy_config(settings)
    config.device = 'cuda:0'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    policy = policy_class()(config)
    report = load_base(policy, settings['assets']['base_path'], settings['assets']['base_sha256'])
    print(f'Loaded {report["loaded_tensors"]} real pretrained tensors', flush=True)
    batch = {'observation.state': torch.zeros(1, 28, device='cuda'),
        **{key: torch.full((1, 3, 256, 256), .5, device='cuda') for key in CAMERAS},
        'observation.language.tokens': torch.ones(1, 200, dtype=torch.long, device='cuda'),
        'observation.language.attention_mask': torch.ones(1, 200, dtype=torch.bool, device='cuda'),
        'action': torch.zeros(1, 10, 4, device='cuda'),
        'action_is_pad': torch.zeros(1, 10, dtype=torch.bool, device='cuda'),
        'action_loss_weight': torch.ones(1, 10, device='cuda')}
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-5)
    before = policy.model.action_out_proj.weight.detach().clone()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, _ = policy(batch)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite synthetic flow loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite synthetic gradient')
    if any(p.grad is not None for p in policy.model.paligemma_with_expert.paligemma.parameters()):
        raise AssertionError('Frozen VLM received gradients')
    optimizer.step()
    changed = not torch.equal(before, policy.model.action_out_proj.weight)
    if not changed:
        raise AssertionError('Action projection did not update')
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    if actions.shape != (1, 10, 4) or not torch.isfinite(actions).all():
        raise AssertionError('Invalid synthetic inference output')
    result = {'status': 'passed', 'numerical_only': True,
        'warning': 'Synthetic images/token IDs/actions. Not real language preprocessing, demo training, or task success.',
        **report, 'flow_loss': float(loss.detach()), 'grad_norm_before_clip': float(norm),
        'action_projection_updated': changed, 'inference_shape': list(actions.shape),
        'trainable_parameters': sum(p.numel() for p in parameters),
        'peak_cuda_memory_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': time.perf_counter() - started}
    args.output.mkdir(parents=True)
    dump(args.output / 'probe.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
