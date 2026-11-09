"""Reconstruct sampled command targets; endpoint parity is not physics replay."""
import numpy as np

JOINT_BOUNDS = np.array([[-np.pi, np.pi], [-1.3089969158, 1.3089969158],
                         [-1.6580628157, 1.6580628157], [-2.0943951607, 1.5707963705]], dtype=np.float32)


def integrator_error(targets, actions, bounds=JOINT_BOUNDS):
    """Replay the benchmark's six float32 1/60 s target updates per 10 Hz action."""
    target = np.asarray(targets, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    current = target[0].copy()
    maximum = 0.
    for i in range(len(target) - 1):
        for _ in range(6):
            current = np.clip(current + actions[i] / 60, bounds[:, 0], bounds[:, 1])
        maximum = max(maximum, float(np.max(np.abs(current - target[i + 1]))))
    return maximum


def relabel_episode(states, actions, limits, fps=10):
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    limits = np.asarray(limits, dtype=np.float64)
    if (fps != 10 or states.ndim != 2 or states.shape[1] != 28 or len(states) < 2
            or actions.shape != (len(states), 4) or limits.shape != (4,)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or not np.isfinite(limits).all() or np.any(limits <= 0)):
        raise ValueError('Expected finite current-v4 states/actions at 10 Hz and four positive limits')
    targets = states[:, :4] + states[:, 8:12]
    if np.max(np.abs(targets - np.clip(targets, JOINT_BOUNDS[:, 0], JOINT_BOUNDS[:, 1]))) > 1e-5:
        raise ValueError('Recorded targets exceed benchmark joint bounds')
    # Do not wrap a target jump into another branch: the deployed target integrator
    # clips the swing joint, rather than wrapping it at +/- pi.
    candidate = np.zeros_like(actions)
    candidate[:-1] = np.diff(targets.astype(np.float64), axis=0) * fps
    if np.any(np.abs(candidate) > limits + 1e-5):
        raise ValueError('Target reconstruction exceeds the selected velocity limits; no clipping is permitted')
    revised = states.copy()
    revised[0, 12:16] = 0
    revised[1:, 12:16] = candidate[:-1]
    after = integrator_error(targets, candidate)
    if after > 1e-3:
        raise ValueError('Reconstructed actions failed continuous bounded integrator parity')
    return revised, candidate, {
        'frames': len(states), 'transitions': len(states) - 1,
        'source_integrator_max_error_rad': integrator_error(targets, actions),
        'relabel_integrator_max_error_rad': after,
        'changed_action_frames': int(np.any(np.abs(candidate - actions) > 1e-5, axis=1).sum()),
        'required_max_abs_rad_s': np.abs(candidate).max(0).tolist(),
        'source_targets_and_other_state_fields_unchanged': True,
        'physics_replay': 'not_performed',
    }
