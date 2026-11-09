"""Separate five-cycle replay assessment; never reuse first-scoop success."""
import numpy as np


def assess_replay(reference, status, trace, *, min_payload=None, max_joint_error=.15):
    if type(min_payload) is not int or min_payload < 1:
        raise ValueError('An explicit positive payload threshold from collection evidence is required')
    expected = np.asarray(reference['expected_states'], dtype=float)
    count = len(reference['actions'])
    full = (status.get('status') == 'completed' and status.get('full_reference_executed') is True
            and status.get('policy_steps_executed') == count and len(trace) == count
            and [r['step'] for r in trace] == list(range(count)))
    if not full:
        return {'success': False, 'full_reference_executed': False, 'cycles_passed': 0,
                'reason': 'missing_or_incomplete_full_reference_trace'}
    # step i carries obs[i] before execution and obs[i+1] afterwards. Compare
    # measured joints to recorded observations, NOT joints to command targets.
    observed = np.asarray([r['input_state'][:4] for r in trace], dtype=float)
    error = observed - expected[:, :4]
    error[:, 0] = np.arctan2(np.sin(error[:, 0]), np.cos(error[:, 0]))
    command = np.asarray([r['joint_command_positions'] for r in trace[:-1]], dtype=float)
    target = expected[1:, :4] + expected[1:, 8:12]
    target_error = float(np.abs(command - target).max())
    measured_error = float(np.abs(error).max())
    pre_payload = int(status.get('initial_deposited_particle_count', 0))
    cycles = []
    for cycle in reference['repeat_cycles']:
        start, end = int(cycle['sample_start']), int(cycle['sample_end'])
        if not 0 <= start < end <= count:
            raise ValueError('Reference cycle boundaries are outside the recorded episode')
        before = int(trace[start - 1]['deposited_particles']) if start else pre_payload
        # Source sample_end is an exclusive observation boundary. trace[j]
        # contains the post-action observation j+1, so end-1 is in trace[end-2].
        after = int(trace[end - 2]['deposited_particles']) if end > 1 else pre_payload
        gain = after - before
        source_payload = cycle.get('metrics', {}).get('payload_particles')
        cycles.append({'scoop_index': int(cycle['scoop_index']), 'net_deposited_particles': gain,
                       'source_payload_particles': source_payload,
                       'success': gain >= min_payload})
    valid = np.isfinite(error).all() and np.isfinite(command).all()
    passed = sum(c['success'] for c in cycles)
    success = bool(valid and len(cycles) == 5 and passed == 5 and measured_error <= max_joint_error
                   and target_error <= 1e-3 and status.get('clipped_action_dimensions') == 0)
    return {'success': success, 'full_reference_executed': True, 'cycles_passed': passed,
            'measured_joint_max_error_rad': measured_error, 'measured_joint_error_limit_rad': max_joint_error,
            'command_target_max_error_rad': target_error, 'command_target_error_limit_rad': 1e-3,
            'min_net_deposited_particles_per_cycle': min_payload, 'cycles': cycles,
            'payload_measurement': 'evaluation net bed-count change; collection uses cycle-local IDs and different bed geometry',
            'reason': 'passed' if success else 'payload_or_trajectory_or_clipping_gate_failed'}
