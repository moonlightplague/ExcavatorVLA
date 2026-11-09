"""Experimental target-slew limits must be explicit, causal and replay gated."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
from excavator_vlm.data.action_limits import (
    BUCKET20_PROFILE, DEFAULT_PROFILE, PROFILES, action_contract,
    check_checkpoint_contract, collection_payload_threshold, require_replay_gate,
)
from excavator_vlm.data.relabel import integrator_error, relabel_episode
from excavator_vlm.data.replay_gate import assess_replay
from excavator_vlm.policies.pi05.config import resolve_settings


class ActionLimitTests(unittest.TestCase):
    def test_old_limits_remain_default_and_relaxation_requires_a_profile(self):
        old = {'action_limits_rad_s': [6, 2.5, 2.5, 6]}
        self.assertEqual(action_contract(old)['profile'], DEFAULT_PROFILE)
        with self.assertRaises(ValueError):
            action_contract({'action_limits_rad_s': [6, 2.5, 2.5, 20]})
        new = {'action_limit_profile': BUCKET20_PROFILE, 'action_limits_rad_s': [6, 2.5, 2.5, 20]}
        self.assertEqual(action_contract(new)['limits_rad_s'][-1], 20)
        for profile in ('unknown', None, []):
            with self.assertRaises(ValueError):
                action_contract({**new, 'action_limit_profile': profile})
        with self.assertRaises(ValueError):
            action_contract({**new, 'action_limits_rad_s': [6, 2.5, 2.5, 21]})

    def test_old_checkpoint_cannot_silently_use_new_limits(self):
        old = {'action_limits_rad_s': PROFILES[DEFAULT_PROFILE]}
        new = {'action_limit_profile': BUCKET20_PROFILE, 'action_limits_rad_s': PROFILES[BUCKET20_PROFILE]}
        check_checkpoint_contract({}, old)
        with self.assertRaises(ValueError):
            check_checkpoint_contract({}, new)
        check_checkpoint_contract({'action_contract': action_contract(new)}, new)

    def test_requested_training_budget_and_retention_are_explicit(self):
        settings = resolve_settings(SUBMODULE / 'configs/pi05/history_c_repeat20k.yaml')
        self.assertEqual(settings['steps'], 20000)
        self.assertEqual(settings['save_freq'], 2000)
        self.assertEqual(settings['training']['keep_checkpoints'], 10)
        self.assertEqual(settings['launcher']['gpu_ids'], [0, 1])
        self.assertEqual(settings['observation_history']['delta_indices'], [-6, -3, 0])
        self.assertEqual(settings['policy']['optimizer_lr'], 2.5e-5)

    def test_relaxed_limits_require_three_complete_physical_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'meta').mkdir()
            info = {'action_limit_profile': BUCKET20_PROFILE, 'action_limits_rad_s': PROFILES[BUCKET20_PROFILE]}
            (root / 'meta/info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError):
                require_replay_gate(root, 'hash')
            quality = root / 'quality.json'
            quality.write_text(json.dumps({'quality_min_dump_particles': 3, 'quality_gate_version': 'collection_v1'}))
            criterion = {'version': 'collection_min_dump_particles_v1', 'min_payload_particles_per_cycle': 3,
                         'quality_gate_version': 'collection_v1', 'source_evidence': [
                             {'path': quality.name, 'sha256': hashlib.sha256(quality.read_bytes()).hexdigest(),
                              'field': 'quality_min_dump_particles'}]}
            gate = {'schema_version': 'excavator_repeat_expert_gate_v2', 'status': 'passed',
                    'dataset_fingerprint': 'hash', 'soil_backend': 'physx', 'action_contract': action_contract(info),
                    'collection_criterion': criterion,
                    'episodes': [{'episode_index': i, 'full_reference_executed': True,
                                  'success': True, 'cycles_passed': 5,
                                  'min_net_deposited_particles_per_cycle': 3} for i in range(3)]}
            for result in gate['episodes']:
                result['evidence'] = []
                for name in ('status', 'trace'):
                    path = root / f"{result['episode_index']}_{name}.json"
                    path.write_text('{}')
                    result['evidence'].append({'path': path.name,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            path = root / 'expert_replay_gate.json'; path.write_text(json.dumps(gate))
            self.assertEqual(require_replay_gate(root, 'hash'), gate)
            with self.assertRaises(ValueError):
                require_replay_gate(root, 'other_hash')
            gate['episodes'][1]['full_reference_executed'] = False
            path.write_text(json.dumps(gate))
            with self.assertRaises(ValueError):
                require_replay_gate(root, 'hash')

    def test_five_cycle_gate_checks_real_motion_and_every_cycle(self):
        reference = {'actions': [[0.] * 4] * 10, 'expected_states': [[0.] * 28] * 10,
                     'repeat_cycles': [{'scoop_index': i, 'sample_start': i * 2,
                                        'sample_end': i * 2 + 2} for i in range(5)]}
        status = {'status': 'completed', 'full_reference_executed': True,
                  'policy_steps_executed': 10, 'clipped_action_dimensions': 0}
        trace = [{'step': i, 'input_state': [0.] * 28, 'joint_command_positions': [0.] * 4,
                  'deposited_particles': ((i + 2) // 2) * 100} for i in range(10)]
        self.assertTrue(assess_replay(reference, status, trace, min_payload=100)['success'])
        trace[4]['input_state'][3] = .2
        self.assertFalse(assess_replay(reference, status, trace, min_payload=100)['success'])
        trace[4]['input_state'][3] = 0
        trace[-2]['deposited_particles'] = 499
        result = assess_replay(reference, status, trace, min_payload=100)
        self.assertFalse(result['success'])
        self.assertEqual(result['cycles_passed'], 4)
        self.assertFalse(assess_replay(reference, status, trace[:-1], min_payload=100)['full_reference_executed'])

    def test_collection_evidence_not_live_defaults_controls_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path = root / 'quality.json'
            path.write_text(json.dumps({'quality_min_dump_particles': 3, 'quality_gate_version': 'archived_v1'}))
            criterion = {'version': 'collection_min_dump_particles_v1', 'min_payload_particles_per_cycle': 3,
                         'quality_gate_version': 'archived_v1', 'source_evidence': [
                             {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                              'field': 'quality_min_dump_particles'}]}
            self.assertEqual(collection_payload_threshold(root, criterion), 3)
            criterion['min_payload_particles_per_cycle'] = 100
            with self.assertRaises(ValueError):
                collection_payload_threshold(root, criterion)
            criterion['min_payload_particles_per_cycle'] = 3
            path.write_text('{}')
            with self.assertRaises(ValueError):
                collection_payload_threshold(root, criterion)

    def test_low_payload_success_and_exclusive_observation_boundary(self):
        reference = {'actions': [[0.] * 4] * 10, 'expected_states': [[0.] * 28] * 10,
                     'repeat_cycles': [{'scoop_index': i, 'sample_start': i * 2,
                                        'sample_end': i * 2 + 2, 'metrics': {'payload_particles': 48}}
                                       for i in range(5)]}
        status = {'status': 'completed', 'full_reference_executed': True,
                  'policy_steps_executed': 10, 'clipped_action_dimensions': 0}
        trace = [{'step': i, 'input_state': [0.] * 28, 'joint_command_positions': [0.] * 4,
                  'deposited_particles': ((i + 2) // 2) * 40} for i in range(10)]
        result = assess_replay(reference, status, trace, min_payload=3)
        self.assertTrue(result['success'])
        self.assertEqual([c['net_deposited_particles'] for c in result['cycles']], [40] * 5)
        self.assertEqual(result['cycles'][-1]['source_payload_particles'], 48)
        self.assertFalse(assess_replay(reference, status, trace, min_payload=100)['success'])
        # The final zero action is outside the recorded observations and must
        # not turn a failed last recorded cycle into a pass.
        trace[-2]['deposited_particles'] = 162
        self.assertFalse(assess_replay(reference, status, trace, min_payload=3)['success'])
        with self.assertRaises(ValueError):
            assess_replay(reference, status, trace)


class RelabelTests(unittest.TestCase):
    def fixture(self):
        states = np.zeros((5, 28), dtype=np.float32)
        states[:, 11] = [0, 0, 1.5, .1, 0]
        return states, np.zeros((5, 4), dtype=np.float32)

    def test_no_clipping_to_hide_unrepresentable_target_resets(self):
        states, actions = self.fixture()
        with self.assertRaisesRegex(ValueError, 'no clipping'):
            relabel_episode(states, actions, PROFILES[DEFAULT_PROFILE])
        revised, candidate, report = relabel_episode(states, actions, PROFILES[BUCKET20_PROFILE])
        np.testing.assert_array_equal(revised[:, :12], states[:, :12])
        np.testing.assert_array_equal(revised[:, 16:], states[:, 16:])
        np.testing.assert_array_equal(revised[1:, 12:16], candidate[:-1])
        np.testing.assert_array_equal(candidate[-1], np.zeros(4))
        self.assertLess(report['relabel_integrator_max_error_rad'], 1e-5)
        self.assertGreater(report['source_integrator_max_error_rad'], 1.)
        self.assertEqual(report['physics_replay'], 'not_performed')
        np.testing.assert_array_equal(actions, np.zeros((5, 4)))

    def test_nonfinite_wrong_fps_and_out_of_bounds_targets_are_rejected(self):
        for bad in ('nan', 'bounds', 'fps'):
            states, actions = self.fixture()
            if bad == 'nan': states[1, 1] = np.nan
            if bad == 'bounds': states[2, 11] = 2.
            with self.assertRaises(ValueError):
                relabel_episode(states, actions, PROFILES[BUCKET20_PROFILE], fps=20 if bad == 'fps' else 10)

    def test_continuous_target_reconstruction_matches_all_steps(self):
        rng = np.random.default_rng(2)
        states = np.zeros((200, 28), dtype=np.float32)
        states[:, 8:12] = np.cumsum(rng.normal(0, .01, (200, 4)), axis=0)
        _, candidate, _ = relabel_episode(states, np.zeros((200, 4)), PROFILES[BUCKET20_PROFILE])
        self.assertLess(integrator_error(states[:, :4] + states[:, 8:12], candidate), 1e-5)


if __name__ == '__main__':
    unittest.main()
