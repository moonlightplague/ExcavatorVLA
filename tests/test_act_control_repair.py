"""A relaxed ACT target-slew profile cannot train without complete replay evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator_vlm.data.action_limits import action_contract
from excavator_vlm.data.contract import file_hash
from excavator_vlm.data.control_repair import require_control_repair_gate


class ControlRepairGateTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'meta').mkdir()
        info = dict(action_limit_profile='excavator_bucket_slew20_v1',
                    action_limits_rad_s=[6., 2.5, 2.5, 20.],
                    relabel={'source_fingerprint': 'source'})
        (root / 'meta/info.json').write_text(json.dumps(info))
        entries = []
        for eid in [0, 22, 1]:
            evidence = []
            for kind in ['status', 'trace', 'reference_controls', 'compatibility']:
                path = root / f'{eid}_{kind}.txt'
                path.write_text(kind)
                evidence.append(dict(kind=kind, path=path.name, sha256=file_hash(path)))
            entries.append(dict(episode_index=eid, cycles_tested=10, full_reference_executed=True,
                numerical_target_parity_passed=True, measured_joint_parity_passed=True,
                all_cycle_deliveries_passed=True, clipped_action_dimensions=0, evidence=evidence))
        gate = dict(schema_version='newton_control_repair_gate_v1', status='passed',
            soil_backend='newton_mpm', source_fingerprint='source', dataset_fingerprint='candidate',
            action_contract=action_contract(info), episodes=entries)
        (root / 'control_replay_gate.json').write_text(json.dumps(gate))
        return gate

    def test_original_profile_needs_no_experimental_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'meta').mkdir()
            (root / 'meta/info.json').write_text(json.dumps({'action_limits_rad_s': [6, 2.5, 2.5, 6]}))
            self.assertIsNone(require_control_repair_gate(root, 'original'))

    def test_only_three_complete_unclipped_replays_pass(self):
        for change in ['missing_parent', 'incomplete_cycle', 'clipped_action', 'failed_delivery']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                gate = self.fixture(root)
                self.assertEqual(require_control_repair_gate(root, 'candidate')['status'], 'passed')
                if change == 'missing_parent':
                    gate['episodes'].pop()
                elif change == 'incomplete_cycle':
                    gate['episodes'][0]['cycles_tested'] = 9
                elif change == 'clipped_action':
                    gate['episodes'][0]['clipped_action_dimensions'] = 1
                else:
                    gate['episodes'][0]['all_cycle_deliveries_passed'] = False
                (root / 'control_replay_gate.json').write_text(json.dumps(gate))
                with self.assertRaises(ValueError):
                    require_control_repair_gate(root, 'candidate')

    def test_tampered_evidence_and_wrong_candidate_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, 'differs'):
                require_control_repair_gate(root, 'other_candidate')
            (root / '0_trace.txt').write_text('altered')
            with self.assertRaisesRegex(ValueError, 'hash'):
                require_control_repair_gate(root, 'candidate')


if __name__ == '__main__':
    unittest.main()
