import json
import tempfile
import unittest
from pathlib import Path

from excavator_common import deployment_contract as contract


class DeploymentContractTests(unittest.TestCase):
    def test_resolves_reversed_articulation_order(self):
        raw = ["bucket_joint", "arm_joint", "boom_joint", "swing_joint"]
        self.assertEqual(contract.resolve_canonical_dof_indices(raw), (3, 2, 1, 0))
        self.assertEqual(contract.canonical_values([10, 20, 30, 40], (3, 2, 1, 0)), [40, 30, 20, 10])

    def test_rejects_missing_or_ambiguous_dof(self):
        with self.assertRaisesRegex(ValueError, "Missing canonical DOF"):
            contract.resolve_canonical_dof_indices(["swing", "boom", "arm"])
        with self.assertRaisesRegex(ValueError, "Ambiguous canonical DOF"):
            contract.resolve_canonical_dof_indices(
                ["swing", "swing_joint", "boom", "arm", "bucket"]
            )

    def test_contract_round_trip(self):
        payload = contract.build_client_contract(10, normalization_hash="abc")
        checked = contract.validate_client_contract(payload)
        self.assertEqual(checked["training_fps"], 10.0)
        self.assertEqual(
            checked["observation_schema"],
            contract.OBSERVATION_SCHEMA_LEGACY_18D,
        )
        self.assertEqual(len(checked["state_names"]), 18)
        self.assertEqual(len(checked["effort_names"]), 0)
        self.assertEqual(len(checked["action_names"]), 4)

    def test_28d_plus_effort_contract_round_trip(self):
        payload = contract.build_client_contract(
            5,
            normalization_hash="new",
            observation_schema=contract.OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
            observation_context={
                "phase_name": "pre_dig",
                "dig_target_xyz": [0.0, 6.5, 0.4],
                "unload_landing_xyz": [-6.8, -7.8, 4.2],
                "task_text": "Excavate one scoop and dump it into the truck bed.",
            },
        )
        checked = contract.validate_client_contract(payload)
        self.assertEqual(
            checked["observation_schema"],
            contract.OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
        )
        self.assertEqual(len(checked["state_names"]), 28)
        self.assertEqual(len(checked["effort_names"]), 4)

    def test_28d_contract_rejects_missing_phase_context(self):
        payload = contract.build_client_contract(
            5,
            observation_schema=contract.OBSERVATION_SCHEMA_28D_PLUS_EFFORT,
            observation_context={
                "dig_target_xyz": [0.0, 6.5, 0.4],
                "unload_landing_xyz": [-6.8, -7.8, 4.2],
                "task_text": "Excavate one scoop.",
            },
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            contract.validate_client_contract(payload)

    def test_load_training_fps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta"
            path.mkdir()
            (path / "info.json").write_text(json.dumps({"fps": 10}), encoding="utf-8")
            fps, resolved = contract.load_training_fps(tmp)
            self.assertEqual(fps, 10.0)
            self.assertEqual(resolved, (path / "info.json").resolve())

    def test_fractional_tick_schedule_has_no_long_term_drift(self):
        scheduler = contract.PhysicsTickScheduler(24)
        ticks = [scheduler.next_ticks() for _ in range(24)]
        self.assertEqual(sum(ticks), 60)
        self.assertTrue(set(ticks).issubset({2, 3}))


if __name__ == "__main__":
    unittest.main()
