from pathlib import Path
import sys
import unittest

ISAAC_REPO = Path(__file__).resolve().parents[2] / "ExcavatorVLA-isaac"
sys.path.insert(0, str(ISAAC_REPO))
import excavator_dataset_tools as export


class ReturnExportTests(unittest.TestCase):
    def meta(self):
        return {"repeat_baseline": {"version": "repeat_flat_fill_v1", "config": {"scoops": 2}},
                "repeat_returns_completed": 2, "source_sampling_audit": {"ok": True},
                "repeat_cycles": [{"scoop_index": 0, "return_start": 8, "sample_end": 10, "returned": True},
                                  {"scoop_index": 1, "return_start": 18, "sample_end": 20, "returned": True}]}

    def test_only_explicit_completed_returns_produce_spans(self):
        self.assertIsNone(export.repeat_return_spans({}))
        self.assertEqual(export.repeat_return_spans(self.meta()), [(0, 8, 10), (1, 18, 20)])
        for key, value in (("repeat_returns_completed", 1), ("source_sampling_audit", {"ok": False})):
            meta = self.meta()
            meta[key] = value
            with self.assertRaises(ValueError):
                export.repeat_return_spans(meta)

    def test_invalid_or_overlapping_cycle_bounds_fail(self):
        for changes in ({"return_start": 8, "sample_end": 8},
                        {"return_start": 9}, {"returned": False}, {"scoop_index": 0}):
            meta = self.meta()
            meta["repeat_cycles"][1].update(changes)
            with self.assertRaises(ValueError):
                export.repeat_return_spans(meta)

    def test_v2_keeps_the_audited_return_contract_and_unknown_versions_fail(self):
        meta = self.meta()
        meta["repeat_baseline"]["version"] = "repeat_flat_fill_v2"
        self.assertEqual(export.repeat_return_spans(meta), [(0, 8, 10), (1, 18, 20)])
        meta["repeat_baseline"]["version"] = "repeat_flat_fill_v3"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            export.repeat_return_spans(meta)

    def test_return_after_last_unload_is_valid_supervision(self):
        stages = [0, 4, 9, 0, 0, 0, 4, 9, 0, 0]
        scoops = [0] * 5 + [1] * 5
        returns = [0, 0, 0, 1, 1, 0, 0, 0, 1, 1]
        self.assertEqual(export.lerobot_stage_transition_errors(stages, scoops, returns), (0, 0))
        self.assertEqual(export.lerobot_stage_transition_errors(stages, scoops), (2, 0))

    def test_return_flag_cannot_hide_stale_lift_or_dig_regression(self):
        self.assertEqual(export.lerobot_stage_transition_errors([9, 7], [0, 0], [0, 1]), (1, 1))
        self.assertEqual(export.lerobot_stage_transition_errors([9, 4], [0, 0], [0, 1]), (1, 1))
        self.assertEqual(export.lerobot_stage_transition_errors([0, 0], [0, 0], [0, 1]), (0, 1))
        self.assertEqual(export.lerobot_stage_transition_errors([9, 0], [0, 1], [0, 1]), (0, 1))

    def test_legacy_next_scoop_and_optional_stats_are_compatible(self):
        self.assertEqual(export.lerobot_stage_transition_errors([0, 9, 0, 9], [0, 0, 1, 1]), (0, 0))
        rows = [{"observation.state": [0.], "action": [0.], "observation.return_to_pile": flag}
                for flag in (0, 1)]
        stats = export.build_lerobot_v3_stats(rows, 1, 1, image_keys=[])
        self.assertEqual(stats["observation.return_to_pile"]["mean"], [.5])


if __name__ == "__main__":
    unittest.main()
