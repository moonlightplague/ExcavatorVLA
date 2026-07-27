import unittest

try:
    import numpy as np
except ImportError:  # Minimal contract-test environments need no numpy.
    np = None

if np is not None:
    from convert_lerobot_stage_dataset import (
        checkpoint_stage_index,
        reconstruct_legacy_state_27d,
        resolve_phase_dimension,
    )


@unittest.skipIf(np is None, "numpy is required for dataset conversion tests")
class ResolvePhaseDimensionTests(unittest.TestCase):
    def test_uses_metadata_name(self):
        states = np.zeros((3, 28), dtype=np.float32)
        names = [f"feature_{index}" for index in range(28)]
        names[7] = "phase_index"

        self.assertEqual(
            resolve_phase_dimension(names, states, 10),
            (7, "metadata"),
        )

    def test_accepts_validated_legacy_dimension_18_when_name_is_missing(self):
        states = np.zeros((4, 28), dtype=np.float32)
        states[:, 18] = [0, 1, 5, 9]

        self.assertEqual(
            resolve_phase_dimension([], states, 10),
            (18, "validated_dimension_18_fallback"),
        )

    def test_rejects_v4_schema_instead_of_deleting_geometry_feature(self):
        states = np.zeros((2, 28), dtype=np.float32)
        names = [f"feature_{index}" for index in range(28)]
        names[18] = "dig_target_from_tip_forward"

        with self.assertRaisesRegex(RuntimeError, "v4 28D"):
            resolve_phase_dimension(names, states, 10)

    def test_rejects_non_integer_dimension_18_fallback(self):
        states = np.zeros((2, 28), dtype=np.float32)
        states[:, 18] = [0.25, 1.75]

        with self.assertRaisesRegex(RuntimeError, "did not contain valid"):
            resolve_phase_dimension([], states, 10)

    def test_preserves_v4_canonical_stage_order(self):
        cases = (
            ({"phase": "curl_to_hold_material", "label": "curl"}, 4),
            ({"phase": "pull_exit_cut", "label": "pull_exit_cut"}, 5),
            ({"phase": "loaded_transit", "label": "clearance_route"}, 8),
            ({"phase": "clearance_route_post_1", "label": "staged_unload"}, 8),
            ({"phase": "unload_to_bin", "label": "unload_to_bin"}, 9),
            ({"phase": "unload_to_bin", "label": "during_dump_direct"}, 9),
            ({"phase": "unload_to_bin", "label": "after_dump_settle_probe"}, 9),
        )
        for sample, expected in cases:
            with self.subTest(sample=sample):
                self.assertEqual(checkpoint_stage_index(sample), expected)

    def test_prefers_raw_canonical_phase_index(self):
        self.assertEqual(
            checkpoint_stage_index(
                {
                    "phase.index": 4,
                    "phase": "curl_to_hold_material",
                    "label": "misleading_old_label",
                }
            ),
            4,
        )

    def test_rejects_invalid_raw_canonical_phase_index(self):
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            checkpoint_stage_index({"phase.index": 10})

    def test_reconstructs_legacy_27d_geometry(self):
        legacy14 = [
            -2.0,
            3.0,
            0.25,
            0.5,
            0.1,
            -0.2,
            -0.3,
            120.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
        ]
        v4 = [0.0] * 28
        v4[:4] = legacy14[3:7]
        v4[4:8] = [0.4, 0.3, 0.2, 0.1]
        v4[18:21] = [1.5, -0.5, 0.25]
        v4[21:24] = [-2.0, 0.75, -0.4]
        v4[24] = np.sin(-0.8)
        v4[25] = np.cos(-0.8)

        result = reconstruct_legacy_state_27d(
            {"obs.state_legacy_14d": legacy14},
            v4,
            legacy14[:2],
            legacy14[2],
            35.0,
        )

        self.assertEqual(len(result), 27)
        np.testing.assert_allclose(result[:14], legacy14, atol=1e-7)
        np.testing.assert_allclose(result[14:18], v4[4:8], atol=1e-7)
        self.assertAlmostEqual(result[26], 35.0)
        self.assertAlmostEqual(result[24] ** 2 + result[25] ** 2, 1.0)


if __name__ == "__main__":
    unittest.main()
