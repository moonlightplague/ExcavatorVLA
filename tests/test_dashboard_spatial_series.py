import os
import tempfile
import unittest

import run_dataset_dashboard_v10 as dashboard


class DashboardSpatialSeriesTests(unittest.TestCase):
    def test_reads_exact_published_vla_episode_from_parquet(self):
        try:
            import pandas as pd
        except Exception as exc:
            self.skipTest(f"pandas unavailable: {exc}")

        state_names = list(dashboard.shared_dataset_tools.LEROBOT_STATE_NAMES_28D)
        effort_names = [
            "swing_measured_effort",
            "boom_measured_effort",
            "arm_measured_effort",
            "bucket_measured_effort",
        ]
        action_names = [
            "swing_cmd_velocity",
            "boom_cmd_velocity",
            "arm_cmd_velocity",
            "bucket_cmd_velocity",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            export_root = os.path.join(tmp, "lerobot_v3")
            data_dir = os.path.join(export_root, "data", "chunk-000")
            meta_dir = os.path.join(export_root, "meta")
            os.makedirs(data_dir, exist_ok=True)
            os.makedirs(meta_dir, exist_ok=True)
            state = [0.0] * len(state_names)
            state[state_names.index("boom_position")] = 0.5
            state[state_names.index("boom_velocity")] = 0.25
            frame = pd.DataFrame(
                [
                    {
                        "episode_index": 0,
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "observation.state": state,
                        "observation.effort": [1.0, 2.0, 3.0, 4.0],
                        "observation.stage_current_id": 0,
                        "action": [0.0, 0.1, 0.0, 0.0],
                    }
                ]
            )
            frame.to_parquet(os.path.join(data_dir, "file-000.parquet"), index=False)
            dashboard.write_json(
                os.path.join(meta_dir, "info.json"),
                {
                    "canonical_phase_names": ["pre_dig"],
                    "action_policy_version": "cmd_velocity_v3_setpoint_aware",
                    "features": {
                        "observation.state": {"names": state_names},
                        "observation.effort": {"names": effort_names},
                        "action": {"names": action_names},
                    },
                },
            )
            dashboard.write_json(
                os.path.join(export_root, "manifest.json"),
                {
                    "state_names": state_names,
                    "effort_names": effort_names,
                    "action_names": action_names,
                    "canonical_phase_names": ["pre_dig"],
                    "action_policy_version": "cmd_velocity_v3_setpoint_aware",
                    "episodes": [
                        {
                            "episode_index": 0,
                            "action_semantic_audit": {"corrected_hold_transitions": 1},
                        }
                    ],
                },
            )

            result = dashboard.dashboard_published_vla_episode(
                tmp,
                {"export_ready": True, "export_episode_index": 0},
            )

            self.assertTrue(result["available"])
            self.assertEqual(result["source"], "published_parquet")
            self.assertEqual(result["frame_count"], 1)
            self.assertAlmostEqual(result["series"]["q_deg"][0][1], 0.5 * 180.0 / dashboard.math.pi)
            self.assertAlmostEqual(result["series"]["action_deg_s"][0][1], 0.1 * 180.0 / dashboard.math.pi)
            self.assertEqual(result["action_semantic_audit"]["corrected_hold_transitions"], 1)

    def test_reads_current_top_level_bucket_xyz(self):
        sample = {
            "bucket.tip": [1.0, 2.0, 3.0],
            "bucket.load": [4.0, 5.0, 6.0],
        }
        self.assertEqual(
            dashboard.dashboard_sample_spatial_xyz(
                sample,
                "bucket.tip",
                [],
                ("bucket_tip_x", "bucket_tip_y", "bucket_tip_z"),
            ),
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(
            dashboard.dashboard_sample_spatial_xyz(
                sample,
                "bucket.load",
                [],
                ("bucket_load_x", "bucket_load_y", "bucket_load_z"),
            ),
            [4.0, 5.0, 6.0],
        )

    def test_falls_back_to_legacy_14d_state(self):
        legacy = list(range(14))
        sample = {"obs.state_legacy_14d": legacy}
        self.assertEqual(
            dashboard.dashboard_sample_spatial_xyz(
                sample,
                "bucket.tip",
                [],
                ("bucket_tip_x", "bucket_tip_y", "bucket_tip_z"),
            ),
            [8.0, 9.0, 10.0],
        )
        self.assertEqual(
            dashboard.dashboard_sample_spatial_xyz(
                sample,
                "bucket.load",
                [],
                ("bucket_load_x", "bucket_load_y", "bucket_load_z"),
            ),
            [11.0, 12.0, 13.0],
        )

    def test_falls_back_to_named_old_state_schema(self):
        names = ["other", "bucket_tip_x", "bucket_tip_y", "bucket_tip_z"]
        sample = {"observation.state": [9.0, 1.5, 2.5, 3.5]}
        self.assertEqual(
            dashboard.dashboard_sample_spatial_xyz(
                sample,
                "bucket.tip",
                names,
                ("bucket_tip_x", "bucket_tip_y", "bucket_tip_z"),
            ),
            [1.5, 2.5, 3.5],
        )


if __name__ == "__main__":
    unittest.main()
