import unittest

import run_dataset_dashboard_v10 as dashboard


class DashboardSpatialSeriesTests(unittest.TestCase):
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
