import os
import tempfile
import unittest
from unittest import mock

import run_dataset_dashboard_v10 as dashboard


class DashboardParallelLauncherTests(unittest.TestCase):
    def test_activity_summary_separates_worker_cost_from_parallel_output(self):
        runs = [
            {
                "activity": {"active": True, "state": "active"},
                "success": 2,
                "run_wall_s": 282.0,
            }
            for _ in range(4)
        ]
        summary = dashboard.summarize_run_activity(runs)
        self.assertEqual(summary["active_count"], 4)
        self.assertEqual(summary["parallel_rate_worker_count"], 4)
        self.assertAlmostEqual(summary["worker_seconds_per_success"], 141.0)
        self.assertAlmostEqual(summary["parallel_seconds_per_success"], 35.25)

    def test_normalize_parallel_collect_config(self):
        config = dashboard.normalize_parallel_collect_config(
            {
                "gpu_ids": "0, 2",
                "workers": 5,
                "success_count": 60,
                "max_attempts": 500,
                "log_mode": "data",
                "fast_sampled_replay": True,
                "shutdown_on_complete": True,
            }
        )
        self.assertEqual(config["gpu_ids"], "0,2")
        self.assertEqual(config["workers"], 5)
        self.assertEqual(config["success_count"], 60)
        self.assertEqual(config["expected_total_successes"], 300)
        self.assertTrue(config["fast_sampled_replay"])
        self.assertTrue(config["shutdown_on_complete"])

    def test_rejects_shell_text_and_invalid_ranges(self):
        with self.assertRaisesRegex(ValueError, "gpu_ids"):
            dashboard.normalize_parallel_collect_config({"gpu_ids": "0; rm -rf /"})
        with self.assertRaisesRegex(ValueError, "workers"):
            dashboard.normalize_parallel_collect_config({"gpu_ids": "0", "workers": 0})
        with self.assertRaisesRegex(ValueError, "max_attempts"):
            dashboard.normalize_parallel_collect_config(
                {
                    "gpu_ids": "0",
                    "workers": 1,
                    "success_count": 100,
                    "max_attempts": 50,
                }
            )

    def test_status_without_launcher_is_idle(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = dashboard.dashboard_parallel_collect_status(tmp)
            self.assertTrue(status["ok"])
            self.assertFalse(status["running"])
            self.assertEqual(status["reason"], "not_started")
            self.assertEqual(status["dataset_root"], os.path.abspath(tmp))

    def test_shutdown_is_default_off_and_requires_explicit_dashboard_opt_in(self):
        config = dashboard.normalize_parallel_collect_config({"gpu_ids": "0"})
        self.assertFalse(config["shutdown_on_complete"])
        previous = os.environ.get("EXCAVATOR_DASHBOARD_ALLOW_SHUTDOWN")
        try:
            os.environ["EXCAVATOR_DASHBOARD_ALLOW_SHUTDOWN"] = "1"
            self.assertTrue(dashboard.dashboard_parallel_shutdown_allowed())
            os.environ["EXCAVATOR_DASHBOARD_ALLOW_SHUTDOWN"] = "0"
            self.assertFalse(dashboard.dashboard_parallel_shutdown_allowed())
        finally:
            if previous is None:
                os.environ.pop("EXCAVATOR_DASHBOARD_ALLOW_SHUTDOWN", None)
            else:
                os.environ["EXCAVATOR_DASHBOARD_ALLOW_SHUTDOWN"] = previous

    def test_parallel_run_worker_identity_matches_batch_worker_and_retry(self):
        identity = dashboard.parallel_run_worker_identity(
            "/tmp/data/run_20260718_155713_dashboard_20260718_155642_worker_07_retry_02",
            "dashboard_20260718_155642",
        )
        self.assertEqual(identity["worker_name"], "worker_07")
        self.assertEqual(identity["worker_index"], 7)
        self.assertEqual(identity["retry_index"], 2)
        self.assertEqual(
            identity["run_suffix"],
            "dashboard_20260718_155642_worker_07_retry_02",
        )
        self.assertIsNone(
            dashboard.parallel_run_worker_identity(
                "/tmp/data/run_20260718_155713_batch_20260718_155642_worker_07",
                "dashboard_20260718_155642",
            )
        )

    def test_launcher_worker_supervisor_resolves_one_launcher_child(self):
        parents = {
            200: 100,
            201: 200,
            202: 201,
            300: 100,
            301: 300,
        }
        with mock.patch.object(dashboard, "linux_process_parent_map", return_value=parents):
            self.assertEqual(
                dashboard.launcher_worker_supervisor_pid(100, [201, 202]),
                200,
            )
            self.assertIsNone(
                dashboard.launcher_worker_supervisor_pid(100, [201, 301]),
            )

    def test_latest_attempt_folder_size_includes_nested_camera_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            older = os.path.join(tmp, "episode_000001")
            latest = os.path.join(tmp, "episode_000002")
            images = os.path.join(latest, "images", "0")
            os.makedirs(older)
            os.makedirs(images)
            with open(os.path.join(older, "meta.json"), "wb") as handle:
                handle.write(b"x" * 5)
            with open(os.path.join(latest, "trajectory.jsonl"), "wb") as handle:
                handle.write(b"y" * 7)
            with open(os.path.join(images, "frame_000000.ppm"), "wb") as handle:
                handle.write(b"z" * 11)

            snapshot = dashboard.latest_attempt_folder_size_snapshot(tmp)

            self.assertEqual(snapshot["name"], "episode_000002")
            self.assertEqual(snapshot["size_bytes"], 18)
            self.assertTrue(snapshot["complete"])


if __name__ == "__main__":
    unittest.main()
