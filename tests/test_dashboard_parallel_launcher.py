import os
import tempfile
import unittest

import run_dataset_dashboard_v10 as dashboard


class DashboardParallelLauncherTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
