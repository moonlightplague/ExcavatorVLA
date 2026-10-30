import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from baseline.episode_metrics import RepeatEpisodeMetrics
from baseline.tests.test_repeat_runtime import fake_runtime


class EpisodeMetricsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.rt = fake_runtime(self.directory.name)
        self.wall = 100.
        self.sim = 10.
        self.rt.dataset_simulation_time_seconds = lambda: self.sim
        self.clock = patch("baseline.episode_metrics.time.perf_counter", side_effect=lambda: self.wall)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.bucket, self.truck = set(), {9}
        # Particle 8 is outside the captured source. Even if it enters the bucket
        # or truck, it must never become source payload through a fallback.
        self.rt.STATE["dataset_initial_pile_particle_mask"][8] = False
        self.rt.sand_region_masks = lambda points: (
            np.ones(10, dtype=bool), np.isin(np.arange(10), list(self.bucket)),
            np.isin(np.arange(10), list(self.truck)))
        self.metrics = RepeatEpisodeMetrics(self.rt, 80.)

    def finish(self, success=True):
        return self.metrics.finalize({"success": success, "status": "trainable" if success else "failed",
                                      "repeat_returns_completed": 2}, {"surface_std_m": .2})

    def test_unique_source_accounting_separates_payload_retained_and_spillage(self):
        self.bucket = {0, 1, 2, 3, 8}
        self.metrics.observe()
        self.bucket, self.truck = {3}, {0, 1, 8, 9}
        result = self.finish()
        self.assertEqual([result[key] for key in ("payload_particles", "scooped_particles", "retained_particles", "spillage_particles")], [2, 4, 1, 1])
        self.assertEqual(result["spillage_fraction"], .25)
        self.assertAlmostEqual(result["payload_mass_kg"], .02)
        self.assertEqual(result["bed_fill"]["surface_std_m"], .2)

    def test_bucket_truck_overlap_is_not_delivered_payload(self):
        self.bucket, self.truck = {1}, {1, 9}
        result = self.finish()
        self.assertEqual(result["retained_particles"], 1)
        self.assertEqual(result["payload_particles"], 0)

    def test_repeat_scoops_count_unique_ids_and_exclude_earlier_deposits_per_cycle(self):
        self.metrics.start_cycle(0)
        self.bucket = {0, 1, 2}
        self.metrics.observe()
        self.bucket, self.truck = set(), {0, 1, 9}
        first = self.metrics.finish_cycle(True, "")
        self.assertEqual(first["payload_particles"], 2)
        self.assertEqual(first["spillage_particles"], 1)
        self.metrics.start_cycle(1)
        self.bucket = {2, 3}
        self.metrics.observe()
        self.bucket, self.truck = set(), {0, 1, 2, 3, 9}
        second = self.metrics.finish_cycle(True, "")
        result = self.finish()
        self.assertEqual(second["payload_particles"], 2)
        self.assertEqual(result["scooped_particles"], 4)
        self.assertEqual(result["payload_particles"], 4)
        self.assertEqual(result["spillage_particles"], 0)

    def test_complete_timing_includes_return_and_separates_preparation(self):
        self.metrics.start_cycle(0)
        self.wall, self.sim = 105., 12.
        self.metrics.finish_cycle(True, "")
        self.metrics.start_cycle(1)
        self.wall, self.sim = 114., 17.
        self.metrics.finish_cycle(True, "")
        self.wall, self.sim = 115., 18.
        result = self.finish()
        self.assertEqual(result["runtime_seconds"], 15.)
        self.assertEqual(result["attempt_runtime_seconds"], 35.)
        self.assertEqual(result["cycle_time_seconds"], 8.)
        self.assertEqual(result["scoop_cycle_time_seconds"]["mean"], 3.5)
        self.assertEqual(result["scoop_cycles"][1]["cycle_time_wall_seconds"], 9.)

    def test_failed_and_empty_episodes_keep_runtime_but_no_completion_time(self):
        self.metrics.start_cycle(0)
        self.wall, self.sim = 108., 13.
        cycle = self.metrics.finish_cycle(False, "bucket_blocked")
        result = self.finish(False)
        self.assertIsNone(result["cycle_time_seconds"])
        self.assertIsNone(cycle["cycle_time_seconds"])
        self.assertEqual(result["runtime_seconds"], 8.)
        self.assertEqual(result["simulated_runtime_seconds"], 3.)
        self.assertIsNone(result["spillage_fraction"])
        self.assertEqual(cycle["failure_reason"], "bucket_blocked")

    def test_invalid_identity_invalidates_counts_instead_of_leaving_stale_success(self):
        self.rt.sand_particle_positions = lambda: np.zeros((9, 3))
        result = self.finish()
        self.assertFalse(result["available"])
        self.assertEqual(result["unavailable_reason"], "source_particle_count_changed")
        self.assertIsNone(result["payload_particles"])
        self.assertIsNone(result["spillage_particles"])

    def test_unknown_mass_is_null(self):
        for mass in (0., None, float("nan")):
            self.rt.sand_particle_mass = lambda: mass
            self.assertIsNone(self.finish()["payload_mass_kg"])


if __name__ == "__main__":
    unittest.main()
