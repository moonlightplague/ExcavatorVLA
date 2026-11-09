import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from baseline.validate_repeat import validate_episode
from baseline.tuning import episode_quality


def fixture(directory):
    directory = Path(directory)
    rows = []
    for index in range(4):
        row = {"i": index, "t": index * .1, "observation.scoop_index": index // 2,
               "observation.state": [0.] * 28, "action": [.2, 0., 0., 0.],
               "task": "Repeated excavation and spreading",
               "phase.index": 9 if index % 2 == 0 else 0,
               "label": "unload" if index % 2 == 0 else "repeat_return_pre_dig"}
        for camera in range(3):
            image = f"camera_{camera}_{index}.ppm"
            (directory / image).write_bytes(b"P6\n1 1\n255\n\xff\x00\x00")
            row[f"observation.images.{camera}"] = image
        rows.append(row)
    bed = {"particle_count": 30, "surface_std_m": .1, "occupied_fraction": .5}
    meta = {"success": True, "status": "trainable", "scoops_completed": 2,
            "repeat_baseline": {"config": {"scoops": 2}}, "all_scoops_success": True,
            "repeat_returns_completed": 2, "source_sampling_audit": {"ok": True},
            "score_summary": {"score": 80},
            "scoop_results": [{"success": True, "bin_gain_from_pile_particles": 10}] * 2,
            "repeat_cycles": [{"scoop_index": scoop, "returned": True,
                               "return_start": scoop * 2 + 1, "sample_end": scoop * 2 + 2,
                               "landing": {"landing": [float(scoop), 0, 3]},
                               "bed_after_return": bed} for scoop in range(2)]}
    return rows, meta


def write(directory, rows, meta):
    (directory / "meta.json").write_text(json.dumps(meta))
    (directory / "trajectory.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


class ValidationTests(unittest.TestCase):
    def test_episode_policy_is_recomputed_and_still_requires_delivery_on_every_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows, meta = fixture(directory)
            config = meta['repeat_baseline']['config']
            config.update(acceptance_profile='episode', episode_min_score=65., episode_max_spill_ratio=.5)
            meta['scoop_results'] = [dict(success=True, strict_success=True, execution_success=True,
                score=90., spill_ratio=.1, freeze_count=0, bin_gain_from_pile_particles=9,
                max_bucket_from_pile_particles=10, lift_bucket_from_pile_particles=10,
                spill_from_lift_particles=1) for _ in range(2)]
            meta['repeat_episode_quality'] = episode_quality(meta['scoop_results'], SimpleNamespace(**config))
            write(directory, rows, meta)
            validate_episode(directory)
            meta['repeat_episode_quality']['weighted_spill_ratio'] = 0.
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, 'disagrees'):
                validate_episode(directory)
            meta['repeat_episode_quality'] = episode_quality(meta['scoop_results'], SimpleNamespace(**config))
            meta['scoop_results'][1]['bin_gain_from_pile_particles'] = 0
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, 'soil delivery'):
                validate_episode(directory)
            # Actual new IDs, rather than the net bed count, establish delivery
            # when an earlier load loses some material during a later return.
            for report in meta['scoop_results']:
                report.update(source_delivered_particles=9, source_captured_particles=10,
                              source_spillage_particles=1, source_score=90.)
            meta['repeat_metrics'] = {'spillage_fraction': .1}
            meta['repeat_episode_quality'] = episode_quality(meta['scoop_results'],
                SimpleNamespace(**config), meta['repeat_metrics'])
            write(directory, rows, meta)
            validate_episode(directory)
            meta['scoop_results'][1]['source_delivered_particles'] = 0
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, 'soil delivery'):
                validate_episode(directory)

    def test_v2_audit_requires_metrics_and_continuous_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows, meta = fixture(directory)
            meta["repeat_baseline"]["version"] = "repeat_flat_fill_v2"
            meta["final_metrics"] = {"samples": 4}
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, "Missing repeat metrics"):
                validate_episode(directory)
            meta["repeat_metrics"] = {
                "metrics_schema_version": "repeat_episode_metrics_v1", "success": True, "available": True,
                "samples": 4, "runtime_seconds": 5., "simulated_runtime_seconds": .4,
                "cycle_time_seconds": .4, "cycle_time_wall_seconds": 5.,
                "spillage_particles": 2, "retained_particles": 1, "scooped_particles": 23,
                "scoop_cycles": [{"success": True}] * 2}
            (directory / "repeat_metrics.json").write_text(json.dumps(meta["repeat_metrics"]))
            write(directory, rows, meta)
            self.assertEqual(validate_episode(directory)["scoops"], 2)
            meta["plan"] = [{"phase": name} for name in ("approach_contact", "insert_cut", "clearance_route_1", "pull_mid_cut", "pull_exit_cut")]
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, "detour interrupts"):
                validate_episode(directory)

    def test_complete_episode_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows, meta = fixture(directory)
            write(directory, rows, meta)
            result = validate_episode(directory)
            self.assertEqual(result["frames"], 4)
            self.assertEqual(result["scoops"], 2)

    def test_missing_final_return_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows, meta = fixture(directory)
            rows[-1]["label"] = "unload_hold"
            rows[-1]["phase.index"] = 9
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, "Return trajectory is missing"):
                validate_episode(directory)

    def test_broken_camera_and_clock_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rows, meta = fixture(directory)
            rows[-1]["t"] = .5
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, "uniform 10 Hz"):
                validate_episode(directory)
            rows[-1]["t"] = .3
            rows[-1]["observation.images.0"] = rows[0]["observation.images.0"]
            write(directory, rows, meta)
            with self.assertRaisesRegex(ValueError, "Reused camera"):
                validate_episode(directory)


if __name__ == "__main__":
    unittest.main()
