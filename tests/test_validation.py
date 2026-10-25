import json
from pathlib import Path
import tempfile
import unittest

from baseline.validate_repeat import validate_episode


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
