import json
import tempfile
import unittest
from pathlib import Path

from scripts import salvage_multi_scoop_prefixes as salvage


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class MultiScoopSalvageTests(unittest.TestCase):
    def test_salvage_removes_failed_tail_and_rebuilds_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run_test"
            episode_dir = run_dir / "episode_000001"
            episode_dir.mkdir(parents=True)
            rows = []
            sand_rows = []
            for index in range(6):
                scoop = index // 2
                image_paths = {}
                for camera in range(3):
                    relative = f"images/{camera}/{index:06d}.ppm"
                    image = episode_dir / relative
                    image.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"P6\n1 1\n255\n\x01\x02\x03")
                    image_paths[f"observation.images.{camera}"] = relative
                row = {
                    "i": index,
                    "t": index * 0.1,
                    "timestamp.wall": 100.0 + index * 0.1,
                    "timestamp.simulation": 10.0 + index * 0.1,
                    "observation.scoop_index": scoop,
                    "phase": "unload_dump_hold_real" if index % 2 else "pre_dig",
                    "obs.q": [0.0, 0.1, 0.2, 0.3],
                    "obs.q_cmd": [0.0, 0.1, 0.2, 0.3],
                    "observation.effort": [1.0, 2.0, 3.0, 4.0],
                    "bucket.tip": [1.0, 2.0, 3.0],
                    "bucket.load": [1.0, 2.0, 3.0],
                    "bucket.pour": [1.0, 2.0, 3.0],
                    "sand": {"bucket_from_pile": 10},
                    **image_paths,
                }
                rows.append(row)
                sand_rows.append(
                    {
                        "scoop_index": scoop,
                        "t_episode": index * 0.1,
                        "sand": {"bucket_from_pile": 10},
                    }
                )
            results = [
                {
                    "scoop_index": 0,
                    "success": True,
                    "score": 80.0,
                    "samples": 2,
                    "max_bucket_from_pile_particles": 100,
                    "lift_bucket_from_pile_particles": 90,
                    "final_bucket_from_pile_particles": 5,
                    "bin_from_pile_end": 85,
                    "spill_from_lift_particles": 0,
                    "raw_region_spill_end": 5,
                    "freeze_count": 0,
                    "phase_metrics": {},
                },
                {
                    "scoop_index": 1,
                    "success": True,
                    "score": 90.0,
                    "samples": 2,
                    "max_bucket_from_pile_particles": 120,
                    "lift_bucket_from_pile_particles": 110,
                    "final_bucket_from_pile_particles": 4,
                    "bin_from_pile_end": 191,
                    "spill_from_lift_particles": 0,
                    "raw_region_spill_end": 7,
                    "freeze_count": 0,
                    "phase_metrics": {},
                },
                {
                    "scoop_index": 2,
                    "success": False,
                    "score": 40.0,
                    "samples": 2,
                    "reason": "execution_failed/test",
                },
            ]
            meta = {
                "episode_id": "episode_000001_test",
                "episode_index": 1,
                "episode_mode": "multi_scoop",
                "dataset_release": "v2.0",
                "schema": "excavator_auto_multi_scoop_v2.0",
                "status": "rejected",
                "success": False,
                "failure_reason": "execution_failed/test",
                "scoops_target": 2,
                "scoops_min": 2,
                "scoops_max": 64,
                "scoop_results": results,
                "camera_episode_summary": {},
                "final_metrics": {"samples": 6},
            }
            score = {
                "success": False,
                "thresholds": {"quality_min_score": 55.0},
                "final_metrics": {"samples": 6},
            }
            events = [
                {
                    "event": "scoop_end",
                    "detail": f"scoop_index={index}; success={index < 2}",
                }
                for index in range(3)
            ]
            write_json(episode_dir / "meta.json", meta)
            write_json(episode_dir / "score.json", score)
            write_jsonl(episode_dir / "trajectory.jsonl", rows)
            write_jsonl(episode_dir / "sand_metrics.jsonl", sand_rows)
            write_jsonl(episode_dir / "events.jsonl", events)

            index_row = {
                "episode_id": meta["episode_id"],
                "episode_index": 1,
                "status": "rejected",
                "success": False,
            }
            write_jsonl(run_dir / "episodes.jsonl", [index_row])
            write_jsonl(run_dir / "rejected_episodes.jsonl", [index_row])
            for filename in salvage.INDEX_FILES:
                path = run_dir / filename
                if not path.exists():
                    write_jsonl(path, [])
            for filename in (
                "segment_dig.jsonl",
                "segment_dig_secure.jsonl",
                "segment_lift_carry.jsonl",
                "segment_unload.jsonl",
            ):
                write_jsonl(
                    run_dir / filename,
                    [{**index_row, "episode_status": "rejected"}],
                )
            write_json(
                run_dir / "summary.json",
                {
                    "episodes_index_rows": 1,
                    "successes": 0,
                    "trainable": 0,
                    "rejections": 1,
                    "failures": 0,
                    "diagnostic": 0,
                },
            )

            dry_run = salvage.salvage_episode(episode_dir, 2, apply=False)
            self.assertTrue(dry_run["eligible"])
            self.assertEqual(dry_run["successful_scoops"], 2)
            self.assertEqual(dry_run["kept_samples"], 4)

            applied = salvage.salvage_episode(episode_dir, 2, apply=True)
            salvage.rebuild_run_indexes(
                run_dir,
                {
                    applied["episode_id"]: (
                        applied["_meta"],
                        applied["_score"],
                    )
                },
            )
            repaired_meta = salvage.read_json(episode_dir / "meta.json")
            repaired_rows = salvage.read_jsonl(episode_dir / "trajectory.jsonl")
            self.assertEqual(repaired_meta["status"], "trainable")
            self.assertEqual(repaired_meta["scoops_target"], 2)
            self.assertFalse(repaired_meta["adaptive_scoop_stop"])
            self.assertEqual(len(repaired_rows), 4)
            self.assertTrue((episode_dir / "images/0/000003.ppm").exists())
            self.assertFalse((episode_dir / "images/0/000004.ppm").exists())
            self.assertTrue(
                (
                    run_dir
                    / ".multi_scoop_salvage_backup"
                    / episode_dir.name
                    / "tail_images"
                    / "images/0/000004.ppm"
                ).exists()
            )
            successful = salvage.read_jsonl(
                run_dir / "successful_episodes.jsonl"
            )
            rejected = salvage.read_jsonl(run_dir / "rejected_episodes.jsonl")
            self.assertEqual(successful[0]["status"], "trainable")
            self.assertEqual(rejected, [])
            summary = salvage.read_json(run_dir / "summary.json")
            self.assertEqual(summary["trainable"], 1)
            self.assertEqual(summary["rejections"], 0)


if __name__ == "__main__":
    unittest.main()
