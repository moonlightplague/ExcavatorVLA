import json
import os
import tempfile
import unittest

import run_dataset_dashboard_v10 as dashboard


class DashboardParallelSuccessTransferTests(unittest.TestCase):
    @staticmethod
    def _write_jsonl(path, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _make_run(self, root, worker, episode_indices):
        run_name = f"run_20260718_120000_dashboard_batch_worker_{worker:02d}"
        run_dir = os.path.join(root, run_name)
        os.makedirs(run_dir, exist_ok=True)
        rows = []
        for episode_index in episode_indices:
            episode_dir = os.path.join(run_dir, f"episode_{episode_index:06d}")
            for camera_index in range(3):
                image_dir = os.path.join(episode_dir, "images", str(camera_index))
                os.makedirs(image_dir, exist_ok=True)
                with open(os.path.join(image_dir, "frame_000000.ppm"), "wb") as handle:
                    handle.write(b"P6\n1 1\n255\n\x00\x00\x00")
            trajectory_path = os.path.join(episode_dir, "trajectory.jsonl")
            self._write_jsonl(
                trajectory_path,
                [
                    {
                        "t": 0.0,
                        "observation.state": [0.0] * 28,
                        "action": [0.0] * 4,
                        "observation.images.0": "images/0/frame_000000.ppm",
                        "observation.images.1": "images/1/frame_000000.ppm",
                        "observation.images.2": "images/2/frame_000000.ppm",
                    }
                ],
            )
            rows.append(
                {
                    "episode_index": episode_index,
                    "episode_id": f"episode_{episode_index:06d}",
                    "status": "trainable",
                    "trajectory": trajectory_path,
                }
            )
        self._write_jsonl(os.path.join(run_dir, "successful_episodes.jsonl"), rows)
        return run_dir

    def test_same_second_parallel_workers_keep_every_success_episode(self):
        with tempfile.TemporaryDirectory() as root:
            run_paths = [
                self._make_run(root, 0, [1, 4]),
                self._make_run(root, 1, [1, 2]),
                self._make_run(root, 2, [1, 2, 3]),
                self._make_run(root, 3, [1, 5]),
            ]
            result = dashboard.dashboard_transfer_success_records(
                root,
                run_paths,
                os.path.join(root, dashboard.SUCCESS_POOL_DIRNAME),
                mode="copy",
            )
            pool = os.path.join(root, dashboard.SUCCESS_POOL_DIRNAME)
            rows = dashboard.load_index(pool, "trainable")
            episode_dirs = [row.get("transferred_episode_dir") for row in rows]

            self.assertEqual(result["total_pool_records"], 9)
            self.assertEqual(len(rows), 9)
            self.assertEqual(len(set(episode_dirs)), 9)
            self.assertEqual(result["cached_skipped"], 0)
            self.assertTrue(all(os.path.isdir(path) for path in episode_dirs))

            repeated = dashboard.dashboard_transfer_success_records(
                root,
                run_paths,
                pool,
                mode="copy",
            )
            repeated_rows = dashboard.load_index(pool, "trainable")
            self.assertEqual(repeated["total_pool_records"], 9)
            self.assertEqual(repeated["cached_skipped"], 9)
            self.assertEqual(len(repeated_rows), 9)

    def test_folder_name_distinguishes_parallel_workers(self):
        row = {"episode_index": 1}
        first = dashboard.success_pool_episode_folder_name(
            "run_20260718_120000_batch_worker_00",
            row,
        )
        second = dashboard.success_pool_episode_folder_name(
            "run_20260718_120000_batch_worker_01",
            row,
        )
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("_ep000001"))
        self.assertTrue(second.endswith("_ep000001"))


if __name__ == "__main__":
    unittest.main()
