"""Append integration tests using small parquet datasets; no Isaac Sim required."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import excavator_dataset_tools as tools


def make_export(root, source="first", task="dig", value=0.0):
    root = Path(root)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    rows = [dict(index=i, episode_index=0, frame_index=i, timestamp=i / 10,
                 task_index=0, action=[value] * 4, **{"observation.state": [value] * 28})
            for i in range(2)]
    pd.DataFrame(rows).to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task])).to_parquet(root / "meta/tasks.parquet")
    episode = {"episode_index": 0, "tasks": [task], "length": 2,
               "dataset_from_index": 0, "dataset_to_index": 2,
               "meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0,
               "data/chunk_index": 0, "data/file_index": 0}
    videos = {}
    for key in tools.LEROBOT_IMAGE_KEYS:
        path = f"videos/{key}/chunk-000/file-000.mp4"
        (root / path).parent.mkdir(parents=True)
        (root / path).write_bytes(b"unchanged-video")
        videos[key] = {"available": True, "frames": 2, "episode_files": [
            {"available": True, "path": path, "chunk_index": 0, "file_index": 0}]}
        for suffix, value_ in [("chunk_index", 0), ("file_index", 0),
                               ("from_timestamp", 0.0), ("to_timestamp", 0.2)]:
            episode[f"videos/{key}/{suffix}"] = value_
    pd.DataFrame([episode]).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    features = {"action": {"names": ["a", "b", "c", "d"]}}
    features.update({key: {"dtype": "video", "shape": tools.LEROBOT_IMAGE_SHAPE}
                     for key in tools.LEROBOT_IMAGE_KEYS})
    info = dict(codebase_version=tools.LEROBOT_CODEBASE_VERSION,
                action_policy_version=tools.LEROBOT_ACTION_POLICY_VERSION,
                action_limits_rad_s=list(tools.vla_observation_contract.ACTION_LIMITS_RAD_S_4D),
                stage_policy_version=tools.LEROBOT_STAGE_POLICY_VERSION,
                effort_policy="raw", state_dim=28, effort_dim=0, fps=10.0,
                total_frames=2, total_episodes=1, total_tasks=1, features=features,
                video_path="videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    tools.write_json(root / "meta/info.json", info)
    tools.write_json(root / "meta/stats.json", tools.build_lerobot_v3_stats(rows, 28, 4))
    return dict(info, export_dir=str(root), vla_training_ready=True)


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(root).rglob("*") if path.is_file()}


class AppendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.destination = self.root / "export"
        make_export(self.destination)

    def append(self, source, task="dig", value=0.0):
        def export(run_dir, output_dir, **kwargs):
            return make_export(output_dir, source, task, value)
        with patch.object(tools, "export_lerobot_dataset", side_effect=export):
            return tools.append_lerobot_dataset(source, str(self.destination))

    def test_dataset_supports_repeated_append_without_manifest(self):
        before = snapshot(self.destination)
        self.append("second", "unload", 2.0)
        result = self.append("third", "dig", 4.0)
        self.assertEqual(result["total_episodes"], 3)
        self.assertFalse((self.destination / "manifest.json").exists())
        self.assertFalse((self.destination / "README.md").exists())
        for name, digest in before.items():
            if "chunk-000" in name:
                self.assertEqual(snapshot(self.destination)[name], digest)
        for chunk in (1, 2):
            metadata = pd.read_parquet(self.destination / f"meta/episodes/chunk-{chunk:03d}/file-000.parquet")
            self.assertEqual(len(metadata), 1)
        stats = tools.read_json(self.destination / "meta/stats.json")
        self.assertEqual(stats["action"]["mean"], [2.0] * 4)
        self.assertTrue(tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)["ok"])

    def test_new_destination_keeps_required_training_metadata(self):
        shutil.rmtree(self.destination)
        self.append("second")
        self.append("third")
        self.assertFalse((self.destination / "manifest.json").exists())
        self.assertFalse((self.destination / "README.md").exists())
        for name in ("info.json", "stats.json", "tasks.parquet"):
            self.assertTrue((self.destination / "meta" / name).is_file())
        self.assertTrue(tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)["ok"])

    def test_repeated_append_preserves_chunks_and_reindexes_tasks_and_stats(self):
        before = snapshot(self.destination)
        result = self.append("second", "unload", 2.0)
        result = self.append("third", "dig", 4.0)
        self.assertEqual(result["last_appended_chunk"], 2)
        self.assertEqual(result["total_frames"], 6)
        self.assertEqual(result["total_episodes"], 3)
        self.assertEqual(result["total_tasks"], 2)
        self.assertEqual(result["export_dir"], str(self.destination))
        for name, digest in before.items():
            if "chunk-000" in name:
                self.assertEqual(snapshot(self.destination)[name], digest)
        data = pd.concat([pd.read_parquet(path) for path in tools._lerobot_parquet_paths(self.destination / "data")])
        self.assertEqual(data["index"].tolist(), list(range(6)))
        self.assertEqual(data["episode_index"].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(data["task_index"].tolist(), [0, 0, 1, 1, 0, 0])
        stats = tools.read_json(self.destination / "meta/stats.json")
        self.assertEqual(stats["action"]["mean"], [2.0] * 4)
        self.assertEqual(stats["action"]["count"], [6])
        self.assertTrue(tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)["ok"])
        metadata = pd.read_parquet(self.destination / "meta/episodes/chunk-002/file-000.parquet")
        self.assertEqual(metadata.iloc[0][f"videos/{tools.LEROBOT_IMAGE_KEYS[0]}/chunk_index"], 2)

    def test_incompatible_batch_keeps_original(self):
        before = snapshot(self.destination)
        def incompatible(run_dir, output_dir, **kwargs):
            result = make_export(output_dir, "second")
            info = tools.read_json(Path(output_dir) / "meta/info.json")
            info["fps"] = 20
            tools.write_json(Path(output_dir) / "meta/info.json", info)
            return result
        with patch.object(tools, "export_lerobot_dataset", side_effect=incompatible):
            with self.assertRaisesRegex(ValueError, "incompatible.*fps"):
                tools.append_lerobot_dataset("second", str(self.destination))
        self.assertEqual(snapshot(self.destination), before)

    def test_failed_export_keeps_original(self):
        before = snapshot(self.destination)
        with patch.object(tools, "export_lerobot_dataset", side_effect=RuntimeError("encoder failed")):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                tools.append_lerobot_dataset("second", str(self.destination))
        self.assertEqual(snapshot(self.destination), before)
        self.assertFalse((self.root / ".export.append.lock").exists())

    def test_failed_publish_restores_original(self):
        before = snapshot(self.destination)
        rename = tools.os.rename
        def fail_publish(source, destination):
            if Path(source).name == "assembled":
                raise OSError("publish failed")
            return rename(source, destination)
        with patch.object(tools.os, "rename", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "publish failed"):
                self.append("second")
        self.assertEqual(snapshot(self.destination), before)

    def test_lock_blocks_parallel_append(self):
        (self.root / ".export.append.lock").touch()
        with self.assertRaisesRegex(RuntimeError, "another append"):
            self.append("second")

    def test_validates_new_chunks(self):
        self.append("second")
        path = self.destination / "data/chunk-001/file-000.parquet"
        data = pd.read_parquet(path)
        data.loc[0, "index"] = 900
        data.to_parquet(path, index=False)
        self.assertFalse(tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)["ok"])

    def test_next_chunk_uses_maximum_existing_number(self):
        (self.destination / "data/chunk-007").mkdir()
        result = self.append("second")
        self.assertEqual(result["last_appended_chunk"], 8)
        self.assertTrue((self.destination / "data/chunk-008/file-000.parquet").exists())

    def test_new_video_and_data_references_are_validated(self):
        self.append("second")
        path = self.destination / "meta/episodes/chunk-001/file-000.parquet"
        episodes = pd.read_parquet(path)
        episodes["data/chunk_index"] = 0
        episodes.to_parquet(path, index=False)
        validation = tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)
        self.assertIn("episodes/wrong_data_reference:1", validation["reasons"])
        video = self.destination / "videos" / tools.LEROBOT_IMAGE_KEYS[0] / "chunk-001/file-000.mp4"
        video.unlink()
        validation = tools.validate_lerobot_v3_export(str(self.destination), tools.LEROBOT_IMAGE_KEYS)
        self.assertTrue(any(reason.startswith("videos/missing_") for reason in validation["reasons"]))

    def test_new_destination_starts_at_zero(self):
        shutil.rmtree(self.destination)
        result = self.append("second")
        self.assertEqual(result["total_episodes"], 1)
        self.assertTrue((self.destination / "data/chunk-000/file-000.parquet").exists())


if __name__ == "__main__":
    unittest.main()
