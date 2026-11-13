from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from baseline.metrics_summary import combine_batch, summarize
from baseline.stream_export import exporter, finish_episode, policy_environment, read_jsonl, verify_run, write_json


def make_episode(run, index=1, status="trainable", payload=8, scooped=10, spillage=2):
    run = Path(run)
    episode = run / f"episode_{index:06d}"
    (episode / "images").mkdir(parents=True)
    (episode / "images/camera.png").write_bytes(b"camera frame")
    for name in ("trajectory.jsonl", "events.jsonl", "sand_metrics.jsonl", "plan_debug.json"):
        (episode / name).write_text("large raw trace")
    success = status == "trainable"
    metrics = {"metrics_schema_version": "repeat_episode_metrics_v1", "available": True,
               "definitions": {}, "success": success, "status": status,
               "payload_particles": payload, "payload_mass_kg": payload * .5,
               "scooped_particles": scooped, "spillage_particles": spillage,
               "spillage_fraction": spillage / scooped, "spillage_mass_kg": spillage * .5,
               "runtime_seconds": 10. * index, "attempt_runtime_seconds": 12. * index,
               "simulated_runtime_seconds": 2. * index,
               "cycle_time_seconds": 2. * index if success else None,
               "cycle_time_wall_seconds": 10. * index if success else None,
               "scoop_cycles": [{"success": True, "runtime_seconds": 3.}],
               "score_summary": {"score": 90.}, "bed_fill": {"occupied_fraction": .5}}
    meta = {"episode_id": f"{run.name}_{index}", "episode_index": index,
            "status": status, "success": success, "repeat_metrics": metrics,
            "plan": ["bulky plan"], "score_summary": {"score": 90.}}
    for name, value in (("meta.json", meta), ("repeat_metrics.json", metrics), ("score.json", {"score": 90.})):
        write_json(episode / name, value)
    row = {"episode_id": meta["episode_id"], "episode_index": index, **metrics}
    for name in ("repeat_metrics.jsonl", "episodes.jsonl", *(["trainable_episodes.jsonl"] if success else [])):
        with (run / name).open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    return episode


def fake_export(run, output_dir, **options):
    destination = Path(output_dir)
    info = destination / "meta/info.json"
    info.parent.mkdir(parents=True, exist_ok=True)
    count = json.loads(info.read_text())["total_episodes"] if info.exists() else 0
    write_json(info, {"total_episodes": count + 1})
    return {"vla_training_ready": True, "validation": {"ok": True}, "total_episodes": count + 1}


class StreamExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="repeat streaming ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / "raw/run_repeat_r1_w0"
        self.destination = self.root / "publication"

    def test_coarse_mass_policy_is_rechecked_before_publication_and_retained_after_cleanup(self):
        from dataclasses import asdict
        from baseline.coarse_acceptance import episode_mass_quality
        from baseline.repeat_runtime import RepeatConfig
        from baseline.tuning import delivery_mass_quality
        cfg = RepeatConfig(scoops=10, acceptance_profile='coarse_mass', min_delivered_kg=50.)
        for retained, valid in ((159, False), (160, True)):
            run = self.root / f'mass_retained_{retained}'
            episode = make_episode(run)
            meta = json.loads((episode / 'meta.json').read_text())
            metrics = meta['repeat_metrics']
            metrics.update(particle_mass_kg=5., payload_particles=retained,
                scoop_cycles=[dict(success=True, payload_particles=20)] * 10)
            model = dict(profile='coarse', active=True, error='', metric_point_mass_kg=5.)
            meta['repeat_baseline'] = dict(config=asdict(cfg), soil=dict(backend='newton_mpm', mpm=model), mpm_final=model)
            meta['scoop_results'] = [dict(success=True, execution_success=True, freeze_count=0)] * 10
            meta['repeat_episode_quality'] = episode_mass_quality(meta['scoop_results'], cfg, metrics)
            meta['repeat_delivery_quality'] = delivery_mass_quality(metrics['scoop_cycles'], 5., 50., 10)
            write_json(episode / 'meta.json', meta)
            write_json(episode / 'repeat_metrics.json', metrics)
            (run / 'repeat_metrics.jsonl').write_text(json.dumps(dict(episode_id=meta['episode_id'],
                episode_index=meta['episode_index'], **metrics)) + '\n')
            function = Mock(side_effect=fake_export)
            if valid:
                finish_episode(run, episode, self.destination, function)
                retained_meta = json.loads((episode / 'meta.json').read_text())
                self.assertEqual(retained_meta['repeat_episode_quality']['deposit_retention_ratio'], .8)
                function.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, 'Coarse mass acceptance'):
                    finish_episode(run, episode, self.destination, function)
                function.assert_not_called()
                self.assertTrue((episode / 'trajectory.jsonl').exists())

    def test_a_small_last_load_cannot_be_published_and_raw_is_retained(self):
        from baseline.tuning import delivery_mass_quality
        for last, valid in ((49, False), (50, True)):
            run = self.root / f'load_{last}'
            episode = make_episode(run)
            meta = json.loads((episode / 'meta.json').read_text())
            metrics = meta['repeat_metrics']
            metrics.update(particle_mass_kg=1., scoop_cycles=[{'payload_particles': 100}, {'payload_particles': last}])
            meta['repeat_baseline'] = {'config': {'scoops': 2, 'min_delivered_kg': 50.}}
            meta['repeat_delivery_quality'] = delivery_mass_quality(metrics['scoop_cycles'], 1., 50., 2)
            write_json(episode / 'meta.json', meta)
            write_json(episode / 'repeat_metrics.json', metrics)
            (run / 'repeat_metrics.jsonl').write_text(json.dumps({'episode_id': meta['episode_id'],
                'episode_index': meta['episode_index'], **metrics}) + '\n')
            function = Mock(side_effect=fake_export)
            if valid:
                finish_episode(run, episode, self.destination, function)
                retained = json.loads((episode / 'meta.json').read_text())
                self.assertEqual(retained['repeat_delivery_quality']['delivered_mass_kg'], [100., 50.])
                function.assert_called_once()
            else:
                with self.assertRaisesRegex(ValueError, 'delivered mass gate'):
                    finish_episode(run, episode, self.destination, function)
                function.assert_not_called()
                self.assertTrue((episode / 'trajectory.jsonl').exists())

    def test_policy_python_restores_environment_before_isaac_setup(self):
        original = {"PATH": "/policy/bin", "PYTHONPATH": "/policy/packages", "PYTHONHOME": None,
                    "LD_LIBRARY_PATH": "/policy/lib", "LD_PRELOAD": None}
        polluted = {"REPEAT_POLICY_ENV": json.dumps(original), "PATH": "/isaac/bin",
                    "PYTHONHOME": "/isaac/python", "LD_PRELOAD": "/isaac/libcarb.so",
                    "ISAAC_PATH": "/isaac", "CUDA_VISIBLE_DEVICES": "1"}
        with patch.dict(os.environ, polluted):
            restored = policy_environment()
        for key, value in original.items():
            self.assertEqual(restored.get(key), value)
        self.assertNotIn("ISAAC_PATH", restored)
        self.assertEqual(restored["CUDA_VISIBLE_DEVICES"], "1")

    def test_export_commits_before_cleanup_and_reentry_never_duplicates(self):
        episode = make_episode(self.run)
        before = (self.run / "repeat_metrics.jsonl").read_bytes()
        def export(run, **options):
            self.assertTrue((episode / "trajectory.jsonl").is_file())
            self.assertTrue((episode / "images/camera.png").is_file())
            self.assertEqual(options["episode_offset"], 0)
            self.assertEqual(options["state_schema"], "current-v4")
            self.assertEqual(options["effort_policy"], "exclude")
            self.assertTrue(options["require_standard"] and options["require_vla"] and options["append"])
            return fake_export(run, **options)
        function = Mock(side_effect=export)
        receipt = finish_episode(self.run, episode, self.destination, function)
        self.assertTrue(receipt["validated"] and receipt["raw_pruned"])
        self.assertFalse((episode / "images").exists())
        self.assertFalse((episode / "trajectory.jsonl").exists())
        self.assertEqual((self.run / "repeat_metrics.jsonl").read_bytes(), before)
        self.assertNotIn("plan", json.loads((episode / "meta.json").read_text()))
        finish_episode(self.run, episode, self.destination, function)
        function.assert_called_once()
        self.assertEqual(verify_run(self.run, 1, self.destination), 1)
        second = make_episode(self.run, 2)
        finish_episode(self.run, second, self.destination, fake_export)
        self.assertEqual(verify_run(self.run, 2, self.destination), 2)

    def test_failed_or_unvalidated_export_preserves_every_raw_file(self):
        for label, function in (("failure", Mock(side_effect=RuntimeError("disk full"))),
                                ("invalid", Mock(return_value={"vla_training_ready": False}))):
            with self.subTest(label=label):
                run = self.root / label
                episode = make_episode(run)
                before = {p.relative_to(episode): p.read_bytes() for p in episode.rglob('*') if p.is_file()}
                with self.assertRaises(RuntimeError):
                    finish_episode(run, episode, self.destination, function)
                after = {p.relative_to(episode): p.read_bytes() for p in episode.rglob('*') if p.is_file()}
                self.assertEqual(before, after)

    def test_rejected_episode_retains_metrics_and_removes_raw_without_export(self):
        episode = make_episode(self.run, status="rejected")
        function = Mock()
        self.assertIsNone(finish_episode(self.run, episode, self.destination, function))
        function.assert_not_called()
        self.assertTrue((episode / "repeat_metrics.json").is_file())
        self.assertTrue((episode / "score.json").is_file())
        self.assertFalse((episode / "images").exists())
        self.assertFalse((episode / "events.jsonl").exists())

    def test_unpersisted_metrics_and_paths_outside_run_cannot_be_pruned(self):
        episode = make_episode(self.run)
        with self.assertRaises(ValueError):
            finish_episode(self.root, episode, self.destination, fake_export)
        link = self.run / 'episode_999999'
        link.symlink_to(episode, target_is_directory=True)
        with self.assertRaises(ValueError):
            finish_episode(self.run, link, self.destination, fake_export)
        (self.run / 'repeat_metrics.jsonl').write_text('')
        with self.assertRaises(ValueError):
            finish_episode(self.run, episode, self.destination, fake_export)
        self.assertTrue((episode / 'images/camera.png').exists())

    def test_workers_wait_for_shared_export_lock(self):
        other_run = self.root / "raw/run_repeat_r1_w1"
        episodes = [(self.run, make_episode(self.run)), (other_run, make_episode(other_run))]
        active, maximum = 0, 0
        guard = threading.Lock()
        def export(run, **options):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(.05)
            result = fake_export(run, **options)
            with guard:
                active -= 1
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(finish_episode, run, episode, self.destination, export) for run, episode in episodes]
            for future in futures:
                future.result(timeout=5)
        self.assertEqual(maximum, 1)
        self.assertEqual(json.loads((self.destination / 'meta/info.json').read_text())['total_episodes'], 2)

    def test_real_export_encodes_and_appends_after_previous_raw_is_pruned(self):
        from baseline.tests.test_validation import fixture, write
        from baseline.validate_repeat import validate_episode, validate_export
        export_function = exporter()
        for index in (1, 2):
            episode = make_episode(self.run, index)
            original = json.loads((episode / 'meta.json').read_text())
            rows, meta = fixture(episode)
            meta.update(episode_id=original['episode_id'], episode_index=index,
                        final_metrics={'samples': 4}, repeat_metrics=original['repeat_metrics'],
                        target_xyz=[0., 7., 0.], unload_landing_xyz=[4., 0., 3.],
                        scene_randomization={'applied': {'truck_yaw_deg': 0.}})
            for row in rows:
                row.update({'obs.q_cmd': [.02 * row['i'], 0., 0., 0.],
                            'bucket.pour': [0., 0., 3.],
                            'phase': 'unload_to_bin' if row['phase.index'] == 9 else 'pre_dig'})
            meta['repeat_baseline']['version'] = 'repeat_flat_fill_v2'
            meta['repeat_metrics'].update(samples=4, scoops_completed=2, returns_completed=2,
                                           retained_particles=0, scooped_particles=10,
                                           scoop_cycles=[{'success': True}] * 2)
            write(episode, rows, meta)
            write_json(episode / 'repeat_metrics.json', meta['repeat_metrics'])
            for name in ('repeat_metrics.jsonl', 'episodes.jsonl', 'trainable_episodes.jsonl'):
                records = read_jsonl(self.run / name)
                records[-1].update(meta['repeat_metrics'])
                if name != 'repeat_metrics.jsonl':
                    records[-1].update(trajectory=str(episode / 'trajectory.jsonl'), meta=str(episode / 'meta.json'))
                (self.run / name).write_text(''.join(json.dumps(row) + '\n' for row in records))
            finish_episode(self.run, episode, self.destination, export_function)
            self.assertFalse((episode / 'trajectory.jsonl').exists())
            retained = validate_episode(episode)
            self.assertEqual(retained['validation_mode'], 'retained_metrics_and_export_receipt')
            self.assertFalse(retained['raw_trajectory_available'])
        self.assertEqual(verify_run(self.run, 2, self.destination), 2)
        publication = validate_export(self.destination)
        self.assertEqual(publication['episodes'], 2)
        self.assertEqual(publication['frames'], 8)
        self.assertEqual(set(publication['camera_frames'].values()), {8})


class MetricsSummaryTests(unittest.TestCase):
    def test_batch_includes_all_workers_and_retry_rounds_without_historical_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "export"
            batch = "repeat_test_123"
            for suffix, number, payload in (("r1_w0", 2, 8), ("r1_w1", 1, 18), ("r2_w0", 1, 28)):
                run = root / f"run_20260101_{batch}_{suffix}"
                for index in range(1, number + 1):
                    episode = make_episode(run, index, payload=payload, scooped=payload + 2)
                    finish_episode(run, episode, destination, fake_export)
                write_json(run / "summary.json", {"attempts": number, "planning_diagnostics": 0})
            make_episode(root / 'run_20260101_repeat_older_r1_w0', payload=999)
            result = combine_batch(root, batch, destination, 4, 'complete')
            self.assertTrue(result['completed'])
            self.assertEqual(len(result['source_files']), 3)
            self.assertEqual(result['exported_episode_count'], 4)
            averages = result['combined']['all_recorded_episodes']['averages']
            self.assertEqual(averages['payload_mass_kg'], (4 + 4 + 9 + 14) / 4)
            pooled = result['combined']['all_recorded_episodes']['pooled_spillage_fraction']
            self.assertAlmostEqual(pooled, 8 / 70)
            self.assertNotAlmostEqual(pooled, averages['spillage_fraction'])

    def test_null_values_have_metric_specific_denominators_and_paired_rates(self):
        result = summarize([
            {'status': 'trainable', 'payload_particles': 8, 'simulated_runtime_seconds': 2,
             'runtime_seconds': 10, 'cycle_time_seconds': 2},
            {'status': 'rejected', 'payload_particles': None, 'simulated_runtime_seconds': 9,
             'runtime_seconds': 20, 'cycle_time_seconds': None},
        ])
        self.assertEqual(result['averages']['runtime_seconds'], 15)
        self.assertEqual(result['averages']['cycle_time_seconds'], 2)
        self.assertEqual(result['non_null_episode_counts']['cycle_time_seconds'], 1)
        self.assertIsNone(result['averages']['spillage_fraction'])
        self.assertEqual(result['pooled_payload_particles_per_sim_second'], 4)
        self.assertIsNone(result['totals']['spillage_particles'])


if __name__ == '__main__':
    unittest.main()
