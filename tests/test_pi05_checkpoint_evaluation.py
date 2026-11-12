"""Checkpoint comparisons must wait for the parent and include every save."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('checkpoint_evaluation', SCRIPTS / 'eval_pi05_checkpoints.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'train'
        self.dataset = self.root / 'dataset'
        dump(self.run / 'resolved_settings.json', {'dataset': {'root': str(self.dataset)},
                                                  'policy': {'chunk_size': 10}})
        dump(self.run / 'provenance.json', {'dataset_fingerprint': 'fixture',
                                           'collection_backend': {'soil_backend': 'physx'}})
        dump(self.run / 'split_manifest.json', {'train': [0], 'validation': [20], 'test': [24]})
        dump(self.run / 'normalization_stats.json', {'action': {'min': [0]}})
        dump(self.dataset / 'backend_provenance.json', {'soil_backend': 'physx'})
        for episode in (0, 20, 24):
            dump(self.dataset / 'references' / f'episode_{episode:04d}.json',
                 {'provenance': {'soil_backend': 'physx', 'source_export_episode': episode + 40}})

    def args(self, *extra):
        return evaluation.parse_args(['--train-dir', str(self.run), '--output', str(self.root / 'eval'), *extra])

    def checkpoint(self, step):
        path = self.run / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
        path.mkdir(parents=True)
        (path / 'model.safetensors').write_bytes(f'fixture {step}'.encode())
        (path / 'normalization_stats.json').write_bytes((self.run / 'normalization_stats.json').read_bytes())
        dump(path / 'benchmark_provenance.json', {'dataset_fingerprint': 'fixture'})
        dump(path / 'checkpoint.json', {'format': 'excavator_pi05_trainable_delta_v1',
             'step': step, 'base_sha256': 'same_base', 'precision_contract': 'same_precision',
             'finetuning': {'mode': 'vlm_lora_expert_full'},
             'delta_sha256': evaluation.digest(path / 'model.safetensors')})
        return path

    def test_default_plan_compares_five_model_checkpoints_in_the_same_scene(self):
        plan = evaluation.build_plan(self.args())
        self.assertEqual([job['checkpoint_step'] for job in plan['jobs']], [2000, 4000, 6000, 8000, 10000])
        self.assertEqual(len({job['output'] for job in plan['jobs']}), 5)
        for job in plan['jobs']:
            command = job['command']
            self.assertEqual(job['reference_episode'], 0)
            self.assertEqual(command[command.index('--soil-backend') + 1], 'physx')
            self.assertEqual(command[command.index('--reference-episode') + 1], str(self.dataset / 'references/episode_0000.json'))
            self.assertIn('--video', command)
            self.assertNotIn('--replay-reference', command)

    def test_missing_later_checkpoint_is_not_silently_skipped(self):
        self.checkpoint(2000)
        self.checkpoint(4000)
        with self.assertRaises(FileNotFoundError):
            evaluation.check_checkpoints(evaluation.build_plan(self.args()))

    def test_validation_reference_and_different_backend_are_distinguished(self):
        args = self.args('--reference-episodes', '20')
        self.assertTrue(all(job['reference_split'] == 'validation' for job in evaluation.build_plan(args)['jobs']))
        dump(self.dataset / 'references/episode_0020.json', {'provenance': {'soil_backend': 'newton_mpm'}})
        with self.assertRaisesRegex(ValueError, 'backend'):
            evaluation.build_plan(args)

    def test_tampered_normalization_data_or_weight_files_are_rejected(self):
        path = self.checkpoint(2000)
        args = self.args('--checkpoint-steps', '2000')
        evaluation.check_checkpoints(evaluation.build_plan(args))
        cases = [('normalization_stats.json', b'{}'),
                 ('benchmark_provenance.json', b'{"dataset_fingerprint":"different"}'),
                 ('model.safetensors', b'changed')]
        for name, replacement in cases:
            with self.subTest(name=name):
                original = (path / name).read_bytes()
                (path / name).write_bytes(replacement)
                try:
                    with self.assertRaises(ValueError):
                        evaluation.check_checkpoints(evaluation.build_plan(args))
                finally:
                    (path / name).write_bytes(original)

    def test_finished_training_does_not_bypass_running_parent_evaluation(self):
        dump(self.root / 'status.json', {'status': 'running', 'training_completed': True,
                                         'phase': 'final_physx_closed_loop_videos', 'pid': os.getpid()})
        with self.assertRaisesRegex(RuntimeError, 'still running'):
            evaluation.wait_for_parent(self.root, wait=False)
        dump(self.root / 'status.json', {'status': 'completed'})
        evaluation.wait_for_parent(self.root, wait=False)

    def test_free_memory_does_not_bypass_an_existing_gpu_compute_process(self):
        with patch.object(evaluation.subprocess, 'check_output', side_effect=[
                '0, GPU-fixture, 23000, 0\n', 'GPU-fixture, 123\n']):
            self.assertFalse(evaluation.gpu_idle(0))

    def test_parallel_repeated_queues_cover_each_checkpoint_once_latest_first(self):
        args = self.args('--gpus', '0,1', '--checkpoint-steps', *map(str, range(2000, 20001, 2000)),
                         '--repeat-cycles', '5', '--rollout-steps', '2500', '--latest-first')
        queues = evaluation.parallel_queues(args)
        self.assertEqual([queue['gpu'] for queue in queues], [0, 1])
        self.assertEqual(queues[0]['plan']['checkpoint_steps'], [20000, 16000, 12000, 8000, 4000])
        self.assertEqual(queues[1]['plan']['checkpoint_steps'], [18000, 14000, 10000, 6000, 2000])
        self.assertEqual(sorted(job['checkpoint_step'] for queue in queues for job in queue['plan']['jobs']),
                         list(range(2000, 20001, 2000)))
        for queue in queues:
            for job in queue['plan']['jobs']:
                self.assertIn('--repeat-cycles', job['command'])
                self.assertNotIn('--replay-reference', job['command'])


if __name__ == '__main__':
    unittest.main()
