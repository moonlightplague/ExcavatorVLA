"""CPU tests of owned experiment outputs and fail-fast orchestration."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_pi05_lora_ab.py'
SPEC = importlib.util.spec_from_file_location('pi05_ab_runner_test_module', SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RunnerTests(unittest.TestCase):
    def test_continuation_budget_and_mismatched_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            (parent / 'status.json').write_text(json.dumps({'status': 'completed'}))
            for arm, mode in (('expert', 'expert_only'), ('lora', 'vlm_lora_expert_full')):
                run = parent / f'{arm}_overfit'
                checkpoint = run / 'checkpoints/002000/pretrained_model'
                checkpoint.mkdir(parents=True)
                (checkpoint / 'checkpoint.json').write_text(json.dumps({'step': 2000, 'finetuning': {'mode': mode}}))
                (checkpoint.parent / 'training_state.pt').write_bytes(b'synthetic-fixture-only')
                (run / 'last_checkpoint.json').write_text(json.dumps({'step': 2000, 'checkpoint': str(checkpoint)}))
                (run / 'provenance.json').write_text(json.dumps(
                    {'batch_contract': {'world_size': 2, 'batch_size': 1, 'accumulation': 4}}))
                (run / 'resolved_settings.json').write_text(json.dumps(
                    {'save_freq': 500, 'benchmark': {'validation_freq': 500}}))
            plan = RUNNER.continuation_plan(parent, 1000)
            self.assertEqual((plan['start_step'], plan['target_step']), (2000, 3000))
            path = parent / 'lora_overfit/last_checkpoint.json'
            changed = json.loads(path.read_text())
            changed['step'] = 1500
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                RUNNER.continuation_plan(parent, 1000)

    def test_user_approved_gpu_sharing_is_memory_guarded(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory) / 'experiment', gpus=[0, 1], steps=3000,
                                      allow_shared_gpus=True)
            experiment = RUNNER.Experiment(args)
            rows = '0, 626, 24564, 8, GPU-A\n1, 7, 24564, 0, GPU-B\n'
            with patch.object(RUNNER.subprocess, 'check_output', return_value=rows), \
                    patch.object(RUNNER.time, 'sleep') as sleep:
                experiment.idle()
                sleep.assert_not_called()

    def test_constant_resume_lr_multiplier(self):
        from excavator_vlm.policies.pi05.train import lr_multiplier
        spec = {'kind': 'constant_resume', 'learning_rate': 2.5e-5, 'base_lr': 2.5e-5}
        for step in (2000, 2001, 2500, 3000):
            self.assertEqual(lr_multiplier(step, spec), 1.)

    def test_stage_logs_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory) / 'experiment', gpus=[0, 1], steps=2000)
            experiment = RUNNER.Experiment(args)
            with patch.object(experiment, 'idle'), patch.object(experiment, 'check_sources'), \
                    patch.object(RUNNER.time, 'sleep'):
                experiment.run('unit_stage', [([sys.executable, '-c', 'print("stage-ok")'], {})])
            self.assertEqual(experiment.state['stages'][0]['status'], 'completed')
            self.assertIn('stage-ok', (args.output / 'logs/unit_stage_0.log').read_text())
            before = (args.output / 'status.json').read_text()
            with self.assertRaises(FileExistsError):
                RUNNER.Experiment(args)
            self.assertEqual(before, (args.output / 'status.json').read_text())

    def test_failures_and_source_drift_stop_the_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(output=Path(directory) / 'experiment', gpus=[0, 1], steps=2000)
            experiment = RUNNER.Experiment(args)
            with patch.object(experiment, 'idle'), patch.object(experiment, 'check_sources'), \
                    patch.object(RUNNER.time, 'sleep'), self.assertRaisesRegex(RuntimeError, 'failed'):
                experiment.run('failing_stage', [([sys.executable, '-c', 'raise SystemExit(7)'], {})])
            experiment.state['source_hashes'] = {}
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                experiment.check_sources()


if __name__ == '__main__':
    unittest.main()
