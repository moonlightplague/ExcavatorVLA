"""Rolling state must preserve scheduled checkpoints and deterministic training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SUBMODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMODULE))
sys.path.insert(0, str(SUBMODULE / 'scripts'))
from excavator_vlm.policies.pi05.checkpoints import FORMAT, retain_checkpoints
from excavator_vlm.policies.pi05.config import resolve_settings
from run_pi05_repeat_history import RepeatHistoryExperiment


class RecoveryTests(unittest.TestCase):
    def checkpoint(self, run, step):
        folder = run / 'checkpoints' / f'{step:06d}' / 'pretrained_model'
        folder.mkdir(parents=True)
        (folder / 'checkpoint.json').write_text(json.dumps({'format': FORMAT, 'step': step}))
        return folder

    def test_rolling_saves_preserve_all_requested_milestones(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (2, 100, 1900, 2000, 2100, 3900, 4000, 4100):
                self.checkpoint(root, step)
            latest = root / 'checkpoints/004100/pretrained_model'
            (root / 'last_checkpoint.json').write_text(json.dumps({'step': 4100, 'checkpoint': str(latest)}))
            retain_checkpoints(root, 10, 2000)
            self.assertEqual(sorted(p.name for p in (root / 'checkpoints').iterdir()), ['002000', '004000', '004100'])

    def test_unknown_and_symlink_directories_are_never_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = self.checkpoint(root, 100)
            (old / 'checkpoint.json').write_text(json.dumps({'format': 'not_our_checkpoint'}))
            self.checkpoint(root, 200)
            (root / 'checkpoints/000050').symlink_to(root / 'checkpoints/000100', target_is_directory=True)
            retain_checkpoints(root, 10, 2000)
            self.assertTrue(old.is_dir())
            self.assertTrue((root / 'checkpoints/000050').is_symlink())

    def test_recovery_option_does_not_change_policy_or_batch_contract(self):
        path = SUBMODULE / 'configs/pi05/history_c_repeat20k.yaml'
        original = resolve_settings(path)
        changed = resolve_settings(path, {'training': {'recovery_save_freq': 100}, 'num_workers': 0})
        self.assertEqual(changed['policy'], original['policy'])
        self.assertEqual(changed['save_freq'], 2000)
        self.assertEqual(changed['training']['keep_checkpoints'], 10)
        self.assertEqual(changed['batch_size'], original['batch_size'])
        self.assertEqual(changed['training']['gradient_accumulation_steps'], 4)
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                resolve_settings(path, {'training': {'recovery_save_freq': value}})

    def test_shared_mode_uses_free_memory_and_never_kills_other_processes(self):
        experiment = object.__new__(RepeatHistoryExperiment)
        experiment.args = SimpleNamespace(allow_shared_gpus=True, gpus=[0, 1])
        with patch('run_pi05_repeat_history.subprocess.check_output', return_value='0, 18550\n1, 18600\n'), \
                patch('run_pi05_repeat_history.time.sleep') as sleep:
            experiment.idle()
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
