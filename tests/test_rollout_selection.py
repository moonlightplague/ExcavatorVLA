"""Selection accounting and dataset overwrite ownership regressions."""
import json
from pathlib import Path
import tempfile
import unittest

from excavator_vlm.rl.select_rollouts import build_selection, parse_args
from excavator_vlm.rl.build_finetune_dataset import prepare_output


class SelectionTests(unittest.TestCase):
    def test_limit_counts_each_rejected_episode_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for seed in range(3):
                folder = run / 'episodes' / str(seed)
                folder.mkdir(parents=True)
                (folder / 'episode_status.json').write_text(json.dumps({
                    'status': 'completed', 'success': True, 'scene_seed': seed,
                    'checkpoint_sha256': 'checkpoint', 'payload_particles': 10 + seed,
                    'policy_steps_executed': 4,
                }))
            result = build_selection(parse_args([
                '--runs', str(run), '--mode', 'top_k', '--top-k', '3', '--max-selected', '1']))
            self.assertEqual(result['totals']['selected'], 1)
            self.assertEqual(result['totals']['rejected'], 2)
            self.assertEqual(result['totals']['rejected_by_reason']['above_limit'], 2)

    def test_overwrite_preserves_unowned_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'data'
            folder.mkdir()
            sentinel = folder / 'source.txt'
            sentinel.write_text('keep')
            with self.assertRaises(ValueError):
                prepare_output(folder, True)
            self.assertEqual(sentinel.read_text(), 'keep')

    def test_overwrite_replaces_owned_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'data'
            folder.mkdir()
            (folder / 'rl_provenance.json').write_text('{}')
            with self.assertRaises(ValueError):
                prepare_output(folder, False)
            prepare_output(folder, True)
            self.assertFalse(folder.exists())


if __name__ == '__main__':
    unittest.main()
