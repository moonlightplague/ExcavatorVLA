"""Validation previews must span episodes; idle ranks must not wait in CUDA."""
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from excavator_vlm.policies.pi05.validation import checkpoint_barrier, checkpoint_group, validation_preview


def control_worker(rank, init_file):
    import torch.distributed as dist
    from excavator_vlm.policies.pi05.train import _gather_rng
    dist.init_process_group('gloo', init_method=f'file://{init_file}', rank=rank,
                            world_size=2, timeout=timedelta(seconds=1))
    accelerator = SimpleNamespace(num_processes=2)
    group = checkpoint_group(accelerator)
    states = _gather_rng(accelerator, group)
    assert len(states) == 2
    if rank == 0:
        time.sleep(2)  # Simulated validation longer than the training PG timeout.
    checkpoint_barrier(accelerator, group)
    # The default training group remains usable after the CPU-only wait.
    value = torch.tensor(rank + 1)
    dist.all_reduce(value)
    assert value.item() == 3
    dist.destroy_process_group(group)
    dist.destroy_process_group()


class ValidationTests(unittest.TestCase):
    def test_preview_uses_actual_rows_and_spans_each_whole_episode(self):
        from datasets import Dataset
        table = Dataset.from_dict({'episode_index': [7, 5, 7, 5, 7, 5, 5],
                                   'frame_index': [20, 30, 0, 10, 10, 0, 20]})
        class Fixture:
            hf_dataset = table
            def __len__(self): return len(table)
            def __getitem__(self, index): raise AssertionError('Must not decode images')
        dataset = Fixture()
        subset, selection = validation_preview(dataset, 2)
        self.assertEqual(subset.indices, [5, 1, 2, 0])
        self.assertEqual(selection['selected_frames'], 4)
        self.assertTrue(selection['partial_evaluation'])
        self.assertEqual(selection['selected_observations'], [
            {'episode_index': 5, 'frame_index': 0}, {'episode_index': 5, 'frame_index': 30},
            {'episode_index': 7, 'frame_index': 0}, {'episode_index': 7, 'frame_index': 20}])
        full, selection = validation_preview(dataset, 0)
        self.assertIs(full, dataset)
        self.assertFalse(selection['partial_evaluation'])
        subset, selection = validation_preview(dataset, 20)
        self.assertEqual(len(subset), 7)
        self.assertFalse(selection['partial_evaluation'])

    def test_two_cpu_ranks_wait_outside_the_training_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(control_worker,
                args=(str(Path(directory) / 'store'),), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
