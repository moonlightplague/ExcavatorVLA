"""CPU-only tests: queued video must not compete with the active training job."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/queue_pi05_video.py'
SPEC = importlib.util.spec_from_file_location('pi05_video_queue_test_module', SCRIPT)
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class VideoQueueTests(unittest.TestCase):
    def test_idle_gpu_requires_enough_memory_and_no_compute_process(self):
        rows = '0, 16819, 24564, GPU-A\n1, 7, 24564, GPU-B\n'
        with patch.object(QUEUE.subprocess, 'check_output', side_effect=[rows, '']):
            self.assertTrue(QUEUE.gpu_available(1))
        with patch.object(QUEUE.subprocess, 'check_output', side_effect=[rows, 'GPU-B, 42\n']):
            self.assertFalse(QUEUE.gpu_available(1))
        with patch.object(QUEUE.subprocess, 'check_output', side_effect=[rows, '']):
            self.assertFalse(QUEUE.gpu_available(0))

    def test_running_training_is_waited_for_without_gpu_or_evaluator_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = argparse.Namespace(wait_for=Path(directory) / 'upstream.json')
            state = {}
            with patch.object(QUEUE, 'read', return_value={'status': 'running', 'phase': 'training'}), \
                    patch.object(QUEUE.time, 'sleep', side_effect=RuntimeError('stop-unit-wait')), \
                    patch.object(QUEUE.subprocess, 'run') as run, \
                    patch.object(QUEUE, 'gpu_available') as gpu, \
                    self.assertRaisesRegex(RuntimeError, 'stop-unit-wait'):
                QUEUE.execute(args, state, lambda: None)
            self.assertEqual(state['status'], 'waiting_for_experiment')
            run.assert_not_called()
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
