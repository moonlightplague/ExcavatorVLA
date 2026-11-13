import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from baseline.benchmark_repeat import summarize, interval, ROOT
from baseline.stream_export import finish_episode
from baseline.tests.test_stream_export import make_episode, fake_export


class BenchmarkTests(unittest.TestCase):
    def test_denominator_includes_planning_failures_and_rejected_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root/'worker_0'
            run = worker/'raw/run_test'
            episode = make_episode(run, 1)
            meta_path = episode/'meta.json'
            meta = json.loads(meta_path.read_text())
            meta['repeat_baseline'] = {'config': {'scoops': 10}}
            meta['scoops_completed'] = meta['repeat_returns_completed'] = 10
            meta['scoop_results'] = [dict(success=True, strict_success=True)]*9 + [dict(success=True, strict_success=False)]
            meta_path.write_text(json.dumps(meta))
            finish_episode(run, episode, root/'export', fake_export)
            make_episode(run, 2, status='rejected')
            (run/'planning_diagnostics.jsonl').write_text(json.dumps({'reason':'planning_failed/reach'})+'\n')
            (worker/'result.json').write_text(json.dumps(dict(run_dir=str(run), attempts=3, successes=1)))
            result = summarize(root, 3)
            self.assertTrue(result['complete'])
            self.assertEqual(result['accepted'], 1)
            self.assertEqual(result['exported'], 1)
            self.assertAlmostEqual(result['acceptance_rate'], 1/3)
            self.assertEqual(result['strict_equivalent_episodes'], 0)
            self.assertFalse(result['target_met'])
            self.assertEqual(result['failures']['planning_failed/reach'], 1)
            self.assertFalse(summarize(root, 4)['complete'])

    def test_small_cohort_interval_exposes_uncertainty(self):
        low, high = interval(2, 4)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertIsNone(interval(0, 0))

    def test_exactly_half_is_not_over_half_and_one_scene_is_not_a_guarantee(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / 'worker_0'
            run = worker / 'raw/run_test'
            episode = make_episode(run, 1)
            finish_episode(run, episode, root / 'export', fake_export)
            (worker / 'result.json').write_text(json.dumps(dict(run_dir=str(run), attempts=2, successes=1)))
            half = summarize(root, 2)
            self.assertEqual(half['export_acceptance_rate'], .5)
            self.assertFalse(half['target_met'])
            self.assertFalse(half['export_rate_above_target_95'])
            (worker / 'result.json').write_text(json.dumps(dict(run_dir=str(run), attempts=1, successes=1)))
            one = summarize(root, 1)
            self.assertTrue(one['target_met'])
            self.assertFalse(one['export_rate_above_target_95'])

    def test_invalid_worker_allocation_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            for allocation in (['--workers-per-gpu', '0'],
                               ['--workers-per-gpu', '2', '--attempts', '2']):
                result = subprocess.run([sys.executable, str(ROOT / 'baseline/benchmark_repeat.py'),
                    '--output', str(output), '--gpu-ids', '0,1', *allocation],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
