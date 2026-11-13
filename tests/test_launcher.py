import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from baseline.collect_repeat import parse_args


ROOT = Path(__file__).resolve().parents[2]


class LauncherTests(unittest.TestCase):
    def test_mass_profile_is_explicit_and_rejects_incompatible_or_invalid_settings(self):
        args, forwarded = parse_args(['--mpm-profile', 'coarse', '--acceptance-profile', 'coarse_mass'])
        self.assertEqual(args.acceptance_profile, 'coarse_mass')
        self.assertEqual(args.min_delivered_kg, 50.)
        self.assertEqual(forwarded, [])
        for argv in (['--mpm-profile', 'fine'], ['--soil-backend', 'physx'],
                     ['--min-delivered-kg', '0'], ['--min-retained-delivery-ratio', 'nan'],
                     ['--min-retained-delivery-ratio', '1.1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--acceptance-profile', 'coarse_mass', *argv])

    def test_cli_rejects_single_scoop(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--scoops-per-episode", "1"])

    def test_cli_preserves_isaac_options(self):
        args, forwarded = parse_args(["--scoops-per-episode", "4", "--scene-seed", "2"])
        self.assertEqual(args.scoops_per_episode, 4)
        self.assertEqual(forwarded, ["--scene-seed", "2"])

    def test_cli_selects_coarse_or_explicit_fine_without_forwarding_the_profile(self):
        with patch.dict(os.environ, EXCAVATOR_MPM_PROFILE="coarse"):
            self.assertEqual(parse_args([])[0].mpm_profile, "coarse")
            args, forwarded = parse_args(["--mpm-profile", "fine", "--mpm-particle-budget", "32000"])
            self.assertEqual(args.mpm_profile, "fine")
            self.assertEqual(args.mpm_particle_budget, 32000)
            self.assertEqual(forwarded, [])

    def test_coarse_defaults_to_fifty_kg_in_every_dig_and_keeps_explicit_overrides(self):
        self.assertEqual(parse_args(['--mpm-profile', 'coarse'])[0].min_delivered_kg, 50.)
        self.assertEqual(parse_args(['--mpm-profile', 'fine'])[0].min_delivered_kg, 0.)
        self.assertEqual(parse_args(['--soil-backend', 'physx'])[0].min_delivered_kg, 0.)
        self.assertEqual(parse_args(['--min-delivered-kg', '70'])[0].min_delivered_kg, 70.)
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--min-delivered-kg', value])

    def run_shell(self, directory, skip_export=False):
        isaac = directory / "fake_isaac"
        calls = directory / "calls.jsonl"
        # Real shell and storage/summary helpers; only simulator/encoding are
        # mocked. Two worker processes exercise the shared filesystem lock.
        isaac.write_text(f"#!{sys.executable}\n" + f"import sys\nsys.path.insert(0, {str(ROOT)!r})\n" + r'''
import json, os, pathlib
from baseline.stream_export import finish_episode
from baseline.tests.test_stream_export import make_episode, fake_export
args = sys.argv[1:]
def arg(key): return args[args.index(key) + 1]
def log(value):
    with open(os.environ['REPEAT_TEST_CALLS'], 'a') as handle:
        handle.write(json.dumps(value) + '\n')
target = int(arg('--success-count'))
path = pathlib.Path(os.environ['EXCAVATOR_AUTO_COLLECT_RESULT_FILE'])
run = pathlib.Path(arg('--dataset-root')) / ('run_fake_' + os.environ['EXCAVATOR_AUTO_RUN_ID_SUFFIX'])
destination = arg('--stream-export-dir')
log({'kind': 'isaac', 'args': args, 'target': target})
def export(run, **options):
    log({'kind': 'export', 'options': options})
    return fake_export(run, **options)
for index in range(1, target + 1):
    episode = make_episode(run, index)
    if not os.environ.get('REPEAT_TEST_SKIP_EXPORT'):
        finish_episode(run, episode, destination, export)
        assert not (episode / 'trajectory.jsonl').exists()
        log({'kind': 'pruned_before_next_episode', 'index': index})
(run / 'summary.json').write_text(json.dumps({'attempts': target, 'planning_diagnostics': 0}))
path.write_text(json.dumps({'run_dir': str(run), 'successes': target, 'requested': target,
                            'completed': True, 'loop_cancelled': False,
                            'loop_exit_reason': 'target_successes_reached'}))
''')
        isaac.chmod(0o755)
        environment = {**os.environ, "ISAAC_PYTHON": str(isaac), "POLICY_PYTHON": sys.executable,
                       "DATA_ROOT": "collected data", "REPEAT_TEST_CALLS": str(calls)}
        if skip_export:
            environment['REPEAT_TEST_SKIP_EXPORT'] = '1'
        return subprocess.run(["bash", str(ROOT / "collect_repeat.sh"), "3", "2", "1,0", "3",
                               "--scene-seed", "8"], env=environment, cwd=directory,
                              capture_output=True, text=True, timeout=30)

    def test_shell_streams_workers_and_writes_combined_averages(self):
        with tempfile.TemporaryDirectory(prefix="repeat launch ") as directory:
            directory = Path(directory)
            calls = directory / "calls.jsonl"
            result = self.run_shell(directory)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in calls.read_text().splitlines()]
            workers = [row for row in records if row["kind"] == "isaac"]
            exports = [row for row in records if row["kind"] == "export"]
            self.assertEqual(sorted(row["target"] for row in workers), [1, 2])
            self.assertEqual(len(exports), 3)
            self.assertEqual(len([r for r in records if r['kind'] == 'pruned_before_next_episode']), 3)
            for worker in workers:
                self.assertIn(str(ROOT / "baseline/collect_repeat.py"), worker["args"])
                self.assertIn("--scoops-per-episode", worker["args"])
                self.assertIn("--scene-seed", worker["args"])
                self.assertIn("--stream-export-dir", worker["args"])
            for export in exports:
                self.assertTrue(export['options']['require_standard'])
                self.assertTrue(export['options']['require_vla'])
                self.assertEqual(export['options']['effort_policy'], 'exclude')
            summary_path, = (directory / 'collected data/metrics').glob('*_summary.json')
            summary = json.loads(summary_path.read_text())
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['exported_episode_count'], 3)
            self.assertEqual(summary['combined']['all_recorded_episodes']['episode_count'], 3)
            self.assertAlmostEqual(summary['combined']['trainable_episodes']['averages']['runtime_seconds'], 40 / 3)

    def test_shell_rejects_worker_success_without_committed_export(self):
        with tempfile.TemporaryDirectory(prefix="repeat missing export ") as directory:
            directory = Path(directory)
            result = self.run_shell(directory, skip_export=True)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            summary_path, = (directory / 'collected data/metrics').glob('*_summary.json')
            summary = json.loads(summary_path.read_text())
            self.assertFalse(summary['completed'])
            self.assertEqual(summary['status'], 'failed')
            self.assertEqual(summary['exported_episode_count'], 0)
            self.assertEqual(len(list((directory / 'collected data/raw').glob('run_*/episode_*/trajectory.jsonl'))), 3)

    def test_shell_rejects_invalid_counts_before_launch(self):
        for args in (["0"], ["1", "0"], ["1", "1", "bogus"], ["1", "1", "0", "1"]):
            result = subprocess.run(["bash", str(ROOT / "collect_repeat.sh"), *args], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
