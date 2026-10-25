import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from baseline.collect_repeat import parse_args


ROOT = Path(__file__).resolve().parents[2]


class LauncherTests(unittest.TestCase):
    def test_cli_rejects_single_scoop(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--scoops-per-episode", "1"])

    def test_cli_preserves_isaac_options(self):
        args, forwarded = parse_args(["--scoops-per-episode", "4", "--scene-seed", "2"])
        self.assertEqual(args.scoops_per_episode, 4)
        self.assertEqual(forwarded, ["--scene-seed", "2"])

    def test_shell_distributes_and_exports_workers_sequentially(self):
        with tempfile.TemporaryDirectory(prefix="repeat launch ") as directory:
            directory = Path(directory)
            isaac = directory / "fake_isaac"
            policy = directory / "fake_policy"
            calls = directory / "calls.jsonl"
            # Real shell launcher, mocked simulator/exporter processes. Spaces
            # in paths exercise quoting and relative DATA_ROOT resolution.
            isaac.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
def arg(key): return args[args.index(key) + 1]
target = int(arg('--success-count'))
path = pathlib.Path(os.environ['EXCAVATOR_AUTO_COLLECT_RESULT_FILE'])
run = pathlib.Path(arg('--dataset-root')) / ('run_' + path.stem)
run.mkdir(parents=True, exist_ok=True)
with open(os.environ['REPEAT_TEST_CALLS'], 'a') as handle:
    handle.write(json.dumps({'kind': 'isaac', 'args': args, 'target': target}) + '\\n')
path.write_text(json.dumps({'run_dir': str(run), 'successes': target,
                            'requested': target, 'completed': True, 'loop_cancelled': False}))
''')
            policy.write_text(f"#!{sys.executable}\n" + '''
import json, os, sys
if len(sys.argv) > 1 and sys.argv[1].endswith('excavator_dataset_tools.py'):
    with open(os.environ['REPEAT_TEST_CALLS'], 'a') as handle:
        handle.write(json.dumps({'kind': 'export', 'args': sys.argv[1:]}) + '\\n')
else:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
            isaac.chmod(0o755)
            policy.chmod(0o755)
            environment = {**os.environ, "ISAAC_PYTHON": str(isaac), "POLICY_PYTHON": str(policy),
                           "DATA_ROOT": str(directory / "collected data"), "REPEAT_TEST_CALLS": str(calls)}
            result = subprocess.run(["bash", str(ROOT / "collect_repeat.sh"), "3", "2", "1,0", "3",
                                     "--scene-seed", "8"], env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            records = [json.loads(line) for line in calls.read_text().splitlines()]
            workers = [row for row in records if row["kind"] == "isaac"]
            exports = [row for row in records if row["kind"] == "export"]
            self.assertEqual(sorted(row["target"] for row in workers), [1, 2])
            self.assertEqual(len(exports), 3)
            for worker in workers:
                self.assertIn(str(ROOT / "baseline/collect_repeat.py"), worker["args"])
                self.assertIn("--scoops-per-episode", worker["args"])
                self.assertIn("--scene-seed", worker["args"])
            for export in exports:
                self.assertIn("--export-require-standard", export["args"])
                self.assertIn("--export-require-vla", export["args"])
                self.assertIn("exclude", export["args"])

    def test_shell_rejects_invalid_counts_before_launch(self):
        for args in (["0"], ["1", "0"], ["1", "1", "bogus"], ["1", "1", "0", "1"]):
            result = subprocess.run(["bash", str(ROOT / "collect_repeat.sh"), *args], capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
