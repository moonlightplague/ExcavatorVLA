import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts" / "run_parallel_linux.sh"


class ParallelLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = LAUNCHER_PATH.read_text(encoding="utf-8-sig")

    def test_worker_runs_in_its_own_process_group(self):
        self.assertIn('setsid "${cmd[@]}"', self.source)
        self.assertIn("child_owns_process_group=1", self.source)

    def test_supervisor_stops_the_complete_worker_group(self):
        self.assertIn('kill -TERM -- "-${child_pid}"', self.source)
        self.assertIn('kill -KILL -- "-${child_pid}"', self.source)
        self.assertIn("stop_supervised_processes", self.source)


if __name__ == "__main__":
    unittest.main()
