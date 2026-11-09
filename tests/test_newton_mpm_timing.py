"""MPM substeps preserve the requested rate across PhysX callback durations."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/"scripts/excavator_app/newton_mpm_soil.py"
spec = importlib.util.spec_from_file_location("standalone_newton_soil", SOURCE)
soil = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soil)


class MPMStepTests(unittest.TestCase):
    def test_120_hz_default_and_60_hz_profile(self):
        for frequency in (60, 120):
            with patch.dict(os.environ, EXCAVATOR_MPM_HZ=str(frequency)):
                for dt in (1/30, 1/60, 1/120):
                    steps = soil.mpm_substeps(dt)
                    self.assertGreaterEqual(steps, 1)
                    self.assertLessEqual(dt/steps, 1/frequency + 1e-10)
        with patch.dict(os.environ, EXCAVATOR_MPM_HZ="120"):
            self.assertEqual(soil.mpm_substeps(1/60), 2)

    def test_bad_rates_and_step_durations_fail_before_simulation(self):
        for frequency in ('0', '30', '300', 'nan', 'inf'):
            with patch.dict(os.environ, EXCAVATOR_MPM_HZ=frequency), self.assertRaises(ValueError):
                soil.mpm_substeps(1/60)
        for dt in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                soil.mpm_substeps(dt)


if __name__ == '__main__':
    unittest.main()
