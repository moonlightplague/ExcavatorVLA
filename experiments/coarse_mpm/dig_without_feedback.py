"""Diagnostic only: identical MPM dig, without soil-to-robot force feedback."""
from pathlib import Path
import runpy
import sys
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
def no_feedback(self,dt):
    self.applied_impulse.fill(0)
soil.NewtonSoil._on_pre_step=no_feedback
print('[DIAGNOSTIC ONLY] Soil feedback disabled; controller and collisions unchanged',flush=True)
runpy.run_path(str(Path(__file__).with_name('dig_graph_test.py')),run_name='__main__')
