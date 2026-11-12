"""Controlled MPM 120/60 Hz measurement. Production source is untouched."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import numpy as np

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from excavator_app.newton_mpm_soil import NewtonSoil

output = Path(os.environ['SUBSTEP_RESULT_DIR'])
output.mkdir(parents=True, exist_ok=False)
original = NewtonSoil.advance
snapshots = []
initial_hash = None
benchmark_start = None
next_sample = 1
completed = False


def advance(self, dt):
    global initial_hash, benchmark_start, next_sample, completed
    if initial_hash is None:
        initial_hash = hashlib.sha256(self.positions.tobytes()).hexdigest()
    original(self, dt)
    if completed:
        return
    if self.sim_time >= next_sample-1e-5:
        now = time.perf_counter()
        p = self.positions
        v = self.state.particle_qd.numpy()
        floor = float(self.runtime['SAND_FLOOR_Z'])
        center = np.median(p[:,:2], axis=0)
        radius = np.linalg.norm(p[:,:2]-center, axis=1)
        row = dict(time=self.sim_time, wall=now, physics_wall=self.physics_wall_seconds,
                   steps=self.steps, substeps=self.solver_substeps,
                   height_max=float(p[:,2].max()-floor), height_p99=float(np.quantile(p[:,2]-floor,.99)),
                   radius_p95=float(np.quantile(radius,.95)),
                   below_floor_2cm=int(np.count_nonzero(p[:,2]<floor-.02)),
                   speed_max=float(np.linalg.norm(v,axis=1).max()),
                   speed_p99=float(np.quantile(np.linalg.norm(v,axis=1),.99)),
                   finite=bool(np.isfinite(p).all() and np.isfinite(v).all()))
        snapshots.append(row)
        if next_sample == 1:
            benchmark_start = row
        print('[SUBSTEP SAMPLE] '+json.dumps(row), flush=True)
        next_sample += 1
        if self.sim_time >= 6-1e-5:
            completed = True
            np.savez_compressed(output/'final_state.npz', positions=p, velocities=v)
            result = dict(hz=int(os.environ['EXCAVATOR_MPM_HZ']), initial_sha256=initial_hash,
                          material=dict(self.material), points=len(p), samples=snapshots,
                          wall_seconds=row['wall']-benchmark_start['wall'],
                          sim_seconds=row['time']-benchmark_start['time'],
                          physics_seconds=row['physics_wall']-benchmark_start['physics_wall'],
                          substeps=row['substeps']-benchmark_start['substeps'],
                          error=self.error, sleep_disabled_for_both=True,
                          recording=bool(os.environ.get('COARSE_VIDEO_DIR')))
            (output/'result.json').write_text(json.dumps(result,indent=2))


NewtonSoil.advance = advance
runpy.run_path(str(Path(__file__).with_name('watch_coarse_mpm.py')), run_name='__main__')
