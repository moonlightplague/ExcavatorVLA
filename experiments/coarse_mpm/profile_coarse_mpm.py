"""Temporary opt-in synchronized timing; no physics settings are changed."""
import json
from pathlib import Path
import runpy
import sys
import time
from collections import defaultdict

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from excavator_app.newton_mpm_soil import NewtonSoil

totals, counts = defaultdict(float), defaultdict(int)
active = False
start_wall = None
done = False
original_init = NewtonSoil.__init__
original_advance = NewtonSoil.advance
original_update = NewtonSoil._on_update


def timed(controller, name, fn):
    def call(*a, **kw):
        if not active:
            return fn(*a, **kw)
        controller.wp.synchronize_device(controller.device)
        start = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            controller.wp.synchronize_device(controller.device)
            totals[name] += time.perf_counter()-start
            counts[name] += 1
    return call


def init(self, *a, **kw):
    original_init(self, *a, **kw)
    for name in ('step', 'project_outside', 'collect_collider_impulses', 'update_particle_frames',
                 '_particles_to_cells', '_rasterize_colliders', '_compute_unconstrained_velocity',
                 '_build_collider_rigidity_operator', '_build_elasticity_system',
                 '_build_plasticity_system', '_load_warmstart', '_solve_rheology',
                 '_save_for_next_warmstart', '_update_particles', '_save_data'):
        setattr(self.solver, name, timed(self, name, getattr(self.solver, name)))


def advance(self, dt):
    global active, start_wall
    active = 60 <= self.steps < 180
    if active and start_wall is None:
        start_wall = time.perf_counter()
    return timed(self, 'advance_total', lambda: original_advance(self, dt))()


def update(self, event):
    global done, active
    timed(self, 'usd_update', lambda: original_update(self, event))()
    if self.steps >= 180 and not done:
        done = True
        active = False
        result = dict(timings_seconds=dict(totals), calls=dict(counts),
                      interval_wall_seconds=time.perf_counter()-start_wall,
                      interval_sim_seconds=2., diagnostics=self.diagnostics(),
                      note='120 physics ticks after 60 warmup ticks; synchronized diagnostic timings include profiling overhead.')
        output = Path(__file__).with_name('profile_result.json')
        output.write_text(json.dumps(result, indent=2))
        print('[PROFILE RESULT] '+json.dumps(result), flush=True)


NewtonSoil.__init__, NewtonSoil.advance, NewtonSoil._on_update = init, advance, update
runpy.run_path(str(Path(__file__).with_name('run_coarse_mpm.py')), run_name='__main__')
