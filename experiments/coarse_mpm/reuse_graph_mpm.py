"""Process-local Newton configuration / solver-core graph A/B test.

No production file or installed package is rewritten. External PhysX force
exchange and post-solve collision corrections remain outside the graph.
"""
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil

mode = os.environ['MPM_EXECUTION_TEST']
if mode not in ('baseline', 'particle_warmstart', 'reuse', 'graph'):
    raise ValueError(mode)
original_init = soil.NewtonSoil.__init__
source = textwrap.dedent(inspect.getsource(original_init))
anchor = 'config = SolverImplicitMPM.Config()'
assert source.count(anchor) == 1
indent = next(line[:len(line)-len(line.lstrip())] for line in source.splitlines() if anchor in line)
settings = ''
if mode != 'baseline':
    settings += "\n            config.warmstart_mode = 'particles'"
if mode in ('reuse', 'graph'):
    settings += "\n            config.max_active_cell_count = int(os.environ.get('MPM_TEST_CELL_CAPACITY', '4096'))"
source = source.replace(anchor, anchor+settings.replace('\n            ', '\n'+indent))
namespace = dict(original_init.__globals__)
exec(compile(source, '<experimental_mpm_init>', 'exec'), namespace)
configured_init = namespace['__init__']


def initialize(self, *a, **kw):
    configured_init(self, *a, **kw)
    solver = self.solver
    print('[EXECUTION MODE]', json.dumps(dict(mode=mode, rebuildable=solver._sparse_rebuildable,
          warmstart=solver._stress_warmstart, capacity=solver.max_active_cell_count)), flush=True)
    if mode not in ('reuse', 'graph'):
        return
    if not solver._sparse_rebuildable:
        raise RuntimeError('Requested reusable sparse grid was not enabled')
    self._execution_graph = None
    self._execution_warm = False
    original_step = solver.step

    def step(state_in, state_out, control, contacts, dt):
        if mode == 'reuse' or not self._execution_warm:
            result = original_step(state_in, state_out, control, contacts, dt)
            self._execution_warm = True
        else:
            signature = (id(state_in), id(state_out), id(control), id(contacts), dt)
            if self._execution_graph is None:
                if not self.wp.is_mempool_enabled(self.device) or not self.wp.is_conditional_graph_supported():
                    raise RuntimeError('Required CUDA graph support unavailable')
                with self.wp.ScopedCapture(device=self.device) as capture:
                    original_step(state_in, state_out, control, contacts, dt)
                self._execution_graph = capture.graph
                self._execution_signature = signature
                print('[MPM CORE GRAPH CAPTURED]', flush=True)
            if signature != self._execution_signature:
                raise RuntimeError('Captured state or timestep changed; refusing stale graph replay')
            self.wp.capture_launch(self._execution_graph)
            result = None
        solver.check_sparse_grid_rebuild_status()
        return result

    solver.step = step


soil.NewtonSoil.__init__ = initialize
runpy.run_path(str(Path(__file__).with_name('compare_substeps.py')), run_name='__main__')
