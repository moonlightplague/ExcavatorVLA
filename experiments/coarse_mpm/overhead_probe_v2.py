"""Independent execution-only experiments and opt-in convergence diagnostics."""
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import sys
import time

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
original_advance=soil.NewtonSoil.advance
original_update=soil.NewtonSoil._on_update
profile=cProfile.Profile()
profiling=False
profile_done=False
usd_seconds=0.
usd_calls=0


def configure_probe(self):
    from omni.kit.viewport.utility import get_active_viewport
    self._probe_view=get_active_viewport()
    if self._probe_view is not None and os.environ.get('MPM_DISPLAY_TEST')=='low':
        self._probe_view.resolution_scale = .5
    print('[VIEW STATE]',bool(self._probe_view.updates_enabled) if self._probe_view else None,flush=True)
    print('[SOLVER SETTINGS]',self.solver.max_iterations,self.solver.tolerance,flush=True)


def update(self,event):
    global usd_seconds,usd_calls
    mode=os.environ.get('MPM_DISPLAY_TEST','normal')
    allowed=mode!='off' and (mode!='half' or self.steps%2==0)
    view=getattr(self,'_probe_view',None)
    if view is not None and mode!='normal':
        view.updates_enabled=allowed
    if (mode=='normal' or allowed) and self.steps % int(os.environ.get('MPM_USD_STRIDE','1')) == 0:
        start=time.perf_counter()
        original_update(self,event)
        if 60<=self.steps<360:
            usd_seconds+=time.perf_counter()-start
            usd_calls+=1


def advance(self,dt):
    global profiling,profile_done
    if os.environ.get('MPM_CPU_PROFILE')=='1' and self.steps==60:
        profile.enable(); profiling=True
    original_advance(self,dt)
    if self.steps>=360 and not profile_done:
        profile_done=True
        out=Path(os.environ['SUBSTEP_RESULT_DIR'])
        if profiling:
            profile.disable()
            profile.dump_stats(str(out/'cpu_profile.prof'))
            text=io.StringIO()
            pstats.Stats(profile,stream=text).strip_dirs().sort_stats('cumulative').print_stats(60)
            pstats.Stats(profile,stream=text).strip_dirs().sort_stats('tottime').print_stats(40)
            (out/'cpu_profile.txt').write_text(text.getvalue())
        scratch=self.solver._scratchpad
        grid=scratch.grid
        stats=dict(grid_cells=int(grid.cell_count()),velocity_nodes=int(scratch.velocity_node_count),
                   strain_nodes=int(scratch.strain_node_count),usd_seconds=usd_seconds,usd_calls=usd_calls)
        if self.solver._sparse_rebuildable:
            stats['active_voxels']=int(grid.cell_grid.get_active_stats().voxel_count)
        (out/'execution_detail.json').write_text(json.dumps(stats,indent=2))


soil.NewtonSoil.advance=advance
soil.NewtonSoil._on_update=update
entry=Path(__file__).with_name('reuse_graph_mpm.py')
text=entry.read_text()
text=text.replace('    configured_init(self, *a, **kw)', '    configured_init(self, *a, **kw)\n    configure_probe(self)')
if os.environ.get('MPM_TIGHT_CONVERGENCE')=='1':
    marker="namespace = dict(original_init.__globals__)"
    text=text.replace(marker,"source = source.replace('config.voxel_size, config.max_iterations, config.tolerance = self.voxel, 150, 1e-4', 'config.voxel_size, config.max_iterations, config.tolerance = self.voxel, 600, 1e-6')\n"+marker)
exec(compile(text,str(entry),'exec'),dict(globals(),__file__=str(entry)))
