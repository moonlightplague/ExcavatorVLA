"""Compare original fine-soil display/MPM resolution without editing launchers."""
import json
import os
from pathlib import Path
import runpy
import sys

folder=Path(__file__).resolve().parent
repo=folder.parents[1]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
original_advance=soil.NewtonSoil.advance
original_run=runpy.run_path
fine=os.environ.get('MPM_RESOLUTION_TEST')=='fine'


def advance(self,dt):
    original_advance(self,dt)
    if self.steps==360:
        stats=self.diagnostics()
        if self.solver._sparse_rebuildable:
            stats['active_voxels']=int(self.solver._scratchpad.grid.cell_grid.get_active_stats().voxel_count)
        stats['cell_capacity']=self.solver.max_active_cell_count
        (Path(os.environ['SUBSTEP_RESULT_DIR'])/'resolution_metadata.json').write_text(json.dumps(stats,indent=2))


def run(path, *args, **kwargs):
    if fine and Path(path).name=='run_coarse_mpm.py':
        source=Path(path).read_text()
        replacements={
            "default=.15)":"default=.075)",
            "default=.20)":"default=.10)",
            "default=.18,":"default=.014,",
            "default=6000)":"default=40000)",
            "'EXCAVATOR_MPM_PROFILE': 'coarse'":"'EXCAVATOR_MPM_PROFILE': 'fine'",
            "assert soil_resolution_settings()['render_material_centers']":"assert not soil_resolution_settings()['render_material_centers']",
            "'one sphere per MPM material point'":"'original passive fine grains'",
        }
        for old,new in replacements.items():
            if source.count(old)!=1:
                raise RuntimeError('Unexpected launcher version: '+old)
            source=source.replace(old,new)
        scope=dict(__name__='__main__',__file__=str(path))
        exec(compile(source,str(path),'exec'),scope)
        return scope
    return original_run(path,*args,**kwargs)


soil.NewtonSoil.advance=advance
runpy.run_path=run
original_run(str(folder/'reuse_graph_mpm.py'),run_name='__main__')
