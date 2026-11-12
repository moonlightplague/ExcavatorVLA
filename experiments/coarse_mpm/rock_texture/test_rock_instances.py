"""Add a visual-only instancer to the existing texture/camera benchmark."""
import json,os,runpy,sys,time
from pathlib import Path
repo=Path(__file__).resolve().parents[3];sys.path.insert(0,str(repo/'scripts'))
sys.path.insert(0,str(Path(__file__).parent))
from excavator_app import newton_mpm_soil as soil
from rock_display import create_rock_display
original_init=soil.NewtonSoil.__init__;original_update=soil.NewtonSoil._on_update

def initialize(self,*a,**kw):
    original_init(self,*a,**kw)
    self._rock_instances,self._rock_metadata=create_rock_display(self)
    self._rock_display_seconds=0.
    self._rock_display_calls=0
    print('[VISUAL INSTANCES]',json.dumps(self._rock_metadata),flush=True)

def update(self,event):
    dirty=self.dirty
    original_update(self,event)
    if not self.closed and dirty:
        from pxr import Vt
        start=time.perf_counter()
        self._rock_instances.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self.positions))
        self._rock_display_seconds+=time.perf_counter()-start
        self._rock_display_calls+=1
    if self.sim_time>=6-1e-5:
        meta=dict(self._rock_metadata,position_write_seconds=self._rock_display_seconds,position_updates=self._rock_display_calls)
        (Path(os.environ['ROCK_TEXTURE_OUTPUT'])/'display.json').write_text(json.dumps(meta,indent=2))
soil.NewtonSoil.__init__=initialize
soil.NewtonSoil._on_update=update
runpy.run_path(str(Path(__file__).with_name('test_texture.py')),run_name='__main__')
