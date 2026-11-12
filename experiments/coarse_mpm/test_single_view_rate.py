"""Experimental MPM clock only; PhysX/control remain 60 Hz. No production changes."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
hz=int(os.environ['RATE_TEST_HZ']);assert hz in (45,60)
h=1./hz
out=Path(os.environ['RATE_TEST_OUTPUT']);out.mkdir(exist_ok=False)
original_init=soil.NewtonSoil.__init__;original_step=soil.NewtonSoil._on_step
original_pre=soil.NewtonSoil._on_pre_step;original_update=soil.NewtonSoil._on_update
# The regular min-one-substep rule cannot express a rate below PhysX's 60 Hz.
# This experiment supplies one full 1/hz solve, with a separate time accumulator.
soil.mpm_substeps=lambda dt: 1
os.environ['EXCAVATOR_MPM_HZ']=str(hz)
start=None;done=False;initial_hash=None;samples=[]

def initialize(self,*a,**kw):
    global initial_hash
    original_init(self,*a,**kw)
    self._rate_time=0.;self._rate_accum=0.;self._rate_ticks=0
    self._rate_pending=np.zeros_like(self.applied_impulse)
    initial_hash=hashlib.sha256(self.positions.tobytes()).hexdigest()
    from pxr import UsdGeom,Gf,Sdf
    from omni.kit.viewport.utility import get_active_viewport
    self._rate_view=get_active_viewport()
    if self._rate_view is None:raise RuntimeError('Single rendering viewport is required')
    x0,x1,y0,y1=self.runtime['current_sand_footprint_bbox_xy']()
    target=Gf.Vec3d((x0+x1)/2,(y0+y1)/2,float(self.runtime['SAND_FLOOR_Z'])+.45)
    cam=UsdGeom.Camera.Define(self.stage,'/World/SingleRateCamera')
    cam.CreateFocalLengthAttr(32.);cam.CreateHorizontalApertureAttr(36.)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
    cam.ClearXformOpOrder()
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(target+Gf.Vec3d(.6,2.6,3.2),target,Gf.Vec3d(0,0,1)).GetInverse())
    self._rate_view.camera_path=Sdf.Path('/World/SingleRateCamera')
    self._rate_view.updates_enabled=True
    self._rate_view.set_texture_resolution((1280,720))
    print('[RATE VIEW]',self._rate_view.get_texture_resolution(),flush=True)

def pre(self,dt):
    original_pre(self,dt)
    if not self.closed and not self.error:
        # Accumulate actual PhysX feedback since the last MPM solve.
        self._rate_pending+=self.applied_impulse

def step(self,dt):
    if self.closed or self.error:return
    self._rate_time+=float(dt);self._rate_ticks+=1;self._rate_accum+=float(dt)
    if self._rate_accum+1e-8<h:return
    if self._rate_accum>=2*h+1e-6:raise RuntimeError('Unexpected driver dt in rate experiment')
    self.applied_impulse=self._rate_pending.copy()
    original_step(self,h)
    self._rate_pending.fill(0)
    self._rate_accum-=h

def update(self,event):
    global start,done
    original_update(self,event)
    if done:return
    if start is None and self._rate_time>=1-1e-6:
        self.wp.synchronize_device(self.device)
        start=(time.perf_counter(),self._rate_time,self.steps,self.physics_wall_seconds,self._rate_ticks)
    if self.error or self._rate_time>=11-1e-6:
        done=True;self.wp.synchronize_device(self.device)
        p=self.positions;v=self.state.particle_qd.numpy()
        wall=time.perf_counter()-start[0] if start else None
        simulated=self._rate_time-start[1] if start else None
        result=dict(hz=hz,wall_seconds=wall,sim_seconds=simulated,rtf=simulated/wall if wall else None,
            measured_mpm_steps=self.steps-start[2] if start else 0,
            measured_driver_steps=self._rate_ticks-start[4] if start else 0,
            physics_seconds=self.physics_wall_seconds-start[3] if start else None,
            initial_sha256=initial_hash,finite=bool(np.isfinite(p).all() and np.isfinite(v).all()),
            height_max=float(p[:,2].max()),height_p99=float(np.quantile(p[:,2],.99)),
            speed_max=float(np.linalg.norm(v,axis=1).max()),
            below_floor_2cm=int(np.count_nonzero(p[:,2]<float(self.runtime['SAND_FLOOR_Z'])-.02)),
            soil_clock_lag=self._rate_time-self.sim_time,viewport_resolution=list(self._rate_view.get_texture_resolution()),
            viewport_camera=str(self._rate_view.camera_path),extra_render_products=0,
            diagnostics=self.diagnostics(),note='Static-pile rate experiment; moving-tool multirate coupling not validated')
        np.savez_compressed(out/'state.npz',positions=p,velocities=v)
        (out/'result.json').write_text(json.dumps(result,indent=2))
        print('[RATE RESULT]',json.dumps(result),flush=True)
        import omni.kit.app
        omni.kit.app.get_app().post_quit(0 if not self.error else 1)
soil.NewtonSoil.__init__=initialize;soil.NewtonSoil._on_step=step
soil.NewtonSoil._on_pre_step=pre;soil.NewtonSoil._on_update=update
entry=repo/'run_vla_train_scene.py'
sys.argv=[str(entry),'--headless','--no-bridge','--disable-export','--active-gpu','0','--physics-gpu','0','--sand-amount','1','--width','1280','--height','720']
runpy.run_path(str(entry),run_name='__main__')
