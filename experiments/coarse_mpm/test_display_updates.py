"""Display-only A/B probes. Physics, feedback and metric USD points stay 60 Hz."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
mode=os.environ['DISPLAY_TEST_MODE'];assert mode in ('baseline','no_duplicate','visual30','combined','render30')
remove_duplicate=mode in ('no_duplicate','combined','render30')
stride=2 if mode in ('visual30','combined','render30') else 1
out=Path(os.environ['DISPLAY_TEST_OUTPUT']);out.mkdir(exist_ok=False)
original_init=soil.NewtonSoil.__init__;original_update=soil.NewtonSoil._on_update
start=None;done=False;initial_hash=None

def initialize(self,*a,**kw):
    global initial_hash
    original_init(self,*a,**kw)
    self._display_metric_count=self._display_visual_count=self._display_hidden_count=0
    self._display_write_seconds=0.
    self._display_last_step=-1
    initial_hash=hashlib.sha256(self.positions.tobytes()).hexdigest()
    from pxr import UsdGeom,Gf,Sdf
    from omni.kit.viewport.utility import get_active_viewport
    self._display_view=get_active_viewport()
    if self._display_view is None:raise RuntimeError('Single rendering viewport is required')
    x0,x1,y0,y1=self.runtime['current_sand_footprint_bbox_xy']()
    target=Gf.Vec3d((x0+x1)/2,(y0+y1)/2,float(self.runtime['SAND_FLOOR_Z'])+.45)
    cam=UsdGeom.Camera.Define(self.stage,'/World/DisplayUpdateCamera')
    cam.CreateFocalLengthAttr(32.);cam.CreateHorizontalApertureAttr(36.)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.));cam.ClearXformOpOrder()
    cam.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(target+Gf.Vec3d(.6,2.6,3.2),target,Gf.Vec3d(0,0,1)).GetInverse())
    self._display_view.camera_path=Sdf.Path('/World/DisplayUpdateCamera')
    self._display_view.updates_enabled=True;self._display_view.set_texture_resolution((1280,720))

def display(self,event):
    if self.closed or not self.dirty:return
    from pxr import Vt
    before=time.perf_counter()
    if mode=='baseline':
        original_update(self,event)
        self._display_metric_count+=1;self._display_hidden_count+=1;self._display_visual_count+=1
    else:
        # Keep the metric USD representation and original collection interface fresh.
        self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.metric_positions))
        self._display_metric_count+=1
        visual_due=self.steps%stride==0 or self.steps<self._display_last_step
        if visual_due:
            if not remove_duplicate:
                self.render.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(self.grains.flatten().numpy()))
                self._display_hidden_count+=1
            self.rock_instances.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(self.positions))
            self._display_visual_count+=1;self.render_updates+=1
        self.dirty=False
        self.render_wall_seconds+=time.perf_counter()-before
        self.publish_status()
    self._display_last_step=self.steps
    self._display_write_seconds+=time.perf_counter()-before
    if mode=='render30':
        self._display_view.updates_enabled=self.steps%2==0

def update(self,event):
    global start,done
    display(self,event)
    if done:return
    if start is None and self.sim_time>=1-1e-6:
        self.wp.synchronize_device(self.device)
        start=dict(wall=time.perf_counter(),time=self.sim_time,steps=self.steps,substeps=self.solver_substeps,
            physics=self.physics_wall_seconds,metric=self._display_metric_count,visual=self._display_visual_count,
            hidden=self._display_hidden_count,write=self._display_write_seconds,frame_info=dict(self._display_view.frame_info))
    if self.error or self.sim_time>=11-1e-6:
        done=True;self.wp.synchronize_device(self.device)
        wall=time.perf_counter()-start['wall'];elapsed=self.sim_time-start['time']
        p=self.positions;v=self.state.particle_qd.numpy()
        metric=np.asarray(self.points.GetPointsAttr().Get())
        assert np.array_equal(metric,self.metric_positions),'Metric USD became stale'
        result=dict(mode=mode,wall_seconds=wall,sim_seconds=elapsed,rtf=elapsed/wall,
            measured_steps=self.steps-start['steps'],measured_substeps=self.solver_substeps-start['substeps'],
            metric_updates=self._display_metric_count-start['metric'],visual_updates=self._display_visual_count-start['visual'],
            hidden_updates=self._display_hidden_count-start['hidden'],display_write_seconds=self._display_write_seconds-start['write'],
            physics_seconds=self.physics_wall_seconds-start['physics'],initial_sha256=initial_hash,
            finite=bool(np.isfinite(p).all() and np.isfinite(v).all()),height_max=float(p[:,2].max()),
            below_floor_2cm=int(np.count_nonzero(p[:,2]<float(self.runtime['SAND_FLOOR_Z'])-.02)),
            viewport_resolution=list(self._display_view.get_texture_resolution()),frame_info_start=start['frame_info'],
            frame_info_end=dict(self._display_view.frame_info),diagnostics=self.diagnostics())
        np.savez_compressed(out/'state.npz',positions=p,velocities=v)
        (out/'result.json').write_text(json.dumps(result,indent=2,default=str))
        print('[DISPLAY RESULT]',json.dumps(result,default=str),flush=True)
        self._display_view.updates_enabled=True
        import omni.kit.app
        omni.kit.app.get_app().post_quit(0 if not self.error else 1)
soil.NewtonSoil.__init__=initialize;soil.NewtonSoil._on_update=update
entry=repo/'run_vla_train_scene.py'
sys.argv=[str(entry),'--headless','--no-bridge','--disable-export','--active-gpu','0','--physics-gpu','0','--sand-amount','1','--width','1280','--height','720']
runpy.run_path(str(entry),run_name='__main__')
