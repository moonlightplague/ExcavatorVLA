"""One existing excavator dig cycle with coarse MPM / CUDA graph and observers."""
import json
import os
from pathlib import Path
import runpy
import sys
import numpy as np

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil

root=Path(os.environ['MPM_DIG_OUTPUT'])
root.mkdir(parents=True,exist_ok=False)
restore=soil.NewtonSoil.restore_initial_state


def restore_without_stale_graph(self):
    # A restored scene can replace collider buffers while keeping state objects.
    self._execution_graph=None
    self._execution_warm=False
    result=restore(self)
    print('[DIG GRAPH INVALIDATED FOR RESET]',result,flush=True)
    return result


soil.NewtonSoil.restore_initial_state=restore_without_stale_graph
original_run=runpy.run_path


def record_and_run():
    original_init=soil.NewtonSoil.__init__
    original_update=soil.NewtonSoil._on_update
    entries={}

    def initialize(self,*a,**kw):
        original_init(self,*a,**kw)
        import omni.replicator.core as rep
        from pxr import UsdGeom,Gf,Usd
        session=root/f'session_{len(entries)+1:02d}'
        session.mkdir()
        bucket=next(i for i,p in enumerate(self.body_paths) if 'bucket' in p.lower())
        low,high=self.body_bounds[bucket]
        robot=self.stage.GetPrimAtPath('/World/URDF_real3')
        base=np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(robot).ExtractTranslation())
        cameras={}
        for name in ('side','overhead'):
            dest=session/name; dest.mkdir()
            path='/World/CoarseDig_'+name
            camera=UsdGeom.Camera.Define(self.stage,path)
            camera.CreateFocalLengthAttr(28.)
            camera.CreateHorizontalApertureAttr(36.)
            camera.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
            op=camera.AddTransformOp()
            product=rep.create.render_product(path,(640,480))
            rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
            cameras[name]=dict(path=dest,op=op,product=product,rgb=rgb)
        entries[id(self)]=dict(session=session,bucket=bucket,base=base,local_center=(low+high)*.5,
                               cameras=cameras,last=-100,frame=0,epoch=0)
        (session/'configuration.json').write_text(json.dumps(self.diagnostics(),indent=2))

    def update(self,event):
        original_update(self,event)
        entry=entries.get(id(self))
        if entry is None or self.closed:return
        from pxr import Gf
        pose=self.state.body_q.numpy()[entry['bucket']]
        center=pose[:3]+soil.rotation(pose[3:])@entry['local_center']
        radial=center-entry['base'];radial[2]=0
        radial/=max(np.linalg.norm(radial),1e-6)
        lateral=np.cross([0.,0.,1.],radial)
        target=center-np.array([0.,0.,.2])
        specs={'side':(target+4*lateral+.6*radial+np.array([0.,0.,1.6]),np.array([0.,0.,1.])),
               'overhead':(center+1.1*radial+np.array([0.,0.,4.8]),radial)}
        for name,(eye,up) in specs.items():
            entry['cameras'][name]['op'].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up)).GetInverse())
        if self.steps<entry['last']:
            entry['last']=-100;entry['epoch']+=1
        if self.steps-entry['last']<6:return
        from PIL import Image
        images={k:np.asarray(v['rgb'].get_data()) for k,v in entry['cameras'].items()}
        if any(p.ndim!=3 or not p.size for p in images.values()):return
        for name,p in images.items():
            Image.fromarray(p[:,:,:3]).save(entry['cameras'][name]['path']/f"{entry['frame']:06d}.png",compress_level=1)
        p=self.positions;v=self.state.particle_qd.numpy()
        state=self.runtime['STATE']
        row=dict(frame=entry['frame'],epoch=entry['epoch'],step=self.steps,time=self.sim_time,
                 bucket_center=center.tolist(),speed_max=float(np.linalg.norm(v,axis=1).max()),
                 below_floor_2cm=int(np.count_nonzero(p[:,2]<float(self.runtime['SAND_FLOOR_Z'])-.02)),
                 last_force=self.last_force.tolist(),applied_impulse=self.applied_impulse.tolist(),
                 finite=bool(np.isfinite(p).all() and np.isfinite(v).all()),error=self.error,
                 progress=state.get('auto_collect_progress',{}),diagnostics=self.diagnostics())
        with (entry['session']/'frames.jsonl').open('a') as f:
            f.write(json.dumps(row,default=str)+'\n')
        entry['last']=self.steps;entry['frame']+=1
        if entry['frame']%60==0:print('[DIG VIDEO]',entry['session'].name,entry['frame'],self.sim_time,flush=True)

    soil.NewtonSoil.__init__,soil.NewtonSoil._on_update=initialize,update
    entry=repo/'run_vla_train_scene.py'
    sys.argv=[str(entry),'--headless','--auto-collect','--no-bridge',
              '--active-gpu','0','--physics-gpu','0',
              '--success-count','1','--max-attempts','1','--sand-amount','1',
              '--dataset-root',str(root/'trial_data'),'--disable-export']
    try:
        original_run(str(entry),run_name='__main__')
    finally:
        (root/'runner_finished.json').write_text(json.dumps(dict(sessions=len(entries))))


def run(path,*a,**kw):
    if Path(path).name=='compare_substeps.py':
        return record_and_run()
    return original_run(path,*a,**kw)


runpy.run_path=run
original_run(str(Path(__file__).with_name('reuse_graph_mpm.py')),run_name='__main__')
