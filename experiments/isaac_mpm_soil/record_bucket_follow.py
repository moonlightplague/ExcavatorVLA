"""Level side and top cameras tracking the actual bucket through a full dig."""
from pathlib import Path
import json,os,runpy,sys
import numpy as np
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
out=Path(os.environ['EXCAVATOR_NEARVIEW_OUTPUT'])
out.mkdir(parents=True,exist_ok=True)
original_init,original_update=soil.NewtonSoil.__init__,soil.NewtonSoil._on_update
recordings={}

def aim(self,entry):
    from pxr import Gf
    # The adapter caches the actual PhysX pose after each physical tick. It
    # remains valid while Isaac tears down its live tensor view at shutdown.
    pose=self.state.body_q.numpy()[entry['bucket']]
    center=pose[:3]+soil.rotation(pose[3:])@entry['local_center']
    radial=center-entry['base']
    radial[2]=0
    radial/=max(float(np.linalg.norm(radial)),1e-6)
    lateral=np.cross([0.,0.,1.],radial)
    target=center-np.array([0.,0.,.20])
    specs={'side':(target+4.0*lateral+.6*radial+np.array([0.,0.,1.6]),np.array([0.,0.,1.])),
           'overhead':(center+1.1*radial+np.array([0.,0.,4.8]),radial)}
    for name,(eye,up) in specs.items():
        entry['cameras'][name]['op'].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(*up)).GetInverse())
    return center.tolist()

def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom,Usd
    bucket=next(i for i,p in enumerate(self.body_paths) if 'bucket' in p.lower())
    low,high=self.body_bounds[bucket]
    robot=self.stage.GetPrimAtPath('/World/URDF_real3')
    base=np.array(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(robot).ExtractTranslation())
    cameras={}
    for name in ['side','overhead']:
        root=out/name;root.mkdir(exist_ok=True)
        path='/World/BucketFollow_'+name
        camera=UsdGeom.Camera.Define(self.stage,path)
        camera.CreateFocalLengthAttr(32.)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
        op=camera.AddTransformOp()
        product=rep.create.render_product(path,(960,720))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        cameras[name]={'root':root,'rgb':rgb,'product':product,'op':op}
    entry={'cameras':cameras,'last_step':-100,'frames':0,'bucket':bucket,'base':base,'local_center':(low+high)*.5}
    recordings[id(self)]=entry
    aim(self,entry)
    (out/'configuration.json').write_text(json.dumps({'soil':self.diagnostics(),'fps':10,'size':[960,720],
      'bucket':self.body_paths[bucket],'bucket_local_bounds':[low.tolist(),high.tolist()],
      'description':'Two level cameras follow actual bucket center and swing heading; no bucket-pitch camera rotation. Fresh actual physics, simulation-time playback.'},indent=2))
    print('[FOLLOW CAMERAS READY]',out,flush=True)

def update(self,event):
    original_update(self,event)
    entry=recordings.get(id(self))
    if entry is None or self.closed:return
    center=aim(self,entry)
    if self.steps-entry['last_step']<6:return
    from PIL import Image
    pixels={k:np.asarray(v['rgb'].get_data()) for k,v in entry['cameras'].items()}
    if any(p.ndim!=3 or p.shape[0]==0 for p in pixels.values()):return
    index=entry['frames']
    for name,p in pixels.items():Image.fromarray(p[:,:,:3]).save(entry['cameras'][name]['root']/f'{index:06d}.png',compress_level=1)
    with (out/'frames.jsonl').open('a') as f:f.write(json.dumps({'frame':index,'time':self.sim_time,'steps':self.steps,'bucket_center':center})+'\n')
    entry['last_step'],entry['frames']=self.steps,index+1
    if index%60==0:print('[FOLLOW FRAME]',index,self.sim_time,flush=True)

soil.NewtonSoil.__init__,soil.NewtonSoil._on_update=initialize,update
sys.argv[0]=str(repo/'run_vla_train_scene.py')
runpy.run_path(str(repo/'run_vla_train_scene.py'),run_name='__main__')
