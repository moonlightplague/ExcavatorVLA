"""Two fixed observer cameras covering the actual dump bed and falling soil."""
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

def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    import omni.replicator.core as rep
    from pxr import Gf,UsdGeom,Usd
    bed=self.stage.GetPrimAtPath('/World/truck/DumpBedCollision/dump_bed_collision')
    if not bed.IsValid():raise RuntimeError('Actual dump-bed collider missing')
    bound=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render]).ComputeWorldBound(bed).ComputeAlignedRange()
    low,high=np.array(bound.GetMin()),np.array(bound.GetMax())
    center=(low+high)*.5
    target=np.array([center[0],center[1],high[2]*.5+.8])
    specs={'side':(target+np.array([5.,-8.,5.]),target,32.),
           'overhead':(np.array([center[0],center[1]-.01,high[2]+11.]),np.array([center[0],center[1],high[2]-1.]),42.)}
    cameras={}
    for name,(eye,look,focal) in specs.items():
        root=out/name
        root.mkdir(exist_ok=True)
        path='/World/UnloadObserver_'+name
        camera=UsdGeom.Camera.Define(self.stage,path)
        camera.CreateFocalLengthAttr(focal)
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
        up=Gf.Vec3d(0,1,0) if name=='overhead' else Gf.Vec3d(0,0,1)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*look),up).GetInverse())
        product=rep.create.render_product(path,(960,720))
        rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product])
        cameras[name]={'root':root,'rgb':rgb,'product':product}
    recordings[id(self)]={'cameras':cameras,'last_step':-100,'frames':0}
    (out/'configuration.json').write_text(json.dumps({'soil':self.diagnostics(),'bed_bounds':[low.tolist(),high.tolist()],
       'cameras':{k: {'eye':v[0].tolist(),'target':v[1].tolist(),'focal':v[2]} for k,v in specs.items()},
       'fps':10,'description':'Fresh actual simulation; two synchronized observer views; simulation-time playback.'},indent=2))
    print('[UNLOAD CAMERAS READY]',out,flush=True)

def update(self,event):
    original_update(self,event)
    entry=recordings.get(id(self))
    if entry is None or self.closed or self.steps-entry['last_step']<6:return
    from PIL import Image
    pixels={k:np.asarray(v['rgb'].get_data()) for k,v in entry['cameras'].items()}
    if any(p.ndim!=3 or p.shape[0]==0 for p in pixels.values()):return
    index=entry['frames']
    for name,p in pixels.items():Image.fromarray(p[:,:,:3]).save(entry['cameras'][name]['root']/f'{index:06d}.png',compress_level=1)
    with (out/'frames.jsonl').open('a') as f:f.write(json.dumps({'frame':index,'time':self.sim_time,'steps':self.steps})+'\n')
    entry['last_step'],entry['frames']=self.steps,index+1
    if index%60==0:print('[UNLOAD FRAME]',index,self.sim_time,flush=True)

soil.NewtonSoil.__init__,soil.NewtonSoil._on_update=initialize,update
sys.argv[0]=str(repo/'run_vla_train_scene.py')
runpy.run_path(str(repo/'run_vla_train_scene.py'),run_name='__main__')
