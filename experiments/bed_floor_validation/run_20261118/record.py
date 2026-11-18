from pathlib import Path
import json, os, sys, traceback
import numpy as np

output = Path(__file__).resolve().parent
project = output.parents[3]
sys.path[:0] = [str(project), str(project / 'ExcavatorVLA-isaac' / 'scripts')]
from baseline import repeat_runtime, collect_repeat

original_install = repeat_runtime.install
def install(rt, config):
    result = original_install(rt, config)
    import omni.kit.app, omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    from PIL import Image
    import builtins
    st = {}
    closeups = output / 'closeups'
    closeups.mkdir(exist_ok=True)
    def capture(event):
        try:
            soil = getattr(builtins, '_EXCAVATOR_MPM_SOIL', None)
            if soil is None or soil.closed or st.get('tick') == (id(soil),soil.steps):
                return
            if soil.steps % 6 != 0:
                return
            st['tick'] = (id(soil),soil.steps)
            stage = soil.stage
            prim = stage.GetPrimAtPath('/World/truck/DumpBedCollision/dump_bed_collision')
            if not prim.IsValid():return
            mat = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            if 'mat' not in st or not np.allclose(mat,st['mat'],atol=1e-7):
                st['mat'] = mat.copy()
                mesh = UsdGeom.Mesh(prim)
                v = np.asarray(mesh.GetPointsAttr().Get()) @ mat[:3,:3] + mat[3,:3]
                tri = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                area = np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]])
                if np.linalg.det(mat[:3,:3])<0: area *= -1
                mask = (area[:,2]>0.98*np.linalg.norm(area,axis=1)) & (v[tri,2].max(axis=1)<v[:,2].min()+.12)
                a = area[mask,2]
                center = (v[tri[mask]].mean(axis=1)*a[:,None]).sum(axis=0)/a.sum()
                long = np.linalg.svd(v[:,:2]-v[:,:2].mean(axis=0), full_matrices=False)[2][0]
                if long[0]<0:long=-long
                st['center'], st['v'], st['tri'] = center,v,tri
                eye = center+np.r_[long*1.15,2.7]+np.array([-long[1]*.38,long[0]*.38,0])
                if 'op' not in st:
                    path='/World/BedFloorValidationCamera'
                    camera=UsdGeom.Camera.Define(stage,path)
                    camera.CreateFocalLengthAttr(31.)
                    camera.CreateHorizontalApertureAttr(36.)
                    camera.CreateClippingRangeAttr(Gf.Vec2f(.05,100.))
                    st['op']=camera.MakeMatrixXform()
                    st['product']=rep.create.render_product(path,(800,600))
                    st['rgb']=rep.AnnotatorRegistry.get_annotator('rgb')
                    st['rgb'].attach([st['product']])
                st['op'].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*(center+np.array([0,0,.10]))),Gf.Vec3d(0,0,1)).GetInverse())
                st['epoch']=st.get('epoch',-1)+1
                np.savez(output/('bed_%d.npz'%st['epoch']),vertices=v,triangles=tri)
                print('[BED OBSERVER]',st['epoch'],center,flush=True)
            center = st['center']
            p=soil.positions
            near = (np.linalg.norm(p[:,:2]-center[:2],axis=1)<3.8)&(p[:,2]>center[2]-.25)&(p[:,2]<center[2]+4.)
            ids=np.flatnonzero(near)
            frame=st.get('frame',0)
            with (output/'sample.jsonl').open('a') as f:
                f.write(json.dumps(dict(frame=frame,epoch=st['epoch'],t=soil.sim_time,step=soil.steps,ids=ids.tolist(),xyz=p[ids].tolist()))+'\n')
            st['frame']=frame+1
            if len(ids):
                pix=np.asarray(st['rgb'].get_data())
                if pix.ndim==3 and pix.size:
                    Image.fromarray(pix[:,:,:3]).save(closeups/('%06d.png'%frame),compress_level=2)
        except Exception:
            if not st.get('warned'):
                traceback.print_exc()
                st['warned']=True
    globals()['observer_subscription'] = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(capture,name='BedFloorRecording')
    return result
repeat_runtime.install=install
collect_repeat.main()

