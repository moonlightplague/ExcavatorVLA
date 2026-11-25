from pathlib import Path
import json, os, sys, time, traceback
import numpy as np

output = Path(__file__).resolve().parent
project = Path('/root/gpufree-data/ExcavatorVLA_new')
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
    rt_append = rt.append_jsonl_dataset
    def append(path, entry, *args, **kw):
        value = rt_append(path, entry, *args, **kw)
        if Path(path).name == 'repeat_cycles.jsonl':
            try:
                from baseline.bed_fill import inside_polygon
                s=builtins._EXCAVATOR_MPM_SOIL
                bed=rt.repeat_baseline.bed
                q=s.positions
                inside=inside_polygon(q[:,:2],bed.polygon)&(q[:,2]>=bed.floor_z-.05)&(q[:,2]<=bed.wall_z+.25)
                ids=np.flatnonzero(inside);dep=s.bed_deposition
                order=int(entry['scoop_index'])+1
                np.savez_compressed(output/f'cycle_{order:02d}.npz',ids=ids,xyz=q[ids],anchors=dep.ids.copy(),
                    effective_floor=dep.surface.heights(q[ids,:2]),polygon=bed.polygon)
                with (output/'cycles.jsonl').open('a') as log:
                    log.write(json.dumps(dict(n=order,t=s.sim_time,wall=time.perf_counter(),cycle=entry,
                        ids_in_bed=len(ids),anchors=len(dep.ids),admitted=dep.admitted,released=dep.released,
                        physics=s.physics_wall_seconds,deposit=dep.elapsed),default=str)+chr(10))
                print('[BED MULTI RETURN]',order,'actual',len(ids),'official',entry['bed_after_return']['particle_count'],flush=True)
            except Exception:
                traceback.print_exc()
        return value
    rt.append_jsonl_dataset=append
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
                short=np.array([-long[1],long[0]])
                axes=np.column_stack((long,short))
                uv=v[:,:2]@axes
                center[:2]=((uv.min(axis=0)+uv.max(axis=0))*.5)@axes.T
                st['center']=center
                eye=center+np.r_[-1.95*short,7.7]
                if 'op' not in st:
                    path='/World/BedFloorValidationCamera'
                    camera=UsdGeom.Camera.Define(stage,path)
                    camera.CreateFocalLengthAttr(32.)
                    camera.CreateHorizontalApertureAttr(36.)
                    camera.CreateClippingRangeAttr(Gf.Vec2f(.05,100.))
                    st['op']=camera.MakeMatrixXform()
                    st['product']=rep.create.render_product(path,(1280,720))
                    st['rgb']=rep.AnnotatorRegistry.get_annotator('rgb')
                    st['rgb'].attach([st['product']])
                st['op'].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*(center+np.array([0,0,.38]))),Gf.Vec3d(0,0,1)).GetInverse())
                st['epoch']=st.get('epoch',-1)+1
                np.savez(output/('bed_%d.npz'%st['epoch']),vertices=v,triangles=tri)
                print('[BED OBSERVER]',st['epoch'],center,flush=True)
            center = st['center']
            p=soil.positions
            near = (np.linalg.norm(p[:,:2]-center[:2],axis=1)<3.8)&(p[:,2]>center[2]-.25)&(p[:,2]<center[2]+4.)
            ids=np.flatnonzero(near)
            frame=st.get('frame',0)
            with (output/'sample.jsonl').open('a') as f:
                f.write(json.dumps(dict(frame=frame,epoch=st['epoch'],t=soil.sim_time,step=soil.steps,phase=rt.STATE.get('last_action_mode',''),scoop=int(rt.STATE.get('dataset_scoop_index',0)),anchors=len(soil.bed_deposition.ids),wall=time.perf_counter(),ids=ids.tolist(),xyz=p[ids].tolist()))+'\n')
            st['frame']=frame+1
            phase=str(rt.STATE.get('last_action_mode',''))
            if 'unload' in phase:
                st['last_dump']=soil.sim_time
            if soil.sim_time-st.get('last_dump',-1.e6)<4.:
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

