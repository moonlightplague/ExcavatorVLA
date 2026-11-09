"""Diagnostic-only real-scene run. Records crossings and projection momentum."""
import os,sys,runpy,json
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
original_init=soil.NewtonSoil.__init__
original_advance=soil.NewtonSoil.advance

def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    wp=self.wp
    from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider,collision_sdf
    @wp.kernel
    def projection_reaction(q:wp.array(dtype=wp.vec3),before:wp.array(dtype=wp.vec3),after:wp.array(dtype=wp.vec3),
                            c:Collider,poses:wp.array(dtype=wp.transform),vel:wp.array(dtype=wp.spatial_vector),
                            dt:float,mass:float,out:wp.array(dtype=wp.float32,ndim=2)):
        i=wp.tid()
        dv=after[i]-before[i]
        if wp.length(dv)>0.000001:
            sdf,n,v,cid,mid=collision_sdf(q[i],-2,c,poses,vel,poses,dt)
            if cid>=0:
                b=c.collider_body_index[cid]
                if b>=0:
                    j=-mass*dv
                    t=wp.cross(q[i]-wp.transform_get_translation(poses[b]),j)
                    for k in range(3):
                        wp.atomic_add(out,b,k,j[k])
                        wp.atomic_add(out,b,k+3,t[k])
                    wp.atomic_add(out,b,6,1.0)
    @wp.kernel
    def crossings(old:wp.array(dtype=wp.vec3),new:wp.array(dtype=wp.vec3),a:wp.transform,b:wp.transform,
                  mesh:wp.uint64,counts:wp.array(dtype=wp.int32)):
        i=wp.tid()
        p=wp.transform_point(wp.transform_inverse(a),old[i])
        q=wp.transform_point(wp.transform_inverse(b),new[i])
        d=q-p
        length=wp.length(d)
        if length>0.0001:
            hit=wp.mesh_query_ray(mesh,p,d/length,length)
            if hit.result and hit.t>0.0001 and hit.t<length-0.0001:
                counts[i]=1
    self.audit_crossings=crossings
    self.audit_old=wp.clone(self.state.particle_q)
    self.audit_hits=wp.zeros(self.model.particle_count,dtype=wp.int32,device=self.device)
    self.audit_projection=np.zeros((len(self.body_paths),7))
    self.audit_out=wp.zeros((len(self.body_paths),7),dtype=wp.float32,device=self.device)
    self.audit_q=wp.clone(self.state.particle_q)
    self.audit_v=wp.clone(self.state.particle_qd)
    original_project=self.solver.project_outside
    def project(state_in,state_out,dt,gap=None):
        self.audit_q.assign(state_in.particle_q)
        self.audit_v.assign(state_in.particle_qd)
        original_project(state_in,state_out,dt,gap)
        c=self.solver._mpm_model.collider
        old_gap=c.query_max_dist
        c.query_max_dist=gap or old_gap
        self.audit_out.zero_()
        wp.launch(projection_reaction,dim=self.model.particle_count,inputs=[self.audit_q,self.audit_v,
            state_out.particle_qd,c,state_in.body_q,state_in.body_qd,dt,self.mass,self.audit_out],device=self.device)
        c.query_max_dist=old_gap
        self.audit_projection+=self.audit_out.numpy()
    self.solver.project_outside=project
    self.audit_root=Path(os.environ['EXCAVATOR_NEARVIEW_OUTPUT'])/'audit'
    self.audit_root.mkdir(exist_ok=True)
    (self.audit_root/'body_config.json').write_text(json.dumps({'bodies':self.body_paths,
        'mass':self.model.body_mass.numpy().tolist(),'inertia':self.model.body_inertia.numpy().tolist(),
        'point_mass':self.mass,'bounds':{str(k):[x.tolist() for x in v] for k,v in self.body_bounds.items()}}))

def advance(self,dt):
    self.audit_old.assign(self.state.particle_q)
    old_pose=self.state.body_q.numpy().copy()
    self.audit_projection.fill(0)
    original_advance(self,dt)
    pose=self.state.body_q.numpy()
    bucket=next(i for i,p in enumerate(self.body_paths) if 'bucket' in p.lower())
    c=self.solver._mpm_model.collider
    cid=int(np.flatnonzero(c.collider_body_index.numpy()==bucket)[0])
    self.audit_hits.zero_()
    self.wp.launch(self.audit_crossings,dim=self.model.particle_count,inputs=[self.audit_old,self.state.particle_q,
        self.wp.transform(*[self.wp.vec3(*old_pose[bucket,:3]),self.wp.quat(*old_pose[bucket,3:])]),
        self.wp.transform(*[self.wp.vec3(*pose[bucket,:3]),self.wp.quat(*pose[bucket,3:])]),
        self.wp.uint64(int(c.collider_mesh.numpy()[cid])),self.audit_hits],device=self.device)
    hits=np.flatnonzero(self.audit_hits.numpy())
    with (self.audit_root/'steps.jsonl').open('a') as f:
        f.write(json.dumps({'step':self.steps,'time':self.sim_time,'crossing_ids':hits.tolist(),
            'projection_impulse':self.audit_projection.tolist(),'grid_impulse':(self.last_force*dt).tolist(),
            'poses':pose.tolist()})+'\n')
    if self.steps%6==0:
        np.savez_compressed(self.audit_root/f'{self.steps:06d}.npz',q=self.positions,pose=pose,
                            metric_ids=self.metric_indices)

soil.NewtonSoil.__init__=initialize
soil.NewtonSoil.advance=advance
runpy.run_path(str(Path(__file__).with_name('record_nearview.py')),run_name='__main__')
