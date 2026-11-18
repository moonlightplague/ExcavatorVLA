from pathlib import Path
import json,re,sys
import numpy as np
import os
here=Path(__file__).resolve().parent
project=here.parents[3]
sys.path.insert(0,str(project/'ExcavatorVLA-isaac'/'scripts'/'excavator_app'))
from soil_rock_display import rock_mesh
rows=[json.loads(s) for s in (here/'sample.jsonl').read_text().splitlines() if s.strip()]
mesh=np.load(here/('bed_%d.npz'%rows[-1]['epoch']))
v,tris=mesh['vertices'],mesh['triangles']
tri=v[tris]
norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
sel=(np.abs(norm[:,2])>.98*np.linalg.norm(norm,axis=1)) & (tri[:,:,2].min(axis=1)>v[:,2].min()+.019) & (tri[:,:,2].max(axis=1)<v[:,2].min()+.11)
tri=tri[sel];a=tri[:,0];u=tri[:,1]-a;w=tri[:,2]-a
det=u[:,0]*w[:,1]-u[:,1]*w[:,0]
def floorpts(pp):
 out=np.full(len(pp),np.nan)
 for i,pt in enumerate(pp):
  d=pt[:2]-a[:,:2]
  s=(d[:,0]*w[:,1]-d[:,1]*w[:,0])/det
  t=(u[:,0]*d[:,1]-u[:,1]*d[:,0])/det
  ok=(s>=-1e-7)&(t>=-1e-7)&(s+t<=1.0000001)
  if np.any(ok):out[i]=(a[:,2]+s*u[:,2]+t*w[:,2])[ok].max()
 return out
rock=[rock_mesh(4109+i,.09)[0] for i in range(4)]
num=int(re.findall(r'\[MPM READY\].*? points=(\d+) ',(here/'run.log').read_text(errors='ignore'))[-1])
rng=np.random.default_rng(4113);types=rng.integers(0,4,num,dtype=np.int32)
quat=rng.normal(size=(num,4));quat/=np.linalg.norm(quat,axis=1)[:,None]
def low(id):
 x,y,z,w=quat[id]
 axis=np.array([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)])
 return float(np.min(rock[types[id]]@axis))
output=[]
for r in rows:
 if not r['ids']:continue
 pp=np.array(r['xyz'])
 delta=pp[:,2]-floorpts(pp)
 floor=np.isfinite(delta)&(delta<.23)&(delta>-.12)
 ids=np.array(r['ids'])[floor];ds=delta[floor]
 vals=ds+np.array([low(i) for i in ids])
 qq=lambda x:np.round(np.percentile(x,[10,50,90])*100,2).tolist() if len(x) else None
 output.append(dict(frame=r['frame'],t=round(r['t'],3),near_floor=len(ids),center_p10_p50_p90_cm=qq(ds),visual_bottom_p10_p50_p90_cm=qq(vals),scope='flat inner floor; centers -12 to +23 cm, all in view region'))
(here/'measurements.json').write_text(json.dumps(output))
print('contact frames',[(v['frame'],v['near_floor']) for v in output[::max(1,len(output)//18)]])
print('last',output[-1] if output else None)

