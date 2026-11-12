"""Shared, low-poly VISUAL rocks. No physics schema or material-point changes."""
import numpy as np

def rock_mesh(seed,radius,segments=12,rings=8):
    rng=np.random.default_rng(seed)
    uv=np.array([(i/segments,j/rings) for j in range(rings+1) for i in range(segments+1)])
    az=uv[:,0]*2*np.pi;theta=uv[:,1]*np.pi
    directions=np.column_stack((np.sin(theta)*np.cos(az),np.sin(theta)*np.sin(az),np.cos(theta)))
    planes=np.r_[np.eye(3),-np.eye(3)]+rng.normal(0,.23,(6,3))
    planes/=np.linalg.norm(planes,axis=1)[:,None]
    distances=rng.uniform(.72,.93,6)
    dots=directions@planes.T
    caps=distances[None,:]/np.maximum(dots,1e-6)
    radial=np.minimum(1.,caps.min(axis=1))
    notch=rng.normal(size=3);notch/=np.linalg.norm(notch)
    radial-=.12*np.exp(-(1-directions@notch)/.075)
    points=directions*radial[:,None]*rng.uniform(.88,1.05,3)
    points*=radius/np.linalg.norm(points,axis=1).max()
    faces=[]
    for j in range(rings):
        for i in range(segments):
            a=j*(segments+1)+i;b=a+1;c=a+segments+1;d=c+1
            if j>0:faces.append([a,c,b])
            if j<rings-1:faces.append([b,c,d])
    faces=np.asarray(faces,dtype=np.int32)
    normals=np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    return points.astype(np.float32),faces,uv.astype(np.float32),normals.astype(np.float32)

def create_rock_display(controller):
    from pxr import UsdGeom,UsdShade,UsdPhysics,Vt,Gf,Sdf
    stage=controller.stage
    path=str(controller.render.GetPath().GetParentPath())+'/RockInstances'
    inst=UsdGeom.PointInstancer.Define(stage,path)
    material=UsdShade.Material.Define(stage,'/World/RockPreviewMaterial')
    targets=[];face_count=0
    for index in range(4):
        points,faces,uv,normals=rock_mesh(4109+index,.5*controller.render_diameter)
        mesh=UsdGeom.Mesh.Define(stage,path+f'/Prototypes/Rock{index}')
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces),3,dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals));mesh.SetNormalsInterpolation(UsdGeom.Tokens.uniform)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        assert not mesh.GetPrim().HasAPI(UsdPhysics.CollisionAPI)
        targets.append(mesh.GetPath());face_count=len(faces)
    inst.CreatePrototypesRel().SetTargets(targets)
    n=len(controller.positions);rng=np.random.default_rng(4113)
    inst.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(rng.integers(0,4,n,dtype=np.int32)))
    inst.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(controller.positions))
    quats=rng.normal(size=(n,4));quats/=np.linalg.norm(quats,axis=1)[:,None]
    inst.CreateOrientationsAttr(Vt.QuathArray([Gf.Quath(float(q[3]),Gf.Vec3h(*map(float,q[:3]))) for q in quats]))
    controller.render.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return inst,dict(prototypes=4,triangles_per_rock=face_count,instances=n,physics_shapes_added=0,orientation='fixed_random_visual_only')
