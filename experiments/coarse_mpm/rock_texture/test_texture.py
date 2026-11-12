"""Visual-only rock material on unchanged native USD Points; production backend untouched."""
import hashlib,json,os,runpy,sys,time
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil
mode=os.environ.get('ROCK_TEXTURE_MODE','texture')
output=Path(os.environ['ROCK_TEXTURE_OUTPUT']);output.mkdir(exist_ok=False)
record=os.environ.get('ROCK_TEXTURE_RECORD')=='1'
camera_enabled=os.environ.get('ROCK_TEXTURE_CAMERA','1')=='1'
assert camera_enabled or not record
original_init=soil.NewtonSoil.__init__;original_update=soil.NewtonSoil._on_update
start=None;done=False;frame=0;next_frame=0.;initial_hash=None

def initialize(self,*a,**kw):
    global initial_hash
    original_init(self,*a,**kw)
    from pxr import UsdGeom,UsdShade,Sdf,Gf
    import omni.replicator.core as rep
    initial_hash=hashlib.sha256(self.positions.tobytes()).hexdigest()
    self._rock_widths=np.asarray(self.render.GetWidthsAttr().Get()).copy()
    if mode!='original':
        material=UsdShade.Material.Define(self.stage,'/World/RockPreviewMaterial')
        shader=UsdShade.Shader.Define(self.stage,'/World/RockPreviewMaterial/Shader')
        shader.CreateImplementationSourceAttr('sourceAsset')
        shader.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'),'mdl')
        shader.SetSourceAssetSubIdentifier('OmniPBR','mdl')
        shader.CreateInput('diffuse_color_constant',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.35,.32,.28))
        shader.CreateInput('reflection_roughness_constant',Sdf.ValueTypeNames.Float).Set(.92)
        if mode=='texture':
            texture=Path(__file__).parent/'assets/rock_relief_basecolor.png'
            shader.CreateInput('diffuse_texture',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture)))
            shader.CreateInput('project_uvw',Sdf.ValueTypeNames.Bool).Set(os.environ.get('ROCK_PROJECT_UV','0')=='1')
            shader.CreateInput('world_or_object',Sdf.ValueTypeNames.Bool).Set(False)
            shader.CreateInput('texture_scale',Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(3.,3.))
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        UsdShade.MaterialBindingAPI.Apply(self.render.GetPrim()).Bind(material)
    if not camera_enabled:return
    x0,x1,y0,y1=self.runtime['current_sand_footprint_bbox_xy']()
    target=Gf.Vec3d((x0+x1)/2,(y0+y1)/2,float(self.runtime['SAND_FLOOR_Z'])+.45)
    eye=target+Gf.Vec3d(.6,2.6,3.2)
    cam=UsdGeom.Camera.Define(self.stage,'/World/RockTextureCamera')
    cam.CreateFocalLengthAttr(32.);cam.CreateHorizontalApertureAttr(36.)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
    self._rock_cam_op=cam.AddTransformOp()
    self._rock_cam_op.Set(Gf.Matrix4d().SetLookAt(eye,target,Gf.Vec3d(0,0,1)).GetInverse())
    self._rock_product=rep.create.render_product(str(cam.GetPath()),(1280,720))
    self._rock_rgb=rep.AnnotatorRegistry.get_annotator('rgb');self._rock_rgb.attach([self._rock_product])
    if record:(output/'frames').mkdir()
    print('[ROCK MATERIAL]',mode,'geometry',self.render.GetPrim().GetTypeName(),flush=True)

def update(self,event):
    global start,done,frame,next_frame
    original_update(self,event)
    if done:return
    if self.error:raise RuntimeError(self.error)
    if start is None and self.sim_time>=1-1e-5:
        self.wp.synchronize_device(self.device)
        start=(time.perf_counter(),self.sim_time,self.physics_wall_seconds,self.steps)
    if record and self.sim_time>=next_frame:
        from PIL import Image
        pixels=np.asarray(self._rock_rgb.get_data())
        if pixels.ndim==3 and pixels.size:
            Image.fromarray(pixels[:,:,:3]).save(output/'frames'/f'{frame:06d}.png',compress_level=1)
            frame+=1;next_frame=self.sim_time+.1-1e-5
    if self.sim_time<6-1e-5:return
    done=True
    self.wp.synchronize_device(self.device)
    wall=time.perf_counter()-start[0]
    assert self.render.GetPrim().GetTypeName()=='Points'
    assert np.array_equal(self._rock_widths,np.asarray(self.render.GetWidthsAttr().Get()))
    result=dict(mode=mode,recording=record,extra_camera=camera_enabled,wall_seconds=wall,sim_seconds=self.sim_time-start[1],
        physics_seconds=self.physics_wall_seconds-start[2],update_fps=(self.steps-start[3])/wall,
        initial_sha256=initial_hash,diagnostics=self.diagnostics(),geometry='Points',widths_unchanged=True,
        finite=bool(np.isfinite(self.positions).all()),below_floor_2cm=int(np.count_nonzero(self.positions[:,2]<float(self.runtime['SAND_FLOOR_Z'])-.02)))
    (output/'result.json').write_text(json.dumps(result,indent=2))
    np.save(output/'final_positions.npy',self.positions)
    from PIL import Image
    if camera_enabled:
        pixels=np.asarray(self._rock_rgb.get_data())
        if pixels.ndim==3 and pixels.size:Image.fromarray(pixels[:,:,:3]).save(output/'preview.png')
    print('[ROCK RESULT]',json.dumps(result),flush=True)
    import omni.kit.app
    omni.kit.app.get_app().post_quit(0)
soil.NewtonSoil.__init__=initialize
soil.NewtonSoil._on_update=update
entry=repo/'run_vla_train_scene.py'
sys.argv=[str(entry),'--headless','--no-bridge','--disable-export','--active-gpu','0','--physics-gpu','0','--sand-amount','1']
runpy.run_path(str(entry),run_name='__main__')
