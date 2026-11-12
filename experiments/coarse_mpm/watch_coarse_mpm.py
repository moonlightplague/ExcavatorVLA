"""GUI benchmark entry; logs wall-clock intervals without changing physics."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from excavator_app.newton_mpm_soil import NewtonSoil

original_update = NewtonSoil._on_update
original_init = NewtonSoil.__init__


def recorded_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if not os.environ.get('COARSE_VIDEO_DIR'):
        return
    import omni.replicator.core as rep
    from pxr import UsdGeom, Gf
    x0, x1, y0, y1 = self.runtime['current_sand_footprint_bbox_xy']()
    center = Gf.Vec3d((x0+x1)/2, (y0+y1)/2, float(self.runtime['SAND_FLOOR_Z'])+.6)
    width = max(x1-x0, y1-y0, 3.)
    eye = center+Gf.Vec3d(.7*width, -width, .7*width)
    camera = UsdGeom.Camera.Define(self.stage, '/World/CoarseMPMVideoCamera')
    camera.CreateFocalLengthAttr(32.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 1000.))
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0,0,1)).GetInverse())
    self._video_product = rep.create.render_product(str(camera.GetPath()), (960,640))
    self._video_rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    self._video_rgb.attach([self._video_product])
    self._video_folder = Path(os.environ['COARSE_VIDEO_DIR'])
    self._video_folder.mkdir(parents=True, exist_ok=False)
    self._video_frame, self._video_time = 0, -1.


def measured_update(self, event):
    original_update(self, event)
    if hasattr(self, '_video_rgb') and self.sim_time <= 8 and self.sim_time-self._video_time >= .1-1e-5:
        import numpy as np
        from PIL import Image
        pixels = np.asarray(self._video_rgb.get_data())
        if pixels.ndim == 3 and pixels.size:
            Image.fromarray(pixels[:,:,:3]).save(self._video_folder/f'{self._video_frame:06d}.png')
            with (self._video_folder/'frames.jsonl').open('a') as f:
                f.write(json.dumps(dict(frame=self._video_frame, sim=self.sim_time, wall=time.time()))+'\n')
            self._video_frame += 1
            self._video_time = self.sim_time
    now = time.perf_counter()
    previous = getattr(self, '_benchmark_last', None)
    if previous is None:
        self._benchmark_last = (now, self.sim_time, self.steps, self.solver_substeps)
        return
    wall = now-previous[0]
    if wall >= 10:
        row = dict(wall_seconds=wall, sim_seconds=self.sim_time-previous[1],
                   rtf=(self.sim_time-previous[1])/wall,
                   physics_steps=self.steps-previous[2],
                   solver_substeps=self.solver_substeps-previous[3],
                   sleeping=self.sleeping, points=len(self.positions),
                   render_points=int(self.grains.size), error=self.error)
        print('[COARSE BENCHMARK] '+json.dumps(row), flush=True)
        self._benchmark_last = (now, self.sim_time, self.steps, self.solver_substeps)


NewtonSoil._on_update = measured_update
NewtonSoil.__init__ = recorded_init
runpy.run_path(str(Path(__file__).with_name('run_coarse_mpm.py')), run_name='__main__')
