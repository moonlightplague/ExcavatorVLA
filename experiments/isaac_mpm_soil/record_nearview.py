"""Record an extra offscreen soil camera while running the original scene.

Only adds a camera and captures images; production soil/controller files and
their parameters remain unchanged. Output is isolated from training data.
"""
from pathlib import Path
import json
import os
import runpy
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from excavator_app import newton_mpm_soil as soil

out = Path(os.environ['EXCAVATOR_NEARVIEW_OUTPUT'])
out.mkdir(parents=True, exist_ok=True)
original_init, original_update = soil.NewtonSoil.__init__, soil.NewtonSoil._on_update
recordings = {}


def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    import omni.replicator.core as rep
    from pxr import Gf, UsdGeom
    index = len(recordings)+1
    root = out/f'nearview_{index:02d}'
    root.mkdir(exist_ok=False)
    center = Gf.Vec3d(float(self.runtime['SAND_CENTER_X']),
                     float(self.runtime['SAND_CENTER_Y']), 0.)
    camera_path = f'/World/SoilResultCamera{index}'
    camera = UsdGeom.Camera.Define(self.stage, camera_path)
    camera.CreateFocalLengthAttr(30.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05, 1000.))
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        center+Gf.Vec3d(4.,-5.,3.2), center+Gf.Vec3d(0.,0.,.65), Gf.Vec3d(0.,0.,1.)).GetInverse())
    product = rep.create.render_product(camera_path, (960,640))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    entry = {'root': root, 'rgb': rgb, 'product': product, 'last_step': -100,
             'frames': 0, 'index': index}
    recordings[id(self)] = entry
    (root/'configuration.json').write_text(json.dumps({
        'camera': camera_path, 'size': [960,640], 'output_fps': 10,
        'center': list(center), 'soil': self.diagnostics(),
        'description': 'Additional camera, fresh real Isaac run; playback uses simulation time, not measured wall-time speed.'
    }, indent=2)+'\n')
    print('[NEARVIEW READY]', root, flush=True)


def update(self, event):
    original_update(self, event)
    entry = recordings.get(id(self))
    if entry is None or self.closed or self.steps-entry['last_step'] < 6:
        return
    from PIL import Image
    import numpy as np
    pixels = np.asarray(entry['rgb'].get_data())
    if pixels.ndim != 3 or pixels.shape[0] == 0:
        return
    index = entry['frames']
    Image.fromarray(pixels[:,:,:3]).save(entry['root']/f'{index:06d}.png', compress_level=1)
    with (entry['root']/'frames.jsonl').open('a') as stream:
        stream.write(json.dumps({'frame': index, 'mpm_steps': self.steps,
            'simulation_seconds': self.sim_time, 'diagnostics': self.diagnostics()})+'\n')
    entry['last_step'], entry['frames'] = self.steps, index+1
    if index % 60 == 0:
        print('[NEARVIEW FRAME]', index, 'sim_t', self.sim_time, flush=True)


soil.NewtonSoil.__init__, soil.NewtonSoil._on_update = initialize, update
sys.argv[0] = str(repo/'run_vla_train_scene.py')
runpy.run_path(str(repo/'run_vla_train_scene.py'), run_name='__main__')
