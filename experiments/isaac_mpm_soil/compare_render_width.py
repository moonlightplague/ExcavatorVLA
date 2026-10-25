"""Render one frozen real scene twice; only the passive USD point width varies."""
from pathlib import Path
import importlib.util
import json
import os
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
out = Path(os.environ['EXCAVATOR_RENDER_COMPARE_OUTPUT'])
out.mkdir(parents=True, exist_ok=True)


def compare_impl(app, runtime, **_kwargs):
    import numpy as np
    import omni.timeline
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Gf, UsdGeom, Vt
    from excavator_app import newton_mpm_soil as soil

    runtime.request_auto_collect(1, max_attempts=1)
    started = time.monotonic()
    controller = None
    while time.monotonic()-started < 180:
        app.update()
        controller = soil.current()
        if controller and controller.error:
            raise RuntimeError(controller.error)
        if controller and controller.sim_time >= 2.5:
            break
    else:
        raise TimeoutError('Soil did not reach the display-comparison state')

    # Stop this disposable preview task. No original training data are used.
    runtime.STATE['running'] = False
    runtime.STATE['auto_collect_stop_requested'] = True
    omni.timeline.get_timeline_interface().pause()
    runtime.cancel_registered_tasks(reason='frozen_render_comparison', keep=set())
    for _ in range(8): app.update()
    print('[WIDTH COMPARE] preview tasks stopped', flush=True)
    assert controller.render.GetPrim().IsValid(), 'Preview cleanup removed the soil'
    center = Gf.Vec3d(float(controller.runtime['SAND_CENTER_X']),
                     float(controller.runtime['SAND_CENTER_Y']), 0.)
    camera_path = '/World/SoilWidthComparisonCamera'
    camera = UsdGeom.Camera.Define(controller.stage, camera_path)
    camera.CreateFocalLengthAttr(30.)
    camera.CreateHorizontalApertureAttr(36.)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.05,1000.))
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
        center+Gf.Vec3d(4.,-5.,3.2), center+Gf.Vec3d(0.,0.,.65), Gf.Vec3d(0.,0.,1.)).GetInverse())
    product = rep.create.render_product(camera_path, (960,640))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    rgb.attach([product])
    print('[WIDTH COMPARE] camera initialized', flush=True)

    def physical_state():
        return [controller.state.particle_q.numpy().copy(),
                controller.state.particle_qd.numpy().copy(), controller.grains.numpy().copy(),
                controller.state.body_q.numpy().copy(), controller.model.particle_mass.numpy().copy()]

    before, step = physical_state(), controller.steps
    images = []
    widths = [max(.05/12,controller.spacing/12), controller.render_diameter]
    for name, diameter in zip(['before_6_25mm.png','after_11mm.png'], widths):
        controller.render.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(
            np.full(controller.grains.size, diameter, dtype=np.float32)))
        for _ in range(16): app.update()
        rep.orchestrator.step(delta_time=0.0, pause_timeline=True, rt_subframes=4)
        pixels = np.asarray(rgb.get_data())
        assert pixels.shape[:2] == (640,960), pixels.shape
        Image.fromarray(pixels[:,:,:3]).save(out/name)
        images.append(pixels[:,:,:3].copy())
        assert controller.steps == step, 'Physics advanced while comparing display widths'
        for old, new in zip(before, physical_state()):
            np.testing.assert_array_equal(old,new)
    difference = np.mean(np.any(images[0] != images[1], axis=2))
    assert difference > .001, 'Width change was not visible in the rendered image'
    result = {'render_widths_m': widths, 'physics_state_identical':True,
              'material_positions_velocities_masses_and_grain_positions_unchanged':True,
              'physics_steps':step, 'changed_image_pixel_fraction':float(difference),
              'diagnostics':controller.diagnostics()}
    (out/'comparison_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RENDER_ONLY_WIDTH_COMPARISON_PASSED', json.dumps(result), flush=True)
    rgb.detach([product])


def compare(*args, **kwargs):
    try:
        return compare_impl(*args, **kwargs)
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise


spec = importlib.util.spec_from_file_location('original_visual_launcher', repo/'run_vla_train_scene.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
launcher.run_auto_collect = compare
launcher.main()
