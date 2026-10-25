"""Bounded live Isaac regression for the production soil/contact adapter.

Run with the existing run_isaac_python.sh. No extra packages or scene assets.
Tests gravity, ground, real PhysX force feedback, pause and callback reset.
"""
import os
from pathlib import Path
import sys
os.environ.setdefault('OMNI_KIT_ALLOW_ROOT', '1')
os.environ.setdefault('EXCAVATOR_MPM_VOXEL_SIZE', '.04')
os.environ.setdefault('EXCAVATOR_MPM_PARTICLE_SPACING', '.025')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'multi_gpu': False, 'active_gpu': 0, 'physics_gpu': 0})
try:
    import numpy as np
    import omni.timeline
    from pxr import UsdPhysics, UsdGeom, UsdLux, Gf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from excavator_app import newton_mpm_soil as soil
    world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(FixedCuboid('/World/ParkingGround', name='parking_ground',
        position=np.array([0.,0.,-.04]), scale=np.array([10.,10.,.08])))
    bucket = world.scene.add(DynamicCuboid('/World/Bucket', name='bucket',
        position=np.array([0., 0., .43]), scale=np.array([.18, .18, .12]), mass=3.0))
    UsdPhysics.MassAPI.Apply(bucket.prim).CreateCenterOfMassAttr(Gf.Vec3f(.04, 0., .025))
    world.reset()
    image_path = os.environ.get('EXCAVATOR_MPM_TEST_IMAGE')
    rgb = None
    if image_path:
        import omni.usd
        import omni.replicator.core as rep
        stage = omni.usd.get_context().get_stage()
        UsdLux.DomeLight.Define(stage, '/World/TestLight').CreateIntensityAttr(1000.)
        camera = UsdGeom.Camera.Define(stage, '/World/TestCamera')
        camera.CreateFocalLengthAttr(30.)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(1.3, -1.6, 1.1), Gf.Vec3d(0, 0, .1), Gf.Vec3d(0, 0, 1)).GetInverse())
        product = rep.create.render_product('/World/TestCamera', (640, 480))
        rgb = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb.attach([product])
    runtime = {'STATE': {}, 'SAND_FLOOR_Z': 0., 'PARTICLE_POINTS_PATH_SUFFIX': 'RealSandParticles',
        'current_sand_footprint_bbox_xy': lambda: (-.25, .25, -.25, .25),
        'is_inside_diggable_xy_batch': lambda x, y: np.ones(x.shape, bool),
        'initial_sand_height_xy_batch': lambda x, y: np.full(x.shape, .25),
        'sand_particle_points': lambda: ([0]*500, [], [], {})}
    soil.create(runtime, '/World/SandSite')
    controller = soil.current()
    # GPU atomic reduction must match the CPU reference, including the torque
    # origin, static exclusions and invalid collider IDs.
    wp = controller.wp
    impulses = np.array([[0,0,-9], [0,2,0], [0,3,0], [99,0,0]], dtype=np.float32)
    positions = np.array([[0,0,0], [2,0,0], [1,0,0], [0,0,0]], dtype=np.float32)
    ids = np.array([0,1,1,-1], dtype=np.int32)
    bodies = np.array([-1,0], dtype=np.int32)
    origins = np.array([[1,0,0]], dtype=np.float32)
    with wp.ScopedDevice(controller.device):
        output = wp.zeros((1,6), dtype=wp.float32)
        wp.launch(controller.wrench_kernel, dim=4, inputs=[
            wp.array(impulses, dtype=wp.vec3), wp.array(positions, dtype=wp.vec3),
            wp.array(ids, dtype=wp.int32), wp.array(bodies, dtype=wp.int32),
            wp.array(np.array([[1,0,0,0,0,0,1]], dtype=np.float32), dtype=wp.transform), output])
        np.testing.assert_allclose(output.numpy(),
            soil.aggregate_wrenches(impulses, positions, ids, bodies, origins, 1), atol=1e-6)
    print('MPM_GPU_WRENCH_REFERENCE_PASSED', flush=True)
    physx_com = controller.body_view.get_coms().numpy()[controller.model_to_view, :3]
    np.testing.assert_allclose(controller.model.body_com.numpy(), physx_com)
    minimum_z, maximum_z, maximum_reaction = 1., 0., 0.
    for i in range(240):
        world.step(render=True)
        if controller.error:
            raise RuntimeError(controller.error)
        minimum_z = min(minimum_z, float(bucket.get_world_pose()[0][2]))
        maximum_z = max(maximum_z, float(bucket.get_world_pose()[0][2]))
        maximum_reaction = max(maximum_reaction,
                              float(controller.last_force[:, 2].max(initial=0.)))
    print('MPM_MOTION_SUMMARY', {'min_z': minimum_z, 'max_z': maximum_z,
          'position': bucket.get_world_pose()[0].tolist(),
          'velocity': bucket.get_linear_velocity().tolist(), 'peak_reaction_N': maximum_reaction}, flush=True)
    assert controller.steps >= 240, controller.diagnostics()
    assert controller.solver_substeps == 2*controller.steps
    assert len(controller.positions) <= 40000
    assert controller.host_contact_values == controller.solver_substeps*6*len(controller.body_paths)
    assert np.isfinite(controller.positions).all()
    assert controller.positions[:, 2].min() > -.05, controller.diagnostics()
    assert maximum_reaction > 5., ('missing soil-to-PhysX force', controller.diagnostics())
    assert minimum_z > .07, ('bucket fell through soil', minimum_z)
    assert maximum_z < .65, ('contact launched a light rigid body', maximum_z)
    final_speed = float(np.linalg.norm(bucket.get_linear_velocity()))
    assert final_speed < .5, ('rigid body did not settle', final_speed)
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    before_pause = controller.steps
    for _ in range(10): app.update()
    assert controller.steps == before_pause, 'MPM advanced while paused'
    if rgb is not None:
        from PIL import Image
        pixels = np.asarray(rgb.get_data())[:, :, :3]
        brown = (pixels[:, :, 0].astype(float) > pixels[:, :, 1]*1.15) & (pixels[:, :, 1].astype(float) > pixels[:, :, 2]*1.15)
        Image.fromarray(pixels).save(image_path)
        assert int(brown.sum()) > 200, ('soil not visible to actual Isaac camera', int(brown.sum()))
        print('MPM_RENDER_PASSED', image_path, 'brown_pixels', int(brown.sum()), flush=True)
    old = controller
    bucket.set_world_pose(position=np.array([2., 0., .43]))
    bucket.set_linear_velocity(np.zeros(3))
    bucket.set_angular_velocity(np.zeros(3))
    soil.create(runtime, '/World/SandSite')
    controller = soil.current()
    assert old.closed and old.subscription is None
    assert controller.steps == 0 and controller.sim_time == 0
    timeline.play()
    world.step(render=True)
    assert controller.steps > 0 and old.steps == before_pause, 'duplicate physics callback after reset'
    for _ in range(240):
        if controller.sleeping:
            break
        world.step(render=True)
    assert controller.sleeping, ('resting soil never sleeps', controller.diagnostics())
    solved = controller.solver_substeps
    for _ in range(10): world.step(render=True)
    assert controller.solver_substeps == solved, 'sleeping soil kept solving'
    timeline.pause()
    controller.cache_initial_state()
    settled = controller.positions.copy()
    runtime['SAND_CENTER_X'], runtime['SAND_CENTER_Y'] = .713, 0.
    runtime['current_sand_footprint_bbox_xy'] = lambda: (.463, .963, -.25, .25)
    assert controller.restore_initial_state(), 'complete-state episode reset failed'
    np.testing.assert_allclose(controller.positions, settled+np.array([.713,0,0]), atol=1e-6)
    assert controller.steps == 0 and not controller.last_force.any()
    timeline.play()
    for _ in range(240):
        world.step(render=True)
        if controller.sleeping: break
    assert controller.sleeping, controller.diagnostics()
    solved = controller.solver_substeps
    bucket.set_world_pose(position=np.array([.713, 0., .2]))
    bucket.set_linear_velocity(np.zeros(3))
    world.step(render=True)
    assert not controller.sleeping and controller.solver_substeps > solved, 'approaching rigid body failed to wake soil'
    # Stress fixed substeps beyond the normal search band. Imported-mesh
    # projection must catch the fast point without raising global substeps.
    particle_q = controller.state.particle_q.numpy()
    particle_v = controller.state.particle_qd.numpy()
    particle_q[0], particle_v[0] = [3.,0.,.2], [0.,0.,-20.]
    controller.state.particle_q.assign(particle_q)
    controller.state.particle_qd.assign(particle_v)
    solved = controller.solver_substeps
    for _ in range(5):
        world.step(render=True)
        assert controller.positions[0, 2] > -.025, ('fast point tunneled through plane/parking floor', controller.positions[0])
    assert controller.solver_substeps-solved == 10, 'fast outlier increased global substeps'
    print('MPM_FAST_GROUND_AND_SNAPSHOT_RESET_PASSED', flush=True)
    print('MPM_PHYSX_REGRESSION_PASSED', {'steps': before_pause, 'minimum_bucket_z': minimum_z,
          'maximum_bucket_z': maximum_z, 'final_bucket_speed': final_speed,
          'peak_upward_reaction_N': maximum_reaction, 'diagnostics': controller.diagnostics()}, flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    print('MPM_PHYSX_REGRESSION_FAILED', flush=True)
    raise
finally:
    try: soil.close()
    except NameError: pass
    app.close()
