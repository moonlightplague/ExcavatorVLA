"""CPU checks for geometry, quadrature mass and rigid-side impulse mapping."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

module_path = Path(__file__).resolve().parents[2]/'scripts/excavator_app/newton_mpm_soil.py'
spec = importlib.util.spec_from_file_location('soil_adapter_under_test', module_path)
soil = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soil)


class SoilAdapterTests(unittest.TestCase):
    def test_quadrature_refinement_preserves_original_scene_mass(self):
        density, source = soil.material_density({'PARTICLE_MASS': .36}, 10000, .0001, 1000)
        self.assertAlmostEqual(density*10000*.0001, 360.)
        self.assertEqual(source, 'preserve_legacy_total_mass')
        self.assertEqual(soil.material_density({}, 10000, .0001, 1000)[0], 1600.)

    def test_fixed_substeps_are_independent_of_particle_speeds(self):
        dt = float(np.float32(1/60))
        self.assertEqual(soil.mpm_substeps(dt), 2)
        self.assertEqual(soil.mpm_substeps(1/120), 1)
        with self.assertRaises(ValueError):
            soil.mpm_substeps(float('nan'))

    def test_full_source_resolution_budget_and_mass(self):
        runtime = {'current_sand_footprint_bbox_xy': lambda: (-1.2, 1.2, -1.2, 1.2),
                   'is_inside_diggable_xy_batch': lambda x, y: np.ones(x.shape, bool),
                   'initial_sand_height_xy_batch': lambda x, y: np.full(x.shape, 3.),
                   'SAND_FLOOR_Z': 0.}
        spacing, voxel, count = soil.choose_resolution(runtime)
        points, volume = soil.sample_material(runtime, spacing, 40000)
        self.assertLessEqual(count, 40000)
        self.assertEqual(len(points), count)
        self.assertGreaterEqual(voxel, .1)
        self.assertAlmostEqual(voxel/spacing, .1/.075)
        self.assertLess(points[:, 0].min(), -1.1)
        self.assertGreater(points[:, 0].max(), 1.1)
        density, _ = soil.material_density({'PARTICLE_MASS': .36}, count, volume, 26411)
        self.assertAlmostEqual(count*volume*density, 26411*.36)
        runtime['initial_sand_height_xy_batch'] = lambda x, y: np.full(x.shape, 9.)
        larger_spacing, larger_voxel, larger_count = soil.choose_resolution(runtime)
        self.assertLessEqual(larger_count, 40000)
        self.assertGreater(larger_spacing, spacing)
        self.assertGreater(larger_voxel, voxel)
        with self.assertRaises(ValueError):
            soil.choose_resolution(runtime, max_count=40001)

    def test_contact_impulse_and_lever_arm(self):
        # A downward impulse on static ground must never reach the bucket.
        impulses = np.array([[0, 0, -9], [0, 2, 0], [0, 3, 0], [99, 0, 0]], float)
        positions = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]], float)
        result = soil.aggregate_wrenches(impulses, positions, np.array([0, 1, 1, -1]),
                                        np.array([-1, 0]), np.array([[1, 0, 0]]), 1)
        np.testing.assert_allclose(result, [[0, 5, 0, 0, 0, 2]])

    def test_usd_face_winding_and_open_faces(self):
        indices = soil.triangulate([4, 3], [0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(indices, [0, 1, 2, 0, 2, 3, 4, 5, 6])
        reverse = soil.triangulate([3], [0, 1, 2], left_handed=True)
        np.testing.assert_array_equal(reverse, [2, 1, 0])
        with self.assertRaises(ValueError):
            soil.triangulate([4], [0, 1, 2])

    def test_sampling_density_mass_floor_and_budget(self):
        runtime = {'current_sand_footprint_bbox_xy': lambda: (-.05, .05, -.05, .05),
                   'is_inside_diggable_xy_batch': lambda x, y: np.ones(x.shape, bool),
                   'initial_sand_height_xy_batch': lambda x, y: np.full(x.shape, .1),
                   'SAND_FLOOR_Z': 0.0}
        positions, volume = soil.sample_material(runtime, .025, 1000)
        self.assertEqual(len(positions), 64)
        self.assertAlmostEqual(len(positions)*volume*1600, 1.6)
        self.assertGreater(positions[:, 2].min(), 0.0)
        self.assertLess(positions[:, 2].max(), .1)
        np.testing.assert_array_equal(soil.sample_material(runtime, .025, 1000)[0], positions)
        with self.assertRaisesRegex(RuntimeError, 'grid is not silently coarsened'):
            soil.sample_material(runtime, .025, 20)

    def test_quaternion_matches_right_handed_rotation(self):
        matrix = soil.rotation([0, 0, np.sqrt(.5), np.sqrt(.5)])
        np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
