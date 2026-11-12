"""CPU-only numerical regression checks; no Isaac, Warp or GPU imports."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

path = Path(__file__).resolve().parents[2]/'scripts/excavator_app/newton_mpm_soil.py'
spec = importlib.util.spec_from_file_location('coarse_soil_test', path)
soil = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soil)


class CoarseMPMTests(unittest.TestCase):
    def test_coarsening_reduces_work_preserves_mass_and_footprint(self):
        runtime = dict(current_sand_footprint_bbox_xy=lambda: (-1.2, 1.2, -1.2, 1.2),
                       is_inside_diggable_xy_batch=lambda x, y: np.ones(x.shape, bool),
                       initial_sand_height_xy_batch=lambda x, y: np.full(x.shape, 3.),
                       SAND_FLOOR_Z=0., PARTICLE_MASS=.36)
        results = []
        for profile in ('fine', 'coarse'):
            with patch.dict(os.environ, {'EXCAVATOR_MPM_PROFILE': profile}, clear=True):
                cfg = soil.soil_resolution_settings()
                spacing, grid, count = soil.choose_resolution(runtime, cfg['max_points'], cfg['spacing'], cfg['voxel'])
                points, volume = soil.sample_material(runtime, spacing, cfg['max_points'])
                density, source = soil.material_density(runtime, count, volume, 26411)
                self.assertAlmostEqual(density*count*volume, 26411*.36)
                self.assertEqual(source, 'preserve_legacy_total_mass')
                self.assertEqual(count, len(points))
                self.assertLess(points[:, 0].min(), -1.05)
                self.assertGreater(points[:, 0].max(), 1.05)
                self.assertGreater(points[:, 2].min(), 0)
                self.assertEqual(soil.mpm_substeps(1/60), 2)
                results.append((count, volume, grid, cfg))
        fine, coarse = results
        self.assertLessEqual(coarse[0], 6000)
        self.assertLessEqual(coarse[0], fine[0]/7)
        self.assertGreater(coarse[1], fine[1]*7)
        self.assertTrue(coarse[3]['render_material_centers'])
        self.assertFalse(fine[3]['render_material_centers'])

    def test_default_fine_is_unchanged(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = soil.soil_resolution_settings()
        self.assertEqual(cfg['max_points'], 40000)
        self.assertAlmostEqual(cfg['spacing'], .075)
        self.assertEqual(cfg['voxel'], .10)
        self.assertEqual(cfg['render_diameter'], .014)

    def test_invalid_configuration_fails_before_gpu(self):
        for values in ({'EXCAVATOR_MPM_PROFILE': 'rigid'},
                       {'EXCAVATOR_MPM_VOXEL_SIZE': 'nan'},
                       {'EXCAVATOR_MPM_PARTICLE_SPACING': '.5'},
                       {'EXCAVATOR_MPM_MAX_PARTICLES': '0'},
                       {'EXCAVATOR_MPM_RENDER_DIAMETER': '-1'}):
            with patch.dict(os.environ, values, clear=True):
                with self.assertRaises(ValueError):
                    soil.soil_resolution_settings()


if __name__ == '__main__':
    unittest.main()
