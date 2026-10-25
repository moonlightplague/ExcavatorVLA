import unittest

import numpy as np

from baseline.bed_fill import BedFillPlanner, FillConfig, inside_polygon


class BedFillTests(unittest.TestCase):
    def make_bed(self, angle=0):
        angle = np.radians(angle)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offset = np.array([8., -4.])
        polygon = np.array([[-3., -1.2], [3., -1.2], [3., 1.2], [-3., 1.2]]) @ rotation.T + offset
        safe = np.array([[-2.4, -.6], [2.4, -.6], [2.4, .6], [-2.4, .6]]) @ rotation.T + offset
        return BedFillPlanner(polygon, safe, 1., 3.)

    def test_rotated_bed_cells_are_safe_and_equivariant(self):
        bed = self.make_bed(67)
        self.assertEqual(len(bed.centers), 15)
        self.assertTrue(inside_polygon(bed.centers, bed.safe_polygon).all())
        self.assertTrue(inside_polygon(bed.centers, bed.polygon[::-1]).all())
        self.assertEqual(len({row["cell_id"] for row in bed.rank([])[0]}), 15)
        original = self.make_bed()
        np.testing.assert_allclose(np.sort(np.linalg.norm(bed.centers - [8., -4.], axis=1)),
                                   np.sort(np.linalg.norm(original.centers - [8., -4.], axis=1)), atol=1e-6)

    def test_real_fill_outranks_motion_and_visits(self):
        bed = self.make_bed()
        first = bed.rank([])[0][0]
        xy = np.array(first["landing"][:2])
        # A tall deposit in the nearest cell must redirect the next landing.
        points = np.tile([*xy, 2.5], (100, 1))
        rows, measurement = bed.rank(points, bucket_xy=xy)
        self.assertNotEqual(rows[0]["cell_id"], first["cell_id"])
        self.assertEqual(measurement["particle_count"], 100)
        self.assertGreater(measurement["surface_std_m"], 0)
        self.assertEqual(measurement["cell_surface_z_m"].count(1.), 14)

    def test_visit_memory_spreads_empty_bed_ties(self):
        bed = self.make_bed(35)
        first = bed.rank([])[0][0]
        bed.commit(first["landing"])
        second = bed.rank([])[0][0]
        self.assertGreater(np.linalg.norm(np.array(first["landing"]) - second["landing"]), .8)

    def test_empty_bed_prefers_center_over_wall_nearest_to_bucket(self):
        bed = self.make_bed()
        first = bed.rank([], bucket_xy=[20, -4])[0][0]
        self.assertLess(first["center_norm"], .1)

    def test_excludes_pile_airborne_and_invalid_points(self):
        bed = self.make_bed()
        points = [[8, -4, 1.5], [8, -4, 5], [0, 0, 1.5], [8, -4, -1], [np.nan, 0, 0]]
        measurement = bed.measure(points)
        self.assertEqual(measurement["particle_count"], 1)
        self.assertGreater(measurement["occupied_fraction"], 0)
        self.assertEqual(sum(measurement["cell_counts"]), 1)

    def test_uniform_fill_is_flatter_than_one_heap(self):
        bed = self.make_bed()
        local_centers = np.array([(x + .5, y + .5) for x in range(5) for y in range(3)])
        xy = (bed.low + local_centers * bed.cell_size) @ bed.axes.T
        uniform = np.column_stack((xy, np.full(15, 1.5)))
        heap = np.tile([*xy[0], 2.5], (15, 1))
        self.assertAlmostEqual(bed.measure(uniform)["surface_std_m"], 0)
        self.assertGreater(bed.measure(heap)["surface_std_m"], 0)

    def test_invalid_geometry_and_config_fail(self):
        for value in (float("nan"), -1, 0):
            with self.assertRaises(ValueError):
                FillConfig(wall_margin_m=value)
        with self.assertRaises(ValueError):
            FillConfig(long_cells=0)
        with self.assertRaises(ValueError):
            BedFillPlanner([[0, 0]] * 3, [[0, 0]] * 3, 1, 3)


if __name__ == "__main__":
    unittest.main()
