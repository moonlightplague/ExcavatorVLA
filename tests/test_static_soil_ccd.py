"""A fast MPM particle must not tunnel through a thin real triangle shell."""
import importlib.util
import os
from pathlib import Path
import unittest

import numpy as np

try:
    import warp as wp
except ImportError:
    wp = None

SOURCE = Path(__file__).resolve().parents[1]/"scripts/excavator_app/newton_mpm_soil.py"
spec = importlib.util.spec_from_file_location("ccd_newton_soil", SOURCE)
soil = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soil)


@unittest.skipIf(wp is None, "Run with Isaac Python and the bundled omni.warp.core on PYTHONPATH")
class StaticCCDTests(unittest.TestCase):
    def test_crossing_entire_floor_shell_is_caught_but_misses_and_outward_segments_are_unchanged(self):
        device = os.environ.get('WARP_TEST_DEVICE', 'cpu')
        # Four-centimeter floor: outward top normals +Z, bottom normals -Z.
        vertices = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                             [-1,-1,-.04],[1,-1,-.04],[1,1,-.04],[-1,1,-.04]], dtype=np.float32)
        indices = [0,1,2,0,2,3,4,6,5,4,7,6]
        mesh = wp.Mesh(wp.array(vertices,dtype=wp.vec3,device=device),
                       wp.array(indices,dtype=wp.int32,device=device))
        previous = np.array([[0,0,.05],[2,0,.05],[0,0,.06]], dtype=np.float32)
        end = np.array([[0,0,-.10],[2,0,-.10],[0,0,.15]], dtype=np.float32)
        velocity = wp.array([[1,0,-9],[1,0,-9],[0,0,5]],dtype=wp.vec3,device=device)
        positions = wp.array(end,dtype=wp.vec3,device=device)
        count = wp.zeros(1,dtype=wp.int32,device=device)
        wp.launch(soil.make_static_ccd_kernel(wp), dim=3, inputs=[
            wp.array(previous,dtype=wp.vec3,device=device), positions, velocity,
            wp.zeros(3,dtype=wp.mat33,device=device), wp.array([mesh.id],dtype=wp.uint64,device=device),
            .002,.5,count], device=device)
        actual = positions.numpy()
        self.assertAlmostEqual(actual[0,2], .002, places=5)
        self.assertGreaterEqual(velocity.numpy()[0,2], 0.)
        np.testing.assert_array_equal(actual[1:], end[1:])
        self.assertEqual(count.numpy()[0], 1)


if __name__ == '__main__':
    unittest.main()
