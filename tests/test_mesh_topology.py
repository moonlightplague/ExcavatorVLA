import unittest

from excavator_common.mesh_topology import closed_mesh_boundary_edge_count


class ClosedMeshTopologyTests(unittest.TestCase):
    def setUp(self):
        self.vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

    def split_vertices(self):
        vertices = [self.vertices[index] for face in self.faces for index in face]
        faces = [list(range(index, index + 3)) for index in range(0, 12, 3)]
        return vertices, faces

    def test_closed_mesh_with_shared_indices(self):
        self.assertEqual(closed_mesh_boundary_edge_count(self.vertices, self.faces), 0)

    def test_closed_mesh_with_separate_vertices_per_face(self):
        vertices, faces = self.split_vertices()
        self.assertEqual(closed_mesh_boundary_edge_count(vertices, faces), 0)
        self.assertEqual(len(vertices), 12)
        self.assertEqual(faces[0], [0, 1, 2])

    def test_missing_face_is_rejected(self):
        vertices, faces = self.split_vertices()
        self.assertEqual(closed_mesh_boundary_edge_count(vertices, faces[:-1]), 3)

    def test_small_real_gap_is_not_welded(self):
        vertices, faces = self.split_vertices()
        vertices[0] = (0, 0, 1e-8)
        self.assertGreater(closed_mesh_boundary_edge_count(vertices, faces), 0)

    def test_nonmanifold_edge_is_rejected(self):
        self.assertEqual(
            closed_mesh_boundary_edge_count(self.vertices, self.faces + self.faces[:1]),
            3,
        )


if __name__ == '__main__':
    unittest.main()
