"""Geometry checks that do not require an Isaac Sim runtime."""


def closed_mesh_boundary_edge_count(vertices, faces):
    """Count edges without exactly two incident faces, joining exact positions.

    Imported meshes can repeat a position for each face to preserve normal/UV
    seams. Those separate indices do not make a geometric hole. Only identical
    positions are joined; nearby vertices remain distinct so real gaps fail.
    The authored vertices and faces are left unchanged for containment tests.
    """
    position_ids = {}
    vertex_ids = []
    for vertex in vertices:
        position = tuple(float(value) for value in vertex)
        vertex_ids.append(position_ids.setdefault(position, len(position_ids)))

    counts = {}
    for face in faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % len(face)]
            key = tuple(sorted((vertex_ids[int(first)], vertex_ids[int(second)])))
            counts[key] = counts.get(key, 0) + 1
    return sum(1 for count in counts.values() if count != 2)
