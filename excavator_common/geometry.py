import math


def clamp_value(value, lo=None, hi=None):
    value = float(value)
    if lo is not None:
        value = max(float(lo), value)
    if hi is not None:
        value = min(float(hi), value)
    return value


def convex_hull_xy(points):
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_signed_area(poly):
    if not poly or len(poly) < 3:
        return 0.0
    area = 0.0
    for a, b in zip(poly, poly[1:] + poly[:1]):
        area += float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
    return 0.5 * area


def polygon_centroid_xy(poly):
    area = polygon_signed_area(poly)
    if abs(area) < 1.0e-9:
        if not poly:
            return (0.0, 0.0)
        return (
            sum(float(p[0]) for p in poly) / float(len(poly)),
            sum(float(p[1]) for p in poly) / float(len(poly)),
        )
    cx = 0.0
    cy = 0.0
    for a, b in zip(poly, poly[1:] + poly[:1]):
        cross = float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
        cx += (float(a[0]) + float(b[0])) * cross
        cy += (float(a[1]) + float(b[1])) * cross
    scale = 1.0 / (6.0 * area)
    return (cx * scale, cy * scale)


def point_in_polygon_xy(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = float(poly[i][0]), float(poly[i][1])
        xj, yj = float(poly[j][0]), float(poly[j][1])
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / max(1.0e-12, yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def circle_polygon_xy(cx, cy, radius, segments=20):
    segments = max(3, int(segments))
    return [
        (
            float(cx) + float(radius) * math.cos(2.0 * math.pi * i / segments),
            float(cy) + float(radius) * math.sin(2.0 * math.pi * i / segments),
        )
        for i in range(segments)
    ]

