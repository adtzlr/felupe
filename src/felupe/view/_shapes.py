# SPDX-License-Identifier: GPL-3.0-or-later

"""Shapes of the decorations of a model, e.g. of rigid contact surfaces, independent
of the plotting backend. A plotting backend draws a shape by its own geometry, e.g.
the sources of PyVista, or by the triangles of :meth:`Shape.triangulate`."""

from dataclasses import dataclass, field

import numpy as np


def plane_axes(normal):
    """Return two orthonormal axes in a plane with a given normal vector. They follow
    the convention of PyVista, e.g. the axes ``(y, z)`` for the normal ``(1, 0, 0)``.
    """

    normx = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    normy = np.array([0.0, 1.0, 0.0])

    if np.allclose(normx, [0, 1, 0]):
        normy = np.array([-1.0, 0.0, 0.0])
    elif np.allclose(normx, [0, -1, 0]):
        normy = np.array([1.0, 0.0, 0.0])

    normz = np.cross(normx, normy)
    normz /= np.linalg.norm(normz)
    normy = np.cross(normz, normx)

    return normy, normz


def symmetry_clip(sym, points):
    """Return the clipping ``((axis, invert), ...)`` at the symmetry planes
    ``x[axis] = 0`` of the flagged axes. The side of the given points is kept, i.e.
    the negative side if there are points with negative coordinates."""

    return tuple(
        (int(axis), bool(np.any(points[:, axis] < 0)))
        for axis in np.where(np.array(sym, dtype=bool))[0]
    )


def _circle_axes(direction):
    "Return the unit direction and two orthonormal axes perpendicular to it."

    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    u, v = plane_axes(direction)

    return direction, u, v


def _clip_polygon(polygon, axis, invert):
    """Clip a polygon (Sutherland-Hodgman) at the plane ``x[axis] = 0``. The positive
    side is kept, or the negative side if ``invert`` is True."""

    sign = -1.0 if invert else 1.0
    distances = sign * polygon[:, axis]
    clipped = []

    for a in range(len(polygon)):
        b = (a + 1) % len(polygon)
        da, db = distances[a], distances[b]

        if da >= 0:
            clipped.append(polygon[a])

        if (da >= 0) != (db >= 0):
            t = da / (da - db)
            clipped.append(polygon[a] + t * (polygon[b] - polygon[a]))

    return np.array(clipped).reshape(-1, 3)


def _area(polygon):
    "Return the area of a planar polygon."

    return np.linalg.norm(np.cross(polygon, np.roll(polygon, -1, axis=0)).sum(0)) / 2


def clip_triangles(points, triangles, clip):
    """Clip triangles at the planes ``x[axis] = 0`` of a list of ``(axis, invert)``
    and return the points and the triangles of the clipped surface."""

    if len(clip) == 0:
        return points, triangles

    new_points = []
    new_triangles = []
    offset = 0

    for triangle in triangles:
        polygon = points[triangle]
        tolerance = 1e-12 * _area(polygon)

        for axis, invert in clip:
            polygon = _clip_polygon(polygon, axis, invert)

            if len(polygon) < 3:
                break

        # degenerated polygons, e.g. an edge on the clipping plane, are not kept
        if len(polygon) >= 3 and _area(polygon) > tolerance:
            fan = [(0, i, i + 1) for i in range(1, len(polygon) - 1)]
            new_points.append(polygon)
            new_triangles.append(np.array(fan) + offset)
            offset += len(polygon)

    if not new_points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)

    return np.vstack(new_points), np.vstack(new_triangles)


def clip_lines(lines, clip):
    """Clip line segments of shape ``(nlines, 2, 3)`` at the planes ``x[axis] = 0`` of
    a list of ``(axis, invert)`` and return the clipped line segments."""

    lines = np.asarray(lines, dtype=float)
    clipped = []

    for start, end in lines:
        keep = True

        for axis, invert in clip:
            sign = -1.0 if invert else 1.0
            ds, de = sign * start[axis], sign * end[axis]

            if ds < 0 and de < 0:
                keep = False
                break

            if (ds < 0) != (de < 0):
                point = start + ds / (ds - de) * (end - start)
                if ds < 0:
                    start = point
                else:
                    end = point

        # a degenerated segment, e.g. clipped at its end point, is not kept
        if keep and not np.allclose(start, end):
            clipped.append([start, end])

    return np.array(clipped).reshape(-1, 2, 3)


def outline(triangles):
    """Return the boundary edges of a triangulated surface, i.e. the edges which belong
    to only one triangle, of shape ``(nedges, 2)``."""

    triangles = np.asarray(triangles).reshape(-1, 3)
    edges = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]
    )
    keys, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)

    return keys[counts == 1]


@dataclass
class Shape:
    """Base class of a shape, optionally clipped at the planes ``x[axis] = 0`` of a
    list of ``(axis, invert)``. The positive side of a plane is kept, or the negative
    side if ``invert`` is True."""

    clip: tuple = field(default=(), kw_only=True)

    def triangulate(self):
        """Return the points of shape ``(npoints, 3)`` and the triangles of shape
        ``(ntriangles, 3)`` of the surface of the (clipped) shape."""

        points, triangles = self._triangles()
        return clip_triangles(points, triangles, self.clip)

    def _triangles(self):
        raise NotImplementedError


@dataclass
class Plane(Shape):
    "A square in a plane with a center point, a normal vector and a size."

    center: np.ndarray
    normal: np.ndarray
    size: float

    def _triangles(self):
        u, v = plane_axes(self.normal)
        a = self.size / 2
        center = np.asarray(self.center, dtype=float)
        points = center + np.array([-a * u - a * v, a * u - a * v, a * u + a * v])
        points = np.vstack([points, center - a * u + a * v])

        return points, np.array([[0, 1, 2], [0, 2, 3]])


@dataclass
class Rectangle(Shape):
    """A rectangle, given by three of its corner points. The fourth corner point is
    opposite to the corner point with the right angle."""

    points: np.ndarray

    def _triangles(self):
        p = np.asarray(self.points, dtype=float)

        # the corner point with the right angle
        angles = [
            np.dot(p[(i + 1) % 3] - p[i], p[(i + 2) % 3] - p[i]) for i in range(3)
        ]
        i = int(np.argmin(np.abs(angles)))
        a, b, c = p[(i + 1) % 3], p[i], p[(i + 2) % 3]

        points = np.array([a, b, c, a + c - b])
        return points, np.array([[0, 1, 2], [0, 2, 3]])


@dataclass
class Cylinder(Shape):
    """A cylinder with a center point, an axis direction, a radius, a height and a
    number of points in circumferential direction."""

    center: np.ndarray
    direction: np.ndarray
    radius: float
    height: float
    resolution: int = 100
    capping: bool = True

    def _triangles(self):
        d, u, v = _circle_axes(self.direction)
        n = self.resolution
        phi = 2 * np.pi * np.arange(n) / n
        circle = self.radius * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
        center = np.asarray(self.center, dtype=float)

        bottom = center - self.height / 2 * d + circle
        top = center + self.height / 2 * d + circle
        points = np.vstack([bottom, top])

        i = np.arange(n)
        j = (i + 1) % n
        triangles = np.vstack(
            [np.stack([i, j, j + n], axis=1), np.stack([i, j + n, i + n], axis=1)]
        )

        if self.capping:
            centers = np.array(
                [center - self.height / 2 * d, center + self.height / 2 * d]
            )
            points = np.vstack([points, centers])
            caps = [np.stack([j, i, np.full(n, 2 * n)], axis=1)]
            caps.append(np.stack([i + n, j + n, np.full(n, 2 * n + 1)], axis=1))
            triangles = np.vstack([triangles, *caps])

        return points, triangles


@dataclass
class Cone(Shape):
    """A cone with a center point (in the middle of its height), a direction from the
    base to the tip, a height, a radius of the base and a number of points of the
    base."""

    center: np.ndarray
    direction: np.ndarray
    height: float
    radius: float
    resolution: int = 6
    capping: bool = True

    def _triangles(self):
        d, u, v = _circle_axes(self.direction)
        n = self.resolution
        phi = 2 * np.pi * np.arange(n) / n
        circle = self.radius * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))
        center = np.asarray(self.center, dtype=float)

        base = center - self.height / 2 * d
        tip = center + self.height / 2 * d
        points = np.vstack([base + circle, tip])

        i = np.arange(n)
        j = (i + 1) % n
        triangles = np.stack([i, j, np.full(n, n)], axis=1)

        if self.capping:
            points = np.vstack([points, base])
            triangles = np.vstack([triangles, np.stack([j, i, np.full(n, n + 1)], 1)])

        return points, triangles


def arrow():
    """Return the points and the triangles of an arrow of length 1 in the
    x-direction, i.e. a cylinder (the shaft) with a cone at the tip."""

    shaft = Cylinder(
        center=(0.325, 0, 0),
        direction=(1, 0, 0),
        radius=0.03,
        height=0.65,
        resolution=6,
    )
    tip = Cone(center=(0.825, 0, 0), direction=(1, 0, 0), height=0.35, radius=0.1)

    points, triangles = shaft.triangulate()
    tip_points, tip_triangles = tip.triangulate()

    return np.vstack([points, tip_points]), np.vstack(
        [triangles, tip_triangles + len(points)]
    )


def arrows(points, directions, scale=1.0):
    """Return the points of shape ``(npoints, 3)`` and the triangles of shape
    ``(ntriangles, 3)`` of arrows by the glyph of :func:`arrow`, given by their start
    points and their vectors (multiplied by a scale), both of shape ``(narrows, 3)``.
    Arrows of zero length are not included."""

    starts = np.asarray(points, dtype=float).reshape(-1, 3)
    vectors = scale * np.asarray(directions, dtype=float).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 0
    starts, vectors, lengths = starts[keep], vectors[keep], lengths[keep]

    glyph, glyph_triangles = arrow()
    frames = []
    for vector, length in zip(vectors, lengths):
        u, v = plane_axes(vector)
        frames.append(np.stack([vector, length * u, length * v]))

    frames = np.array(frames).reshape(-1, 3, 3)
    positions = starts[:, None] + np.einsum("pj,ajk->apk", glyph, frames)
    offsets = len(glyph) * np.arange(len(starts))
    triangles = (glyph_triangles[None] + offsets[:, None, None]).reshape(-1, 3)

    return positions.reshape(-1, 3), triangles
