# SPDX-License-Identifier: GPL-3.0-or-later

from dataclasses import dataclass

import numpy as np
import pytest

import felupe as fem
from felupe.view._shapes import (
    Cone,
    Cylinder,
    Plane,
    Rectangle,
    Shape,
    clip_lines,
    clip_triangles,
    plane_axes,
    symmetry_clip,
)


def area(points, triangles):
    "The area of a triangulated surface."

    a = points[triangles[:, 1]] - points[triangles[:, 0]]
    b = points[triangles[:, 2]] - points[triangles[:, 0]]
    return np.linalg.norm(np.cross(a, b), axis=1).sum() / 2


def same_points(p, q):
    "Check if two arrays contain the same (unique) points."

    p = np.unique(np.round(p, 6), axis=0)
    q = np.unique(np.round(q, 6), axis=0)
    return p.shape == q.shape and np.allclose(p, q)


def test_plane_axes():
    for normal in [(1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (-1, 0.3, 0.2)]:
        u, v = plane_axes(normal)
        n = np.array(normal) / np.linalg.norm(normal)
        assert np.allclose([u @ u, v @ v, u @ v, u @ n, v @ n], [1, 1, 0, 0, 0])

    assert np.allclose(plane_axes((1, 0, 0)), [[0, 1, 0], [0, 0, 1]])


def test_triangulate():
    plane = Plane(center=(1, 2, 3), normal=(0, 0, -1), size=2.0)
    points, triangles = plane.triangulate()
    assert np.isclose(area(points, triangles), 4.0)
    assert np.allclose(points[:, 2], 3)

    rectangle = Rectangle(points=[[1, 0, 0], [0, 0, 0], [0, 0, 3]])
    points, triangles = rectangle.triangulate()
    assert np.isclose(area(points, triangles), 3.0)
    assert same_points(points, [[0, 0, 0], [1, 0, 0], [0, 0, 3], [1, 0, 3]])

    n = 64
    cylinder = Cylinder(center=(0, 0, 0), direction=(0, 0, 2), radius=0.5, height=2.0)
    points, triangles = cylinder.triangulate()
    polygon = n * np.sin(np.pi / n)  # the circumference of the polygon / diameter
    assert len(points) == 2 * 100 + 2
    assert np.allclose(np.linalg.norm(points[:200, :2], axis=1), 0.5)

    cylinder = Cylinder(
        center=(0, 0, 0), direction=(0, 0, 1), radius=0.5, height=2, resolution=n
    )
    points, triangles = cylinder.triangulate()
    caps = 2 * n * 0.25 * np.sin(2 * np.pi / n) / 2
    assert np.isclose(area(points, triangles), polygon * 2 + caps)

    cylinder.capping = False
    assert np.isclose(area(*cylinder.triangulate()), polygon * 2)

    cone = Cone(center=(0, 0, 0), direction=(1, 0, 0), height=2.0, radius=1.0)
    points, triangles = cone.triangulate()
    assert np.allclose(points[6], [1, 0, 0])  # the tip
    assert np.allclose(points[:6, 0], -1)  # the base

    cone.capping = False
    assert len(cone.triangulate()[1]) == 6

    with pytest.raises(NotImplementedError):
        Shape().triangulate()


def test_triangulate_like_pyvista():
    pv = pytest.importorskip("pyvista")

    for normal in [(1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (-1, 0.3, 0.2)]:
        points, triangles = Plane(
            center=(0.1, 0.2, 0.3), normal=normal, size=1.3
        )._triangles()
        surface = pv.Plane(
            center=(0.1, 0.2, 0.3),
            direction=normal,
            i_resolution=1,
            j_resolution=1,
            i_size=1.3,
            j_size=1.3,
        )
        assert same_points(points, surface.points)

    for corners in [
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
    ]:
        points, triangles = Rectangle(points=corners).triangulate()
        assert same_points(points, pv.Rectangle(np.array(corners, dtype=float)).points)

    options = dict(center=(1, 2, 3), direction=(0, 1, 1), radius=0.5, height=2.0)
    for capping in [False, True]:
        points, triangles = Cylinder(**options, capping=capping).triangulate()
        surface = pv.Cylinder(**options, capping=capping).triangulate()
        assert np.isclose(area(points, triangles), surface.area)

    options = dict(center=(0, 0, 0), direction=(0, 0, 1), height=0.2, radius=0.07)
    points, triangles = Cone(**options).triangulate()
    assert np.isclose(area(points, triangles), pv.Cone(**options).triangulate().area)

    # clipped shapes
    options = dict(center=(0.5, 0, 1.5), normal=(0, 0, -1), size=2.0)
    points, triangles = Plane(**options, clip=((1, False),)).triangulate()
    surface = pv.Plane(
        center=(0.5, 0, 1.5),
        direction=(0, 0, -1),
        i_resolution=1,
        j_resolution=1,
        i_size=2,
        j_size=2,
    ).clip("y", invert=False, origin=(0, 0, 0))
    assert np.isclose(area(points, triangles), surface.area)

    options = dict(center=(0.5, 0, 1.5), direction=(0, 1, 0), radius=0.5, height=1.3)
    cylinder = Cylinder(**options, resolution=120, capping=False, clip=((0, True),))
    surface = pv.Cylinder(**options, resolution=120, capping=False).clip(
        "x", invert=True, origin=(0, 0, 0)
    )
    assert np.isclose(area(*cylinder.triangulate()), surface.area)


def test_clip():
    # the positive side is kept (or the negative side, if inverted)
    lines = [
        [[-1, 0, 0], [1, 0, 0]],
        [[-1, 0, 0], [-0.5, 0, 0]],
        [[1, 0, 0], [0, 0, 0]],
    ]
    clipped = clip_lines(lines, ((0, False),))
    assert np.allclose(clipped, [[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 0]]])
    assert np.allclose(
        clip_lines(lines, ((0, True),)),
        [[[-1, 0, 0], [0, 0, 0]], [[-1, 0, 0], [-0.5, 0, 0]]],
    )
    assert clip_lines(lines, ((0, False), (1, True))).shape == (2, 2, 3)
    assert clip_lines(lines[1:2], ((0, False),)).shape == (0, 2, 3)

    points = np.array([[-1, 0, 0], [1, 0, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    assert clip_triangles(points, triangles, ()) == (points, triangles)
    assert np.isclose(area(*clip_triangles(points, triangles, ((0, False),))), 1.0)

    new_points, new_triangles = clip_triangles(points, triangles, ((1, True),))
    assert new_points.shape == (0, 3) and new_triangles.shape == (0, 3)

    points = np.array([[0, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=float)
    assert symmetry_clip((True, False, True), points) == ((0, True), (2, False))
    assert symmetry_clip((False, False, False), points) == ()


def test_custom_shape_by_pyvista():
    pytest.importorskip("pyvista")

    @dataclass
    class Triangle(Shape):
        "A custom shape with one triangle."

        def _triangles(self):
            points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
            return points, np.array([[0, 1, 2]])

    mesh = fem.Rectangle(n=3)
    plotter = mesh.plot(off_screen=True)
    backend = fem.view._backends.backend_of(plotter)

    # without clipping, partly clipped and fully clipped (not shown)
    for clip in [(), ((1, False),), ((1, True),)]:
        assert backend.add_shape(plotter, Triangle(clip=clip), color="red") is plotter
