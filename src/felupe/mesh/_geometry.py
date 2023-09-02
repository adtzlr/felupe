# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from ._line_rectangle_cube import cube_hexa, line_line, rectangle_quad
from ._mesh import Mesh
from ._tools import concatenate


class Line(Mesh):
    "A line shaped 1d-mesh with lines between ``a`` and ``b`` with ``n`` points."

    def __init__(self, a=0, b=1, n=2):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = line_line(a, b, n)

        super().__init__(points, cells, cell_type)


class Rectangle(Mesh):
    """A rectangular 2d-mesh with quads between ``a`` and ``b`` with ``n``
    points per axis."""

    def __init__(self, a=(0, 0), b=(1, 1), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = rectangle_quad(a, b, n)

        super().__init__(points, cells, cell_type)


class Cube(Mesh):
    """A cube shaped 3d-mesh with hexahedrons between ``a`` and ``b`` with ``n``
    points per axis."""

    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = cube_hexa(a, b, n)

        super().__init__(points, cells, cell_type)


class Grid(Mesh):
    """A grid shaped 3d-mesh with hexahedrons. Basically a wrapper for
    :func:`numpy.meshgrid` with  default ``indexing="ij"``."""

    def __init__(self, *xi, indexing="ij", **kwargs):
        shape = np.array([len(x) for x in xi])
        n = shape if len(shape) > 1 else shape[0]

        M = [None, line_line, rectangle_quad, cube_hexa][len(xi)]
        points, cells, cell_type = M(n=n)

        super().__init__(points, cells, cell_type)

        self.points = (
            np.vstack(
                [g.T.ravel() for g in np.meshgrid(*xi, indexing=indexing, **kwargs)]
            )
            .astype(float)
            .T
        )


class RectangleArbitraryOrderQuad(Mesh):
    """A rectangular 2d-mesh with arbitrary order quads between ``a`` and ``b`` with
    ``n`` points per axis."""

    def __init__(self, a=(0, 0), b=(1, 1), order=2):
        yv, xv = np.meshgrid(
            np.linspace(a[1], b[1], order + 1),
            np.linspace(a[0], b[0], order + 1),
            indexing="ij",
        )

        points = np.vstack((xv.flatten(), yv.flatten())).T

        # search vertices
        xmin = min(points[:, 0])
        ymin = min(points[:, 1])

        xmax = max(points[:, 0])
        ymax = max(points[:, 1])

        def search_vertice(p, x, y):
            return np.where(np.logical_and(p[:, 0] == x, p[:, 1] == y))[0][0]

        def search_edge(p, a, x):
            return np.where(p[:, a] == x)[0][1:-1]

        v1 = search_vertice(points, xmin, ymin)
        v2 = search_vertice(points, xmax, ymin)
        v3 = search_vertice(points, xmax, ymax)
        v4 = search_vertice(points, xmin, ymax)

        vertices = [v1, v2, v3, v4]

        mask1 = np.ones_like(points[:, 0], dtype=bool)
        mask1[vertices] = 0
        # points_no_verts = points[mask1]

        e1 = search_edge(points, 1, ymin)
        e2 = search_edge(points, 0, xmax)
        e3 = search_edge(points, 1, ymax)
        e4 = search_edge(points, 0, xmin)

        edges = np.hstack((e1, e2, e3, e4))

        mask2 = np.ones_like(points[:, 0], dtype=bool)
        mask2[np.hstack((vertices, edges))] = 0
        # points_no_verts_edges = points[mask2]

        face = np.arange(len(points))[mask2]

        cells = np.hstack((vertices, edges, face)).reshape(1, -1)

        super().__init__(points, cells, cell_type="VTK_LAGRANGE_QUADRILATERAL")


class CubeArbitraryOrderHexahedron(Mesh):
    """A cube shaped 3d-mesh with arbitrary order hexahedrons between ``a`` and ``b``
    with ``n`` points per axis."""

    def __init__(self, a=(0, 0, 0), b=(1, 1, 1), order=2):
        zv, yv, xv = np.meshgrid(
            np.linspace(a[2], b[2], order + 1),
            np.linspace(a[1], b[1], order + 1),
            np.linspace(a[0], b[0], order + 1),
            indexing="ij",
        )

        points = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T

        # search vertices
        xmin = min(points[:, 0])
        ymin = min(points[:, 1])
        zmin = min(points[:, 2])

        xmax = max(points[:, 0])
        ymax = max(points[:, 1])
        zmax = max(points[:, 2])

        def search_vertice(p, x, y, z):
            return np.where(
                np.logical_and(np.logical_and(p[:, 0] == x, p[:, 1] == y), p[:, 2] == z)
            )[0][0]

        def search_edge(p, a, b, x, y):
            return np.where(np.logical_and(p[:, a] == x, p[:, b] == y))[0][1:-1]

        def search_face(p, a, x, vertices, edges):
            face = np.where(points[:, a] == x)[0]
            mask = np.zeros_like(p[:, 0], dtype=bool)
            mask[face] = 1
            mask[np.hstack((vertices, edges))] = 0
            return np.arange(len(p[:, 0]))[mask]

        v1 = search_vertice(points, xmin, ymin, zmin)
        v2 = search_vertice(points, xmax, ymin, zmin)
        v3 = search_vertice(points, xmax, ymax, zmin)
        v4 = search_vertice(points, xmin, ymax, zmin)

        v5 = search_vertice(points, xmin, ymin, zmax)
        v6 = search_vertice(points, xmax, ymin, zmax)
        v7 = search_vertice(points, xmax, ymax, zmax)
        v8 = search_vertice(points, xmin, ymax, zmax)

        vertices = [v1, v2, v3, v4, v5, v6, v7, v8]

        mask1 = np.ones_like(points[:, 0], dtype=bool)
        mask1[vertices] = 0
        # points_no_verts = points[mask1]

        e1 = search_edge(points, 1, 2, ymin, zmin)
        e2 = search_edge(points, 0, 2, xmax, zmin)
        e3 = search_edge(points, 1, 2, ymax, zmin)
        e4 = search_edge(points, 0, 2, xmin, zmin)

        e5 = search_edge(points, 1, 2, ymin, zmax)
        e6 = search_edge(points, 0, 2, xmax, zmax)
        e7 = search_edge(points, 1, 2, ymax, zmax)
        e8 = search_edge(points, 0, 2, xmin, zmax)

        e9 = search_edge(points, 0, 1, xmin, ymin)
        e10 = search_edge(points, 0, 1, xmax, ymin)
        e11 = search_edge(points, 0, 1, xmin, ymax)
        e12 = search_edge(points, 0, 1, xmax, ymax)

        edges = np.hstack((e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12))

        mask2 = np.ones_like(points[:, 0], dtype=bool)
        mask2[np.hstack((vertices, edges))] = 0
        # points_no_verts_edges = points[mask2]

        f1 = search_face(points, 0, xmin, vertices, edges)
        f2 = search_face(points, 0, xmax, vertices, edges)
        f3 = search_face(points, 1, ymin, vertices, edges)
        f4 = search_face(points, 1, ymax, vertices, edges)
        f5 = search_face(points, 2, zmin, vertices, edges)
        f6 = search_face(points, 2, zmax, vertices, edges)

        faces = np.hstack((f1, f2, f3, f4, f5, f6))

        mask3 = np.ones_like(points[:, 0], dtype=bool)
        mask3[np.hstack((vertices, edges, faces))] = 0
        volume = np.arange(len(points))[mask3]

        cells = np.hstack((vertices, edges, faces, volume)).reshape(1, -1)

        super().__init__(points, cells, cell_type="VTK_LAGRANGE_HEXAHEDRON")


class Circle(Mesh):
    """A circular shaped 2d-mesh with quads and ``n`` points on the circumferential
    edge of a 45-degree section. 90-degree ``sections`` are placed at given angles in
    degree."""

    def __init__(
        self,
        radius=1,
        centerpoint=[0, 0],
        n=2,
        sections=[0, 90, 180, 270],
        value=0.15,
        exponent=2,
        decimals=10,
    ):
        rect = Rectangle(b=(0.5, 0.5), n=(n, n))
        right = rect.points[:, 0] == 0.5

        rect = rect.add_runouts(values=[-value], axis=0, exponent=exponent)
        rect.points[:, 0] = rect.points[:, 1].reshape(n, n, order="F").ravel()

        line = Line(n=n)
        phi = np.linspace(np.pi / 4, 0, n)

        bottom = line.copy(points=rect.points[right][::-1])
        top = line.copy(points=np.vstack([np.cos(phi), np.sin(phi)]).T)
        face = bottom.fill_between(top, n=n)

        quarter = concatenate([face, face.mirror(normal=[-1, 1]), rect])
        circle = concatenate([quarter.rotate(alpha, 2) for alpha in sections]).sweep(
            decimals=decimals
        )

        circle.points *= radius
        circle.points += np.array(centerpoint)

        super().__init__(
            points=circle.points, cells=circle.cells, cell_type=circle.cell_type
        )


class Triangle(Mesh):
    """A triangular shaped 2d-mesh with quads and ``n`` points at the edges of the three
    sub-quadrilaterals."""

    def __init__(
        self,
        a=(0, 0),
        b=(1, 0),
        c=(0, 1),
        n=2,
        decimals=10,
    ):
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)

        sections = []

        centerpoint = (a + b + c) / 3
        centerpoints = {"ab": (a + b) / 2, "bc": (b + c) / 2, "ac": (a + c) / 2}

        line = Line(n=n)

        # section (connected to point) a
        x1 = np.linspace(a[0], centerpoints["ac"][0], n)
        y1 = np.linspace(a[1], centerpoints["ac"][1], n)

        left = line.copy(points=np.vstack([x1, y1]).T)

        x2 = np.linspace(centerpoints["ab"][0], centerpoint[0], n)
        y2 = np.linspace(centerpoints["ab"][1], centerpoint[1], n)

        middle = line.copy(points=np.vstack([x2, y2]).T)

        sections.append(middle.fill_between(left, n=n))

        # section (connected to point) b
        x3 = np.linspace(b[0], centerpoints["bc"][0], n)
        y3 = np.linspace(b[1], centerpoints["bc"][1], n)

        right = line.copy(points=np.vstack([x3, y3]).T)

        sections.append(right.fill_between(middle, n=n))

        # section (connected to point) c
        x4 = np.linspace(centerpoints["ac"][0], c[0], n)
        y4 = np.linspace(centerpoints["ac"][1], c[1], n)

        top = line.copy(points=np.vstack([x4, y4]).T)

        x5 = np.linspace(centerpoint[0], centerpoints["bc"][0], n)
        y5 = np.linspace(centerpoint[1], centerpoints["bc"][1], n)

        bottom = line.copy(points=np.vstack([x5, y5]).T)

        sections.append(bottom.fill_between(top, n=n))

        # combine sections
        triangle = concatenate(sections).sweep(decimals=decimals)

        super().__init__(
            points=triangle.points, cells=triangle.cells, cell_type=triangle.cell_type
        )
