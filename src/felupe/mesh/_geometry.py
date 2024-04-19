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

from ..element._lagrange import lagrange_hexahedron, lagrange_quad
from ._line_rectangle_cube import cube_hexa, line_line, rectangle_quad
from ._mesh import Mesh
from ._tools import concatenate


class Point(Mesh):
    """A mesh with a single vertex point located at ``a``.

    Parameters
    ----------
    a : float, optional
        Vertex point coordinate of the mesh (default is 0.0).

    Examples
    --------
    >>> import felupe as fem

    >>> mesh = fem.Point(a=-2.1)
    >>> mesh
    <felupe Mesh object>
      Number of points: 1
      Number of cells:
        vertex: 1

    >>> mesh.points
    array([[-2.1]])

    >>> mesh.cells
    array([[0]])
    """

    def __init__(self, a=0.0):
        self.a = a

        points = np.array([a]).reshape(1, -1)
        cells = np.array([[0]])
        cell_type = "vertex"

        super().__init__(points, cells, cell_type)


class Line(Mesh):
    """A 1d-mesh with lines between ``a`` and ``b`` with ``n`` points.

    Parameters
    ----------
    a : float, optional
        Left end point of the mesh (default is 0.0).
    b : float, optional
        Right end point of the mesh (default is 1.0).
    n : int, optional
        Number of points (default is 2).

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Line(a=-2.1, b=3.5, n=3)
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 3
      Number of cells:
        line: 2

    >>> mesh.points
    array([[-2.1],
           [ 0.7],
           [ 3.5]])

    >>> mesh.cells
    array([[0, 1],
           [1, 2]])
    """

    def __init__(self, a=0.0, b=1.0, n=2):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = line_line(a, b, n)

        super().__init__(points, cells, cell_type)


class Rectangle(Mesh):
    """A rectangular 2d-mesh with quads between ``a`` and ``b`` with ``n``
    points per axis.

    Parameters
    ----------
    a : 2-tuple of float, optional
        Lower-left end point of the mesh (default is (0.0, 0.0)).
    b : 2-tuple of float, optional
        Upper-right end point of the mesh (default is (1.0, 1.0)).
    n : int or 2-tuple of int, optional
        Number of points per axis (default is (2, 2)).

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Rectangle(a=(-1.2, 0.5), b=(4.5, 7.3), n=3)
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 9
      Number of cells:
        quad: 4

    >>> mesh.points
    array([[-1.2 ,  0.5 ],
           [ 1.65,  0.5 ],
           [ 4.5 ,  0.5 ],
           [-1.2 ,  3.9 ],
           [ 1.65,  3.9 ],
           [ 4.5 ,  3.9 ],
           [-1.2 ,  7.3 ],
           [ 1.65,  7.3 ],
           [ 4.5 ,  7.3 ]])

    >>> mesh.cells
    array([[0, 1, 4, 3],
           [1, 2, 5, 4],
           [3, 4, 7, 6],
           [4, 5, 8, 7]])
    """

    def __init__(self, a=(0.0, 0.0), b=(1.0, 1.0), n=(2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = rectangle_quad(a, b, n)

        super().__init__(points, cells, cell_type)


class Cube(Mesh):
    """A cube shaped 3d-mesh with hexahedrons between ``a`` and ``b`` with ``n``
    points per axis.

    Parameters
    ----------
    a : 3-tuple of float, optional
        Lower-left end point of the mesh (default is (0.0, 0.0, 0.0)).
    b : 3-tuple of float, optional
        Upper-right end point of the mesh (default is (1.0, 1.0, 1.0)).
    n : int or 3-tuple of int, optional
        Number of points per axis (default is (2, 2, 2)).

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Cube(a=(-1.2, 0.5, 6.2), b=(4.5, 7.3, 9.3), n=(3, 2, 2))
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 12
      Number of cells:
        hexahedron: 2

    >>> mesh.points
    array([[-1.2 ,  0.5 ,  6.2 ],
           [ 1.65,  0.5 ,  6.2 ],
           [ 4.5 ,  0.5 ,  6.2 ],
           [-1.2 ,  7.3 ,  6.2 ],
           [ 1.65,  7.3 ,  6.2 ],
           [ 4.5 ,  7.3 ,  6.2 ],
           [-1.2 ,  0.5 ,  9.3 ],
           [ 1.65,  0.5 ,  9.3 ],
           [ 4.5 ,  0.5 ,  9.3 ],
           [-1.2 ,  7.3 ,  9.3 ],
           [ 1.65,  7.3 ,  9.3 ],
           [ 4.5 ,  7.3 ,  9.3 ]])

    >>> mesh.cells
    array([[ 0,  1,  4,  3,  6,  7, 10,  9],
           [ 1,  2,  5,  4,  7,  8, 11, 10]])
    """

    def __init__(self, a=(0.0, 0.0, 0.0), b=(1.0, 1.0, 1.0), n=(2, 2, 2)):
        self.a = a
        self.b = b
        self.n = n

        points, cells, cell_type = cube_hexa(a, b, n)

        super().__init__(points, cells, cell_type)


class Grid(Mesh):
    """A grid shaped 3d-mesh with hexahedrons. Basically a wrapper for
    :func:`numpy.meshgrid` with  default ``indexing="ij"``.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import numpy as np
       >>> import felupe as fem
       >>>
       >>> x1 = np.linspace(0, 2, 3)**2
       >>> x2 = np.sqrt(np.linspace(0, 1, 3))
       >>>
       >>> mesh = fem.mesh.Grid(x1, x2)
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 9
      Number of cells:
        quad: 4

    >>> mesh.points
    array([[0.        , 0.        ],
           [1.        , 0.        ],
           [4.        , 0.        ],
           [0.        , 0.70710678],
           [1.        , 0.70710678],
           [4.        , 0.70710678],
           [0.        , 1.        ],
           [1.        , 1.        ],
           [4.        , 1.        ]])

    >>> mesh.cells
    array([[0, 1, 4, 3],
           [1, 2, 5, 4],
           [3, 4, 7, 6],
           [4, 5, 8, 7]])

    See Also
    --------
    numpy.meshgrid : Return a list of coordinate matrices from coordinate vectors.
    """

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


class RectangleArbitraryOrderQuad(Rectangle):
    """A rectangular 2d-mesh with an arbitrarr-order Lagrange quad between ``a`` and
    ``b``.
    """

    def __init__(self, a=(0.0, 0.0), b=(1.0, 1.0), order=2):
        super().__init__(a=a, b=b, n=order + 1)
        self.update(
            cells=lagrange_quad(order=order).reshape(1, -1),
            cell_type="VTK_LAGRANGE_QUADRILATERAL",
        )


class CubeArbitraryOrderHexahedron(Cube):
    """A rectangular 2d-mesh with an arbitrarr-order Lagrange hexahedron between ``a``
    and ``b``.
    """

    def __init__(self, a=(0.0, 0.0, 0.0), b=(1.0, 1.0, 1.0), order=2):
        super().__init__(a=a, b=b, n=order + 1)
        self.update(
            cells=lagrange_hexahedron(order=order).reshape(1, -1),
            cell_type="VTK_LAGRANGE_HEXAHEDRON",
        )


class Circle(Mesh):
    """A circular shaped 2d-mesh with quads and ``n`` points on the circumferential
    edge of a 45-degree section. 90-degree ``sections`` are placed at given angles in
    degree.

    Parameters
    ----------
    radius : float, optional
        Lower-left end point of the mesh (default is (0.0, 0.0)).
    centerpoint : 2-list of float, optional
        Coordinates of the origin where the circle is centered (default is [1.0, 1.0]).
    n : int, optional
        Number of points per axis for a quarter of the embedded rectangle
        (default is 2).
    sections : list of int or float, optional
        Rotation angles in deg where quarter circles (sections) are placed (default is
        [0, 90, 180, 270]).
    value : float
        First shape parameter of the embedded rectangle (default is 0.15).
    exponent : int
        Second shape parameter of the embedded rectangle (default is 2).
    decimals : int
        Decimals used for rounding point coordinates to avoid non-connected sections
        (default is 10).

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Circle()
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 17
      Number of cells:
        quad: 12

    >>> mesh.points
    array([[-1.        ,  0.        ],
           [-0.70710678, -0.70710678],
           [-0.70710678,  0.70710678],
           [-0.5       ,  0.        ],
           [-0.425     , -0.425     ],
           [-0.425     ,  0.425     ],
           [ 0.        , -1.        ],
           [ 0.        , -0.5       ],
           [ 0.        ,  0.        ],
           [ 0.        ,  0.5       ],
           [ 0.        ,  1.        ],
           [ 0.425     , -0.425     ],
           [ 0.425     ,  0.425     ],
           [ 0.5       ,  0.        ],
           [ 0.70710678, -0.70710678],
           [ 0.70710678,  0.70710678],
           [ 1.        ,  0.        ]])

    >>> mesh.cells
    array([[12, 13, 16, 15],
           [15, 10,  9, 12],
           [ 8, 13, 12,  9],
           [ 5,  9, 10,  2],
           [ 2,  0,  3,  5],
           [ 8,  9,  5,  3],
           [ 4,  3,  0,  1],
           [ 1,  6,  7,  4],
           [ 8,  3,  4,  7],
           [11,  7,  6, 14],
           [14, 16, 13, 11],
           [ 8,  7, 11, 13]])
    """

    def __init__(
        self,
        radius=1.0,
        centerpoint=[0.0, 0.0],
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
    sub-quadrilaterals.

    Parameters
    ----------
    a : 2-tuple of float, optional
        First end point of the mesh (default is (0.0, 0.0)).
    b : 2-tuple of float, optional
        Second end point of the mesh (default is (1.0, 0.0)).
    c : 2-tuple of float, optional
        Third end point of the mesh (default is (0.0, 1.0)).
    n : int, optional
        Number of points per axis (default is 2).
    decimals : int
        Decimals used for rounding point coordinates to avoid non-connected sections
        (default is 10).

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> mesh = fem.mesh.Triangle(a=(0.3, 0.2), b=(1.2, 0.1), c=(0.1, 0.9), n=3)
       >>> mesh.plot().show()

    >>> mesh
    <felupe Mesh object>
      Number of points: 19
      Number of cells:
        quad: 12

    >>> mesh.points
    array([[0.1       , 0.9       ],
           [0.15      , 0.725     ],
           [0.2       , 0.55      ],
           [0.25      , 0.375     ],
           [0.3       , 0.2       ],
           [0.36666667, 0.475     ],
           [0.37083333, 0.5875    ],
           [0.375     , 0.7       ],
           [0.44583333, 0.325     ],
           [0.525     , 0.175     ],
           [0.53333333, 0.4       ],
           [0.59166667, 0.45      ],
           [0.64166667, 0.275     ],
           [0.65      , 0.5       ],
           [0.75      , 0.15      ],
           [0.78333333, 0.2875    ],
           [0.925     , 0.3       ],
           [0.975     , 0.125     ],
           [1.2       , 0.1       ]])

    >>> mesh.cells
    array([[14, 12,  8,  9],
           [12, 10,  5,  8],
           [ 9,  8,  3,  4],
           [ 8,  5,  2,  3],
           [18, 16, 15, 17],
           [16, 13, 11, 15],
           [17, 15, 12, 14],
           [15, 11, 10, 12],
           [10, 11,  6,  5],
           [11, 13,  7,  6],
           [ 5,  6,  1,  2],
           [ 6,  7,  0,  1]])
    """

    def __init__(
        self,
        a=(0.0, 0.0),
        b=(1.0, 0.0),
        c=(0.0, 1.0),
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
