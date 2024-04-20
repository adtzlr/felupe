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

from copy import deepcopy as copy
from string import ascii_lowercase as alphabet

import numpy as np
from scipy.special import factorial

from ._base import Element


def lagrange_line(order):
    "Return the cell-connectivity for an arbitrary-order Lagrange line."

    vertices = np.array([0, order])
    edge = np.arange(order + 1)[1:-1]

    return np.concatenate([vertices, edge])


def lagrange_quad(order):
    "Return the cell-connectivity for an arbitrary-order Lagrange quad."
    # points on a unit rectangle
    x = np.linspace(0, 1, order + 1)
    points = np.vstack([p.ravel() for p in np.meshgrid(x, x, indexing="ij")][::-1]).T

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

    e1 = search_edge(points, 1, ymin)
    e2 = search_edge(points, 0, xmax)
    e3 = search_edge(points, 1, ymax)
    e4 = search_edge(points, 0, xmin)

    edges = np.hstack((e1, e2, e3, e4))

    mask2 = np.ones_like(points[:, 0], dtype=bool)
    mask2[np.hstack((vertices, edges))] = 0

    face = np.arange(len(points))[mask2]
    return np.hstack((vertices, edges, face))


def lagrange_hexahedron(order):
    "Return the cell-connectivity for an arbitrary-order Lagrange hexahedron."

    x = np.linspace(0, 1, order + 1)
    points = np.vstack([p.ravel() for p in np.meshgrid(x, x, x, indexing="ij")][::-1]).T

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
    e11 = search_edge(points, 0, 1, xmax, ymax)
    e12 = search_edge(points, 0, 1, xmin, ymax)

    edges = np.hstack((e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12))

    mask2 = np.ones_like(points[:, 0], dtype=bool)
    mask2[np.hstack((vertices, edges))] = 0

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

    return np.hstack((vertices, edges, faces, volume))


class ArbitraryOrderLagrange(Element):
    r"""A n-dimensional Lagrange finite element (e.g. line, quad or hexahdron) of
    arbitrary order.

    Notes
    -----

    **Polynomial shape functions**

    The basis function vector is generated with row-stacking of the individual lagrange
    polynomials. Each polynomial defined in the interval :math:`[-1,1]` is a function of
    the parameter :math:`r`. The curve parameters matrix :math:`\boldsymbol{A}` is of
    symmetric shape due to the fact that for each evaluation point :math:`r_j` exactly
    one basis function :math:`h_j(r)` is needed.

    ..  math::

        \boldsymbol{h}(r) = \boldsymbol{A}^T \boldsymbol{r}(r)

    **Curve parameter matrix**

    The evaluation of the curve parameter matrix :math:`\boldsymbol{A}` is carried out
    by boundary conditions. Each shape function :math:`h_i` has to take the value of one
    at the associated nodal coordinate :math:`r_i` and zero at all other point
    coordinates.

    ..  math::

        \boldsymbol{A}^T \boldsymbol{R} &=
            \boldsymbol{I} \qquad \text{with} \qquad \boldsymbol{R} =
            \begin{bmatrix}\boldsymbol{r}(r_1) & \boldsymbol{r}(r_2) & \dots &
            \boldsymbol{r}(r_p)\end{bmatrix}

        \boldsymbol{A}^T &= \boldsymbol{R}^{-1}


    **Interpolation and partial derivatives**

    The approximation of nodal unknowns :math:`\hat{\boldsymbol{u}}` as a function of
    the parameter :math:`r` is evaluated as

    ..  math::

        \boldsymbol{u}(r) \approx \hat{\boldsymbol{u}}^T \boldsymbol{h}(r)

    For the calculation of the partial derivative of the interpolation field w.r.t. the
    parameter :math:`r` a simple shift of the entries of the parameter vector is enough.
    This shifted parameter vector is denoted as :math:`\boldsymbol{r}^-`. A minus
    superscript indices the negative shift of the vector entries by :math:`-1`.

    ..  math::

        \frac{\partial \boldsymbol{u}(r)}{\partial r} &\approx
            \hat{\boldsymbol{u}}^T \frac{\partial \boldsymbol{h}(r)}{\partial r}

        \frac{\partial \boldsymbol{h}(r)}{\partial r} &=
            \boldsymbol{A}^T \boldsymbol{r}^-(r) \qquad \text{with} \qquad r_0^- =
            0 \qquad \text{and} \qquad r_i^- = \frac{r^{(i-1)}}{(i-1)!} \qquad
            \text{for} \qquad  i=(1 \dots p)


    n-dimensional shape functions
    *****************************

    Multi-dimensional shape function matrices
    :math:`\boldsymbol{H}_{2D}, \boldsymbol{H}_{3D}` are simply evaluated as dyadic
    (outer) vector products of one-dimensional shape function vectors. The multi-
    dimensional shape function vector is a one-dimensional representation (flattened
    version) of the multi-dimensional shape function matrix.

    ..  math::

        \boldsymbol{H}_{2D}(r,s) &= \boldsymbol{h}(r) \otimes \boldsymbol{h}(s)

        \boldsymbol{H}_{3D}(r,s,t) &= \boldsymbol{h}(r) \otimes \boldsymbol{h}(s)
            \otimes \boldsymbol{h}(t)

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.ArbitraryOrderLagrangeElement(order=4, dim=2)
       >>> element.plot().show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.ArbitraryOrderLagrangeElement(order=3, dim=3)
       >>> element.plot().show()
    """

    def __init__(self, order, dim, interval=(-1, 1), permute=True):
        self._order = order
        self._nshape = order + 1
        self._npoints = self._nshape**dim
        self._nbasis = self._npoints
        self._interval = interval

        self.permute = None
        if permute:
            self.permute = [None, lagrange_line, lagrange_quad, lagrange_hexahedron][
                dim
            ](order)

        super().__init__(shape=(self._npoints, dim))

        # init curve-parameter matrix
        n = self._nshape
        self._AT = np.linalg.inv(
            np.array([self._polynomial(p, n) for p in self._points(n)])
        ).T

        # indices for outer product in einstein notation
        # idx = ["a", "b", "c", ...][:dim]
        # subscripts = "a,b,c -> abc"
        self._idx = [letter for letter in alphabet][: self.dim]
        self._subscripts = ",".join(self._idx) + "->" + "".join(self._idx)

        # init points
        grid = np.meshgrid(*np.tile(self._points(n), (dim, 1)), indexing="ij")[::-1]
        self.points = np.vstack([point.ravel() for point in grid]).T

        if self.permute is not None:
            self.points = self.points[self.permute]

        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "VTK_LAGRANGE"

        if dim == 1:
            self.cell_type += "_LINE"
        elif dim == 2:
            self.cell_type += "_QUADRILATERAL"
        elif dim == 3:
            self.cell_type += "_HEXAHEDRON"

    def function(self, r):
        "Return the shape functions at given coordinate vector r."
        n = self._nshape

        # 1d - basis function vectors per axis
        fun = [self._AT @ self._polynomial(ra, n) for ra in r]
        h = np.einsum(self._subscripts, *fun).ravel("F")

        if self.permute is not None:
            h = h[self.permute]

        return h

    def gradient(self, r):
        "Return the gradient of shape functions at given coordinate vector r."
        n = self._nshape

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        # shifted 1d - basis function vectors per axis
        k = [self._AT @ np.append(0, self._polynomial(ra, n)[:-1]) for ra in r]

        # init output
        dhdr = np.zeros((n**self.dim, self.dim))

        # loop over columns
        for i in range(self.dim):
            g = copy(h)
            g[i] = k[i]
            dhdr[:, i] = np.einsum(self._subscripts, *g).ravel("F")

        if self.permute is not None:
            dhdr = dhdr[self.permute]

        return dhdr

    def _points(self, n):
        "Equidistant n-points in interval [-1, 1]."
        return np.linspace(*self._interval, n)

    def _polynomial(self, r, n):
        "Lagrange-Polynomial of order n evaluated at coordinate vector r."
        m = np.arange(n)
        return r**m / factorial(m)
