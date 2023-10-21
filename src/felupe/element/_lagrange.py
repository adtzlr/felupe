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
    """

    def __init__(self, order, dim, interval=(-1, 1)):
        self._order = order
        self._nshape = order + 1
        self._npoints = self._nshape**dim
        self._nbasis = self._npoints
        self._interval = interval

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

    def function(self, r):
        "Return the shape functions at given coordinate vector r."
        n = self._nshape

        # 1d - basis function vectors per axis
        h = [self._AT @ self._polynomial(ra, n) for ra in r]

        return np.einsum(self._subscripts, *h).ravel("F")

    def gradient(self, r):
        "Return the grgadient of shape functions at given coordinate vector r."
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

        return dhdr

    def _points(self, n):
        "Equidistant n-points in interval [-1, 1]."
        return np.linspace(*self._interval, n)

    def _polynomial(self, r, n):
        "Lagrange-Polynomial of order n evaluated at coordinate vector r."
        m = np.arange(n)
        return r**m / factorial(m)
