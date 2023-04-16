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

from ._base import Element
from ._lagrange import ArbitraryOrderLagrange


class ConstantQuad(Element):
    r"""Quadrilateral element with constant shape functions.

    ..  code-block::

                      ^ s
         3 (-1/ 1)    |            2 ( 1/ 1)
          o-----------|-----------o
          |           |           |
          |           |           |
          |           |           |
          |           |           |
          |      -----|-----------|-----> r
          |           |           |
          |           |           |
          |                       |
          |                       |
          o-----------------------o
        0 (-1/-1)                  1 ( 1/-1)

    Attributes
    ----------
    points : ndarray
        Array with point locations in natural coordinate system
    """

    def __init__(self):
        super().__init__(shape=(1, 2))
        self.points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

    def function(self, rst):
        r"""Constant quadrilateral - shape functions.

        ..  math::

            \boldsymbol{h}(\boldsymbol{r}) = \begin{bmatrix}
                1
            \end{bmatrix}

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for shape function evaluation

        Returns
        -------
        ndarray
            Shape functions evaluated at given location
        """
        return np.array([1])

    def gradient(self, rst):
        r"""Constant quadrilateral - gradient of shape functions.

        ..  math::

            \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{r}} =
            \begin{bmatrix}
                0
            \end{bmatrix}

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for gradient of shape function evaluation

        Returns
        -------
        ndarray
            Gradient of shape functions evaluated at given location
        """
        return np.array([[0, 0]])


class Quad(Element):
    r"""Quadrilateral element with linear shape functions.

    ..  code-block::

                      ^ s
         3 (-1/ 1)    |            2 ( 1/ 1)
          o-----------|-----------o
          |           |           |
          |           |           |
          |           |           |
          |           |           |
          |      -----|-----------|-----> r
          |           |           |
          |           |           |
          |                       |
          |                       |
          o-----------------------o
        0 (-1/-1)                  1 ( 1/-1)


    Attributes
    ----------
    points : ndarray
        Array with point locations in natural coordinate system
    """

    def __init__(self):
        super().__init__(shape=(4, 2))
        self.points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)

    def function(self, rs):
        r"""Linear quadrilateral - shape functions.

        ..  math::

            \boldsymbol{h}(\boldsymbol{r}) = \frac{1}{4} \begin{bmatrix}
                (1-r)(1-s) \\ (1+r)(1-s) \\ (1+r)(1+s) \\ (1-r)(1+s)
            \end{bmatrix}

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for shape function evaluation

        Returns
        -------
        ndarray
            Shape functions evaluated at given location
        """
        r, s = rs
        return (
            np.array(
                [
                    (1 - r) * (1 - s),
                    (1 + r) * (1 - s),
                    (1 + r) * (1 + s),
                    (1 - r) * (1 + s),
                ]
            )
            * 0.25
        )

    def gradient(self, rs):
        r"""Linear quadrilateral - gradient of shape functions.

        ..  math::

            \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{r}} =
            \frac{1}{4} \begin{bmatrix}
                -(1-s) & -(1-r) \\
                 (1-s) & -(1+r) \\
                 (1+s) &  (1+r) \\
                -(1+s) &  (1-r)
            \end{bmatrix}

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for gradient of shape function evaluation

        Returns
        -------
        ndarray
            Gradient of shape functions evaluated at given location
        """

        r, s = rs
        return (
            np.array(
                [
                    [-(1 - s), -(1 - r)],
                    [(1 - s), -(1 + r)],
                    [(1 + s), (1 + r)],
                    [-(1 + s), (1 - r)],
                ]
            )
            * 0.25
        )


class QuadraticQuad(Element):
    r"""Quadratic serendipity quadrilateral element.

    ..  code-block::

                      ^ s
         3 (-1/ 1)    |6 ( 0/ 1)   2 ( 1/ 1)
          o-----------o-----------o
          |           |           |
          |           |           |
          |           |           |
          |7 (-1/ 0)  |           |5 ( 1/ 0)
          o      -----|-----------o-----> r
          |           |           |
          |           |           |
          |                       |
          |                       |
          o-----------o-----------o
        0 (-1/-1)     4 ( 0/-1)    1 ( 1/-1)


    Attributes
    ----------
    points : ndarray
        Array with point locations in natural coordinate system
    """

    def __init__(self):
        super().__init__(shape=(8, 2))
        self.points = np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
            ],
            dtype=float,
        )

    def function(self, rs):
        r"""Quadratic serendipity quadrilateral - shape functions."""
        r, s = rs
        ra, sa = self.points.T

        h = (1 + ra * r) * (1 + sa * s) * (ra * r + sa * s - 1) / 4
        h[ra == 0] = (1 - r**2) * (1 + sa[ra == 0] * s) / 2
        h[sa == 0] = (1 + ra[sa == 0] * r) * (1 - s**2) / 2

        return h

    def gradient(self, rs):
        r"""Quadratic serendipity quadrilateral - gradient of shape functions.

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for gradient of shape function evaluation

        Returns
        -------
        ndarray
            Gradient of shape functions evaluated at given location
        """

        r, s = rs
        ra, sa = self.points.T

        dhdr = (
            ra * (1 + sa * s) * (ra * r + sa * s - 1) / 4
            + (1 + ra * r) * (1 + sa * s) * ra / 4
        )

        dhdr[ra == 0] = -2 * r * (1 + sa[ra == 0] * s) / 2
        dhdr[sa == 0] = ra[sa == 0] * (1 - s**2) / 2

        dhds = (1 + ra * r) * sa * (ra * r + sa * s - 1) / 4 + (1 + ra * r) * (
            1 + sa * s
        ) * sa / 4

        dhds[ra == 0] = (1 - r**2) * sa[ra == 0] / 2
        dhds[sa == 0] = (1 + ra[sa == 0] * r) * -2 * s / 2

        return np.vstack([dhdr, dhds]).T


class BiQuadraticQuad(Element):
    r"""Bi-Quadratic Lagrange quadrilateral element.

    ..  code-block::

                      ^ s
         3 (-1/ 1)    |6 ( 0/ 1)   2 ( 1/ 1)
          o-----------o-----------o
          |           |           |
          |           |           |
          |           |           |
          |7 (-1/ 0)  |8 ( 0/ 0)  |5 ( 1/ 0)
          o      -----o-----------o-----> r
          |           |           |
          |           |           |
          |                       |
          |                       |
          o-----------o-----------o
        0 (-1/-1)     4 ( 0/-1)    1 ( 1/-1)


    Attributes
    ----------
    points : ndarray
        Array with point locations in natural coordinate system
    """

    def __init__(self):
        super().__init__(shape=(9, 2))

        self._lagrange = ArbitraryOrderLagrange(order=2, dim=2)

        self._vertices = np.array([0, 2, 8, 6])
        self._edges = np.array([1, 5, 7, 3])
        self._faces = np.array([4])
        self._volume = np.array([], dtype=int)

        self._permute = np.concatenate(
            (self._vertices, self._edges, self._faces, self._volume)
        )

        self.points = self._lagrange.points[self._permute]

    def function(self, rs):
        r"""Bi-Quadratic Lagrange quadrilateral - shape functions."""

        return self._lagrange.function(rs)[self._permute]

    def gradient(self, rs):
        r"""Bi-Quadratic Lagrange quadrilateral - gradient of shape functions.

        Arguments
        ---------
        rs : ndarray
            Point as coordinate vector for gradient of shape function evaluation

        Returns
        -------
        ndarray
            Gradient of shape functions evaluated at given location
        """

        return self._lagrange.gradient(rs)[self._permute, :]
