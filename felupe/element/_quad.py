# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np

from ._base import Element


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
        0 (-1,-1)                  1 ( 1/-1)

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
        0 (-1,-1)                  1 ( 1/-1)


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
