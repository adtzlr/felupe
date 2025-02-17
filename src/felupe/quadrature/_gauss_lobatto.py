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

from string import ascii_lowercase

import numpy as np

from ._scheme import Scheme


def gauss_lobatto(deg):
    r"""Gauss-Lobatto quadrature.

    Computes the sample points and weights for Gauss-Lobatto quadrature. These sample
    points and weights will correctly integrate polynomials of degree
    :math:`2 \cdot deg - 3` or less over the interval :math:`[-1, 1]` with the weight
    function :math:`f(x) = 1`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 2.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.
    """
    if deg == 2:
        x = np.array([-1.0, 1.0])
        y = np.array([1.0, 1.0])

    elif deg == 3:
        x = np.array([-1.0, 0.0, 1.0])
        y = np.array([1.0, 4.0, 1.0]) / 3

    elif deg == 4:
        a = np.sqrt(0.2)
        x = np.array([-1.0, -a, a, 1.0])
        y = np.array([1.0, 5.0, 5.0, 1.0]) / 6

    elif deg == 5:
        a = np.sqrt(3 / 7)
        x = np.array([-1.0, -a, 0.0, a, 1.0])
        y = np.array([0.1, 49 / 90, 32 / 45, 49 / 90, 0.1])

    elif deg == 6:
        a = np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21)
        b = np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21)
        c = (14 + np.sqrt(7)) / 30
        d = (14 - np.sqrt(7)) / 30
        x = np.array([-1.0, -b, -a, a, b, 1.0])
        y = np.array([1 / 15, d, c, c, d, 1 / 15])

    elif deg == 7:
        a = np.sqrt(5 / 11 - 2 / 11 * np.sqrt(5 / 3))
        b = np.sqrt(5 / 11 + 2 / 11 * np.sqrt(5 / 3))
        c = (124 + 7 * np.sqrt(15)) / 350
        d = (124 - 7 * np.sqrt(15)) / 350
        x = np.array([-1.0, -b, -a, 0, a, b, 1.0])
        y = np.array([1 / 21, d, c, 256 / 525, c, d, 1 / 21])

    else:
        raise ValueError("deg must be a positive integer (2 <= deg <= 7)")

    return x, y


class GaussLobatto(Scheme):
    r"""An arbitrary-`order` Gauss-Lobatto quadrature rule of dimension 1, 2 or 3 on
    the interval :math:`[-1, 1]`.

    Parameters
    ----------
    order : int
        The number of sample points :math:`n` minus two. The quadrature rule integrates
        degree :math:`2n-3` polynomials exactly.
    dim : int
        The dimension of the quadrature region.
    permute : bool, optional
        Permute the quadrature points according to the cell point orderings (default is
        True). This is supported for two and three dimensions as well as first and
        second order schemes. Otherwise this flag is silently ignored.

    Notes
    -----

    The approximation is given by

    ..  math::

        \int_{-1}^1 f(x) dx \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q`.

    """

    def __init__(self, order: int, dim: int):
        if dim not in [1, 2, 3]:
            raise ValueError("Wrong dimension.")

        x, w = gauss_lobatto(2 + order)

        points = (
            np.stack(np.meshgrid(*([x] * dim), indexing="ij"))[::-1].reshape(dim, -1).T
        )

        idx = list(ascii_lowercase)[:dim]
        weights = np.einsum(", ".join(idx), *([w] * dim)).ravel()

        super().__init__(points, weights)


class GaussLobattoBoundary(GaussLobatto):
    r"""An arbitrary-`order` Gauss-Lobatto quadrature rule of `dim` 1, 2 or 3 on the
    interval ``[-1, 1]``.

    Parameters
    ----------
    order : int
        The number of sample points :math:`n` minus two. The quadrature rule integrates
        degree :math:`2n-3` polynomials exactly.
    dim : int
        The dimension of the quadrature region.
    permute : bool, optional
        Permute the quadrature points according to the cell point orderings (default is
        True).

    Notes
    -----

    The approximation is given by

    ..  math::

        \int_{-1}^1 f(x) dx \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q`.

    """

    def __init__(self, order: int, dim: int):
        super().__init__(order=order, dim=dim - 1)

        # reset dimension
        self.dim = dim

        if self.dim == 2 or self.dim == 3:
            # quadrature points projected onto first edge of a quad
            #                          or onto first face of a hexahedron
            self.points = np.hstack((self.points, -np.ones((len(self.points), 1))))

        else:
            raise ValueError("Given dimension not implemented (must be 2 or 3).")
