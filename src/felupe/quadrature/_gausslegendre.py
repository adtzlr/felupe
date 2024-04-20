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

from ..element._lagrange import lagrange_hexahedron, lagrange_quad
from ._scheme import Scheme


class GaussLegendre(Scheme):
    r"""An arbitrary-`order` Gauss-Legendre quadrature rule of dimension 1, 2 or 3 on
    the interval :math:`[-1, 1]`.

    Parameters
    ----------
    order : int
        The number of sample points :math:`n` minus one. The quadrature rule integrates
        degree :math:`2n-1` polynomials exactly.
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

    Examples
    --------
    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.Line()
       >>> quadrature = fem.GaussLegendre(order=2, dim=1)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticQuad()
       >>> quadrature = fem.GaussLegendre(order=2, dim=2)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticHexahedron()
       >>> quadrature = fem.GaussLegendre(order=2, dim=3)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>>
       >>> element = fem.ArbitraryOrderLagrangeElement(order=5, dim=2)
       >>> quadrature = fem.GaussLegendre(order=5, dim=2)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()
    """

    def __init__(self, order: int, dim: int, permute: bool = True):
        if dim not in [1, 2, 3]:
            raise ValueError("Wrong dimension.")

        x, w = np.polynomial.legendre.leggauss(1 + order)

        points = (
            np.stack(np.meshgrid(*([x] * dim), indexing="ij"))[::-1].reshape(dim, -1).T
        )

        idx = list(ascii_lowercase)[:dim]
        weights = np.einsum(", ".join(idx), *([w] * dim)).ravel()
        sort = None

        if permute and order == 1 and dim == 2:
            sort = [0, 1, 3, 2]

        elif permute and order == 1 and dim == 3:
            sort = [0, 1, 3, 2, 4, 5, 7, 6]

        elif permute and order == 2 and dim == 2:
            sort = [0, 2, 8, 6, 1, 5, 7, 3, 4]

        elif permute and order == 2 and dim == 3:
            vertices = np.array([0, 2, 8, 6, 18, 20, 26, 24])
            edges = np.array([1, 5, 7, 3, 19, 23, 25, 21, 9, 11, 17, 15])
            faces = np.array([12, 14, 10, 16, 4, 22])
            volume = np.array([13])

            sort = np.concatenate((vertices, edges, faces, volume))

        elif permute and order > 2 and dim == 2:
            sort = lagrange_quad(order=order)

        elif permute and order > 2 and dim == 3:
            sort = lagrange_hexahedron(order=order)

        if sort is not None:
            points = points[sort]
            weights = weights[sort]

        super().__init__(points, weights)

    def inv(self):
        "Return the inverse quadrature scheme."

        points = self.points.copy()
        points[self.points != 0] = 1 / points[self.points != 0]

        return Scheme(points, self.weights)


class GaussLegendreBoundary(GaussLegendre):
    r"""An arbitrary-`order` Gauss-Legendre quadrature rule of `dim` 1, 2 or 3 on the
    interval ``[-1, 1]``.

    Parameters
    ----------
    order : int
        The number of sample points :math:`n` minus one. The quadrature rule integrates
        degree :math:`2n-1` polynomials exactly.
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

    Examples
    --------
    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticQuad()
       >>> quadrature = fem.GaussLegendreBoundary(order=2, dim=2)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> element = fem.QuadraticHexahedron()
       >>> quadrature = fem.GaussLegendreBoundary(order=2, dim=3)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    .. pyvista-plot::
       :include-source: True

       >>> import felupe as fem
       >>>
       >>> element = fem.ArbitraryOrderLagrangeElement(order=5, dim=3)
       >>> quadrature = fem.GaussLegendreBoundary(order=5, dim=3)
       >>> quadrature.plot(
       ...     plotter=element.plot(add_point_labels=False, show_points=False),
       ...     weighted=True,
       ... ).show()

    """

    def __init__(self, order: int, dim: int, permute: bool = True):
        super().__init__(order=order, dim=dim - 1, permute=permute)

        # reset dimension
        self.dim = dim

        if self.dim == 2 or self.dim == 3:
            # quadrature points projected onto first edge of a quad
            #                          or onto first face of a hexahedron
            self.points = np.hstack((self.points, -np.ones((len(self.points), 1))))

        else:
            raise ValueError("Given dimension not implemented (must be 2 or 3).")
