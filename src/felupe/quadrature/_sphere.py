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

from ._scheme import Scheme


class BazantOh(Scheme):
    r"""Quadrature scheme for a numeric integration of the surface of a unit sphere
    [1]_.

    Parameters
    ----------
    n : int, optional
        The number of quadrature points (default is 21).

    Notes
    -----
    The approximation is given by

    ..  math::

        \int_{\partial V} f(x) dA \approx \sum f(x_q) w_q

    with quadrature points :math:`x_q` and corresponding weights :math:`w_q`.

    Examples
    --------

    .. pyvista-plot::
       :force_static:

       >>> import felupe as fem
       >>> import pyvista as pv
       >>>
       >>> quadrature = fem.BazantOh(n=21)
       >>>
       >>> plotter = quadrature.plot(weighted=True)
       >>> sphere = pv.Sphere(radius=1).clip(normal="z", invert=False)
       >>> actor = plotter.add_mesh(sphere, opacity=0.3, color="white")
       >>> plotter.show()

    References
    ----------
    .. [1] Bazant, Z. P., & Oh, B. H. (1986). Efficient Numerical Integration on
       the Surface of a Sphere. ZAMM ‐ Journal of Applied Mathematics and
       Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik, 66(1),
       37-49. https://doi.org/10.1002/zamm.19860660108
    """

    def __init__(self, n: int = 21):
        schemes = {
            21: self._scheme_21,
        }
        points, weights = schemes[n]()
        super().__init__(points=points, weights=weights)

    def _scheme_21(self):
        "2x21-point scheme (degree 9, orthogonal symmetries)."

        a = np.sqrt(2) / 2
        b = 0.836095596749
        c = 0.387907304067

        points = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, a, a],
                [0, -a, a],
                [a, 0, a],
                [-a, 0, a],
                [a, a, 0],
                [-a, a, 0],
                [b, c, c],
                [-b, c, c],
                [b, -c, c],
                [-b, -c, c],
                [c, b, c],
                [-c, b, c],
                [c, -b, c],
                [-c, -b, c],
                [c, c, b],
                [-c, c, b],
                [c, -c, b],
                [-c, -c, b],
            ]
        )

        w1 = 0.0265214244093
        w2 = 0.0199301476312
        w3 = 0.0250712367487

        weights = 2 * np.concatenate(
            [np.repeat(w1, 3), np.repeat(w2, 6), np.repeat(w3, 12)]
        )

        return points, weights
