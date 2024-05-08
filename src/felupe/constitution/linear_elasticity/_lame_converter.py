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


def lame_converter(E, nu):
    r"""Convert the pair of given material parameters Young's modulus :math:`E` and
    Poisson ratio :math:`\nu` to first and second Lamé - constants :math:`\lambda` and
    :math:`\mu`.

    Notes
    -----

    ..  math::

        \lambda &= \frac{E \nu}{(1 + \nu) (1 - 2 \nu)}

        \mu &= \frac{E}{2 (1 + \nu)}

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson ratio.

    Returns
    -------
    lmbda : float
        First Lamé - constant.
    mu : float
        Second Lamé - constant (shear modulus).
    """

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return lmbda, mu
