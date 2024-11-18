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
from tensortrax.math import trace
from tensortrax.math.linalg import det


def neo_hooke(C, mu):
    r"""Strain energy function of the isotropic hyperelastic
    `Neo-Hookean <https://en.wikipedia.org/wiki/Neo-Hookean_solid>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        Shear modulus.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-nh`.

    ..  math::
        :label: psi-nh

        \psi = \frac{\mu}{2} \left(\text{tr}\left(\hat{\boldsymbol{C}}\right) - 3\right)

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  pyvista-plot::
        :context:

        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the hyperelastic material.

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = mat.Hyperelastic(mat.models.hyperelastic.neo_hooke, mu=1.0)
        >>> ax = umat.plot(incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """
    return mu / 2 * (det(C) ** (-1 / 3) * trace(C) - 3)
