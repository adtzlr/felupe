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
from ._morph_uniaxial import morph_uniaxial
from .microsphere import affine_force_statevars


def morph_representative_directions(F, statevars, p, ε=1e-6):
    """First Piola-Kirchhoff stress tensor of the
    `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ model formulation [1]_,
    implemented by the concept of
    `representative directions <https://nbn-resolving.org/urn:nbn:de:bsz:ch1-qucosa-114428>`_
    [2]_, [3]_.

    Parameters
    ----------
    F : tensortrax.Tensor or jax.Array
        Deformation gradient tensor.
    statevars : array
        Vector of stacked state variables (CTS, λ - 1, SA1, SA2).
    p : list of float
        A list which contains the 8 material parameters.
    ε : float, optional
        A small stabilization parameter (default is 1e-6).

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the material.

    ..  pyvista-plot::
        :context:

        >>> umat = mat.Material(
        ...     mat.models.lagrange.morph_representative_directions,
        ...     p=[0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117],
        ...     nstatevars=84,
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps(
        ...        # [1, 2, 1, 2.75, 1, 3.5, 1, 4.2, 1, 4.8, 1, 4.8, 1],
        ...        [1, 2.75, 1, 2.75],
        ...        num=20,
        ...    ),
        ...    ps=None,
        ...    bx=None,
        ... )

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    References
    ----------
    .. [1] D. Besdo and J. Ihlemann, "A phenomenological constitutive model for
       rubberlike materials and its numerical applications", International Journal
       of Plasticity, vol. 19, no. 7. Elsevier BV, pp. 1019–1036, Jul. 2003. doi:
       `10.1016/s0749-6419(02)00091-8 <https://doi.org/10.1016/s0749-6419(02)00091-8>`_.

    .. [2] M. Freund, "Verallgemeinerung eindimensionaler Materialmodelle für die
       Finite-Elemente-Methode", Dissertation, Technische Universität Chemnitz,
       Chemnitz, 2013.

    .. [3] C. Miehe, S. Göktepe and F. Lulei, "A micro-macro approach to rubber-like
       materials - Part I: the non-affine micro-sphere model of rubber elasticity",
       Journal of the Mechanics and Physics of Solids, vol. 52, no. 11. Elsevier BV, pp.
       2617–2660, Nov. 2004. doi:
       `10.1016/j.jmps.2004.03.011 <https://doi.org/10.1016/j.jmps.2004.03.011>`_.

    See Also
    --------
    felupe.constitution.tensortrax.models.lagrange.morph : Strain energy function of the
        MORPH model formulation.
    felupe.constitution.jax.models.lagrange.morph : Strain energy function of the MORPH
        model formulation.
    """

    def f(λ, statevars, **kwargs):
        dψdλ, statevars_new = morph_uniaxial(λ, statevars, **kwargs)
        return 5 * dψdλ, statevars_new

    return affine_force_statevars(F, statevars, f=f, kwargs={"p": p, "ε": ε})
