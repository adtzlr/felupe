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
from .microsphere import affine_stretch_statevars
from tensortrax.math import array, abs as tensor_abs, maximum, sqrt, exp
from tensortrax.math.special import try_stack


def morph_representative_directions(C, statevars, p, ε=1e-8):
    """Strain energy function of the
    `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ model formulation [1]_,
    implemented by the concept of
    `representative directions <https://nbn-resolving.org/urn:nbn:de:bsz:ch1-qucosa-114428>`_
    [2]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    statevars : array
        Vector of stacked state variables (CTS, λ - 1, SA1, SA2).
    p : list of float
        A list which contains the 8 material parameters.
    ε : float, optional
        A small stabilization parameter (default is 1e-8).

    Examples
    --------
    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.morph_representative_directions,
        ...     p=[0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117],
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps(
        ...        [1, 2, 1, 2.75, 1, 3.5, 1, 4.2, 1, 4.8, 1, 4.8, 1],
        ...        num=50,
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
    """

    def morph_uniaxial(λ, statevars, p, ε=1e-8):
        """Return the force (per undeformed area) for a given longitudinal stretch in
        uniaxial incompressible tension or compression for the MORPH material
        formulation [1]_, [2]_.

        Parameters
        ----------
        λ : tensortrax.Tensor
            Longitudinal stretch of uniaxial incompressible deformation.
        statevars : array
            Vector of stacked state variables (CTS, λ - 1, SA1, SA2).
        p : list of float
            A list which contains the 8 material parameters.
        ε : float, optional
            A small stabilization parameter (default is 1e-8).

        References
        ----------
        .. [1] D. Besdo and J. Ihlemann, "A phenomenological constitutive model for
           rubberlike materials and its numerical applications", International Journal
           of Plasticity, vol. 19, no. 7. Elsevier BV, pp. 1019–1036, Jul. 2003. doi:
           `10.1016/s0749-6419(02)00091-8 <https://doi.org/10.1016/s0749-6419(02)00091-8>`_.

        .. [2] M. Freund, "Verallgemeinerung eindimensionaler Materialmodelle für die
           Finite-Elemente-Methode", Dissertation, Technische Universität Chemnitz,
           Chemnitz, 2013.

        """
        CTSn = array(statevars[:21], like=C, shape=(21,))
        λn = array(statevars[21:42], like=C, shape=(21,)) + 1
        SA1n = array(statevars[42:63], like=C, shape=(21,))
        SA2n = array(statevars[63:], like=C, shape=(21,))

        CT = tensor_abs(λ**2 - 1 / λ)
        CTS = maximum(CT, CTSn)

        L1 = 2 * (λ**3 / λn - λn**2) / 3
        L2 = (λn**2 / λ**3 - 1 / λn) / 3
        LT = tensor_abs(L1 - L2)

        sigmoid = lambda x: 1 / sqrt(1 + x**2)
        α = p[0] + p[1] * sigmoid(p[2] * CTS)
        β = p[3] * sigmoid(p[2] * CTS)
        γ = p[4] * CTS * (1 - sigmoid(CTS / p[5]))

        L1_LT = L1 / (ε + LT)
        L2_LT = L2 / (ε + LT)
        CT_CTS = CT / (ε + CTS)

        SL1 = (γ * exp(p[6] * L1_LT * CT_CTS) + p[7] * L1_LT) / λ**2
        SL2 = (γ * exp(p[6] * L2_LT * CT_CTS) + p[7] * L2_LT) * λ

        SA1 = (SA1n + β * LT * SL1) / (1 + β * LT)
        SA2 = (SA2n + β * LT * SL2) / (1 + β * LT)

        dψdλ = (2 * α + SA1) * λ - (2 * α + SA2) / λ**2
        statevars_new = try_stack([CTS, (λ - 1), SA1, SA2], fallback=statevars)

        return 5 * dψdλ.real_to_dual(λ), statevars_new

    return affine_stretch_statevars(
        C, statevars, f=morph_uniaxial, kwargs={"p": p, "ε": ε}
    )
