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
from tensortrax.math import abs as tensor_abs
from tensortrax.math import array, exp, maximum, sqrt
from tensortrax.math.special import try_stack


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
    CTSn = array(statevars[:21], like=λ, shape=(21,))
    λn = array(statevars[21:42], like=λ, shape=(21,)) + 1
    SA1n = array(statevars[42:63], like=λ, shape=(21,))
    SA2n = array(statevars[63:84], like=λ, shape=(21,))

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

    return dψdλ, statevars_new
