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

from ....math import cdya_ik, dev, dya, identity, trace


def linear_elastic_viscoelastic(dε, εn, σn, ζn, λ, μ, G, τ, Δt, **kwargs):
    r"""3D linear-elastic viscoelastic material formulation to be used in
    :class:`~felupe.MaterialStrain`.

    Arguments
    ---------
    dε : ndarray
        Strain increment.
    εn : ndarray
        Old strain tensor.
    σn : ndarray
        Old stress tensor.
    ζn : list
        List of old state variables.
    λ : float
        First Lamé-constant.
    μ : float
        Second Lamé-constant (shear modulus).
    G : list of float
        List of deviatoric viscoelastic shear moduli.
    τ : list of float
        List of time constants for deviatoric viscoelastic shear moduli.
    Δt : float
        Time increment.
    **kwargs
        Additional keyword arguments.

        tangent : bool
            A flag to evaluate the elasticity tensor.

    Returns
    -------
    dσdε : ndarray
        Elasticity tensor.
    σ : ndarray
        (New) stress tensor.
    ζ : list
        List of new state variables.

    Notes
    -----
    The stress consists of a long-term elastic and a deviatoric viscoelastic stress
    part, see Eq. :eq:`total-stress`.

    ..  math::
        :label: total-stress

        \boldsymbol{\sigma} = \boldsymbol{\sigma}_e + \boldsymbol{\sigma}_v

    The long-term elastic part is given in Eq. :eq:`elastic-stress`.

    ..  math::
        :label: elastic-stress

        \boldsymbol{\sigma}_e = 2 \mu\ \boldsymbol{\varepsilon}
            + \lambda \operatorname{tr}(\boldsymbol{\varepsilon}) \boldsymbol{1}

    The i-th viscous part is given in Eq. :eq:`visco-stress`,

    ..  math::
        :label: visco-stress

        \boldsymbol{\sigma}_{v,i} = a_i\ \boldsymbol{\sigma}_{v,i}^n + b_i
            \operatorname{dev} \left(
                \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_n
            \right)

    along with the coefficients as denoted in Eq. :eq:`visco-coeff`.

    ..  math::
        :label: visco-coeff

        a_i &= \exp \left( -\Delta t / \tau_i \right)

        b_i &= \frac{2 G_i\ \tau_i (1 - a_i)}{\Delta t}

        \bar{\mu} &= \mu + \sum_{i=1}^N \frac{b_i}{2}

        \bar{\lambda} &= \lambda - \frac{2}{3} \left( \bar{\mu} - \mu \right)

    The total fourth-order elasticity tensor is given in Eq. :eq:`fourth-order-visco`.

    ..  math::
        :label: fourth-order-visco

        \mathbb{C} = 2 \bar{\mu} \ \boldsymbol{1} \odot \boldsymbol{1} +
                \bar{\lambda} \boldsymbol{1} \otimes \boldsymbol{1}

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.MaterialStrain(
        ...     material=fem.linear_elastic_viscoelastic,
        ...     λ=2.0,
        ...     μ=1.0,
        ...     G=[3.0, 2.0],
        ...     τ=[10.0, 100.0],
        ...     Δt=5.0,
        ...     statevars=((2, 3, 3),),
        ... )
        >>> ux = fem.math.linsteps([1, 2, 1, 2, 1, 2, 1], num=20)
        >>> ax = umat.plot(ux=ux, bx=None, ps=None)

    See Also
    --------
    MaterialStrain : A strain-based user-defined material definition with a given
        function for the stress tensor and the (fourth-order) elasticity tensor.

    """

    # helpers
    eye = identity(dim=3, shape=(1, 1))

    # new elastic stress (old stress σn is not used because of included viscoelasticity)
    ε = εn + dε
    σ = 2 * μ * ε + λ * trace(ε) * eye

    # update deviatoric viscoelastic stress
    dev_dε = dev(dε)

    a = [np.exp(-Δt / τi) for τi in τ]
    b = [2 * Gi * (1 - ai) * τi / Δt for Gi, ai, τi in zip(G, a, τ)]

    # update of new deviatoric viscoelastic stress
    σvn = ζn[0]
    σv = σvn.copy()
    for i, σvni in enumerate(ζn[0]):
        σv[i] = a[i] * σvni + b[i] * dev_dε

    # evaluate elasticity tensor
    if kwargs["tangent"]:
        μ_eff = μ + np.sum(b, axis=0) / 2
        λ_eff = λ - 2 / 3 * (μ_eff - μ)
        dσdε = 2 * μ_eff * cdya_ik(eye, eye) + λ_eff * dya(eye, eye)
    else:
        dσdε = None

    # new total stress
    σ += np.sum(σv, axis=0)

    # new state variables
    ζ = [σv]

    return dσdε, σ, ζ
