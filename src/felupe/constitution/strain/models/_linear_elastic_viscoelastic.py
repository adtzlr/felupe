# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ....math import cdya_ik, dev, dya, identity, trace


def linear_elastic_viscoelastic(dε, εn, σn, ζn, λ, μ, G, τ, dtime, **kwargs):
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
    dtime : float
        Time increment :math:`\Delta t`. If it is not given as material parameter of
        :class:`~felupe.MaterialStrain`, the material is time-dependent, i.e. the time
        increment is an input of the material, see the examples.

        ..  deprecated:: 12.0.0
            The former name of this argument, ``Δt``, is still taken as material
            parameter ``dtime`` by :class:`~felupe.MaterialStrain`, but it emits a
            :class:`DeprecationWarning`. It will be removed in 13.0.0. Use
            ``dtime=0.5`` instead of ``Δt=0.5``.
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

        \dot{\boldsymbol{\sigma}}_{v,i}
            + \frac{1}{\tau_i} \boldsymbol{\sigma}_{v,i} &=
            2 G_i \operatorname{dev}\left( \dot{\boldsymbol{\varepsilon}} \right)

        \boldsymbol{\sigma}_{v,i} &= a_i\ \boldsymbol{\sigma}_{v,i}^n + b_i
            \operatorname{dev} \left(
                \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_n
            \right)

    with the coefficients as denoted in Eq. :eq:`visco-coeff`.

    ..  math::
        :label: visco-coeff

        a_i &= \exp \left( -\Delta t / \tau_i \right)

        b_i &= \frac{2 G_i\ \tau_i (1 - a_i)}{\Delta t}

    For a time increment of zero, there is no relaxation (:math:`a_i = 1`) and the
    viscoelastic parts respond with their (instantaneous) shear moduli, i.e. the limit
    :math:`b_i = 2 G_i` is taken for :math:`\Delta t \rightarrow 0`.

    The total fourth-order elasticity tensor is given in Eq. :eq:`fourth-order-visco`.

    ..  math::
        :label: fourth-order-visco

        \mathbb{C} = 2 \bar{\mu} \ \boldsymbol{1} \odot \boldsymbol{1} +
                \bar{\lambda}\ \boldsymbol{1} \otimes \boldsymbol{1}

    The effective coefficients are summarized in Eq. :eq:`visco-coeff-tangent`.

    ..  math::
        :label: visco-coeff-tangent

        \bar{\mu} &= \mu + \sum_{i=1}^N \frac{b_i}{2}

        \bar{\lambda} &= \lambda - \frac{2}{3} \left( \bar{\mu} - \mu \right)

    Examples
    --------
    The material is time-dependent, i.e. the time increment is an input of the
    material, like the deformation gradient. The load case of the force-stretch curve
    includes the (absolute) times, here with a time increment of 5 per stretch
    increment.

    ..  plot::

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> umat = fem.MaterialStrain(
        ...     material=fem.linear_elastic_viscoelastic,
        ...     λ=2.0,
        ...     μ=1.0,
        ...     G=[3.0, 2.0],
        ...     τ=[10.0, 100.0],
        ...     statevars=((2, 3, 3),),
        ... )
        >>> stretches = fem.math.linsteps([1, 2, 1, 2, 1, 2, 1], num=20)
        >>> times = 5.0 * np.arange(len(stretches))
        >>> ax = umat.plot(ux=[stretches, times], bx=None, ps=None)

    The time increment is given by the keyword-argument ``dtime`` of the methods
    ``gradient()`` and ``hessian()``, e.g. by the time increment of a solid body which
    is updated by a :class:`~felupe.TimeStep`.

    >>> umat.time_dependent
    True
    >>>
    >>> F = np.diag([1.1, 1.0, 1.0]).reshape(3, 3, 1, 1)
    >>> statevars = np.zeros((36, 1, 1))
    >>> P, statevars_new = umat.gradient([F, statevars], dtime=5.0)

    A time increment which is given as material parameter, e.g. ``dtime=5.0``, is a
    constant material parameter.

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

    # relaxation factors and effective shear moduli with x = Δt / τ, i.e.
    # b = 2 G τ (1 - a) / Δt = 2 G (1 - exp(-x)) / x with the limit b = 2 G for x → 0
    x = [dtime / τi for τi in τ]
    a = [np.exp(-xi) for xi in x]
    b = [2 * Gi * (-np.expm1(-xi) / xi if xi > 0 else 1.0) for Gi, xi in zip(G, x)]

    # Alternative method: backward-Euler
    # a = [1 / (1 + Δt / τi) for τi in τ]
    # b = [2 * Gi * ai for Gi, ai in zip(G, a)]

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
