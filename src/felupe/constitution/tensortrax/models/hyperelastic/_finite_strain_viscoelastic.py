# SPDX-License-Identifier: GPL-3.0-or-later
from tensortrax.math import base, no_variation, trace
from tensortrax.math.linalg import det, inv
from tensortrax.math.special import from_triu_1d, triu_1d


def finite_strain_viscoelastic(C, statevars, mu, eta, dtime):
    r"""Multiplicative
    `finite strain viscoelastic <https://doi.org/10.1016/j.cma.2013.07.004>`_ [1]_
    material formulation.

    Notes
    -----
    The material formulation is built upon the multiplicative decomposition of the
    distortional part of the deformation gradient tensor into an elastic and an
    inelastic part, see Eq. :eq:`multiplicative-split`.

    ..  math::
        :label: multiplicative-split

        \hat{\boldsymbol{F}} &= \boldsymbol{F}_e \boldsymbol{F}_i
            \ (= \hat{\boldsymbol{F}_e} \boldsymbol{F}_i)

        \boldsymbol{C}_e &= \boldsymbol{F}_e^T \boldsymbol{F}_e

        \boldsymbol{C}_i &= \boldsymbol{F}_i^T \boldsymbol{F}_i

        \text{tr}\left( \boldsymbol{C}_e \right) &= \text{tr}\left(
            \hat{\boldsymbol{C}} \boldsymbol{C}_i^{-1}
        \right)

    The components of the inelastic right Cauchy-Green deformation tensor are used as
    state variables with the evolution equation and its explicit update formula as given
    in Eq. :eq:`evolution` [1]_. The elastic part of the multiplicative decomposition of
    the deformation gradient tensor is also enforced to be an unimodular tensor which
    leads to the constraint :math:`\det(\boldsymbol{F_i})=1`. Hence, the inelastic right
    Cauchy-Green deformation tensor must be an unimodular tensor
    :math:`\det(\boldsymbol{C_i})=1`.

    ..  math::
        :label: evolution

        \dot{\boldsymbol{C}}_i &= \frac{\mu}{\eta}\ \hat{\boldsymbol{C}}

        \boldsymbol{C}_i &= \hat{\overline{\boldsymbol{C}_{i,n}
            + \frac{\Delta t \mu}{\eta} \hat{\boldsymbol{C}}}}

    The distortional part of the strain energy density per unit undeformed volume is
    assumed to be of a Neo-Hookean form, see Eq. :eq:`nh-w`.

    ..  math::
        :label: nh-w

        \hat{\psi} = \frac{\mu}{2} \left( \text{tr}\left(
            \hat{\boldsymbol{C}} \boldsymbol{C}_i^{-1}
        \right) - 3 \right)

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, dtime=1.0, nstatevars=6
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps([1, 1.5, 1, 2, 1], num=[5, 5, 10, 10]),
        ...    ps=None,
        ...    bx=None,
        ... )

    References
    ----------
    ..  [1] A. V. Shutov, R. Landgraf, and J. Ihlemann, "An explicit solution for
        implicit time stepping in multiplicative finite strain viscoelasticity",
        Computer Methods in Applied Mechanics and Engineering, vol. 265. Elsevier BV,
        pp. 213–225, Oct. 2013. doi:
        `10.1016/j.cma.2013.07.004 <https://doi.org/10.1016/j.cma.2013.07.004>`_.

    """
    J3 = det(C) ** (-1 / 3)

    # update of state variables by evolution equation
    eye = base.eye(C)
    Cin = eye + from_triu_1d(statevars[:6], like=C)
    Ci = Cin + (mu / eta * dtime) * (J3 * C)
    Ci = no_variation(det(Ci) ** (-1 / 3) * Ci)

    # first invariant of elastic part of right Cauchy-Green deformation tensor
    I1 = J3 * trace(inv(Ci) @ C)

    # strain energy function
    ψ = mu / 2 * (I1 - 3)

    # new state variables
    statevars_new = triu_1d(Ci - eye)
    return ψ, statevars_new
