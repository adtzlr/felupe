# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import log, sqrt, trace
from tensortrax.math.linalg import det


def van_der_waals(C, mu, limit, a, beta):
    r"""Strain energy function of the
    `Van der Waals <https://doi.org/10.1016/0032-3861(81)90200-7>`_ [1]_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        Initial shear modulus.
    limit : float
        Limiting stretch :math:`\lambda_m` at which the polymer chain network becomes
        locked.
    a : float
        Attractive interactions between the quasi-particles.
    beta : float
        Mixed-Invariant factor: 0 for pure I1- and 1 for pure I2-contribution.

    Examples
    --------
    First, choose the desired automatic differentiation backend

    ..  plot::
        :context: close-figs

        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    and create the hyperelastic material.

    ..  plot::
        :context: close-figs

        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.van_der_waals,
        ...     mu=1.0,
        ...     beta=0.1,
        ...     a=0.5,
        ...     limit=5.0
        ... )
        >>> ax = umat.plot(incompressible=True)

    References
    ----------
    ..  [1] H.-G. Kilian, "Equation of state of real networks", Polymer, vol. 22, no. 2.
        Elsevier BV, pp. 209–217, Feb. 1981. doi:
        `10.1016/0032-3861(81)90200-7 <https://www.doi.org/10.1016/0032-3861(81)90200-7>`_.

    """
    J3 = det(C) ** (-1 / 3)
    I1 = J3 * trace(C)
    I2 = (trace(C) ** 2 - J3**2 * trace(C @ C)) / 2
    Im = (1 - beta) * I1 + beta * I2
    Im += 1e-4
    eta = sqrt((Im - 3) / (limit**2 - 3))
    return mu * (
        -(limit**2 - 3) * (log(1 - eta) + eta) - 2 / 3 * a * ((Im - 3) / 2) ** (3 / 2)
    )
