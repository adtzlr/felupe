# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, eigvalsh


def storakers(C, mu, alpha, beta):
    r"""Strain energy function of the Storåkers isotropic hyperelastic
    `foam <https://doi.org/10.1016/0022-5096(86)90033-5>`_ material formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of moduli.
    alpha : list of float
        List of stretch exponents.
    beta : list of float
        List of coefficients for the degree of compressibility.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-foam`

    ..  math::
        :label: psi-foam

        \psi = \sum_i \frac{2 \mu_i}{\alpha^2_i} \left[
            \lambda_1^{\alpha_i} +
            \lambda_2^{\alpha_i} +
            \lambda_3^{\alpha_i} - 3
            + \frac{1}{\beta_i} \left( J^{-\alpha_i \beta_i} - 1 \right)
        \right]

    The sum of the moduli :math:`\mu_i` is equal to the initial shear modulus
    :math:`\mu`, see Eq. :eq:`shear-modulus-foam`,

    ..  math::
        :label: shear-modulus-foam

        \mu = \sum_i \mu_i

    and the initial bulk modulus is given in Eq. :eq:`bulk-modulus-foam`.

    ..  math::
        :label: bulk-modulus-foam

        K = \sum_i 2 \mu_i \left( \frac{1}{3} + \beta_i \right)

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

        >>> import felupe as fem
        >>>
        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.storakers,
        ...     mu=[4.5 * (1.85 / 2), -4.5 * (-9.2 / 2)],
        ...     alpha=[1.85, -9.2],
        ...     beta=[0.92, 0.92],
        ... )
        >>> ax = umat.plot(
        ...     ux=fem.math.linsteps([1, 2], 15),
        ...     ps=fem.math.linsteps([1, 1], 15),
        ...     bx=fem.math.linsteps([1, 1], 9),
        ... )

    References
    ----------
    .. [1] B. Storåkers, "On material representation and constitutive branching in
       finite compressible elasticity", Journal of the Mechanics and Physics of Solids,
       vol. 34, no. 2. Elsevier BV, pp. 125–145, Jan. 1986. doi:
       10.1016/0022-5096(86)90033-5.
    """

    λ2 = eigvalsh(C)

    return tsum(
        [
            2 * μ / α**2 * (tsum(λ2 ** (α / 2)) - 3 + (det(C) ** (-α * β / 2) - 1) / β)
            for μ, α, β in zip(mu, alpha, beta)
        ]
    )
