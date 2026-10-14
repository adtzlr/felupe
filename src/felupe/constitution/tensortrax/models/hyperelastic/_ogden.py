# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import sum as tsum
from tensortrax.math.linalg import det, eigvalsh


def ogden(C, mu, alpha):
    r"""Strain energy function of the isotropic hyperelastic
    `Ogden <https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)>`_ material
    formulation.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of moduli.
    alpha : list of float
        List of stretch exponents.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-ogden`

    ..  math::
        :label: psi-ogden

        \psi = \sum_i \frac{2 \mu_i}{\alpha^2_i} \left(
            \lambda_1^{\alpha_i} + \lambda_2^{\alpha_i} + \lambda_3^{\alpha_i} - 3
        \right)

    The sum of the moduli :math:`\mu_i` is equal to the initial shear modulus
    :math:`\mu`, see Eq. :eq:`shear-modulus-ogden`.

    ..  math::
        :label: shear-modulus-ogden

        \mu = \sum_i \mu_i

    Examples
    --------

    ..  plot::
        :context: close-figs

        >>> # import felupe.constitution.jax as mat
        >>> import felupe.constitution.tensortrax as mat

    ..  plot::

        >>> umat = mat.Hyperelastic(
        ...     mat.models.hyperelastic.ogden, mu=[1, 0.2], alpha=[1.7, -1.5]
        ... )
        >>> ax = umat.plot(incompressible=True)

    """

    wC = det(C) ** (-1 / 3) * eigvalsh(C)
    return tsum([2 * m / a**2 * (tsum(wC ** (a / 2)) - 3) for m, a in zip(mu, alpha)])
