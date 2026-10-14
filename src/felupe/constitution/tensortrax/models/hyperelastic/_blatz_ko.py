# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import sqrt, trace
from tensortrax.math.linalg import det


def blatz_ko(C, mu):
    r"""Strain energy function of the Blatz-Ko isotropic hyperelastic
    `foam <https://doi.org/10.1122/1.548937>`_ material formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        The shear modulus.

    Notes
    -----
    The Poisson ratio of the Blatz-Ko model formulation is :math:`\nu = 0.25`. The
    strain energy function is given in Eq. :eq:`psi-blatz-ko`

    ..  math::
        :label: psi-blatz-ko

        \psi = \frac{\mu}{2} \left(\frac{I_2}{I_3} + 2 \sqrt{I_3} - 5 \right)

    The shear modulus :math:`\mu` is related to young's modulus as denoted in Eq.
    :eq:`shear-modulus-blatz-ko`.

    ..  math::
        :label: shear-modulus-blatz-ko

        \mu = \frac{2 E}{5}

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

        >>> umat = mat.Hyperelastic(mat.models.hyperelastic.blatz_ko, mu=1.0)
        >>> ax = umat.plot()

    References
    ----------
    .. [1] P. J. Blatz and W. L. Ko, "Application of Finite Elastic Theory to the
       Deformation of Rubbery Materials", Transactions of the Society of Rheology, vol.
       6, no. 1. Society of Rheology, pp. 223–252, Mar. 01, 1962. doi: 10.1122/1.548937.
    """

    I1 = trace(C)
    I2 = (I1**2 - trace(C @ C)) / 2
    I3 = det(C)

    return mu * (I2 / I3 + 2 * sqrt(I3) - 5)
