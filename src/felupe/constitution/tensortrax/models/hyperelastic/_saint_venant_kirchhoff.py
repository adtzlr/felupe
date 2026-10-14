# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import base, log, trace
from tensortrax.math import sum as tsum
from tensortrax.math.linalg import eigvalsh


def saint_venant_kirchhoff(C, mu, lmbda, k=2):
    r"""Strain energy function of the isotropic hyperelastic
    `Saint-Venant Kirchhoff <https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant-Kirchhoff_model>`_
    material formulation.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : float
        Second Lamé constant (shear modulus).
    lmbda : float
        First Lamé constant (shear modulus).
    k : float, optional
        Strain exponent (default is 2). If 2, the Green-Lagrange strain measure is used.
        For any other value, the family of Seth-Hill strains is used.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-svk`

    ..  math::
        :label: psi-svk

        \psi = \mu I_2 + \lambda \frac{I_1^2}{2}

    with the first and second invariant of the Green-Lagrange strain tensor
    :math:`\boldsymbol{E} = \frac{1}{2} (\boldsymbol{C} - \boldsymbol{1})`, see Eq.
    :eq:`invariants-svk`.

    ..  math::
        :label: invariants-svk

        I_1 &= \text{tr}\left( \boldsymbol{E} \right)

        I_2 &= \boldsymbol{E} : \boldsymbol{E}

    Examples
    --------
    ..  warning::
        The Saint-Venant Kirchhoff material formulation is unstable for large strains.

    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.saint_venant_kirchhoff, mu=1.0, lmbda=20.0)
        >>> ax = umat.plot(incompressible=False)

    """
    eye = base.eye

    if k == 2:
        E = (C - eye(C)) / 2

        I1 = trace(E)
        I2 = trace(E @ E)

    else:
        λ2 = eigvalsh(C)
        if k == 0:
            Ek = log(λ2) / 2
        else:
            Ek = (λ2 ** (k / 2) - 1) / k

        I1 = tsum(Ek)
        I2 = tsum(Ek**2)

    return mu * I2 + lmbda * I1**2 / 2
