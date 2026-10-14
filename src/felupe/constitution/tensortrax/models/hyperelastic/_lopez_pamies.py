# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import sum as tsum
from tensortrax.math import trace
from tensortrax.math.linalg import det


def lopez_pamies(C, mu, alpha):
    r"""Strain energy function of the isotropic hyperelastic
    `Lopez-Pamies <https://doi.org/10.1016/j.crme.2009.12.007>`_ material
    formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : list of float
        List of moduli.
    alpha : list of float
        List of invariant exponents.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-lp`

    ..  math::
        :label: psi-lp

        \psi = \sum_{r=1}^M \frac{3^{1-\alpha_r}}{2 \alpha_r} \mu_r \left(
            \hat{I}_1^{\alpha_r} - 3^{\alpha_r}
        \right)

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariant-lp`.

    ..  math::
        :label: invariant-lp

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    The sum of the moduli :math:`\mu_r` is equal to the initial shear modulus
    :math:`\mu`, see Eq. :eq:`shear-modulus-lp`.

    ..  math::
        :label: shear-modulus-lp

        \mu = \sum_r \mu_r

    Examples
    --------

    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.lopez_pamies, mu=[0.2699,  0.00001771], alpha=[1.08, 4.40]
        ... )
        >>>
        >>> ux = fem.math.linsteps([0.6, 7], num=50)
        >>> ps = fem.math.linsteps([1, 7], num=50)
        >>> bx = fem.math.linsteps([1, 5], num=50)
        >>>
        >>> ax = umat.plot(ux=ux, ps=ps, bx=bx, incompressible=True)

    References
    ----------
    .. [1] O. Lopez-Pamies, "A new I1-based hyperelastic model for rubber elastic
       materials", Comptes Rendus. Mécanique, vol. 338, no. 1. Cellule MathDoc/Centre
       Mersenne, pp. 3–11, Dec. 23, 2009. doi:
       `10.1016/j.crme.2009.12.007 <https://doi.org/10.1016/j.crme.2009.12.007>`_.

    """
    I1 = det(C) ** (-1 / 3) * trace(C)
    ψr = lambda μr, αr, I1: 3 ** (1 - αr) / (2 * αr) * μr * (I1**αr - 3**αr)

    return tsum([ψr(μr, αr, I1) for μr, αr in zip(mu, alpha)])
