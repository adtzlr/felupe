# SPDX-License-Identifier: GPL-3.0-or-later

from tensortrax.math import log, trace
from tensortrax.math.linalg import det


def anssari_benam_bucchi(C, mu, N):
    r"""Strain energy function of the isotropic hyperelastic generalized Neo-Hookean
    `Anssari-Benam Bucchi <https://doi.org/10.1016/j.ijnonlinmec.2020.103626>`_ material
    formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    mu : float
        Modulus :math:`\mu = nkT` - this is not the infinitesimal shear modulus.
    N : float
        Number of Kuhn segments of a chain.

    Notes
    -----
    The strain energy function is given in Eq. :eq:`psi-abb`

    ..  math::
        :label: psi-abb

        \psi = \mu N \left( \frac{1}{6N} \left( \hat{I}_1 - 3 \right)
             - \ln \left( \frac{\hat{I}_1 - 3N}{3 - 3N} \right) \right)

    with the first main invariant of the distortional part of the right
    Cauchy-Green deformation tensor, see Eq. :eq:`invariant-abb`.

    ..  math::
        :label: invariant-abb

        \hat{I}_1 = J^{-2/3} \text{tr}\left( \boldsymbol{C} \right)

    The initial shear modulus :math:`\mu_0` is given in Eq. :eq:`shear-modulus-abb`.

    ..  math::
        :label: shear-modulus-abb

        \mu_0 = \mu \frac{1 - 3N}{3 - 3N}

    Examples
    --------

    ..  plot::

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(fem.anssari_benam_bucchi, mu=0.29, N=26.8)
        >>>
        >>> ux = fem.math.linsteps([0.6, 5], num=50)
        >>> ps = fem.math.linsteps([1, 5], num=50)
        >>> bx = fem.math.linsteps([1, 3], num=50)
        >>>
        >>> ax = umat.plot(ux=ux, ps=ps, bx=bx, incompressible=True)

    References
    ----------
    .. [1] A. Anssari-Benam and A. Bucchi, "A generalised neo-Hookean strain energy
       function for application to the finite deformation of elastomers", International
       Journal of Non-Linear Mechanics, vol. 128. Elsevier BV, p. 103626, Jan. 2021.
       doi: `10.1016/j.ijnonlinmec.2020.103626 <https://doi.org/10.1016/j.ijnonlinmec.2020.103626>`_.

    """
    I1 = det(C) ** (-1 / 3) * trace(C)
    return mu * N * ((I1 - 3) / (6 * N) - log((I1 - 3 * N) / (3 - 3 * N)))
