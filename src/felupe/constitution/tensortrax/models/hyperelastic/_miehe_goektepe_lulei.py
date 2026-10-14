# SPDX-License-Identifier: GPL-3.0-or-later

from .microsphere import langevin, linear, nonaffine_stretch, nonaffine_tube


def miehe_goektepe_lulei(C, mu, N, U, p, q):
    """Strain energy function of the isotropic hyperelastic
    `micro-sphere <https://doi.org/10.1016/j.jmps.2004.03.011>`_ model formulation [1]_.

    Parameters
    ----------
    C : tensortrax.Tensor or jax.Array
        Right Cauchy-Green deformation tensor.
    mu : float
        Shear modulus (ground state stiffness).
    N : float
        Number of chain segments (chain locking response).
    U : float
        Tube geometry parameter (3D locking characteristics).
    p : float
        Non-affine stretch parameter (additional constraint stiffness).
    q : float
        Non-affine tube parameter (shape of constraint stress).

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
        ...     mat.models.hyperelastic.miehe_goektepe_lulei,
        ...     mu=0.1475,
        ...     N=3.273,
        ...     p=9.31,
        ...     U=9.94,
        ...     q=0.567,
        ... )
        >>> ux = ps = fem.math.linsteps([1, 2], num=50)
        >>> bx = fem.math.linsteps([1, 1.5], num=50)
        >>> ax = umat.plot(ux=ux, ps=ps, bx=bx, incompressible=True)

    References
    ----------
    .. [1] C. Miehe, S. Göktepe and F. Lulei, "A micro-macro approach to rubber-like
       materials - Part I: the non-affine micro-sphere model of rubber elasticity",
       Journal of the Mechanics and Physics of Solids, vol. 52, no. 11. Elsevier BV, pp.
       2617–2660, Nov. 2004. doi:
       `10.1016/j.jmps.2004.03.011 <https://doi.org/10.1016/j.jmps.2004.03.011>`_.
    """

    kwargs_stretch = {"mu": mu, "N": N}
    kwargs_tube = {"mu": mu * N * U}

    return nonaffine_stretch(
        C, p=p, f=langevin, kwargs=kwargs_stretch
    ) + nonaffine_tube(C, q=q, f=linear, kwargs=kwargs_tube)
