# SPDX-License-Identifier: GPL-3.0-or-later

from ..lagrange import morph_uniaxial
from .microsphere import affine_stretch_statevars


def morph_representative_directions(C, statevars, p, ε=1e-8):
    """Strain energy function of the
    `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ model formulation [1]_,
    implemented by the concept of
    `representative directions <https://nbn-resolving.org/urn:nbn:de:bsz:ch1-qucosa-114428>`_
    [2]_, [3]_.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    statevars : array
        Vector of stacked state variables (CTS, λ - 1, SA1, SA2).
    p : list of float
        A list which contains the 8 material parameters.
    ε : float, optional
        A small stabilization parameter (default is 1e-8).

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>> import felupe.constitution.tensortrax.models as models
        >>>
        >>> umat = fem.Hyperelastic(
        ...     models.hyperelastic.morph_representative_directions,
        ...     p=[0.011, 0.408, 0.421, 6.85, 0.0056, 5.54, 5.84, 0.117],
        ...     nstatevars=84,
        ... )
        >>> ax = umat.plot(
        ...    incompressible=True,
        ...    ux=fem.math.linsteps(
        ...        # [1, 2, 1, 2.75, 1, 3.5, 1, 4.2, 1, 4.8, 1, 4.8, 1],
        ...        [1, 2.75, 1, 2.75],
        ...        num=20,
        ...    ),
        ...    ps=None,
        ...    bx=None,
        ... )

    References
    ----------
    .. [1] D. Besdo and J. Ihlemann, "A phenomenological constitutive model for
       rubberlike materials and its numerical applications", International Journal
       of Plasticity, vol. 19, no. 7. Elsevier BV, pp. 1019–1036, Jul. 2003. doi:
       `10.1016/s0749-6419(02)00091-8 <https://doi.org/10.1016/s0749-6419(02)00091-8>`_.

    .. [2] M. Freund, "Verallgemeinerung eindimensionaler Materialmodelle für die
       Finite-Elemente-Methode", Dissertation, Technische Universität Chemnitz,
       Chemnitz, 2013.

    .. [3] C. Miehe, S. Göktepe and F. Lulei, "A micro-macro approach to rubber-like
       materials - Part I: the non-affine micro-sphere model of rubber elasticity",
       Journal of the Mechanics and Physics of Solids, vol. 52, no. 11. Elsevier BV, pp.
       2617–2660, Nov. 2004. doi:
       `10.1016/j.jmps.2004.03.011 <https://doi.org/10.1016/j.jmps.2004.03.011>`_.

    See Also
    --------
    felupe.morph : Strain energy function of the MORPH model formulation.
    felupe.morph_representative_directions : First Piola-Kirchhoff stress tensor of the
        MORPH model formulation, implemented by the concept of representative
        directions.
    """

    def f(λ, statevars, **kwargs):
        dψdλ, statevars_new = morph_uniaxial(λ, statevars, **kwargs)
        return 5 * dψdλ.real_to_dual(λ), statevars_new

    return affine_stretch_statevars(C, statevars, f=f, kwargs={"p": p, "ε": ε})
