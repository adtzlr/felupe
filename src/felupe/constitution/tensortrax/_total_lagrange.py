# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from tensortrax import Tensor


def total_lagrange(material):
    r"""Decorate a second Piola-Kirchhoff stress Total-Lagrange material formulation as
    a first Piola-Kirchoff stress function.

    Notes
    -----
    The Green-Lagrange strain tensor and its first variation are given in Eq.
    :eq:`green_lagrange_strain_tl`.

    ..  math::
        :label: green_lagrange_strain_tl

        \boldsymbol{E} &= \frac{1}{2} \left(
            \boldsymbol{F}^T \boldsymbol{F} - \boldsymbol{1}
        \right)

        \delta \boldsymbol{E} &= \frac{1}{2} \left(
            \delta \boldsymbol{F}^T \boldsymbol{F} +
            \boldsymbol{F}^T \delta \boldsymbol{F}
        \right)

    This will lead to the variations of the strain energy density function per unit
    undeformed volume, see Eq. :eq:`total_lagrange_variation`.

    ..  math::
        :label: total_lagrange_variation

        \delta \psi &= \boldsymbol{S} : \delta \boldsymbol{E}

        \delta \psi &= \boldsymbol{F} \boldsymbol{S} : \delta \boldsymbol{F}

    Finally, the first Piola-Kirchhoff stress tensor is given by Eq.
    :eq:`first_piola_kirchhoff_stress`.

    ..  math::
        :label: first_piola_kirchhoff_stress

        \boldsymbol{P} = \boldsymbol{F} \boldsymbol{S}

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>> import felupe.constitution.tensortrax as mat
        >>> import tensortrax.math as tm
        >>>
        >>> @fem.total_lagrange
        ... def neo_hooke_total_lagrange(F, mu=1):
        ...     C = F.T @ F
        ...     S = mu * tm.special.dev(tm.linalg.det(C)**(-1/3) * C) @ tm.linalg.inv(C)
        ...     return S
        >>>
        >>> umat = mat.Material(neo_hooke_total_lagrange, mu=1)

    See Also
    --------
    felupe.constitution.tensortrax.Hyperelastic : A hyperelastic material definition
        with a given function for the strain energy density function per unit undeformed
        volume with Automatic Differentiation.
    felupe.constitution.tensortrax.Material : A material definition with a given
        function for the partial derivative of the strain energy function w.r.t. the
        deformation gradient tensor with Automatic Differentiation.
    """

    @wraps(material)
    def first_piola_kirchhoff_stress(F, *args, **kwargs):
        # evaluate the second Piola-Kirchhoff stress
        res = material(F, *args, **kwargs)

        # check if the material formulation returns state variables and extract
        # the second Piola-Kirchhoff stress tensor
        if isinstance(res, Tensor):
            S = res
            statevars_new = None
        else:
            S, statevars_new = res

        # first Piola-Kirchhoff stress tensor
        P = F @ S

        if statevars_new is None:
            return P
        else:
            return P, statevars_new

    return first_piola_kirchhoff_stress
