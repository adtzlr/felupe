# SPDX-License-Identifier: GPL-3.0-or-later

from functools import wraps

from tensortrax import Tensor
from tensortrax.math.linalg import det, inv


def updated_lagrange(material):
    r"""Decorate a Cauchy-stress Updated-Lagrange material formulation as a first Piola-
    Kirchoff stress function.

    Notes
    -----
    The equilibrium equations for statics are given in Eq. :eq:`equilibrium_statics`.

    ..  math::
        :label: equilibrium_statics

        \operatorname{div} \boldsymbol{\sigma} + \boldsymbol{b} = \boldsymbol{0}

    The weak form of the equilibrium equations for statics is given in Eq.
    :eq:`equilibrium_statics_weak`.

    ..  math::
        :label: equilibrium_statics_weak

        \int_v \operatorname{div} \boldsymbol{\sigma} \cdot \delta \boldsymbol{u} \ \mathrm{d}v
            + \int_v \boldsymbol{b} \cdot \delta \boldsymbol{u} \ \mathrm{d}v &= 0

        - \int_v \boldsymbol{\sigma} :
            \frac{\partial \delta \boldsymbol{u}}{\partial \boldsymbol{x}} \ \mathrm{d}v
            + \int_{\partial v} \left( \boldsymbol{\sigma} \cdot \boldsymbol{n} \right )
            \cdot \delta \boldsymbol{u} \ \mathrm{d}a
            + \int_v \boldsymbol{b} \cdot \delta \boldsymbol{u} \ \mathrm{d}v &= 0

    This leads to the virtual work of internal forces, see Eq.
    :eq:`virtual_work_internal`.

    ..  math::
        :label: virtual_work_internal

        \delta W_{\text{int}} = -\int_v
            \boldsymbol{\sigma} :
            \frac{\partial \delta \boldsymbol{u}}{\partial \boldsymbol{x}} \ \mathrm{d}v

    The variation of the total potential energy of internal forces is given in Eq.
    :eq:`updated_lagrange_variation`.

    ..  math::
        :label: updated_lagrange_variation

        \delta \Pi &=  \int_v
            \boldsymbol{\sigma} :
            \frac{\partial \delta \boldsymbol{u}}{\partial \boldsymbol{x}} \ \mathrm{d}v

        \delta \Pi &=  \int_V
            \boldsymbol{\sigma} : \delta \boldsymbol{F} \boldsymbol{F}^{-1} \ J \mathrm{d}V

        \delta \Pi &=  \int_V
            \boldsymbol{P} : \delta \boldsymbol{F} \ \mathrm{d}V

    Finally, the first Piola-Kirchhoff stress tensor is given by Eq.
    :eq:`first_piola_kirchhoff_stress`.

    ..  math::
        :label: first_piola_kirchhoff_stress

        \boldsymbol{P} = J \boldsymbol{\sigma} \boldsymbol{F}^{-T}

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>> import felupe.constitution.tensortrax as mat
        >>> import tensortrax.math as tm
        >>>
        >>> @fem.updated_lagrange
        ... def neo_hooke_updated_lagrange(F, mu=1):
        ...     J = tm.linalg.det(F)
        ...     b = F @ F.T
        ...     σ = mu * tm.special.dev(J**(-2/3) * b) / J
        ...     return σ
        >>>
        >>> umat = mat.Material(neo_hooke_updated_lagrange, mu=1)

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
        # evaluate the Cauchy stress
        res = material(F, *args, **kwargs)

        # check if the material formulation returns state variables and extract
        # the Cauchy stress tensor
        if isinstance(res, Tensor):
            σ = res
            statevars_new = None
        else:
            σ, statevars_new = res

        # first Piola-Kirchhoff stress tensor
        J = det(F)
        P = J * σ @ inv(F).T

        if statevars_new is None:
            return P
        else:
            return P, statevars_new

    return first_piola_kirchhoff_stress
