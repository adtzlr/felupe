# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""
from functools import wraps

from tensortrax import Tensor


def total_lagrange(material):
    r"""Decorate a second Piola-Kirchhoff stress Total-Lagrange material formulation as
    a first Piola-Kirchoff stress function.

    Notes
    -----
    ..  math::

        \delta \psi = \boldsymbol{F} \boldsymbol{S} : \delta \boldsymbol{F}

    Examples
    --------
    >>> import felupe as fem
    >>> import felupe.constitution.tensortrax as mat
    >>> import tensortrax.math as tm
    >>>
    >>> @fem.total_lagrange
    >>> def neo_hooke_total_lagrange(F, mu=1):
    >>>     C = F.T @ F
    >>>     S = mu * tm.special.dev(tm.linalg.det(C)**(-1/3) * C) @ tm.linalg.inv(C)
    >>>     return S
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
