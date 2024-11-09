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

import jax
import jax.numpy as jnp


def updated_lagrange(material):
    r"""Decorate a Cauchy-stress Updated-Lagrange material formulation as a first Piola-
    Kirchoff stress function.

    Notes
    -----
    ..  math::

        \delta \psi =  J \boldsymbol{\sigma} \boldsymbol{F}^{-T} : \delta \boldsymbol{F}

    Examples
    --------
    >>> import felupe as fem
    >>> import felupe.constitution.jax as mat
    >>> import jax.numpy as jnp
    >>>
    >>> @fem.updated_lagrange
    >>> def neo_hooke_updated_lagrange(F, mu=1):
    >>>     J = jnp.linalg.det(F)
    >>>     b = F @ F.T
    >>>     dev = lambda b: b - jnp.trace(b) / 3 * jnp.eye(3)
    >>>     τ = mu * dev(J**(-2/3) * b)
    >>>     return τ / J
    >>>
    >>> umat = mat.Material(neo_hooke_updated_lagrange, mu=1)

    See Also
    --------
    felupe.constitution.jax.Hyperelastic : A hyperelastic material definition with a
        given function for the strain energy density function per unit undeformed volume
        with Automatic Differentiation provided by jax.
    felupe.constitution.jax.Material : A material definition with a given function for
        the partial derivative of the strain energy function w.r.t. the deformation
        gradient tensor with Automatic Differentiation provided by jax.
    """

    @wraps(material)
    def first_piola_kirchhoff_stress(F, *args, **kwargs):
        # evaluate the Cauchy stress
        res = material(F, *args, **kwargs)

        # check if the material formulation returns state variables and extract
        # the Cauchy stress tensor
        if isinstance(res, jax.Array):
            σ = res
            statevars_new = None
        else:
            σ, statevars_new = res

        # first Piola-Kirchhoff stress tensor
        J = jnp.linalg.det(F)
        P = J * σ @ jnp.linalg.inv(F).T

        if statevars_new is None:
            return P
        else:
            return P, statevars_new

    return first_piola_kirchhoff_stress
