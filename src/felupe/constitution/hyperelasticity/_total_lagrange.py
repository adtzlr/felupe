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
from tensortrax.math.special import ddot


def total_lagrange(material):
    r"""Decorate a Total-Lagrange material formulation as a (fake) strain energy density
    function where only its gradient and hessian are defined.

    Notes
    -----
    ..  math::

        \delta \psi = \frac{1}{2} \boldsymbol{S} : \delta \boldsymbol{C}

    Examples
    --------
    >>> import felupe as fem
    >>> import tensortrax.math as tm
    >>>
    >>> @fem.total_lagrange
    >>> def neo_hooke_total_lagrange(C, mu=1):
    >>>     S = mu * tm.special.dev(tm.linalg.det(C)**(-1/3) * C) @ tm.linalg.inv(C)
    >>>     return S
    >>>
    >>> umat = fem.Hyperelastic(neo_hooke_total_lagrange, mu=1)

    See Also
    --------
    felupe.Hyperelastic : A hyperelastic material definition with a given function for
        the strain energy density function per unit undeformed volume with Automatic
        Differentiation.
    """

    @wraps(material)
    def strain_energy_density(C, *args, **kwargs):
        # evaluate the second Piola-Kirchhoff stress
        res = material(C, *args, **kwargs)

        # check if the material formulation returns state variables and extract
        # the second Piola-Kirchhoff stress tensor
        if isinstance(res, Tensor):
            S = res
            statevars_new = None
        else:
            S, statevars_new = res

        # create a (fake) strain energy density
        W = (S / 2).real_to_dual(C, mul=ddot)

        if statevars_new is None:
            return W
        else:
            return W, statevars_new

    return strain_energy_density
