# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import tensortrax as tr

from ._user_materials import UserMaterial
from ..math import (
    dot,
    transpose,
    cdya_ik,
)


class UserMaterialHyperelastic(UserMaterial):
    """A user-defined hyperelastic material definition with a given function
    for the strain energy function with Automatic Differentiation provided by
    ``tensortrax``.

    Take this code-block as template:

    ..  code-block::

        import tensortrax.math as tm

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)

        umat = fem.UserMaterialHyperelastic(neo_hooke, mu=1)

    See the `documentation of tensortrax <https://github.com/adtzlr/tensortrax>`_
    for further details.

    """

    def __init__(self, fun, parallel=False, **kwargs):
        self.fun = fun
        self.parallel = parallel
        super().__init__(
            stress=self._stress, elasticity=self._elasticity, nstatevars=0, **kwargs
        )

    def _stress(self, x, **kwargs):
        F = x[0]
        C = dot(transpose(F), F)
        S = tr.gradient(self.fun, wrt=0, ntrax=2, parallel=self.parallel, sym=True)(
            C, **kwargs
        )
        return [dot(F, 2 * S), None]

    def _elasticity(self, x, **kwargs):
        F = x[0]
        C = dot(transpose(F), F)
        D, S, W = tr.hessian(
            self.fun, wrt=0, ntrax=2, full_output=True, parallel=self.parallel, sym=True
        )(C, **kwargs)
        A = np.einsum("iI...,kK...,IJKL...->iJkL...", F, F, 4 * D)
        A += cdya_ik(np.eye(3), 2 * S)
        return [A]
