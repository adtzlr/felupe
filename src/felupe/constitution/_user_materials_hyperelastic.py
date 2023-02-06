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

from ..math import cdya_ik, dot, transpose
from ._user_materials import UserMaterial


class UserMaterialHyperelastic(UserMaterial):
    """A user-defined hyperelastic material definition with a given function
    for the strain energy function with Automatic Differentiation provided by
    ``tensortrax``.

    Take this code-block as template

    ..  code-block::

        import tensortrax.math as tm

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)

        umat = fem.UserMaterialHyperelastic(neo_hooke, mu=1)

    and this code-block for material formulations with state variables:

    ..  code-block::

        import tensortrax.math as tm

        def viscoelastic(C, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            Cu = det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Ci = tm.special.from_triu_1d(Cin, like=C) + mu / eta * dtime * Cu
            Ci = det(Ci) ** (-1 / 3) * Ci

            # first invariant of elastic part of right Cauchy-Green deformation tensor
            I1 = tm.trace(Cu @ inv(Ci))

            # first Piola-Kirchhoff stress tensor and state variable
            return mu / 2 * (I1 - 3), tm.special.triu_1d(Ci)

        umat = fem.UserMaterialHyperelastic(
            viscoelastic, mu=1, eta=1, dtime=1, nstatevars=6
        )

    See the `documentation of tensortrax <https://github.com/adtzlr/tensortrax>`_
    for further details.

    """

    def __init__(self, fun, nstatevars=0, parallel=False, **kwargs):

        if nstatevars > 0:
            # split the original function into two sub-functions
            self.fun = tr.take(fun, item=0)
            self.fun_statevars = tr.take(fun, item=1)
        else:
            self.fun = fun

        self.parallel = parallel

        super().__init__(
            stress=self._stress,
            elasticity=self._elasticity,
            nstatevars=nstatevars,
            **kwargs
        )

    def _stress(self, x, **kwargs):
        F = x[0]

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        S = tr.gradient(self.fun, wrt=0, ntrax=2, parallel=self.parallel, sym=True)(
            C, *statevars, **kwargs
        )
        if self.nstatevars > 0:
            statevars_new = tr.function(
                self.fun_statevars, wrt=0, ntrax=2, parallel=self.parallel
            )(C, *statevars, **kwargs)
        else:
            statevars_new = None
        return [dot(F, 2 * S), statevars_new]

    def _elasticity(self, x, **kwargs):
        F = x[0]

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        D, S, W = tr.hessian(
            self.fun, wrt=0, ntrax=2, full_output=True, parallel=self.parallel, sym=True
        )(C, *statevars, **kwargs)
        A = np.einsum("iI...,kK...,IJKL...->iJkL...", F, F, 4 * D)
        A += cdya_ik(np.eye(3), 2 * S)
        return [A]
