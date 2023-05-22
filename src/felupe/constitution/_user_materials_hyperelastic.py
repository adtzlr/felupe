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

import numpy as np
import tensortrax as tr

from ..math import cdya_ik, dot, transpose
from ._user_materials import Material


class Hyperelastic(Material):
    """A user-defined hyperelastic material definition with a given function
    for the strain energy function with Automatic Differentiation provided by
    ``tensortrax``.

    Take this code-block as template

    ..  code-block::

        import tensortrax.math as tm

        def neo_hooke(C, mu):
            "Strain energy function of the Neo-Hookean material formulation."
            return mu / 2 * (tm.linalg.det(C) ** (-1/3) * tm.trace(C) - 3)

        umat = fem.Hyperelastic(neo_hooke, mu=1)

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

            # strain energy function and state variable
            return mu / 2 * (I1 - 3), tm.special.triu_1d(Ci)

        umat = fem.Hyperelastic(
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
            **kwargs,
        )

    def _stress(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        dWdC = tr.gradient(self.fun, wrt=0, ntrax=2, parallel=self.parallel, sym=True)(
            C, *statevars, **kwargs
        )
        if self.nstatevars > 0:
            statevars_new = tr.function(
                self.fun_statevars, wrt=0, ntrax=2, parallel=self.parallel
            )(C, *statevars, **kwargs)
        else:
            statevars_new = None
        return [dot(F, 2 * dWdC), statevars_new]

    def _elasticity(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        C = dot(transpose(F), F)
        d2WdCdC, dWdC, W = tr.hessian(
            self.fun, wrt=0, ntrax=2, full_output=True, parallel=self.parallel, sym=True
        )(C, *statevars, **kwargs)
        A = 4 * np.einsum(
            "iI...,kK...,IJKL...->iJkL...", F, F, np.ascontiguousarray(d2WdCdC)
        )
        B = cdya_ik(np.eye(3), 2 * dWdC)
        return [np.sum(np.broadcast_arrays(A, B), axis=0)]


class MaterialAD(Material):
    """A user-defined material definition with a given function for the partial
    derivative of the strain energy function w.r.t. the deformation gradient tensor
    with Automatic Differentiation provided by ``tensortrax``.

    Take this code-block as template

    ..  code-block::

        import tensortrax.math as tm

        def neo_hooke(F, mu):
            "First Piola-Kirchhoff stress of the Neo-Hookean material formulation."

            C = tm.dot(tm.transpose(F), F)
            Cu = tm.linalg.det(C) ** (-1/3) * C

            return mu * F @ tm.special.dev(Cu) @ tm.linalg.inv(C)

        umat = fem.MaterialAD(neo_hooke, mu=1)

    and this code-block for material formulations with state variables:

    ..  code-block::

        import tensortrax.math as tm

        def viscoelastic(F, Cin, mu, eta, dtime):
            "Finite strain viscoelastic material formulation."

            # unimodular part of the right Cauchy-Green deformation tensor
            C = tm.dot(tm.transpose(F), F)
            Cu = tm.linalg.det(C) ** (-1 / 3) * C

            # update of state variables by evolution equation
            Ci = tm.special.from_triu_1d(Cin, like=C) + mu / eta * dtime * Cu
            Ci = tm.linalg.det(Ci) ** (-1 / 3) * Ci

            # second Piola-Kirchhoff stress tensor
            S = mu * tm.special.dev(Cu @ tm.linalg.inv(Ci)) @ tm.linalg.inv(C)

            # first Piola-Kirchhoff stress tensor and state variable
            return F @ S, tm.special.triu_1d(Ci)

        umat = fem.MaterialAD(
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
            **kwargs,
        )

    def _stress(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        dWdF = tr.function(self.fun, wrt=0, ntrax=2, parallel=self.parallel)(
            F, *statevars, **kwargs
        )
        if self.nstatevars > 0:
            statevars_new = tr.function(
                self.fun_statevars, wrt=0, ntrax=2, parallel=self.parallel
            )(F, *statevars, **kwargs)
        else:
            statevars_new = None
        return [dWdF, statevars_new]

    def _elasticity(self, x, **kwargs):
        F = np.ascontiguousarray(x[0])

        if self.nstatevars > 0:
            statevars = (x[1],)
        else:
            statevars = ()

        d2WdFdF = tr.jacobian(self.fun, wrt=0, ntrax=2, parallel=self.parallel)(
            F, *statevars, **kwargs
        )
        return [d2WdFdF]
