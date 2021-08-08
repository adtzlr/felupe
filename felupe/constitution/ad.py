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

from functools import partial
import numpy as np
import casadi as ca


def apply(x, fun, fun_shape, trailing_axes=2):
    "Helper function for the calculation of fun(x)."

    # get shape of trailing axes
    ax = x[0].shape[-trailing_axes:]

    def rshape(z):
        if len(z.shape) == trailing_axes:
            return z.reshape(1, -1, order="F")
        else:
            return z.reshape(z.shape[0], -1, order="F")

    # reshape input
    y = [rshape(z) for z in x]

    # map function to reshaped input
    out = np.array(fun.map(np.product(ax))(*y))

    return out.reshape(*fun_shape, *ax, order="F")


class Material:
    def __init__(self, W, *args, **kwargs):
        "Material class (u) with Automatic Differentiation."

        # init deformation gradient
        F = ca.SX.sym("F", 3, 3)

        # gradient and hessian of strain ernergy function W
        d2WdFdF, dWdF = ca.hessian(W(F, *args, **kwargs), F)
        # dWdF = ca.jacobian(W(F, *args, **kwargs), F)
        # d2WdFdF = ca.jacobian(dWdF, F)

        # generate casadi function objects
        self._f_P = ca.Function("P", [F], [dWdF])
        self._f_A = ca.Function("A", [F], [d2WdFdF])

        self.P = self.f

    def _modify(self, F, eps=1e-5):
        G = F.copy()
        G[0, 0] += eps
        G[1, 1] -= eps
        return G

    # functions for stress P and elasticity A
    def f(self, F, modify=True):
        if modify:
            F = self._modify(F)
        fF = apply([F], fun=self._f_P, fun_shape=(3, 3))
        return fF

    def A(self, F, modify=True):
        if modify:
            F = self._modify(F)
        AFF = apply([F], fun=self._f_A, fun_shape=(3, 3, 3, 3))
        return AFF


class Materialup:
    def __init__(self, W, *args, **kwargs):
        "Material class (u,p) with Automatic Differentiation."

        # init deformation gradient
        F = ca.SX.sym("F", 3, 3)
        p = ca.SX.sym("p", 1)

        # gradient and hessian of strain ernergy function W
        w = W(F, p, *args, **kwargs)

        d2WdF2, dWdF = ca.hessian(w, F)
        d2Wdp2, dWdp = ca.hessian(w, p)

        d2WdFdp = ca.jacobian(dWdF, p)

        # generate casadi function objects
        self._f_PF = ca.Function("PF", [F, p], [dWdF])
        self._f_Pp = ca.Function("Pp", [F, p], [dWdp])

        self._f_AFF = ca.Function("AFF", [F, p], [d2WdF2])
        self._f_App = ca.Function("App", [F, p], [d2Wdp2])
        self._f_AFp = ca.Function("AFp", [F, p], [d2WdFdp])

    def _modify(self, F, eps=1e-5):
        G = F.copy()
        G[0, 0] += eps
        G[1, 1] -= eps
        return G

    # functions for stress P and elasticity A
    def f(self, F, p, modify=True):
        if modify:
            F = self._modify(F)
        fF = apply([F, p], fun=self._f_PF, fun_shape=(3, 3))
        fp = apply([F, p], fun=self._f_Pp, fun_shape=())
        return [fF, fp]

    def A(self, F, p, modify=True):
        if modify:
            F = self._modify(F)
        AFF = apply([F, p], fun=self._f_AFF, fun_shape=(3, 3, 3, 3))
        App = apply([F, p], fun=self._f_App, fun_shape=(1,))
        AFp = apply([F, p], fun=self._f_AFp, fun_shape=(3, 3))
        return [AFF, AFp, App]


class MaterialupJ:
    def __init__(self, W, *args, **kwargs):
        "Material class (u,p,J) with Automatic Differentiation."

        # init deformation gradient
        F = ca.SX.sym("F", 3, 3)
        p = ca.SX.sym("p", 1)
        J = ca.SX.sym("J", 1)

        # gradient and hessian of strain ernergy function W
        w = W(F, p, J, *args, **kwargs)

        d2WdF2, dWdF = ca.hessian(w, F)
        d2Wdp2, dWdp = ca.hessian(w, p)
        d2WdJ2, dWdJ = ca.hessian(w, J)

        d2WdFdp = ca.jacobian(dWdF, p)
        d2WdFdJ = ca.jacobian(dWdF, J)
        d2WdpdJ = ca.jacobian(dWdp, J)

        # generate casadi function objects
        self._f_PF = ca.Function("PF", [F, p, J], [dWdF])
        self._f_Pp = ca.Function("Pp", [F, p, J], [dWdp])
        self._f_PJ = ca.Function("PJ", [F, p, J], [dWdJ])

        self._f_AFF = ca.Function("AFF", [F, p, J], [d2WdF2])
        self._f_App = ca.Function("App", [F, p, J], [d2Wdp2])
        self._f_AJJ = ca.Function("AJJ", [F, p, J], [d2WdJ2])
        self._f_AFp = ca.Function("AFp", [F, p, J], [d2WdFdp])
        self._f_AFJ = ca.Function("AFp", [F, p, J], [d2WdFdJ])
        self._f_ApJ = ca.Function("ApJ", [F, p, J], [d2WdpdJ])

    def _modify(self, F, eps=1e-5):
        G = F.copy()
        G[0, 0] += eps
        G[1, 1] -= eps
        return G

    # functions for stress P and elasticity A
    def f(self, F, p, J, modify=True):
        if modify:
            F = self._modify(F)
        fF = apply([F, p, J], fun=self._f_PF, fun_shape=(3, 3))
        fp = apply([F, p, J], fun=self._f_Pp, fun_shape=(1,))
        fJ = apply([F, p, J], fun=self._f_PJ, fun_shape=(1,))
        return [fF, fp, fJ]

    def A(self, F, p, J, modify=True):
        if modify:
            F = self._modify(F)
        AFF = apply([F, p, J], fun=self._f_AFF, fun_shape=(3, 3, 3, 3))
        App = apply([F, p, J], fun=self._f_App, fun_shape=(1,))
        AJJ = apply([F, p, J], fun=self._f_AJJ, fun_shape=(1,))
        AFp = apply([F, p, J], fun=self._f_AFp, fun_shape=(3, 3))
        AFJ = apply([F, p, J], fun=self._f_AFJ, fun_shape=(3, 3))
        ApJ = apply([F, p, J], fun=self._f_ApJ, fun_shape=(1,))
        return [AFF, AFp, AFJ, App, ApJ, AJJ]
