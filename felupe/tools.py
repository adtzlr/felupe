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

from numpy.linalg import norm
import numpy as np

from .math import grad, identity


def defgrad(displacement):
    "Calculate the deformation gradient of a given displacement field."
    dudX = grad(displacement)
    return identity(dudX) + dudX


class Result:
    def __init__(
        self, x, f=None, K=None, converged=None, iterations=None, ref_x=None, ref_f=None
    ):
        "Result class."
        self.x = x
        self.f = f
        self.K = K
        self.converged = converged
        self.iterations = iterations
        self.ref_f = ref_f
        self.ref_x = ref_x


def newtonrhapson(
    fun,
    x,
    jac,
    solve=np.linalg.solve,
    maxiter=8,
    norm=norm,
    tol_f=np.inf,
    tol_x=1e-2,
    dof0=slice(None, None, None),
    dof1=slice(None, None, None),
    pre=lambda x: x,
    post=lambda x: x,
    args=(),
    kwargs={},
):
    "General-purpose Newton-Rhapson algorithm."

    # get pre-processed initial unknowns "y" from "x0"
    y = pre(x, *args, **kwargs)

    # evaluate function and jacobian at initial state "y"
    f = fun(y, *args, **kwargs)
    K = jac(y, *args, **kwargs)

    # init converged flag
    converged = False

    # iteration loop
    for iteration in range(maxiter):

        # solve linear system and update solution
        dx = solve(K, -f)
        x += dx

        # get pre-processed updated unknowns "y" from "x"
        y = pre(x, *args, **kwargs)

        # evaluate function and jacobian at updated state "y"
        f = fun(y, *args, **kwargs)

        # postprocess function "f" to "g"
        g = post(f, *args, **kwargs)

        # get reference values of "f" and "x"
        ref_f = norm(f[dof0])
        ref_x = norm(x[dof0])

        if ref_f == 0:
            ref_f = 1

        if ref_x == 0:
            ref_x = 1

        norm_f = norm(f[dof1]) / ref_f
        norm_x = norm(dx.ravel()[dof1]) / ref_x

        if norm_f < tol_f and norm_x < tol_x:
            converged = True
            break

        else:
            K = jac(y, *args, **kwargs)

    return Result(x, g, K, converged, 1 + iteration, ref_x, ref_f)
