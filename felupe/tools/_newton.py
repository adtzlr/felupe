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

import inspect

import numpy as np
from scipy.sparse.linalg import spsolve

from ..math import norm
from .._assembly import IntegralForm, IntegralFormMixed
from .. import solve as fesolve


class Result:
    def __init__(self, x, fun=None, jac=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations


def fun(x, umat, parallel=False):
    "Force residuals from assembly of equilibrium (weak form)."

    if "mixed" in str(type(x)):

        L = IntegralFormMixed(
            fun=umat.gradient(x.extract()),
            v=x,
            dV=x[0].region.dV,
        )

    else:

        L = IntegralForm(
            fun=umat.gradient(x.extract()), v=x, dV=x.region.dV, grad_v=True
        )

    return L.assemble(parallel=parallel).toarray()[:, 0]


def jac(x, umat, parallel=False):
    "Tangent stiffness matrix from assembly of linearized equilibrium."

    if "mixed" in str(type(x)):

        a = IntegralFormMixed(
            fun=umat.hessian(x.extract()),
            v=x,
            dV=x[0].region.dV,
            u=x,
        )

    else:

        a = IntegralForm(
            fun=umat.hessian(x.extract()),
            v=x,
            dV=x.region.dV,
            u=x,
            grad_v=True,
            grad_u=True,
        )

    return a.assemble(parallel=parallel)


def solve(A, b, x, dof1, dof0, ext0=None, solver=spsolve):
    "Solve partitioned system."

    system = fesolve.partition(x, A, dof1, dof0, -b)

    return fesolve.solve(*system, ext0, solver=solver)


def check(dx, x, f, tol):
    "Check result."
    return np.all(norm(dx) < tol)


def newtonrhapson(
    x0,
    fun=fun,
    jac=jac,
    solve=solve,
    maxiter=16,
    update=lambda x, dx: x + dx,
    check=check,
    args=(),
    kwargs={},
    kwargs_solve={},
    kwargs_check={},
    tol=np.sqrt(np.finfo(float).eps),
    umat=None,
    dof1=None,
    dof0=None,
    ext0=None,
    solver=spsolve,
    export_jac=False,
):
    """
    General-purpose Newton-Rhapson algorithm
    ========================================

    (Nonlinear) equilibrium equations `f`, as a function `f(x)` of the
    unknowns `x`, are solved by linearization of `f` at given unknowns `x0`.

        f(x0) = 0                                          (1)

        f(x0 + dx)     =  f(x0) + (df/dx)(x0) dx (= 0)     (2)
        (df/dx)(x0) dx = -f(x0)

        dx = solve(df/dx(x0), -f(x0))                      (3)

         x = x0 + dx                                       (4)

    Repeated evaluations of Eq.(3) and Eq.(4) lead to an incrementally updated
    solution of `x` which is shown in equation (4). Herein `xn` refer to the
    inital unknowns whereas `x` to the updated unknowns (the subscript `n+1`
    is dropped for readability).

        dx = solve(df/dx(xn), -f(xn))                      (5)

         x = xn + dx                                       (6)

    Eq.(5) and Eq.(6) are repeated until `check(dx, x, f)` returns `True`.

    """

    # copy x0
    x = x0
    # x = deepcopy(x0)

    if umat is not None:
        kwargs["umat"] = umat

    # pre-evaluate function at given unknowns "x"
    f = fun(x, *args, **kwargs)

    # iteration loop
    for iteration in range(maxiter):

        # evaluate jacobian at unknowns "x"
        K = jac(x, *args, **kwargs)

        # solve linear system and update solution
        sig = inspect.signature(solve)

        for key, value in zip(
            ["x", "dof1", "dof0", "ext0", "solver"], [x, dof1, dof0, ext0, solver]
        ):
            if key in sig.parameters:
                kwargs_solve[key] = value

        dx = solve(K, -f, **kwargs_solve)
        x = update(x, dx)

        # evaluate function at unknowns "x"
        f = fun(x, *args, **kwargs)

        # check success of solution
        success = check(dx, x, f, tol, **kwargs_check)

        if success:
            break

    Res = Result(x=x, fun=f, success=success, iterations=1 + iteration)

    if export_jac:
        Res.jac = K

    return Res
