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


class Result:
    def __init__(self, x, y=None, fun=None, jac=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.y = y
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations


def newtonrhapson(
    fun,
    x0,
    jac,
    solve=np.linalg.solve,
    maxiter=8,
    pre=lambda x: x,
    update=lambda x, dx: x + dx,
    check=lambda dx, x, f: np.linalg.norm(dx) < np.finfo(float).eps,
    args=(),
    kwargs={},
    kwargs_solve={},
    kwargs_check={},
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

    # pre-evaluate function at given unknowns "x"
    f = fun(pre(x), *args, **kwargs)

    # iteration loop
    for iteration in range(maxiter):

        # evaluate jacobian at unknowns "x"
        K = jac(pre(x), *args, **kwargs)

        # solve linear system and update solution
        result = solve(K, -f, **kwargs_solve)
        x = update(x, result)

        # evaluate function at unknowns "x"
        f = fun(pre(x), *args, **kwargs)

        # check success of solution
        success = check(result, x, f, **kwargs_check)

        if success:
            break

    Res = Result(x=x, y=pre(x), fun=f, success=success, iterations=1 + iteration)

    if export_jac:
        Res.jac = K

    return Res
