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

from copy import deepcopy

from numpy.linalg import norm
import numpy as np

from .math import grad, identity, interpolate
from . import solve as solvetools


def defgrad_upJ(fields):
    """Calculate the deformation gradient of a given list of fields
    and evaluate all other field values at integration points."""
    dudX = grad(fields[0])
    F = identity(dudX) + dudX
    return F, *[interpolate(f) for f in fields[1:]]


def defgrad(field):
    "Calculate the deformation gradient of a given displacement field."
    dudX = grad(field)
    return identity(dudX) + dudX


def strain(field):
    "Calculate strains of a given displacement field."
    dudX = grad(field)
    dim = dudX.shape[1]

    eps_normal = np.array([dudX[i, i] for i in range(dim)])

    if dim > 1:
        eps_shear = [dudX[0, 1] + dudX[1, 0]]
    if dim > 2:
        eps_shear.append(dudX[1, 2] + dudX[2, 1])
        eps_shear.append(dudX[2, 0] + dudX[0, 2])

    if dim > 1:
        return np.concatenate((eps_normal, eps_shear), axis=0)

    else:
        return eps_normal


def update(x, dx):
    "Update field values."

    if isinstance(x, tuple) or isinstance(x, list):
        for x, dx in zip(x, dx):
            x += dx
    else:
        x += dx

    return x


def solve(K, f, field, dof0, dof1, ext):
    "Solve linear equation system K dx = b"
    system = solvetools.partition(field, K, dof1, dof0, -f)
    dx = solvetools.solve(*system, ext)
    return dx


def solve_mixed(K, f, fields, dof0, dof1, ext, unstack):
    "Solve linear equation system K dx = b"
    system = solvetools.partition(fields, K, dof1, dof0, -f)
    dfields = np.split(solvetools.solve(*system, ext), unstack)
    return dfields


def check(dx, x, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3):
    "Check if solution dx is valid."

    # get reference values of "f" and "x"
    ref_f = 1 if np.linalg.norm(f[dof0]) == 0 else np.linalg.norm(f[dof0])
    ref_x = 1 if np.linalg.norm(x[dof0]) == 0 else np.linalg.norm(x[dof0])

    norm_f = np.linalg.norm(f[dof1]) / ref_f
    norm_x = np.linalg.norm(dx.ravel()[dof1]) / ref_x

    if norm_f < tol_f and norm_x < tol_x:
        success = True
    else:
        success = False

    return success


class Result:
    def __init__(self, x, fun=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.fun = fun
        self.success = success
        self.iterations = iterations


def newtonrhapson(
    fun,
    x,
    jac,
    solve=np.linalg.solve,
    maxiter=8,
    pre=lambda x: x,
    update=lambda x, dx: x + dx,
    check=lambda dx, x, f: norm(dx) < np.finfo(float).eps,
    args=(),
    kwargs={},
    kwargs_solve={},
    kwargs_check={},
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

    return Result(deepcopy(x), f, success, 1 + iteration)
