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
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from ..math import norm
from .._assembly import IntegralFormMixed
from .. import solve as fesolve


class Result:
    def __init__(self, x, fun=None, jac=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations


def fun_bodies(bodies, x=None, parallel=False, jit=False):
    "Force residuals from assembly of equilibrium (weak form)."

    # init keyword arguments
    kwargs = {"parallel": parallel, "jit": jit}

    # assemble vector of first body
    vector = bodies[0].assemble.vector(field=x, **kwargs)

    # loop over other bodies
    for body in bodies[1:]:

        # assemble vector
        r = body.assemble.vector(field=x, **kwargs)

        # check and reshape vector
        if r.shape != vector.shape:
            r.resize(*vector.shape)

        # add vector
        vector += r

    return vector.toarray()[:, 0]


def jac_bodies(bodies, parallel=False, jit=False):
    "Tangent stiffness matrix from assembly of linearized equilibrium."

    # init keyword arguments
    kwargs = {"parallel": parallel, "jit": jit}

    # assemble matrix of first body
    matrix = bodies[0].assemble.matrix(**kwargs)

    # loop over other bodies
    for body in bodies[1:]:

        # assemble matrix
        K = body.assemble.matrix(**kwargs)

        # check and reshape matrix
        if K.shape != matrix.shape:
            K.resize(*matrix.shape)

        # add matrix
        matrix += K

    return matrix


def fun(x, umat, parallel=False, jit=False, grad=True, add_identity=True, sym=False):
    "Force residuals from assembly of equilibrium (weak form)."

    return (
        IntegralFormMixed(
            fun=umat.gradient(x.extract(grad=grad, add_identity=add_identity, sym=sym)),
            v=x,
            dV=x.region.dV,
        )
        .assemble(parallel=parallel, jit=jit)
        .toarray()[:, 0]
    )


def jac(x, umat, parallel=False, jit=False, grad=True, add_identity=True, sym=False):
    "Tangent stiffness matrix from assembly of linearized equilibrium."

    return IntegralFormMixed(
        fun=umat.hessian(x.extract(grad=grad, add_identity=add_identity, sym=sym)),
        v=x,
        dV=x.region.dV,
        u=x,
    ).assemble(parallel=parallel, jit=jit)


def solve(A, b, x, dof1, dof0, offsets=None, ext0=None, solver=spsolve):
    "Solve partitioned system."

    system = fesolve.partition(x, A, dof1, dof0, -b)
    dx = fesolve.solve(*system, ext0, solver=solver)

    return dx


def check(dx, x, f, tol):
    "Check result."
    return np.sum(norm(dx)), np.all(norm(dx) < tol)


def newtonrhapson(
    x0=None,
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
    bodies=None,
    dof1=None,
    dof0=None,
    ext0=None,
    solver=spsolve,
    export_jac=False,
    verbose=True,
    timing=True,
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

    if timing:
        time_start = perf_counter()

    if x0 is not None:
        # copy x0
        x = x0
    else:
        # copy field of body
        x = bodies[0].field

    if umat is not None:
        kwargs["umat"] = umat

    # pre-evaluate function at given unknowns "x"
    if bodies is not None:
        f = fun_bodies(bodies, x, *args, **kwargs)
    else:
        f = fun(x, *args, **kwargs)

    if verbose:
        print()
        print("Newton-Rhapson solver")
        print("=====================")
        print()
        print("| # |  norm(dx) |")
        print("|---|-----------|")

    # iteration loop
    for iteration in range(maxiter):

        # evaluate jacobian at unknowns "x"
        if bodies is not None:
            K = jac_bodies(bodies, *args, **kwargs)
        else:
            K = jac(x, *args, **kwargs)

        # solve linear system and update solution
        sig = inspect.signature(solve)

        keys = ["x", "dof1", "dof0", "ext0", "solver"]
        values = [x, dof1, dof0, ext0, solver]

        for key, value in zip(keys, values):

            if key in sig.parameters:
                kwargs_solve[key] = value

        dx = solve(K, -f, **kwargs_solve)
        x = update(x, dx)

        # evaluate function at unknowns "x"
        if bodies is not None:
            f = fun_bodies(bodies, x, *args, **kwargs)
        else:
            f = fun(x, *args, **kwargs)

        # check success of solution
        norm, success = check(dx, x, f, tol, **kwargs_check)

        if verbose:
            print("|%2d | %1.3e |" % (1 + iteration, norm))

        if success:
            if not timing:
                print("\nSolution converged in %d iterations.\n" % (iteration + 1))
            break

        if np.isnan(norm):
            raise ValueError("Norm of unknowns is NaN.")

    if 1 + iteration == maxiter and not success:
        raise ValueError("Maximum number of iterations reached (not converged).\n")

    Res = Result(x=x, fun=f, success=success, iterations=1 + iteration)

    if export_jac:
        Res.jac = K

    if timing:
        time_finish = perf_counter()
        print(
            "\nSolution converged in %d iterations within %1.4g seconds.\n"
            % (iteration + 1, time_finish - time_start)
        )

    return Res
