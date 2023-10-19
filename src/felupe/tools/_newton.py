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

import inspect
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .. import solve as fesolve
from .._assembly import IntegralForm
from ..math import norm


class Result:
    def __init__(self, x, fun=None, jac=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations


def fun_items(items, x, parallel=False):
    "Force residuals from assembly of equilibrium (weak form)."

    # init keyword arguments
    kwargs = {"parallel": parallel}

    # link field of items with global field
    [item.field.link(x) for item in items]

    # init vector with shape from global field
    shape = (np.sum(x.fieldsizes), 1)
    vector = csr_matrix(shape)

    for body in items:
        # assemble vector
        r = body.assemble.vector(field=body.field, **kwargs)

        # check and reshape vector
        if r.shape != shape:
            r.resize(*shape)

        # add vector
        vector += r

    return vector.toarray()[:, 0]


def jac_items(items, x, parallel=False):
    "Tangent stiffness matrix from assembly of linearized equilibrium."

    # init keyword arguments
    kwargs = {"parallel": parallel}

    # init matrix with shape from global field
    shape = (np.sum(x.fieldsizes), np.sum(x.fieldsizes))
    matrix = csr_matrix(shape)

    for body in items:
        # assemble matrix
        K = body.assemble.matrix(**kwargs)

        # check and reshape matrix
        if K.shape != matrix.shape:
            K.resize(*shape)

        # add matrix
        matrix += K

    return matrix


def fun(x, umat, parallel=False, grad=True, add_identity=True, sym=False):
    "Force residuals from assembly of equilibrium (weak form)."

    return (
        IntegralForm(
            fun=umat.gradient(x.extract(grad=grad, add_identity=add_identity, sym=sym))[
                :-1
            ],
            v=x,
            dV=x.region.dV,
        )
        .assemble(parallel=parallel)
        .toarray()[:, 0]
    )


def jac(x, umat, parallel=False, grad=True, add_identity=True, sym=False):
    "Tangent stiffness matrix from assembly of linearized equilibrium."

    return IntegralForm(
        fun=umat.hessian(x.extract(grad=grad, add_identity=add_identity, sym=sym)),
        v=x,
        dV=x.region.dV,
        u=x,
    ).assemble(parallel=parallel)


def solve(A, b, x, dof1, dof0, offsets=None, ext0=None, solver=spsolve):
    "Solve partitioned system."

    system = fesolve.partition(x, A, dof1, dof0, -b)
    dx = fesolve.solve(*system, ext0, solver=solver)

    return dx


def check(dx, x, f, xtol, ftol, dof1=None, dof0=None, items=None, eps=1e-3):
    "Check result."

    def sumnorm(x):
        return np.sum(norm(x))

    xnorm = sumnorm(dx)

    if dof1 is None:
        dof1 = slice(None)

    if dof0 is None:
        dof0 = slice(0, 0)

    fnorm = sumnorm(f[dof1]) / (eps + sumnorm(f[dof0]))
    success = fnorm < ftol and xnorm < xtol

    if success and items is not None:
        for item in items:
            [item.results.update_statevars() for item in items]

    return xnorm, fnorm, success


def update(x, dx):
    "Update field."
    # x += dx # in-place
    return x + dx


def newtonrhapson(
    x0=None,
    fun=fun,
    jac=jac,
    solve=solve,
    maxiter=16,
    update=update,
    check=check,
    args=(),
    kwargs={},
    tol=np.sqrt(np.finfo(float).eps),
    items=None,
    dof1=None,
    dof0=None,
    ext0=None,
    solver=spsolve,
    verbose=True,
):
    """
    General-purpose Newton-Rhapson algorithm

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

    if verbose:
        runtimes = [perf_counter()]
        soltimes = []

    if x0 is not None:
        # take x0
        x = x0

    else:
        # obtain field of first body
        x = items[0].field

    kwargs_solve = {}
    sig = inspect.signature(solve)

    # pre-evaluate function at given unknowns "x"
    if items is not None:
        f = fun_items(items, x, *args, **kwargs)
    else:
        f = fun(x, *args, **kwargs)

    if verbose:
        print()
        print("Newton-Rhapson solver")
        print("=====================")
        print()
        print("| # | norm(fun) |  norm(dx) |")
        print("|---|-----------|-----------|")

    # iteration loop
    for iteration in range(maxiter):
        # evaluate jacobian at unknowns "x"
        if items is not None:
            K = jac_items(items, x, *args, **kwargs)
        else:
            K = jac(x, *args, **kwargs)

        # create keyword-arguments for solving the linear system
        keys = ["x", "dof1", "dof0", "ext0", "solver"]
        values = [x, dof1, dof0, ext0, solver]

        for key, value in zip(keys, values):
            if key in sig.parameters:
                kwargs_solve[key] = value

        # solve linear system and update solution
        if verbose:
            soltime_start = perf_counter()

        dx = solve(K, -f, **kwargs_solve)

        if verbose:
            soltime_end = perf_counter()
            soltimes.append([soltime_start, soltime_end])

        x = update(x, dx)

        # evaluate function at unknowns "x"
        if items is not None:
            f = fun_items(items, x, *args, **kwargs)
        else:
            f = fun(x, *args, **kwargs)

        # check success of solution
        xnorm, fnorm, success = check(
            dx=dx, x=x, f=f, xtol=np.inf, ftol=tol, dof1=dof1, dof0=dof0, items=items
        )

        if verbose:
            print("|%2d | %1.3e | %1.3e |" % (1 + iteration, fnorm, xnorm))

        if success:
            break

        if np.any(np.isnan([xnorm, fnorm])):
            raise ValueError("Norm of unknowns is NaN.")

    if 1 + iteration == maxiter and not success:
        raise ValueError("Maximum number of iterations reached (not converged).\n")

    Res = Result(x=x, fun=f, success=success, iterations=1 + iteration)

    if verbose:
        runtimes.append(perf_counter())
        runtime = np.diff(runtimes)[0]
        soltime = np.diff(soltimes).sum()
        print(
            "\nConverged in %d iterations (Assembly: %1.4g s, Solve: %1.4g s).\n"
            % (iteration + 1, runtime - soltime, soltime)
        )

    return Res
