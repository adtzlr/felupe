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
from ..assembly import IntegralForm
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
    "Assemble the sparse system vector for each item."

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
    "Assemble the sparse system matrix for each item."

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
    r"""Find a root of a real function using the Newton-Raphson method.

    Parameters
    ----------
    x0 : felupe.Field, ndarray or None (optional)
        Array or Field with values of unknowns at a valid starting point (default is
        None).
    fun : callable, optional
        Callable which assembles the vector-valued function. Additional args and kwargs
        are passed.
    jac : callable, optional
        Callable which assembles the matrix-valued Jacobian. Additional args and kwargs
        are passed.
    solve : callable, optional
        Callable which prepares (partitions) the linear equation system and solves it.
        If a keyword-argument from the list ``["x", "dof1", "dof0", "ext0", "solver"]``
        is found in the function-signature, then these arguments are passed to
        ``solve``.
    maxiter : int, optional
        Maximum number of function iterations (default is 16).
    update : callable, optional
        Callable to update the unknowns.
    check : callable, optional
        Callable to the check the result.
    tol : float, optional
        Tolerance value to check if the function has converged (default is 1.490e-8).
    items : list or None, optional
        List with items which provide methods for assembly, e.g. like felupe.SolidBody
        or felupe.Pressure (default is None).
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None).
    dof0 : ndarray or None, optional
        1d-array of int with all prescribed degress of freedom (default is None).
    ext0 : ndarray or None, optional
        Field values at mesh-points for the prescribed components of the unknowns based
        on ``dof0`` (default is None).
    solver : callable, optional
        A sparse or dense solver (default is scipy.sparse.linalg.spsolve).
    verbose : bool or int, optional
        Verbosity level: False or 0 for no output, True or 1 for a progress bar and
        2 for a text-based output.

    Returns
    -------
    felupe.tools.Result
        The result object.

    Notes
    -----
    Nonlinear equilibrium equations :math:`f(x)` as a function of the unknowns :math:`x`
    are solved by linearization of :math:`f` at a valid starting point of given unknowns
    :math:`x_0`.

    ..  math::

        f(x_0) = 0

    The linearization is given by

    ..  math::

        f(x_0 + dx) \approx f(x_0) + K(x_0) \ dx \ (= 0)

    with the Jacobian, evaluated at given unknowns :math:`x_0`,

    ..  math::

        K(x_0) = \frac{\partial f}{\partial x}(x_0)

    and is rearranged to an equation system with left- and right-hand sides.

    ..  math::

        K(x_0) \ dx = -f(x_0)

    After a solution is found, the unknowns are updated.

    ..  math::

        dx &= \text{solve} \left( K(x_0), -f(x_0) \right)

        x &= x_0 + dx

    Repeated evaluations lead to an incrementally updated solution of :math:`x`. Herein
    :math:`x_n` refer to the inital unknowns whereas :math:`x` are the updated unknowns
    (the subscript :math:`(\bullet)_{n+1}` is dropped for readability).

    ..  math::
        :name: eq:newton-solve

        dx &= \text{solve} \left( K(x_n), -f(x_n) \right)

         x &= x_n + dx

    Then, the nonlinear equilibrium equations are evaluated with the updated unknowns
    :math:`f(x)`. The procedure is repeated until convergence is reached.

    """

    if verbose:
        runtimes = [perf_counter()]
        soltimes = []

    if x0 is not None:
        x = x0

    else:
        x = items[0].field

    kwargs_solve = {}
    sig = inspect.signature(solve)

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

        if verbose:
            soltime_start = perf_counter()

        dx = solve(K, -f, **kwargs_solve)

        if verbose:
            soltime_end = perf_counter()
            soltimes.append([soltime_start, soltime_end])

        x = update(x, dx)

        if items is not None:
            f = fun_items(items, x, *args, **kwargs)
        else:
            f = fun(x, *args, **kwargs)

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
