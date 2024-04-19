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
import os
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from .. import solve as fesolve
from ..assembly import IntegralForm
from ..math import norm


class NewtonResult:
    r"""A data class which represents the result found by Newton's method. All
    parameters are available as attributes.

    Parameters
    ----------
    x : felupe.FieldContainer or ndarray
        Array or Field container with values at a solution found by Newton's method.
    fun : ndarray or None, optional
        Values of objective function (default is None).
    jac : ndarray or None, optional
        Values of the Jacobian of the objective function (default is None).
    success : bool or None, optional
        A boolean flag which is True if the solution converged (default is None).
    iterations : int or None, optional
        Number of iterations until solution converged (default is None).
    xnorms : array of float or None, optional
        List with norms of the values of the solution (default is None).
    fnorms : float or None, optional
        List with norms of the objective function (default is None).

    Notes
    -----
    The objective function's norm is relative based on the function values on the
    prescribed degrees of freedom :math:`(\bullet)_0`. A small number
    :math:`\varepsilon` is added to avoid numeric instabilities.

    ..  math::

        \text{norm}(\boldsymbol{f}) = \frac{||\boldsymbol{f}_1||}
            {\varepsilon + ||\boldsymbol{f}_0||}

    """

    def __init__(
        self,
        x,
        fun=None,
        jac=None,
        success=None,
        iterations=None,
        xnorms=None,
        fnorms=None,
    ):
        self.x = x
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations
        self.xnorms = xnorms
        self.fnorms = fnorms


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
    verbose=None,
):
    r"""Find a root of a real function using the Newton-Raphson method.

    Parameters
    ----------
    x0 : felupe.FieldContainer, ndarray or None (optional)
        Array or Field container with values of unknowns at a valid starting point
        (default is None).
    fun : callable, optional
        Callable which assembles the vector-valued objective function. Additional args
        and kwargs are passed. Function signature of a user-defined function has to be
        ``fun = lambda x, *args, **kwargs: f``.
    jac : callable, optional
        Callable which assembles the matrix-valued Jacobian. Additional args and kwargs
        are passed. Function signature of a user-defined Jacobian has to be
        ``jac = lambda x, *args, **kwargs: K``.
    solve : callable, optional
        Callable which prepares the linear equation system and solves it. If a keyword-
        argument from the list ``["x", "dof1", "dof0", "ext0", "solver"]`` is found in
        the function-signature, then these arguments are passed to ``solve``.
    maxiter : int, optional
        Maximum number of function iterations (default is 16).
    update : callable, optional
        Callable to update the unknowns. Function signature must be
        ``update = lambda x0, dx: x``.
    check : callable, optional
        Callable to the check the result.
    tol : float, optional
        Tolerance value to check if the function has converged (default is 1.490e-8).
    items : list or None, optional
        List with items which provide methods for assembly, e.g. like
        :class:`felupe.SolidBody` (default is None).
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None).
    dof0 : ndarray or None, optional
        1d-array of int with all prescribed degress of freedom (default is None).
    ext0 : ndarray or None, optional
        Field values at mesh-points for the prescribed components of the unknowns based
        on ``dof0`` (default is None).
    solver : callable, optional
        A sparse or dense solver (default is :func:`scipy.sparse.linalg.spsolve`). For a
        more performant alternative install PyPardiso and use :func:`pypardiso.spsolve`.
    verbose : bool or int or None, optional
        Verbosity level to control how messages are printed during evaluation. If
        1 or True and ``tqdm`` is installed, a progress bar is shown. If ``tqdm`` is
        missing or verbose is 2, more detailed text-based messages are printed.
        Default is None. If None, verbosity is set to True. If None and the
        environmental variable FELUPE_VERBOSE is set and its value is not ``true``,
        then logging is turned off.

    Returns
    -------
    felupe.tools.NewtonResult
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

    Examples
    --------
    >>> import felupe as fem

    >>> region = fem.RegionHexahedron(fem.Cube(n=6))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(field, move=0.2, clamped=True)
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
    >>> res = fem.newtonrhapson(items=[solid], **loadcase)  # doctest: +ELLIPSIS
    <BLANKLINE>
    Newton-Rhapson solver
    =====================
    <BLANKLINE>
    | # | norm(fun) |  norm(dx) |
    |---|-----------|-----------|
    | 1 | 7.553e-02 | 1.898e+00 |
    | 2 | 1.310e-03 | 5.091e-02 |
    | 3 | 3.086e-07 | 6.698e-04 |
    | 4 | 2.255e-14 | 1.527e-07 |
    <BLANKLINE>
    Converged in 4 iterations ...
    <BLANKLINE>

    Newton's method had success

    >>> res.success
    True

    and 4 iterations were needed to converge within the specified tolerance.

    >>> res.iterations
    4

    The norm of the objective function for all active degrees of freedom is lower than
    3e-15.

    >>> np.linalg.norm(res.fun[loadcase["dof1"]])
    2.7384964752762237e-15

    """
    if verbose is None:
        FELUPE_VERBOSE = os.environ.get("FELUPE_VERBOSE")
        if FELUPE_VERBOSE is None:
            verbose = True
        else:
            verbose = FELUPE_VERBOSE == "true"

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

    xnorms, fnorms = [], []

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
        xnorms.append(xnorm)
        fnorms.append(fnorm)

        if verbose:
            print("|%2d | %1.3e | %1.3e |" % (1 + iteration, fnorm, xnorm))

        if success:
            break

        if np.any(np.isnan([xnorm, fnorm])):
            raise ValueError("Norm of unknowns is NaN.")

    if 1 + iteration == maxiter and not success:
        raise ValueError("Maximum number of iterations reached (not converged).\n")

    Res = NewtonResult(
        x=x,
        fun=f,
        success=success,
        iterations=1 + iteration,
        xnorms=xnorms,
        fnorms=fnorms,
    )

    if verbose:
        runtimes.append(perf_counter())
        runtime = np.diff(runtimes)[0]
        soltime = np.diff(soltimes).sum()
        print(
            "\nConverged in %d iterations (Assembly: %1.4g s, Solve: %1.4g s).\n"
            % (iteration + 1, runtime - soltime, soltime)
        )

    return Res
