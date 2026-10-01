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
from ._event_dispatcher import Context, EventDispatcher


class IterationState:
    r"""A class to keep track of the state of an iteration during evaluation.

    Parameters
    ----------
    iteration : int or None, optional
        The (zero-based) index of the current iteration (default is None).
    fnorm : float or None, optional
        The norm of the objective function of the current iteration (default is None).
    xnorm : float or None, optional
        The norm of the increment of the current iteration (default is None).
    success : bool or None, optional
        A flag if the current iteration has converged (default is None).
    tol : float or None, optional
        The tolerance of the Newton-Raphson method (default is None).
    error : bool or None, optional
        A flag if an error occured in the current iteration (default is None).
    x : felupe.FieldContainer or ndarray or None, optional
        The unknowns (default is None). Before the update of an iteration, these are
        the unknowns of the last accepted iteration :math:`x_n`. After the update, these
        are the updated unknowns :math:`x`.
    dx : ndarray or None, optional
        The increment of the unknowns of the current iteration (default is None).
    fun : ndarray or None, optional
        The values of the objective function (the residuals), evaluated at the unknowns
        ``x`` (default is None).
    jac : sparse matrix or ndarray or None, optional
        The Jacobian of the objective function of the current iteration, evaluated at
        the unknowns of the last accepted iteration :math:`x_n` (default is None).
    alpha : float or None, optional
        The step length which scales the Newton increment of the current iteration,
        e.g. of a line search (default is None). None means that the full Newton
        increment is applied.
    updated : bool, optional
        A flag if the unknowns (and the values of the objective function) are already
        updated in the current iteration (default is False).

    Notes
    -----
    One state is created by :func:`~felupe.newtonraphson` and its attributes are
    updated in-place during the iterations. Hence, a plugin may keep a reference to the
    state and attributes set by a plugin persist until they are updated by the
    Newton-Raphson method. The attributes ``alpha`` and ``updated`` are reset in the
    beginning of each iteration.

    ..  list-table:: Attributes of the state in the hooks of an iteration.
        :header-rows: 1

        * - Hook
          - ``x``
          - ``fun``
          - ``jac``
          - ``dx``
          - ``updated``
        * - ``before_iteration``
          - :math:`x_n`
          - :math:`f(x_n)`
          - previous
          - previous
          - False
        * - ``before_linear_solve``
          - :math:`x_n`
          - :math:`f(x_n)`
          - :math:`K(x_n)`
          - previous
          - False
        * - ``after_linear_solve``
          - :math:`x_n`
          - :math:`f(x_n)`
          - :math:`K(x_n)`
          - :math:`dx`
          - False
        * - ``after_iteration``
          - :math:`x`
          - :math:`f(x)`
          - :math:`K(x_n)`
          - :math:`dx`
          - True

    A plugin may modify the update of the unknowns in the ``after_linear_solve`` hook.
    Either the increment ``dx`` is replaced, e.g. by a scaled increment, which is then
    used by the Newton-Raphson method to update the unknowns ``x = update(x_n, dx)`` and
    to evaluate the objective function ``f(x)``. Or the plugin performs the update on
    its own, e.g. by the callables ``update`` and ``fun`` of the
    :class:`~felupe.tools.Context`. Then, the plugin has to set the updated unknowns
    ``x``, the values of the objective function ``fun`` evaluated at the updated
    unknowns, the applied increment ``dx`` and the flag ``updated=True``. The
    Newton-Raphson method re-uses the values of the objective function and no
    additional assembly is performed. If the objective function is evaluated for more
    than one trial, the accepted trial must be evaluated last, because the (temporary)
    state variables of the items belong to the last evaluation.

    ..  note::

        The evaluation of the objective function for a list of items links the fields
        of the items to the evaluated unknowns. If the unknowns of the last accepted
        iteration :math:`x_n` are the field container of an item (e.g. in the first
        iteration of a job), the value arrays of :math:`x_n` are replaced by the value
        arrays of the evaluated trial. A plugin which evaluates more than one trial
        must restore the value arrays of :math:`x_n` before the next trial is created.
        If a plugin does not perform the update, the value arrays of :math:`x_n` are
        restored by the Newton-Raphson method.

    See Also
    --------
    felupe.Plugin : Base class for plugins.
    felupe.tools.Context : A class to keep track of the context of a Job during
        evaluation.
    felupe.LinesearchPlugin : A backtracking line search for the Newton-Raphson method.
    """

    def __init__(
        self,
        iteration=None,
        fnorm=None,
        xnorm=None,
        success=None,
        tol=None,
        error=None,
        x=None,
        dx=None,
        fun=None,
        jac=None,
        alpha=None,
        updated=False,
    ):
        self.iteration = iteration
        self.fnorm = fnorm
        self.xnorm = xnorm
        self.success = success
        self.tol = tol
        self.error = error
        self.x = x
        self.dx = dx
        self.fun = fun
        self.jac = jac
        self.alpha = alpha
        self.updated = updated


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


def spresize(a, shape):
    """Increase (pad) the size of a sparse vector or matrix with zeros to a given shape.

    Parameters
    ----------
    a : csr_matrix
        Sparse vector or matrix to be resized.
    shape : tuple of int
        Target shape for the sparse vector or matrix.

    Returns
    -------
    csr_matrix
        Resized sparse vector or matrix with zeros added to match the target shape.

    Notes
    -----
    This function will increase the size of the sparse vector or matrix by adding zeros
    to match the target shape. It will raise a ValueError if the current shape exceeds
    the target shape.
    """

    if a.shape == shape:
        return a

    if any(ai > si for ai, si in zip(a.shape, shape)):
        raise ValueError(
            f"The assembled item has shape {a.shape}, which exceeds the global "
            f"shape {shape}. The item's field container has more degrees of "
            "freedom than the top-level field container."
        )

    a.resize(*shape)  # in-place
    return a


def _snapshot(x):
    """Return the value arrays of the fields of a field container (or None). The fields
    of the items are linked to the evaluated unknowns. If the unknowns are the field
    container of an item (e.g. in the first iteration of a job), the value arrays of
    the unknowns are replaced by the linked value arrays (in-place)."""
    fields = getattr(x, "fields", None)
    if fields is None:
        return None
    return [field.values for field in fields]


def _restore(x, snapshot):
    "Restore the value arrays of the fields of a field container."
    if snapshot is not None:
        for field, values in zip(x.fields, snapshot):
            field.values = values


def fun_items(items, x, parallel=False):
    "Assemble the sparse system vector for each item."

    # init keyword arguments
    kwargs = {"parallel": parallel}

    # link field of items with global field
    for item in items:
        item.field.link(x)

    # init vector with shape from global field
    shape = (np.sum(x.fieldsizes), 1)
    vector = csr_matrix(shape)

    for body in items:
        # assemble vector
        r = body.assemble.vector(field=body.field, **kwargs)

        if body.assemble.multiplier is not None:
            r *= body.assemble.multiplier

        # reshape vector
        r = spresize(r, shape)

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

        if body.assemble.multiplier is not None:
            K *= body.assemble.multiplier

        # reshape matrix
        K = spresize(K, shape)

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
            item.results.update_statevars()

    return xnorm, fnorm, success


def update(x, dx):
    "Update field."
    # x += dx # in-place
    return x + dx


def newtonraphson(
    x0=None,
    fun=fun,
    jac=jac,
    solve=solve,
    maxiter=16,
    update=update,
    check=check,
    args=(),
    kwargs=None,
    tol=np.sqrt(np.finfo(float).eps),
    items=None,
    dof1=None,
    dof0=None,
    ext0=None,
    solver=spsolve,
    verbose=None,
    callback=None,
    callback_kwargs=None,
    progress_bar=None,
    tqdm="tqdm",
    dispatcher=None,
    plugins=None,
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
        Callable to check the result with signature
        ``check = lambda xnorm, fnorm, success: dx, x, **kwargs``.
    tol : float, optional
        Tolerance value to check if the function has converged (default is 1.490e-8).
    items : list or None, optional
        List with items which provide methods for assembly, e.g. like
        :class:`felupe.SolidBody` (default is None).
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None).
    dof0 : ndarray or None, optional
        1d-array of int with all prescribed degrees of freedom (default is None).
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
    callback : callable or None, optional
        An optional callback function with function signature
        ``callback = lambda dx, x, iteration, xnorm, fnorm, success: None``, which is
        called after each completed iteration. Default is None.
    progress_bar : tqdm or None, optional
        Use an existing instance of a progress bar if verbose is True. If None and
        verbose is True, a new bar is created. Default is None.
    tqdm : str, optional
        If verbose is True, choose a backend for ``tqdm`` (``"tqdm"``, ``"auto"`` or
        ``"notebook"``). Default is ``"tqdm"``.
    dispatcher: EventDispatcher or None, optional
        An optional EventDispatcher to trigger events during evaluation. Default is
        None.
    plugins : list or None, optional
        A list of plugins with hooks to be used during evaluation, e.g. a
        :class:`~felupe.LinesearchPlugin`. If a ``dispatcher`` is given, the plugins
        are added to a copy of the dispatcher for this evaluation. Default is None.

    Returns
    -------
    felupe.tools.NewtonResult
        The result object.

    Notes
    -----
    Nonlinear equilibrium equations :math:`f(x)` as a function of the unknowns :math:`x`
    are solved by linearization of :math:`f` at a valid starting point of given unknowns
    :math:`x_0`, see Eq. :eq:`newton-x0`.

    ..  math::
        :label: newton-x0

        f(x_0) = 0

    The linearization is given in Eq. :eq:`newton-lin`

    ..  math::
        :label:  newton-lin

        f(x_0 + dx) \approx f(x_0) + K(x_0) \ dx \ (= 0)

    with the Jacobian as in Eq. :eq:`newton-jac`, evaluated at given unknowns
    :math:`x_0`,

    ..  math::
        :label: newton-jac

        K(x_0) = \frac{\partial f}{\partial x}(x_0)

    and is rearranged to an equation system with left- and right-hand sides, see Eq.
    :eq:`newton-lhs-rhs`.

    ..  math::
        :label: newton-lhs-rhs

        K(x_0) \ dx = -f(x_0)

    After a solution is found, the unknowns are updated, see Eq. :eq:`newton-update-0`.

    ..  math::
        :label: newton-update-0

        dx &= \text{solve} \left( K(x_0), -f(x_0) \right) \nonumber

        x &= x_0 + dx

    Repeated evaluations lead to an incrementally updated solution of :math:`x`. Herein,
    :math:`x_n` refer to the initial unknowns whereas :math:`x` are the updated unknowns,
    see Eq. :eq:`newton-update`.

    ..  note::
        The subscript :math:`(\bullet)_{n+1}` is dropped for easier readability.

    ..  math::
        :label: newton-update

        dx &= \text{solve} \left( K(x_n), -f(x_n) \right) \nonumber

         x &= x_n + dx

    Then, the nonlinear equilibrium equations are evaluated with the updated unknowns
    :math:`f(x)`. The procedure is repeated until convergence is reached.

    Plugins may modify the update of the unknowns in the ``after_linear_solve`` hook,
    see :class:`~felupe.tools.IterationState`. E.g., a backtracking line search
    (:class:`~felupe.LinesearchPlugin`) scales the Newton increment by a step length
    :math:`\alpha \in (0, 1]`, see Eq. :eq:`newton-update-alpha`.

    ..  math::
        :label: newton-update-alpha

        x = x_n + \alpha\ dx

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> region = fem.RegionHexahedron(fem.Cube(n=6))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(
    ...     field, move=0.2, clamped=True, return_loadcase=True
    ... )
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
    >>> res = fem.newtonraphson(items=[solid], **loadcase, verbose=2)  # doctest: +ELLIPSIS
     ...
    Newton-Raphson solver
    =====================
    <BLANKLINE>
    | # | norm(fun) |  norm(dx) |
    |---|-----------|-----------|
    | 1 | 7.553e-02 | 1.898e+00 |
    | 2 | 1.310e-03 | 5.091e-02 |
    | 3 | 3.086e-07 | 6.698e-04 |
    | 4 | ...e-14 | ...e-07 |
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
    if dispatcher is None:
        from ..plugins import ProgressPlugin

        progress = ProgressPlugin(verbose=verbose, tqdm=tqdm)
        dispatcher = EventDispatcher(plugins=[*(plugins or []), progress])

    elif plugins is not None:
        # don't modify the given dispatcher (e.g. of a job)
        dispatcher = EventDispatcher(plugins=[*dispatcher.plugins, *plugins])

    if x0 is not None:
        x = x0

    else:
        x = items[0].field

    kwargs_solve = {}
    sig = inspect.signature(solve)

    if kwargs is None:
        kwargs = {}

    def evaluate(x):
        "Evaluate the objective function for given unknowns."
        if items is not None:
            return fun_items(items, x, *args, **kwargs)
        return fun(x, *args, **kwargs)

    def converged(dx, x, f):
        "Check the convergence without side effects (no update of state variables)."
        return check(
            dx=dx, x=x, f=f, xtol=np.inf, ftol=tol, dof1=dof1, dof0=dof0, items=None
        )

    context = Context(
        items=items,
        dof1=dof1,
        dof0=dof0,
        ext0=ext0,
        fun=evaluate,
        update=update,
        check=converged,
    )

    # one state for all iterations, its attributes are updated in-place
    state = IterationState(tol=tol, x=x)
    dispatcher.trigger("before_newton", context, state)

    f = evaluate(x)
    state.fun = f

    xnorms, fnorms = [], []

    # iteration loop
    for iteration in range(maxiter):

        state.iteration = iteration
        state.alpha = None
        state.updated = False

        dispatcher.trigger("before_iteration", context, state)

        if items is not None:
            K = jac_items(items, x, *args, **kwargs)
        else:
            K = jac(x, *args, **kwargs)

        state.jac = K

        # create keyword-arguments for solving the linear system
        keys = ["x", "dof1", "dof0", "ext0", "solver"]
        values = [x, dof1, dof0, ext0, solver]

        for key, value in zip(keys, values):
            if key in sig.parameters:
                kwargs_solve[key] = value

        dispatcher.trigger("before_linear_solve", context, state)

        dx = solve(K, -f, **kwargs_solve)

        if np.any(np.isnan(dx)):
            raise ValueError(
                "Solution contains NaN values. Newton-Raphson method failed."
            )

        state.dx = dx

        # plugins may replace the increment or perform the update on their own
        snapshot = _snapshot(x)
        dispatcher.trigger("after_linear_solve", context, state)

        dx = state.dx

        if state.updated:
            # re-use the updated unknowns and the objective function of a plugin
            x, f = state.x, state.fun
        else:
            # the unknowns may be linked to a trial, evaluated by a plugin
            _restore(x, snapshot)
            x = update(x, dx)
            f = evaluate(x)

        xnorm, fnorm, success = check(
            dx=dx, x=x, f=f, xtol=np.inf, ftol=tol, dof1=dof1, dof0=dof0, items=items
        )
        xnorms.append(xnorm)
        fnorms.append(fnorm)

        if callback is not None:
            if callback_kwargs is None:
                callback_kwargs = {}

            callback(dx, x, iteration, xnorm, fnorm, success)

        isnan = np.any(np.isnan([xnorm, fnorm]))
        abort = 1 + iteration == maxiter and not success
        error = isnan or abort

        state.x = x
        state.fun = f
        state.updated = True
        state.xnorm = xnorm
        state.fnorm = fnorm
        state.success = success
        state.error = error

        dispatcher.trigger("after_iteration", context, state)

        if success:
            break

    if abort:
        raise ValueError("Maximum number of iterations reached (not converged).\n")

    Res = NewtonResult(
        x=x,
        fun=f,
        success=success,
        iterations=1 + iteration,
        xnorms=xnorms,
        fnorms=fnorms,
    )

    dispatcher.trigger("after_newton", context, state)

    return Res
