# SPDX-License-Identifier: GPL-3.0-or-later

import inspect
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix

from .. import solve as fesolve
from ..assembly import IntegralForm
from ..field import FieldContainer
from ..field._container import shared_geometry_memo
from ..math import norm
from ._event_dispatcher import Context, EventDispatcher


class NewtonConvergenceError(ValueError):
    r"""An error which is raised if the Newton-Raphson method fails to converge.

    This error is raised by :func:`~felupe.newtonraphson` if the solution of the
    linearized equation system contains NaN values or if the maximum number of
    iterations is reached, by a :class:`~felupe.LinesearchPlugin` if no acceptable
    step length is found and by a :class:`~felupe.CutbackPlugin` if a failed substep
    is not recovered.

    Notes
    -----
    It is a subclass of :class:`ValueError`. Hence, existing code which catches a
    :class:`ValueError` is not affected. Plugins like the
    :class:`~felupe.CutbackPlugin` catch this error to recover a failed substep,
    while other errors, e.g. a :class:`ValueError` of mismatching shapes of arrays,
    are raised without a recovery.

    Examples
    --------
    ..  code-block::

        try:
            res = fem.newtonraphson(items=[solid], ...)
        except fem.NewtonConvergenceError:
            ...  # e.g. reduce the load increment
    """


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
        The tolerance of the Newton-Raphson method (default is None). If no tolerance
        is given to :func:`~felupe.newtonraphson`, the default tolerance depends on the
        data types of the linear equation system and of the unknowns. It is set after
        the assembly of the Jacobian in the first iteration, i.e. it is None in the
        hooks ``before_newton`` and ``before_iteration`` of the first iteration.
    atol : float or None, optional
        The absolute tolerance of the norm of the objective function on the active
        degrees of freedom of the current iteration (default is None). If no absolute
        tolerance is given to :func:`~felupe.newtonraphson`, it is the bound of the
        round-off errors of the objective function, which is evaluated by the check of
        the convergence of each iteration (for the updated unknowns), see
        :func:`~felupe.newtonraphson`. Then, it is None in the hooks ``before_newton``,
        ``before_iteration``, ``before_linear_solve`` and ``after_linear_solve`` of the
        first iteration.
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
    state variables of the items belong to the last evaluation. The reference norm of
    the check of the convergence (the forces between the items, see
    :func:`~felupe.newtonraphson`) is also taken from the last evaluation; it is
    only used for the values of the objective function which were returned by the
    last evaluation.

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
        atol=None,
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
        self.atol = atol
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
    The norms of the objective function are the norms of the objective function on the
    active degrees of freedom :math:`(\bullet)_1`, relative to the sum of a reference
    norm :math:`f_\text{ref}` and the ratio of the absolute and the relative
    tolerances, see :func:`~felupe.newtonraphson`.

    ..  math::

        \text{norm}(\boldsymbol{f}) = \frac{||\boldsymbol{f}_1||}
            {f_\text{ref} + \text{atol} / \text{tol}}

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


def embed(a, shape, field=None, x=None):
    """Embed an assembled sparse vector or matrix of an item into the shape of the
    global (top-level) field container.

    Parameters
    ----------
    a : sparse matrix
        Assembled sparse vector or matrix of an item, w.r.t. the degrees of freedom of
        the field container of the item.
    shape : tuple of int
        Target shape for the sparse vector or matrix.
    field : FieldContainer or None, optional
        The field container of the item (default is None).
    x : FieldContainer or None, optional
        The global (top-level) field container (default is None).

    Returns
    -------
    csr_matrix
        The sparse vector or matrix w.r.t. the global degrees of freedom.

    Notes
    -----
    If the field container of the item has ``slots`` (the indices of its fields in the
    global field container, e.g. set by :func:`~felupe.field.merge`), the degrees of
    freedom of the item are mapped to the global degrees of freedom. The sizes of the
    fields must be equal to the sizes of their fields in the global field container.
    Otherwise, the sparse vector or matrix is resized (padded with zeros) by
    :func:`spresize`, i.e. the fields of the item are the first fields of the global
    field container.
    """

    slots = getattr(field, "slots", None)

    if slots is None or x is None:
        return spresize(a, shape)

    sizes = [x.fieldsizes[slot] for slot in slots]

    if sizes != list(field.fieldsizes):
        raise ValueError(
            f"The fields of the item have the sizes {list(field.fieldsizes)}, which "
            f"don't match the sizes {sizes} of their fields {slots} in the global field "
            "container."
        )

    offsets = np.concatenate([[0], np.cumsum(x.fieldsizes)])
    dofs = np.concatenate(
        [offsets[slot] + np.arange(size) for slot, size in zip(slots, sizes)]
    )

    if any(ai > len(dofs) for ai in a.shape[: 1 if shape[1] == 1 else 2]):
        raise ValueError(
            f"The assembled item has shape {a.shape}, which exceeds the number of "
            f"degrees of freedom {len(dofs)} of the field container of the item."
        )

    a = a.tocoo()
    rows = dofs[a.row]
    cols = a.col if shape[1] == 1 else dofs[a.col]

    return csr_matrix((a.data, (rows, cols)), shape=shape)


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


def fun_items(items, x, parallel=False, return_reference=False):
    r"""Assemble the sparse system vector for each item and return the sum of the
    vectors of all items (the objective function).

    Parameters
    ----------
    items : list
        List with items which provide methods for assembly, e.g. like
        :class:`felupe.SolidBody`.
    x : felupe.FieldContainer
        The global (top-level) field container with the unknowns.
    parallel : bool, optional
        A flag to assemble the vectors in parallel (default is False).
    return_reference : bool, optional
        A flag to return the reference norm of the items in addition to the objective
        function (default is False).

    Returns
    -------
    ndarray or tuple of (ndarray, float)
        The objective function :math:`\boldsymbol{f} = \sum_i \boldsymbol{f}_i` and, if
        ``return_reference`` is True, the reference norm of the items
        :math:`\max_i ||\boldsymbol{f} - \boldsymbol{f}_i||`.

    Notes
    -----
    The reference norm of the items is the largest norm of the objective function
    without one of the items. This is the norm of the forces which the other items
    exert on the item, e.g. the forces of a :class:`~felupe.ContactRigidPlane` or of a
    :class:`~felupe.SolidBodyPressure` on a :class:`~felupe.SolidBody`. It is zero for a
    single item.
    """

    # init keyword arguments
    kwargs = {"parallel": parallel}

    # link field of items with global field
    for item in items:
        item.field.link(x)

    # init vector with shape from global field
    shape = (np.sum(x.fieldsizes), 1)

    # assume a matrix type based on the first field values
    dtype = x.fields[0].values.dtype
    vector = csr_matrix(shape, dtype=dtype)
    vectors = []

    for body in items:
        # assemble vector
        r = body.assemble.vector(field=body.field, **kwargs)

        if body.assemble.multiplier is not None:
            r *= body.assemble.multiplier

        # embed the vector into the global degrees of freedom
        r = embed(r, shape, body.field, x)

        # add vector
        vector += r

        if return_reference:
            vectors.append(r)

    f = vector.toarray()[:, 0]

    if return_reference:
        reference = 0.0

        if len(vectors) > 1:
            reference = max(
                float(np.linalg.norm(f - r.toarray()[:, 0])) for r in vectors
            )

        return f, reference

    return f


def jac_items(items, x, parallel=False):
    "Assemble the sparse system matrix for each item."

    # init keyword arguments
    kwargs = {"parallel": parallel}

    # init matrix with shape from global field
    shape = (np.sum(x.fieldsizes), np.sum(x.fieldsizes))

    # assume a matrix type based on the first field values
    dtype = x.fields[0].values.dtype
    matrix = csr_matrix(shape, dtype=dtype)

    for body in items:
        # assemble matrix
        K = body.assemble.matrix(**kwargs)

        if body.assemble.multiplier is not None:
            K *= body.assemble.multiplier

        # embed the matrix into the global degrees of freedom
        K = embed(K, shape, body.field, x)

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


def solve(A, b, x, dof1, dof0, offsets=None, ext0=None, solver=fesolve.SuperLUSolver()):
    "Solve partitioned system."

    system = fesolve.partition(x, A, dof1, dof0, -b)
    dx = fesolve.solve(*system, ext0, solver=solver)

    return dx


def check(
    dx, x, f, xtol, ftol, dof1=None, dof0=None, items=None, atol=0.0, reference=0.0
):
    r"""Check the convergence of the Newton-Raphson method (without side effects).

    Parameters
    ----------
    dx : ndarray
        The increment of the unknowns.
    x : felupe.FieldContainer or ndarray
        The updated unknowns (not used).
    f : ndarray
        The values of the objective function, evaluated at the updated unknowns.
    xtol : float
        The tolerance of the norm of the increment.
    ftol : float
        The relative tolerance of the norm of the objective function.
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None). If None,
        all degrees of freedom are active.
    dof0 : ndarray or None, optional
        1d-array of int with all prescribed degrees of freedom (default is None).
    items : list or None, optional
        A list of items (not used). The state variables of the items are updated by
        :func:`~felupe.newtonraphson` after a converged iteration, see
        :func:`update_statevars`.
    atol : float, optional
        The absolute tolerance of the norm of the objective function on the active
        degrees of freedom, in the units of the objective function (default is 0.0).
    reference : float, optional
        An additional reference norm of the objective function, e.g. the reference norm
        of the items, see :func:`fun_items` (default is 0.0).

    Returns
    -------
    xnorm : float
        The norm of the increment.
    fnorm : float
        The norm of the objective function on the active degrees of freedom, relative to
        the sum of the reference norm and the ratio of the absolute and the relative
        tolerances.
    success : bool
        A flag if the iteration has converged, i.e. if ``fnorm < ftol`` and
        ``xnorm < xtol``.

    Notes
    -----
    The reference norm :math:`f_\text{ref}` is the larger of the norm of the objective
    function on the prescribed degrees of freedom (the reaction forces) and the given
    reference norm. The iteration has converged if the norm of the objective function on
    the active degrees of freedom is lower than the absolute tolerance plus the relative
    tolerance times the reference norm.

    ..  math::

        \text{fnorm} = \frac{||\boldsymbol{f}_1||}
            {f_\text{ref} + \text{atol} / \text{ftol}} < \text{ftol}
            \quad \Leftrightarrow \quad
            ||\boldsymbol{f}_1|| < \text{atol} + \text{ftol}\ f_\text{ref}

    """

    def sumnorm(x):
        return np.sum(norm(x))

    xnorm = sumnorm(dx)

    if dof1 is None:
        dof1 = slice(None)

    if dof0 is None:
        dof0 = slice(0, 0)

    reference = max(sumnorm(f[dof0]), reference)
    denominator = reference + (atol / ftol if atol > 0 and ftol > 0 else 0.0)
    numerator = sumnorm(f[dof1])

    if denominator > 0:
        fnorm = numerator / denominator
    else:
        # an exact solution converges, also without a reference and a tolerance
        fnorm = 0.0 if numerator == 0 else np.inf

    success = bool(fnorm < ftol and xnorm < xtol)

    return xnorm, fnorm, success


def _dtype(a):
    "Return the data type of an array-like object."
    dtype = getattr(a, "dtype", None)
    return np.asarray(a).dtype if dtype is None else dtype


def _values(x):
    "Return a list with the value arrays of the unknowns, in the order of the dofs."
    fields = getattr(x, "fields", None)
    if fields is None:
        return [x]
    return [field.values for field in fields]


def _epsilon(A, b, x=None):
    """Return the machine epsilon of the lowest precision of the linear equation system
    with the Jacobian ``A`` and the objective function ``b`` and of the unknowns ``x``,
    see :func:`default_tol`."""

    system = np.result_type(_dtype(A), _dtype(b))

    if not np.issubdtype(system, np.inexact):
        system = np.dtype(float)

    dtypes = [system]

    if x is not None:
        values = _values(x)
        dtypes += [t for t in map(_dtype, values) if np.issubdtype(t, np.inexact)]

    return max(np.finfo(dtype).eps for dtype in dtypes)


def default_tol(A, b, x=None):
    r"""Return the default tolerance of the Newton-Raphson method for a linear equation
    system with the Jacobian ``A``, the objective function ``b`` and the unknowns
    ``x``.

    The tolerance is the square root of the machine epsilon of the lowest precision of
    the linear equation system and of the unknowns, e.g. ``1.490e-8`` for ``float64``
    and ``3.453e-4`` if the linear equation system or the unknowns are ``float32``. The
    precision of the linear equation system is given by the common data type of ``A``
    and ``b``. Data types without a machine epsilon, e.g. integers, use the tolerance
    of ``float64``. The precision of the unknowns is given by the data types of the
    values of all fields of a field container or by the data type of an array. Data
    types of the unknowns without a machine epsilon are ignored.
    """

    return float(np.sqrt(_epsilon(A, b, x)))


def _coordinates(x):
    """Return the coordinates of the points of the meshes for the fields with as many
    components as the dimension of their meshes, e.g. of displacements, and zeros for
    the other fields, in the order of the degrees of freedom (or None for an array)."""

    fields = getattr(x, "fields", None)

    if fields is None:
        return None

    coordinates = []

    for field in fields:
        mesh = getattr(getattr(field, "region", None), "mesh", None)
        points = getattr(mesh, "points", None)

        if points is not None and np.shape(points) == np.shape(field.values):
            coordinates.append(np.ravel(points))
        else:
            coordinates.append(np.zeros(np.size(field.values)))

    return np.concatenate(coordinates)


def default_atol(A, b, x, dof1=None, factor=10.0):
    r"""Return the default absolute tolerance of the Newton-Raphson method, a bound of
    the round-off errors of the objective function on the active degrees of freedom,
    for the Jacobian ``A``, the objective function ``b`` and the unknowns ``x``.

    Parameters
    ----------
    A : sparse matrix or ndarray
        The Jacobian of the objective function.
    b : ndarray
        The values of the objective function.
    x : felupe.FieldContainer or ndarray
        The unknowns.
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None). If None,
        all degrees of freedom are active.
    factor : float, optional
        A safety factor (default is 10).

    Returns
    -------
    float
        The bound of the round-off errors of the objective function on the active
        degrees of freedom, in the units of the objective function. If the absolute
        values of the Jacobian can't be evaluated, e.g. for a linear operator, it is
        zero.

    Notes
    -----
    The objective function, e.g. the internal forces of a solid body, is a sum of many
    contributions, which cancel each other out in a state of equilibrium. Its round-off
    errors are of the order of the machine epsilon :math:`\varepsilon` times the
    absolute values of the contributions. These are estimated by the products of the
    absolute values of the Jacobian and of the unknowns, which include the round-off
    errors of the gradients of the unknowns, e.g. of a rigid body motion, and by the
    product of the Jacobian and the coordinates :math:`\boldsymbol{X}` of the points
    of the meshes, which are the forces of a deformation gradient of
    :math:`\boldsymbol{F} = \boldsymbol{1} + \boldsymbol{1}`. The deformation gradient
    :math:`\boldsymbol{F} = \boldsymbol{1} + \partial \boldsymbol{u} / \partial
    \boldsymbol{X}` is represented with an absolute precision of
    :math:`\varepsilon\ \boldsymbol{1}`, also in a force-free state.

    ..  math::

        \text{atol} = c\ \left|\left| \left(
            \varepsilon_x\ |\boldsymbol{K}|\ |\boldsymbol{x}|
            + \varepsilon_K\ |\boldsymbol{K}\ \boldsymbol{X}|
            \right)_1 \right|\right|

    The coordinates are only taken for the fields with as many components as the
    dimension of their meshes, e.g. for displacements, and they are zero for other
    fields and for unknowns which are given as array. Each row of the products has the
    unit of the objective function, also for mixed field formulations. The machine
    epsilon :math:`\varepsilon_x` is the one of the lowest precision of the linear
    equation system and of the unknowns and :math:`\varepsilon_K` is the one of the
    precision of the linear equation system, in which the deformation gradient is
    evaluated, see :func:`default_tol`. E.g., for fields in single-precision on a
    region in double-precision, the deformation gradient is evaluated in
    double-precision. The safety factor is :math:`c=10`. Unlike a fixed absolute
    tolerance, this bound is invariant to the units of the model.

    If the Jacobian is not a matrix whose number of columns is the number of the
    unknowns, e.g. for the batched Jacobians of local problems at the quadrature points
    (see :func:`~felupe.math.solve_2d`), the absolute tolerance is the upper bound
    :math:`c\ \varepsilon_x\ ||\boldsymbol{K}||\ ||\boldsymbol{x}||` (with the
    Frobenius norm of the Jacobian) and the active degrees of freedom are not
    considered.
    """

    if dof1 is None:
        dof1 = slice(None)

    try:
        absA = abs(A)
    except TypeError:
        return 0.0

    absx = np.abs(np.concatenate([np.ravel(values) for values in _values(x)]))
    shape = getattr(A, "shape", np.shape(A))

    if len(shape) != 2 or shape[1] != absx.size:
        # e.g. batched Jacobians of local problems at the quadrature points, which
        # are solved by `math.solve_2d()`: an upper bound by the norms
        bound = np.linalg.norm(np.asarray(absA)) * np.linalg.norm(absx)
        return float(factor * _epsilon(A, b, x) * bound)

    # round-off errors of the gradients of the unknowns, e.g. of a rigid body motion,
    # in the lowest precision of the linear equation system and of the unknowns
    estimate = _epsilon(A, b, x) * np.ravel(np.asarray(absA @ absx))

    # round-off errors of the deformation gradient F = 1 + du/dX, which is evaluated
    # in the precision of the linear equation system (e.g. of the region)
    coordinates = _coordinates(x)

    if coordinates is not None:
        forces = np.ravel(np.asarray(A @ coordinates))
        estimate = estimate + _epsilon(A, b) * np.abs(forces)

    return float(factor * np.linalg.norm(estimate[dof1]))


def update_statevars(items):
    """Accept the trial state variables of the items for a converged solution.

    The trial state variables of the last evaluation of the items become the state
    variables of the converged solution, by ``item.results.update_statevars()``, see
    :class:`~felupe.mechanics.Item`.
    """

    for item in items:
        item.results.update_statevars()


def update(x, dx):
    "Update field."
    # x += dx # in-place

    if isinstance(x, FieldContainer):
        # the new field container shares the regions, meshes and indices with the old
        # one, only the field containers, the fields and their values are copied
        x = deepcopy(x, shared_geometry_memo(x))
        x += dx
        return x

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
    tol=None,
    atol=None,
    items=None,
    dof1=None,
    dof0=None,
    ext0=None,
    solver=fesolve.SuperLUSolver(),
    verbose=None,
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
        Callable to check the convergence with the signature
        ``xnorm, fnorm, success = check(dx, x, f, xtol, ftol, dof1, dof0, items, atol,
        reference)``. The state variables of the items are updated by the
        Newton-Raphson method after a converged iteration, i.e. a custom ``check``
        doesn't update them.

        ..  versionchanged:: 12.1.0
            The arguments ``atol`` and ``reference`` are passed to ``check``.

    tol : float or None, optional
        Relative tolerance value to check if the function has converged. If None, the
        tolerance is the square root of the machine epsilon of the lowest precision of
        the linear equation system (the common data type of the Jacobian and the
        objective function) and of the unknowns, e.g. ``1.490e-8`` for ``float64`` and
        ``3.453e-4`` if the linear equation system or the unknowns are ``float32``, see
        the notes below. Default is None.

        ..  versionchanged:: 12.0.0
            The default tolerance depends on the data types of the linear equation
            system and of the unknowns. Before, it was ``1.490e-8`` for all data types.

    atol : float or None, optional
        Absolute tolerance value of the norm of the objective function on the active
        degrees of freedom, in the units of the objective function, e.g. of forces. If
        None, it is a bound of the round-off errors of the objective function, which is
        evaluated in each iteration, see the notes below. Default is None.

        ..  versionadded:: 12.1.0

    items : list or None, optional
        List with items which provide methods for assembly, e.g. like
        :class:`felupe.SolidBody` (default is None). See
        :class:`~felupe.mechanics.Item` for the interface of an item.
    dof1 : ndarray or None, optional
        1d-array of int with all active degrees of freedom (default is None).
    dof0 : ndarray or None, optional
        1d-array of int with all prescribed degrees of freedom (default is None).
    ext0 : ndarray or None, optional
        Field values at mesh-points for the prescribed components of the unknowns based
        on ``dof0`` (default is None).
    solver : callable, optional
        A sparse or dense solver with the signature ``x = solver(A, b)`` (default is
        :class:`~felupe.solve.SuperLUSolver`). For a more performant alternative install
        PyPardiso and use :func:`pypardiso.spsolve`.
    verbose : bool or int or None, optional
        Verbosity level to control how messages are printed during evaluation. If
        1 or True and ``tqdm`` is installed, a progress bar is shown. If ``tqdm`` is
        missing or verbose is 2, more detailed text-based messages are printed.
        Default is None. If None, verbosity is set to True. If None and the
        environmental variable FELUPE_VERBOSE is set and its value is not ``true``,
        then logging is turned off.
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

        ..  versionchanged:: 12.0.0
            The arguments ``callback``, ``callback_kwargs`` and ``progress_bar`` are
            removed. A plugin with the hook ``after_iteration`` replaces the
            ``callback``, see the notes below.

    Returns
    -------
    felupe.tools.NewtonResult
        The result object.

    Raises
    ------
    NewtonConvergenceError
        If the solution of the linearized equation system contains NaN values or if
        the maximum number of iterations is reached (a subclass of
        :class:`ValueError`).

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
    :math:`f(x)`. The procedure is repeated until convergence is reached, i.e. until the
    norm of the objective function on the active degrees of freedom :math:`f_1(x)` is
    lower than the absolute tolerance plus the relative tolerance times a reference
    norm, see Eq. :eq:`newton-check`.

    ..  math::
        :label: newton-check

        ||f_1(x)|| < \text{atol} + \text{tol}\ f_\text{ref}

    The reference norm is the larger of the norm of the objective function on the
    prescribed degrees of freedom :math:`f_0(x)` (the reaction forces) and, for a list
    of items, the largest norm of the objective function without one of the items, see
    Eq. :eq:`newton-reference`, where :math:`f_i(x)` is the assembled vector of the
    item :math:`i`.

    ..  math::
        :label: newton-reference

        f_\text{ref} = \max \left( ||f_0(x)||, \max_i ||f(x) - f_i(x)|| \right)

    The norm of the objective function without one of the items is the norm of the
    forces which the other items exert on this item, e.g. the forces of a
    :class:`~felupe.ContactRigidPlane` or of a :class:`~felupe.SolidBodyPressure` on a
    :class:`~felupe.SolidBody`. Hence, the criterion is relative also for forces which
    are not reacted on the prescribed degrees of freedom, e.g. for a solid body which is
    supported by a contact or for a self-equilibrated pressure on a solid body with
    statically determinate supports. It is zero for a single item, i.e. the reference
    norm is the norm of the reaction forces.

    In a (nearly) force-free state, e.g. for a rigid body motion of a solid body before
    it gets in contact, the reference norm is zero up to round-off errors. Then, the
    absolute tolerance is used. The objective function is a sum of many contributions,
    e.g. of the cells of a solid body, which cancel each other out in a state of
    equilibrium. Its round-off errors are of the order of the machine epsilon
    :math:`\varepsilon` times the absolute values of these contributions, which depend
    on the units and on the stiffness of the model. If no absolute tolerance is given,
    it is a bound of the round-off errors of the objective function on the active
    degrees of freedom, see Eq. :eq:`newton-atol`, with a safety factor :math:`c=10`.
    The first product of the absolute values of the Jacobian of the iteration and of
    the updated unknowns, e.g. of the displacements, estimates the round-off errors of
    the gradients of the unknowns, e.g. of a rigid body motion. The second product of
    the Jacobian and the coordinates :math:`X` of the points of the meshes (only for
    the fields with as many components as the dimension of their meshes, e.g. for
    displacements) are the forces of a deformation gradient of
    :math:`\boldsymbol{F} = \boldsymbol{1} + \boldsymbol{1}`, i.e. it estimates the
    round-off errors of the deformation gradient
    :math:`\boldsymbol{F} = \boldsymbol{1} + \partial \boldsymbol{u} / \partial
    \boldsymbol{X}`, which is represented with an absolute precision of
    :math:`\varepsilon_K \boldsymbol{1}`, e.g. in the undeformed state. Each row of the
    products has the unit of the objective function, also for mixed field
    formulations.

    ..  math::
        :label: newton-atol

        \text{atol} = c \left|\left| \left(
            \varepsilon_x\ |K(x_n)|\ |x| + \varepsilon_K\ |K(x_n)\ X| \right)_1
            \right|\right|

    It is evaluated by the check of the convergence of each iteration and it is
    invariant to the units of the model. The machine epsilon :math:`\varepsilon_x` is
    the one of the lowest precision of the linear equation system and of the unknowns
    and :math:`\varepsilon_K` is the one of the precision of the linear equation
    system, see below. E.g., for fields in single-precision on a region in
    double-precision, the deformation gradient is evaluated in double-precision. For
    batched Jacobians of local problems at the quadrature points, e.g. solved by
    :func:`~felupe.math.solve_2d`, the absolute tolerance is the upper bound
    :math:`c\ \varepsilon_x\ ||K(x_n)||\ ||x||`. A function without a reference norm,
    i.e. without prescribed degrees of freedom and without items, is checked by the
    absolute tolerance only, e.g. a double root may require a given absolute
    tolerance. The norms
    of the objective function of the iterations, e.g. ``NewtonResult.fnorms``, are
    relative norms, see Eq. :eq:`newton-fnorm`, i.e. the iteration has converged if
    the relative norm is lower than the relative tolerance.

    ..  math::
        :label: newton-fnorm

        \text{norm}(f) = \frac{||f_1(x)||}{f_\text{ref} + \text{atol} / \text{tol}}

    ..  versionchanged:: 12.1.0
        The reference norm includes the forces between the items and the absolute
        tolerance is a bound of the round-off errors of the objective function. Before,
        the iteration has converged if :math:`||f_1(x)|| / (10^{-3} + ||f_0(x)||)` was
        lower than the tolerance, i.e. with an absolute tolerance of
        :math:`10^{-3}\ \text{tol}` in the units of the objective function. Hence,
        e.g. a rigid body motion of a stiff solid body before it gets in contact did
        not converge, because the round-off errors of the forces exceeded the
        absolute tolerance, and models with small forces converged too early, because
        the absolute tolerance exceeded the relative tolerance times the reaction
        forces.

    The attainable norm of the objective function is limited by the precision of its
    data type and by the precision of the unknowns. An increment which is smaller than
    the spacing of the values of the unknowns is lost by the update, e.g. of fields in
    single-precision (``float32``) on a region in double-precision (``float64``). Even
    the exact solution, rounded to the precision of the unknowns, does not fulfil the
    equilibrium equations within the precision of the objective function. Hence, the
    default tolerance is the square root of the machine epsilon :math:`\varepsilon` of
    the lowest precision of the linear equation system and of the unknowns, see Eq.
    :eq:`newton-tol`. The precision of the linear equation system is given by the
    common data type of the Jacobian :math:`K(x_n)` and of the objective function
    :math:`f(x_n)`. The precision of the unknowns is given by the data types of the
    values of all fields of a field container or by the data type of an array. The
    default tolerance is evaluated after the assembly of the Jacobian in the first
    iteration.

    ..  math::
        :label: newton-tol

        \text{tol} = \sqrt{\varepsilon}

    This is approximately ``1.490e-8`` for double-precision (``float64``) and
    ``3.453e-4`` for single-precision (``float32``), e.g. of fields cast to ``float32``
    on a region in ``float32`` or ``float64``. The data type of the solution of the
    linear equation system is not considered, i.e. a linear solver in single-precision
    for an equation system and unknowns in double-precision uses the tolerance of
    double-precision. A lower precision within the evaluation of the objective
    function, e.g. of a constitutive material in single-precision for unknowns and a
    region in double-precision, is not detected.

    Plugins may modify the update of the unknowns in the ``after_linear_solve`` hook,
    see :class:`~felupe.tools.IterationState`. E.g., a backtracking line search
    (:class:`~felupe.LinesearchPlugin`) scales the Newton increment by a step length
    :math:`\alpha \in (0, 1]`, see Eq. :eq:`newton-update-alpha`.

    ..  math::
        :label: newton-update-alpha

        x = x_n + \alpha\ dx

    A plugin with the hook ``after_iteration`` replaces the ``callback`` (removed in
    12.0.0), which was called after each completed iteration. The arguments of the
    callback are attributes of the state of the iteration, see
    :class:`~felupe.tools.IterationState`. The same plugin is used for the iterations
    of all substeps of a :class:`~felupe.Job`, e.g.
    ``Job(steps, plugins=[NewtonCallback()])`` replaces
    ``Job.evaluate(callback=callback)``. A callable plugin can't be used here, because
    it is only called after each completed substep of a job (and never by
    :func:`~felupe.newtonraphson` alone).

    ..  code-block:: python

        def callback(dx, x, iteration, xnorm, fnorm, success):
            ...

        # removed in 12.0.0
        # res = fem.newtonraphson(items, **loadcase, callback=callback)

        # use a plugin instead
        class NewtonCallback(fem.Plugin):
            def after_iteration(self, context, state):
                callback(
                    state.dx,
                    state.x,
                    state.iteration,
                    state.xnorm,
                    state.fnorm,
                    state.success,
                )

        res = fem.newtonraphson(items, **loadcase, plugins=[NewtonCallback()])

    The plugin gets the same values as the callback, after each completed iteration.

    Examples
    --------
    >>> import numpy as np
    >>> import felupe as fem
    >>>
    >>> region = fem.RegionHexahedron(fem.Cube(n=6))
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(
    ...     field, move=0.2, clamped=True, return_loadcase=True
    ... )
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=2.0), field=field)
    >>> res = fem.newtonraphson(items=[solid], **loadcase, verbose=2)  # doctest: +ELLIPSIS
     _______  _______  ___ ...
    Newton-Raphson solver
    =====================
    <BLANKLINE>
    | # | norm(fun) |  norm(dx) |
    |---|-----------|-----------|
    | 1 | 7.615e-02 | 1.898e+00 |
    | 2 | 1.321e-03 | 5.091e-02 |
    | 3 | 3.112e-07 | 6.698e-04 |
    | 4 | ...e-14 | ...e-07 |
    <BLANKLINE>
    Converged in 4 iterations ...
    <BLANKLINE>

    Newton's method had success

    >>> print(res.success)
    True

    and 4 iterations were needed to converge within the specified tolerance.

    >>> res.iterations
    4

    The norm of the objective function for all active degrees of freedom is lower than
    3e-15.

    >>> print(np.linalg.norm(res.fun[loadcase["dof1"]]) < 3e-15)
    True

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

    # the objective function and the reference norm of the items of the last evaluation
    last = {"fun": None, "reference": 0.0}

    def evaluate(x):
        "Evaluate the objective function for given unknowns."
        if items is not None:
            f, reference = fun_items(items, x, *args, return_reference=True, **kwargs)
            last["fun"], last["reference"] = f, reference
            return f
        return fun(x, *args, **kwargs)

    def reference(f):
        "Return the reference norm of the items for given values of the objective."
        return last["reference"] if f is last["fun"] else 0.0

    def absolute_tol(x, f):
        "Return the absolute tolerance for the updated unknowns of the iteration."
        if atol is not None:
            return atol
        return default_atol(state.jac, f, x, dof1=dof1)

    def converged(dx, x, f):
        "Check the convergence for given unknowns, e.g. of a trial of a plugin."
        return check(
            dx=dx,
            x=x,
            f=f,
            xtol=np.inf,
            ftol=tol,
            dof1=dof1,
            dof0=dof0,
            items=None,
            atol=absolute_tol(x, f),
            reference=reference(f),
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
    state = IterationState(tol=tol, atol=atol, x=x)
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

        if tol is None:
            # the default tolerance depends on the data types of the linear equation
            # system, which are known after the assembly of the Jacobian, and of the
            # unknowns
            tol = state.tol = default_tol(K, f, x)

        # create keyword-arguments for solving the linear system
        keys = ["x", "dof1", "dof0", "ext0", "solver"]
        values = [x, dof1, dof0, ext0, solver]

        for key, value in zip(keys, values):
            if key in sig.parameters:
                kwargs_solve[key] = value

        dispatcher.trigger("before_linear_solve", context, state)

        dx = solve(K, -f, **kwargs_solve)

        if np.any(np.isnan(dx)):
            raise NewtonConvergenceError(
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

        # the absolute tolerance depends on the updated unknowns of the iteration
        state.atol = absolute_tol(x, f)

        xnorm, fnorm, success = check(
            dx=dx,
            x=x,
            f=f,
            xtol=np.inf,
            ftol=tol,
            dof1=dof1,
            dof0=dof0,
            items=items,
            atol=state.atol,
            reference=reference(f),
        )

        if success and items is not None:
            # accept the trial state variables of the items for the converged solution
            update_statevars(items)

        xnorms.append(xnorm)
        fnorms.append(fnorm)

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
        raise NewtonConvergenceError(
            "Maximum number of iterations reached (not converged).\n"
        )

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
