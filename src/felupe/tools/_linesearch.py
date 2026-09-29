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

import numpy as np

from ..math import det


def _flatten(dx):
    "Return a flat 1d-array of a (list of) array(s)."
    if isinstance(dx, (list, tuple)):
        return np.concatenate([np.ravel(d) for d in dx])
    return np.ravel(dx)


def _scale(dx, alpha):
    "Scale a (list of) array(s) by a step length. Returns ``dx`` itself for alpha=1."
    if alpha == 1.0:
        return dx
    if isinstance(dx, (list, tuple)):
        return [alpha * d for d in dx]
    return alpha * dx


def _first_gradient_flag(grad):
    "Return the gradient-flag of the first field for a given ``grad``-argument."
    if grad is None:
        return True
    if isinstance(grad, (list, tuple)):
        return bool(grad[0]) if len(grad) > 0 else True
    return bool(grad)


def _is_square_tensor(A):
    "Check if an array is a batch of (2, 2) or (3, 3) second-order tensors."
    return (
        isinstance(A, np.ndarray)
        and A.ndim > 2
        and A.shape[0] == A.shape[1]
        and A.shape[0] in (2, 3)
    )


def _first_extracted_field(field):
    "Return the first extracted field of a field container."
    fields = field.extracted_fields
    if callable(fields):
        fields = fields()
    return fields[0]


def deformation_gradients(x, items=None):
    r"""Return a list with the deformation gradients of all items (or of the field
    container, if no items are given), evaluated for the unknowns ``x``.

    Parameters
    ----------
    x : felupe.FieldContainer or ndarray
        The (trial) unknowns.
    items : list or None, optional
        A list of items, e.g. :class:`~felupe.SolidBody` (default is None).

    Returns
    -------
    list of ndarray
        The deformation gradients :math:`\boldsymbol{F}` at the quadrature points of
        all cells for all items with a deformation gradient.

    Notes
    -----
    The fields of the items are linked to the unknowns ``x``. The deformation gradient
    of an item is evaluated by the first extracted field of the item's field container
    (in the same way as the kinematics of the item), without modifying the internal
    state of the item. Only items whose first kinematic quantity is a second-order
    tensor (with a gradient-flag of the first field, if given) are considered. This
    includes plane-strain, axisymmetric and mixed-field formulations. Items without a
    deformation gradient, e.g. trusses, point loads or multi-point constraints, are
    skipped.
    """

    Fs = []

    if items is None:
        if hasattr(x, "extracted_fields"):
            F = _first_extracted_field(x).extract(
                grad=True, sym=False, add_identity=True
            )
            if _is_square_tensor(F):
                Fs.append(F)
        return Fs

    for item in items:
        field = getattr(item, "field", None)
        results = getattr(item, "results", None)
        kinematics = getattr(results, "kinematics", None)

        if field is None or kinematics is None or not hasattr(field, "link"):
            continue

        if isinstance(kinematics, (list, tuple)):
            if len(kinematics) == 0:
                continue
            kinematics = kinematics[0]

        if not _is_square_tensor(kinematics):
            continue

        if not _first_gradient_flag(getattr(item, "grad", None)):
            continue

        if not hasattr(field, "extracted_fields"):
            continue

        field.link(x)
        F = _first_extracted_field(field).extract(
            grad=True, sym=False, add_identity=True
        )

        if _is_square_tensor(F):
            Fs.append(F)

    return Fs


class LineSearchState:
    r"""The state of a trial of a line search, passed to the acceptance criteria of a
    :class:`~felupe.tools.LineSearch`. All parameters are available as attributes.

    Parameters
    ----------
    x_old : felupe.FieldContainer or ndarray
        The unknowns of the last accepted iteration.
    x_trial : felupe.FieldContainer or ndarray
        The trial unknowns, ``x_trial = update(x_old, alpha * dx)``.
    f_old : ndarray
        The objective function (residuals) of the last accepted iteration.
    alpha : float
        The trial step length.
    dx : ndarray
        The (full) Newton increment.
    items : list or None
        The list of items (or None).
    dof1 : ndarray or None
        The active degrees of freedom (or None).
    dof0 : ndarray or None
        The prescribed degrees of freedom (or None).
    df0 : ndarray or None
        The linearized change of the objective function due to the increment of the
        prescribed degrees of freedom, :math:`\boldsymbol{K}\ d\boldsymbol{x}_0`, or
        None if the prescribed degrees of freedom are not changed.
    iteration : int or None
        The index of the Newton iteration.
    halving : int
        The number of halvings of the step length.
    fun : callable
        A callable which assembles the objective function ``f = fun(x)``.
    check : callable or None
        A callable ``converged = check(dx, x, f)`` which checks the convergence of a
        trial without side effects.

    Notes
    -----
    The objective function of the trial is only assembled on demand, i.e. on the first
    access of the attribute :attr:`f_trial`. The result is cached. Hence, criteria
    which do not need the residuals (like the admissibility of the deformation) should
    not access :attr:`f_trial` and should be placed first in the list of criteria.
    """

    def __init__(
        self,
        x_old,
        x_trial,
        f_old,
        alpha,
        dx,
        items=None,
        dof1=None,
        dof0=None,
        df0=None,
        iteration=None,
        halving=0,
        fun=None,
        check=None,
    ):
        self.x_old = x_old
        self.x_trial = x_trial
        self.f_old = f_old
        self.alpha = alpha
        self.dx = dx
        self.items = items
        self.dof1 = dof1
        self.dof0 = dof0
        self.df0 = df0
        self.iteration = iteration
        self.halving = halving

        self._fun = fun
        self._check = check
        self._f_trial = None
        self._converged = None

    @property
    def assembled(self):
        "A flag if the objective function of the trial is already assembled."
        return self._f_trial is not None

    @property
    def f_trial(self):
        "The objective function (residuals) of the trial (assembled on demand)."
        if self._f_trial is None:
            # rejected trials may lead to invalid floating point operations, e.g. in
            # inadmissible states. non-finite residuals are rejected by the line-search
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                self._f_trial = self._fun(self.x_trial)
        return self._f_trial

    @property
    def finite(self):
        "A flag if the objective function of the trial is finite (assembled on demand)."
        return bool(np.all(np.isfinite(self.f_trial)))

    @property
    def converged(self):
        """A flag if the trial satisfies the convergence criterion of the Newton-Raphson
        method (assembled on demand)."""
        if self._converged is None:
            if self._check is None or not self.finite:
                self._converged = False
            else:
                self._converged = bool(
                    self._check(_scale(self.dx, self.alpha), self.x_trial, self.f_trial)
                )
        return self._converged


class LineSearch:
    r"""A backtracking line search for the Newton-Raphson method.

    Parameters
    ----------
    admissible : bool, optional
        A flag to reject trials with non-positive determinants of the deformation
        gradients, see Eq. :eq:`linesearch-admissible` (default is True).
    residual : bool, optional
        A flag to reject trials without a sufficient decrease of the norm of the
        residuals, see Eq. :eq:`linesearch-residual` (default is True).
    c : float, optional
        The parameter of the sufficient-decrease condition of the residuals (default is
        1e-4).
    max_halvings : int, optional
        The maximum number of halvings of the step length (default is 10). The smallest
        step length is :math:`\alpha = 2^{-\text{max\_halvings}}`.
    criteria : list of callable or None, optional
        A list of additional acceptance criteria, called after the built-in criteria.
        A criterion has the function signature ``criterion(state) -> bool``, where
        ``state`` is a :class:`~felupe.tools.LineSearchState`. Default is None.

    Notes
    -----
    Starting with a step length of :math:`\alpha = 1`, the Newton increment
    :math:`d\boldsymbol{x}` is scaled and the trial unknowns are evaluated, see Eq.
    :eq:`linesearch-update`.

    ..  math::
        :label: linesearch-update

        \boldsymbol{x}(\alpha) = \boldsymbol{x}_n + \alpha\ d\boldsymbol{x}

    The step length is halved, :math:`\alpha \leftarrow \alpha / 2`, until all
    acceptance criteria are fulfilled. If the maximum number of halvings is exceeded, a
    :class:`ValueError` is raised. The acceptance criteria are evaluated in the order
    of the list :attr:`criteria` and the evaluation stops at the first rejection.

    **Admissibility** (``admissible=True``): The determinants of the deformation
    gradients at all quadrature points of all cells of all items must be positive, see
    Eq. :eq:`linesearch-admissible`.

    ..  math::
        :label: linesearch-admissible

        \det \boldsymbol{F}(\boldsymbol{x}(\alpha)) > 0

    The deformation gradient is evaluated by the first extracted field of the field
    container of each item. Hence, mixed-field formulations as well as plane-strain and
    axisymmetric fields are supported. Items without a deformation gradient are
    skipped. This criterion is evaluated first because it does not require an assembly
    of the residuals.

    **Sufficient decrease of the residuals** (``residual=True``): The norm of the
    residuals of the active degrees of freedom :math:`\boldsymbol{r}_1` must decrease
    sufficiently, see Eq. :eq:`linesearch-residual`. This is an Armijo-type condition
    for the merit function :math:`\frac{1}{2} ||\boldsymbol{r}_1||^2`.

    ..  math::
        :label: linesearch-residual

        ||\boldsymbol{r}_1(\alpha)|| \le (1 - c\ \alpha)\ ||\boldsymbol{r}_1(0)||

    If the Newton increment changes the prescribed degrees of freedom
    :math:`d\boldsymbol{x}_0` (e.g. in the first iteration of a new substep), the
    residuals of the last iteration are not a valid reference because they belong to
    the old boundary conditions. Hence, the remaining part of the increment of the
    prescribed degrees of freedom is added by its linearization with the (partitioned)
    Jacobian :math:`\boldsymbol{K}_{10}` of the last iteration, see Eq.
    :eq:`linesearch-residual-effective`.

    ..  math::
        :label: linesearch-residual-effective

        \boldsymbol{r}_1(\alpha) = \boldsymbol{f}_1(\boldsymbol{x}(\alpha))
            + (1 - \alpha)\ \boldsymbol{K}_{10}\ d\boldsymbol{x}_0

    The Newton increment is a descent direction of this merit function. Without
    prescribed increments, Eq. :eq:`linesearch-residual-effective` simplifies to
    :math:`\boldsymbol{r}_1(\alpha) = \boldsymbol{f}_1(\boldsymbol{x}(\alpha))`.
    Trials with non-finite residuals are always rejected and a trial which already
    satisfies the convergence criterion of the Newton-Raphson method is always
    accepted by this criterion.

    The residuals of the accepted trial are re-used for the next iteration of the
    Newton-Raphson method. The accepted trial is always the last evaluated trial.
    Hence, the (temporary) state variables of the items belong to the accepted trial.

    Examples
    --------
    A coarse cube is compressed by 70% in a single step. Without a line search, the
    first Newton increment leads to inadmissible deformation gradients and to NaN
    values in the stresses.

    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=3)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>> boundaries, loadcase = fem.dof.uniaxial(
    ...     field, move=-0.7, clamped=True, return_loadcase=True
    ... )
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    >>>
    >>> res = fem.newtonraphson(
    ...     items=[solid], linesearch=fem.tools.LineSearch(), verbose=0, **loadcase
    ... )
    >>> print(res.success)
    True

    The accepted step lengths are stored in the result.

    >>> res.alphas
    [0.5, 1.0, 1.0, 1.0, 1.0]

    A custom criterion is a callable which takes a
    :class:`~felupe.tools.LineSearchState` and returns a boolean.

    >>> def max_increment(state):
    ...     return state.alpha * abs(state.dx).max() <= 0.5
    >>>
    >>> linesearch = fem.tools.LineSearch(criteria=[max_increment])

    See Also
    --------
    felupe.newtonraphson : Find a root of a real function using the Newton-Raphson
        method.
    felupe.tools.LineSearchState : The state of a trial of a line search.
    """

    def __init__(
        self,
        admissible=True,
        residual=True,
        c=1e-4,
        max_halvings=10,
        criteria=None,
    ):
        self.admissible = admissible
        self.residual = residual
        self.c = c
        self.max_halvings = int(max_halvings)

        if self.max_halvings < 0:
            raise ValueError("The maximum number of halvings must not be negative.")

        self.criteria = []

        if admissible:
            self.criteria.append(self.check_admissible)

        if residual:
            self.criteria.append(self.check_residual)

        if criteria is not None:
            self.criteria.extend(criteria)

    def check_admissible(self, state):
        r"""Return True if the determinants of the deformation gradients of all items,
        evaluated for the trial unknowns, are positive, see Eq.
        :eq:`linesearch-admissible`. Does not assemble the residuals."""

        for F in deformation_gradients(state.x_trial, state.items):
            with np.errstate(invalid="ignore"):
                if not np.all(det(F) > 0):
                    return False

        return True

    def check_residual(self, state):
        r"""Return True if the norm of the (effective) residuals of the active degrees
        of freedom decreases sufficiently, see Eqs. :eq:`linesearch-residual` and
        :eq:`linesearch-residual-effective`."""

        if not state.finite:
            return False

        if state.converged:
            return True

        dof1 = state.dof1
        if dof1 is None:
            dof1 = slice(None)

        r_old = state.f_old[dof1]
        r_trial = state.f_trial[dof1]

        if state.df0 is not None:
            df0 = state.df0[dof1]
            r_old = r_old + df0
            r_trial = r_trial + (1 - state.alpha) * df0

        norm_old = np.linalg.norm(r_old)
        norm_trial = np.linalg.norm(r_trial)

        return bool(norm_trial <= (1 - self.c * state.alpha) * norm_old)

    @staticmethod
    def _linearized_prescribed_change(jac, dx, dof1, dof0):
        "Return the linearized change of the residuals due to prescribed increments."

        if jac is None or dof0 is None or dof1 is None:
            return None

        dxflat = _flatten(dx)
        dx0 = np.zeros_like(dxflat)
        dx0[dof0] = dxflat[dof0]

        if not np.any(dx0):
            return None

        return np.asarray(jac @ dx0).ravel()

    @staticmethod
    def _snapshot(x):
        """Return the value arrays of the fields of the unknowns. Items may be linked to
        the trials and the unknowns may be the field of an item (e.g. if no ``x0`` is
        given). Linking replaces the value arrays of the fields of the unknowns."""
        fields = getattr(x, "fields", None)
        if fields is None:
            return None
        return [field.values for field in fields]

    @staticmethod
    def _restore(x, snapshot):
        "Restore the value arrays of the fields of the unknowns."
        if snapshot is not None:
            for field, values in zip(x.fields, snapshot):
                field.values = values

    @staticmethod
    def _reset_buffers(items):
        """Reset the re-used output arrays of the gradients of the items. They may
        contain non-finite values of a rejected trial."""

        if items is None:
            return

        for item in items:
            results = getattr(item, "results", None)
            if results is not None and hasattr(results, "gradient"):
                results.gradient = None

    def __call__(
        self,
        x,
        dx,
        f,
        fun,
        update,
        jac=None,
        items=None,
        dof1=None,
        dof0=None,
        check=None,
        iteration=None,
    ):
        r"""Perform the line search and return the accepted trial.

        Parameters
        ----------
        x : felupe.FieldContainer or ndarray
            The unknowns of the last accepted iteration.
        dx : ndarray
            The Newton increment.
        f : ndarray
            The objective function (residuals) of the last accepted iteration.
        fun : callable
            A callable which assembles the objective function ``f = fun(x)``.
        update : callable
            A callable which updates the unknowns ``x = update(x, dx)``. It must return
            a new object and must not modify ``x`` in-place.
        jac : sparse matrix or None, optional
            The Jacobian of the last accepted iteration (default is None).
        items : list or None, optional
            The list of items (default is None).
        dof1 : ndarray or None, optional
            The active degrees of freedom (default is None).
        dof0 : ndarray or None, optional
            The prescribed degrees of freedom (default is None).
        check : callable or None, optional
            A callable ``converged = check(dx, x, f)`` without side effects (default is
            None).
        iteration : int or None, optional
            The index of the Newton iteration (default is None).

        Returns
        -------
        x_trial : felupe.FieldContainer or ndarray
            The unknowns of the accepted trial.
        f_trial : ndarray
            The objective function (residuals) of the accepted trial.
        alpha : float
            The accepted step length.
        """

        df0 = None
        if self.residual:
            df0 = self._linearized_prescribed_change(jac, dx, dof1, dof0)

        alpha = 1.0
        rejected_by = None
        snapshot = self._snapshot(x)

        for halving in range(self.max_halvings + 1):
            # the unknowns may be linked to a rejected trial (via the items)
            self._restore(x, snapshot)
            x_trial = update(x, _scale(dx, alpha))

            if x_trial is x:
                raise ValueError(
                    "The update function modified the unknowns in-place. The line "
                    "search requires an update function which returns a new object, "
                    "e.g. `update = lambda x, dx: x + dx`."
                )

            state = LineSearchState(
                x_old=x,
                x_trial=x_trial,
                f_old=f,
                alpha=alpha,
                dx=dx,
                items=items,
                dof1=dof1,
                dof0=dof0,
                df0=df0,
                iteration=iteration,
                halving=halving,
                fun=fun,
                check=check,
            )

            rejected_by = None
            for criterion in self.criteria:
                if not criterion(state):
                    rejected_by = getattr(criterion, "__name__", repr(criterion))
                    break

            # the residuals of the accepted trial are always assembled. If the trial
            # was already assembled by a criterion, the cached residuals are re-used.
            if rejected_by is None:
                if state.finite:
                    return x_trial, state.f_trial, alpha
                rejected_by = "finite residuals"

            if state.assembled and not state.finite:
                self._reset_buffers(items)

            alpha /= 2

        # restore the unknowns and link the items to the unknowns of the last accepted
        # iteration (e.g. for a subsequent cutback)
        self._restore(x, snapshot)

        if items is not None and hasattr(x, "fields"):
            for item in items:
                field = getattr(item, "field", None)
                if field is not None and hasattr(field, "link"):
                    field.link(x)

        where = "" if iteration is None else f" in Newton iteration {1 + iteration}"
        raise ValueError(
            "Line search failed: no acceptable step length found after "
            f"{self.max_halvings} halvings{where} (smallest step length "
            f"{2.0 ** -self.max_halvings:1.3e}, last trial rejected by "
            f"`{rejected_by}`)."
        )
