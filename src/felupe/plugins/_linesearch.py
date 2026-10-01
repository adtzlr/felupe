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
from ..tools._newton import NewtonConvergenceError, _restore, _snapshot
from ._plugin import Plugin


def _is_deformation_gradient(A):
    """Check if an array is a batch of (2, 2) or (3, 3) second-order tensors at the
    quadrature points of all cells, i.e. an array with the shape ``(i, j, q, c)``."""
    return (
        isinstance(A, np.ndarray)
        and A.ndim == 4
        and A.shape[0] == A.shape[1]
        and A.shape[0] in (2, 3)
    )


def _deformation_gradient(field):
    "Return the deformation gradient of the first extracted field of a container."
    return field.extracted_fields()[0].extract(grad=True, sym=False, add_identity=True)


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
    (in the same way as the kinematics of the item), without modifying the results of
    the item. Only items whose first kinematic quantity is a second-order tensor at the
    quadrature points of all cells are considered. This includes plane-strain,
    axisymmetric and mixed-field formulations. Items without a deformation gradient,
    e.g. trusses, point loads or multi-point constraints, are skipped.
    """

    Fs = []

    if items is None:
        if hasattr(x, "extracted_fields"):
            F = _deformation_gradient(x)
            if _is_deformation_gradient(F):
                Fs.append(F)
        return Fs

    for item in items:
        kinematics = getattr(getattr(item, "results", None), "kinematics", None)

        if isinstance(kinematics, (list, tuple)):
            kinematics = kinematics[0]

        # the first kinematic quantity of an item without a gradient of its first
        # field has the shape (dim, q, c)
        if not _is_deformation_gradient(kinematics):
            continue

        item.field.link(x)
        Fs.append(_deformation_gradient(item.field))

    return Fs


class LinesearchTrial:
    r"""A trial of a line search, passed to the acceptance criteria of a
    :class:`~felupe.LinesearchPlugin`. All parameters are available as attributes.

    Parameters
    ----------
    context : felupe.tools.Context
        The context of the Newton-Raphson method with the items, the degrees of freedom
        and the callables ``fun``, ``update`` and ``check``.
    state : felupe.tools.IterationState
        The state of the current Newton iteration with the unknowns ``state.x``, the
        values of the objective function ``state.fun`` of the last accepted iteration,
        the Jacobian ``state.jac`` and the full Newton increment ``state.dx``.
    x : felupe.FieldContainer or ndarray
        The trial unknowns, ``x = update(state.x, alpha * state.dx)``.
    alpha : float
        The trial step length.
    dx : ndarray
        The (scaled) increment of the trial, ``dx = alpha * state.dx``.
    halving : int
        The number of halvings of the step length.
    df0 : ndarray or None, optional
        The linearized change of the objective function due to the increment of the
        prescribed degrees of freedom, :math:`\boldsymbol{K}\ d\boldsymbol{x}_0`, or
        None if the prescribed degrees of freedom are not changed (default is None).

    Notes
    -----
    The objective function of the trial is only evaluated on demand, i.e. on the first
    access of the attribute :attr:`fun`. The result is cached. Hence, criteria which do
    not need the residuals (like the admissibility of the deformation) should not access
    :attr:`fun` and should be placed first in the list of criteria.
    """

    def __init__(self, context, state, x, alpha, dx, halving, df0=None):
        self.context = context
        self.state = state
        self.x = x
        self.alpha = alpha
        self.dx = dx
        self.halving = halving
        self.df0 = df0

        self._fun = None
        self._converged = None

    @property
    def evaluated(self):
        "A flag if the objective function of the trial is already evaluated."
        return self._fun is not None

    @property
    def fun(self):
        "The values of the objective function (residuals) of the trial (on demand)."
        if self._fun is None:
            # rejected trials may lead to invalid floating point operations, e.g. in
            # inadmissible states. non-finite residuals are rejected by the line-search
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                self._fun = self.context.fun(self.x)
        return self._fun

    @property
    def finite(self):
        "A flag if the objective function of the trial is finite (evaluated on demand)."
        return bool(np.all(np.isfinite(self.fun)))

    @property
    def converged(self):
        """A flag if the trial satisfies the convergence criterion of the Newton-Raphson
        method (evaluated on demand)."""
        if self._converged is None:
            if self.context.check is None or not self.finite:
                self._converged = False
            else:
                self._converged = bool(self.context.check(self.dx, self.x, self.fun)[2])
        return self._converged


class LinesearchPlugin(Plugin):
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
        A criterion has the function signature ``criterion(trial) -> bool``, where
        ``trial`` is a :class:`~felupe.plugins.LinesearchTrial`. Default is None.

    Attributes
    ----------
    alphas : list of list of float
        The accepted step lengths of all iterations for each evaluation of the
        Newton-Raphson method, e.g. for each substep of a :class:`~felupe.Job`.

    Notes
    -----
    The line search is performed in the ``after_linear_solve`` hook of
    :func:`~felupe.newtonraphson`. Starting with a step length of :math:`\alpha = 1`,
    the Newton increment :math:`d\boldsymbol{x}` is scaled and the trial unknowns are
    evaluated, see Eq. :eq:`linesearch-update`.

    ..  math::
        :label: linesearch-update

        \boldsymbol{x}(\alpha) = \boldsymbol{x}_n + \alpha\ d\boldsymbol{x}

    The step length is halved, :math:`\alpha \leftarrow \alpha / 2`, until all
    acceptance criteria are fulfilled. If the maximum number of halvings is exceeded, a
    :class:`~felupe.NewtonConvergenceError` is raised. The acceptance criteria are
    evaluated in the order of the list :attr:`criteria` and the evaluation stops at
    the first rejection.

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

    The accepted trial is handed back to the Newton-Raphson method, see
    :class:`~felupe.tools.IterationState`, and its residuals are re-used, i.e. no
    additional assembly is required for the accepted step. The accepted trial is always
    the last evaluated trial. Hence, the (temporary) state variables of the items belong
    to the accepted trial. A given ``update`` function of the Newton-Raphson method is
    called with the scaled increment, ``update(x_n, alpha * dx)``, and must return a
    new object. The scaled increment is passed to ``check`` and ``callback``.

    ..  note::

        The line search is not optimal for a
        :class:`~felupe.SolidBodyNearlyIncompressible`. Its internal fields, the
        pressure and the volume ratio, are statically condensed and updated by the full
        Newton increment in each evaluation. They are neither scaled by the step length
        nor included in the assembled residuals (they cancel out), but they are included
        in the tangent stiffness matrix. Hence, the Newton increment of this three-field
        formulation does not match the residual criterion of the line search, which may
        fail for large load increments. The admissibility criterion is not affected. For
        a line search with (nearly) incompressible materials, a mixed-field formulation
        with the pressure and the volume ratio as fields of the field container is
        recommended. Then, all fields are scaled by the step length and all residuals
        are checked.

        ..  code-block:: python

            field = fem.FieldsMixed(region, n=3)
            solid = fem.SolidBody(fem.ThreeFieldVariation(umat), field)

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
    >>> boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    >>> solid = fem.SolidBody(umat=fem.NeoHooke(mu=1.0, bulk=5.0), field=field)
    >>>
    >>> move = fem.math.linsteps([0, -0.7], num=1)
    >>> step = fem.Step(
    ...     items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    ... )
    >>>
    >>> linesearch = fem.LinesearchPlugin()
    >>> job = fem.Job(steps=[step], plugins=[linesearch]).evaluate(verbose=0)

    The accepted step lengths are stored in the plugin, one list for each substep.

    >>> linesearch.alphas
    [[1.0], [0.5, 1.0, 1.0, 1.0, 1.0]]

    The line search is also available for :func:`~felupe.newtonraphson`.

    >>> field, loadcase = ... # doctest: +SKIP
    >>> res = fem.newtonraphson(
    ...     items=[solid], plugins=[fem.LinesearchPlugin()], **loadcase
    ... ) # doctest: +SKIP

    A custom criterion is a callable which takes a
    :class:`~felupe.plugins.LinesearchTrial` and returns a boolean.

    >>> def max_increment(trial):
    ...     return abs(trial.dx).max() <= 0.5
    >>>
    >>> linesearch = fem.LinesearchPlugin(criteria=[max_increment])

    See Also
    --------
    felupe.newtonraphson : Find a root of a real function using the Newton-Raphson
        method.
    felupe.plugins.LinesearchTrial : A trial of a line search.
    felupe.tools.IterationState : A class to keep track of the state of an iteration.
    felupe.ThreeFieldVariation : Hu-Washizu hydrostatic-volumetric selective
        three-field variation.
    """

    # note: a plugin must not be callable, otherwise it is dispatched as simple
    # callable plugin in the ``after_substep`` hook (see ``EventDispatcher``).

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

        self.alphas = []

    def check_admissible(self, trial):
        r"""Return True if the determinants of the deformation gradients of all items,
        evaluated for the trial unknowns, are positive, see Eq.
        :eq:`linesearch-admissible`. Does not assemble the residuals."""

        for F in deformation_gradients(trial.x, trial.context.items):
            with np.errstate(invalid="ignore"):
                if not np.all(det(F) > 0):
                    return False

        return True

    def check_residual(self, trial):
        r"""Return True if the norm of the (effective) residuals of the active degrees
        of freedom decreases sufficiently, see Eqs. :eq:`linesearch-residual` and
        :eq:`linesearch-residual-effective`."""

        # non-finite residuals are rejected: `converged` is False and the comparison of
        # non-finite norms is False
        if trial.converged:
            return True

        dof1 = trial.context.dof1
        if dof1 is None:
            dof1 = slice(None)

        r_old = trial.state.fun[dof1]
        r_trial = trial.fun[dof1]

        if trial.df0 is not None:
            df0 = trial.df0[dof1]
            r_old = r_old + df0
            r_trial = r_trial + (1 - trial.alpha) * df0

        norm_old = np.linalg.norm(r_old)
        norm_trial = np.linalg.norm(r_trial)

        return bool(norm_trial <= (1 - self.c * trial.alpha) * norm_old)

    @staticmethod
    def _linearized_prescribed_change(jac, dx, dof1, dof0):
        "Return the linearized change of the residuals due to prescribed increments."

        if jac is None or dof0 is None or dof1 is None:
            return None

        dxflat = np.ravel(dx)
        dx0 = np.zeros_like(dxflat)
        dx0[dof0] = dxflat[dof0]

        if not np.any(dx0):
            return None

        return np.asarray(jac @ dx0).ravel()

    def before_newton(self, context, state):
        self.alphas.append([])

    def after_linear_solve(self, context, state):
        """Perform the line search and hand the accepted trial back to the
        Newton-Raphson method, see :class:`~felupe.tools.IterationState`."""

        # this hook requires the context and the state of `newtonraphson()`
        if context.fun is None or context.update is None or state.dx is None:
            raise TypeError(
                "The line search requires a context with the callables `fun` and "
                "`update` and a state with the increment `dx` of the Newton-Raphson "
                "method, see `felupe.newtonraphson()`."
            )

        x, dx = state.x, state.dx

        df0 = None
        if self.residual:
            df0 = self._linearized_prescribed_change(
                state.jac, dx, context.dof1, context.dof0
            )

        alpha = 1.0
        rejected_by = None
        snapshot = _snapshot(x)

        for halving in range(self.max_halvings + 1):
            # the unknowns may be linked to a rejected trial (via the items)
            _restore(x, snapshot)

            dx_trial = alpha * dx
            x_trial = context.update(x, dx_trial)

            if x_trial is x:
                raise ValueError(
                    "The update function modified the unknowns in-place. The line "
                    "search requires an update function which returns a new object, "
                    "e.g. `update = lambda x, dx: x + dx`."
                )

            trial = LinesearchTrial(
                context=context,
                state=state,
                x=x_trial,
                alpha=alpha,
                dx=dx_trial,
                halving=halving,
                df0=df0,
            )

            rejected_by = None
            for criterion in self.criteria:
                if not criterion(trial):
                    rejected_by = getattr(criterion, "__name__", repr(criterion))
                    break

            # the residuals of the accepted trial are always evaluated. If the trial
            # was already evaluated by a criterion, the cached residuals are re-used.
            if rejected_by is None:
                if trial.finite:
                    state.x = x_trial
                    state.fun = trial.fun
                    state.dx = dx_trial
                    state.alpha = alpha
                    state.updated = True

                    self.alphas[-1].append(alpha)
                    return

                rejected_by = "finite residuals"

            alpha /= 2

        # restore the unknowns and link the items to the unknowns of the last accepted
        # iteration (e.g. for a subsequent cutback)
        _restore(x, snapshot)

        for item in context.items or []:
            item.field.link(x)

        raise NewtonConvergenceError(
            "Line search failed: no acceptable step length found after "
            f"{self.max_halvings} halvings in Newton iteration {1 + state.iteration} "
            f"(smallest step length "
            f"{2.0 ** -self.max_halvings:1.3e}, last trial rejected by "
            f"`{rejected_by}`)."
        )
