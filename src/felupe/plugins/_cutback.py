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

from ._plugin import Plugin


def interpolate(values0, values1, t):
    r"""Return the values of a ramp, linearly interpolated between the values of the
    previous substep and the values of the current substep for a given load factor,
    see Eq. :eq:`cutback-interpolation`.

    Parameters
    ----------
    values0 : dict
        The values of the ramp of the previous substep.
    values1 : dict
        The values of the ramp of the current substep.
    t : float
        The load factor :math:`t \in (0, 1]`.

    Returns
    -------
    dict
        The interpolated values of the ramp. For a load factor of one, the values of
        the current substep are returned (without interpolation).
    """

    if t >= 1.0:
        return dict(values1)

    values = {}
    for key, value1 in values1.items():
        value0 = np.asarray(values0[key])
        values[key] = value0 + t * (np.asarray(value1) - value0)

    return values


class CutbackPlugin(Plugin):
    r"""A cutback of the increment of failed substeps. If the Newton-Raphson method
    of a substep fails, the substep is subdivided into smaller increments.

    Parameters
    ----------
    factor : float, optional
        The factor :math:`0 < f < 1` by which the increment is reduced after a failed
        attempt (default is 0.5).
    max_cutbacks : int, optional
        The maximum number of cutbacks of a substep (default is 5). The smallest
        increment is :math:`\Delta t = f^{\text{max\_cutbacks}}`.
    growth : float, optional
        The factor :math:`g \ge 1` by which the increment is increased after a
        converged increment (default is 1.0), limited by the remaining part of the
        substep. With the default value, the increment is kept constant.
    exceptions : type or tuple of type, optional
        The errors of the Newton-Raphson method which lead to a cutback (default is
        ``(ValueError, ArithmeticError)``). Errors of other types are raised without a
        cutback. The default types include the errors of
        :func:`~felupe.newtonraphson` if the maximum number of iterations is reached
        or if the solution contains NaN values, the error of a failed
        :class:`~felupe.LinesearchPlugin` as well as errors of :mod:`numpy.linalg`.

    Attributes
    ----------
    load_factors : list of list of float
        The load factors of the converged increments for each substep, e.g.
        ``[1.0]`` for a substep without a cutback or ``[0.5, 1.0]`` for a substep
        which was subdivided into two increments. The list of a substep which was not
        recovered ends with a load factor lower than one (or is empty).
    cutbacks : list of int
        The number of cutbacks for each substep.

    Notes
    -----
    The plugin requires a :class:`~felupe.Job` (or the hooks of
    :meth:`~felupe.Step.generate`). Before each substep, a checkpoint of the unknowns
    and of the (history-dependent) state of the items is created, see
    :meth:`checkpoint`. If the Newton-Raphson method of a substep raises one of the
    given ``exceptions``, the checkpoint is restored and the substep is subdivided into
    increments with load factors :math:`t \in (0, 1]`, starting with the reduced
    increment :math:`\Delta t = f`. The values of the ramp :math:`\boldsymbol{v}(t)`
    of an increment are linearly interpolated between the values of the previous
    substep :math:`\boldsymbol{v}_0` and the values of the current substep
    :math:`\boldsymbol{v}_1`, see Eq. :eq:`cutback-interpolation`.

    ..  math::
        :label: cutback-interpolation

        \boldsymbol{v}(t) = \boldsymbol{v}_0 + t\ \left(
            \boldsymbol{v}_1 - \boldsymbol{v}_0 \right)

    After a converged increment, a new checkpoint is created, the load factor is
    increased :math:`t \leftarrow t + \Delta t` and the next increment is
    :math:`\Delta t \leftarrow \min(g\ \Delta t, 1 - t)`. After a failed attempt, the
    checkpoint of the last converged increment is restored and the increment is reduced,
    :math:`\Delta t \leftarrow f\ \Delta t`. If the maximum number of cutbacks is
    exceeded, a :class:`ValueError` is raised, which is caused by the error of the last
    attempt. Then, the unknowns and the items are restored to the last converged
    increment. The last increment of a substep always uses the values of the ramp of
    the substep (without interpolation).

    The increments are internal to the substep, i.e. only the result of the last
    increment is handed back to the :class:`~felupe.Step` and the hooks
    ``after_substep`` of other plugins (e.g. writers of result files) are only called
    once per substep. However, the hooks of the Newton-Raphson method are called for
    each increment.

    **State of the items**: A checkpoint consists of the values of the unknowns and
    of the checkpoints of all items (and ramped items) which provide the methods
    ``checkpoint()`` and ``restore(checkpoint)``, e.g.
    :meth:`SolidBody.checkpoint() <felupe.SolidBody.checkpoint>` and
    :meth:`SolidBodyNearlyIncompressible.checkpoint()
    <felupe.SolidBodyNearlyIncompressible.checkpoint>`. This includes the internal
    fields (pressure and volume ratio) of a
    :class:`~felupe.SolidBodyNearlyIncompressible`, which are updated in each
    evaluation of the Newton-Raphson method, as well as the time of a
    :class:`~felupe.thermal.TimeStep` and the state of the frictional contact of a
    :class:`~felupe.ContactRigidPlane`. The state variables of material models are
    only updated after a converged increment. Hence, the increments of a subdivided
    substep are also increments of the loading path of history-dependent materials,
    e.g. for viscoelasticity or plasticity. A custom item with a state which is
    modified during the Newton iterations has to provide the methods ``checkpoint()``
    and ``restore(checkpoint)``.

    ..  note::

        The time increment of a rate-dependent material model, e.g. the parameter
        ``dtime`` of :func:`~felupe.constitution.finite_strain_viscoelastic`, is not
        part of the ramp. Hence, it is not scaled by the cutback, i.e. each increment
        of a subdivided substep uses the full time increment of the material. A time
        which is part of the ramp, like the time of a
        :class:`~felupe.thermal.TimeStep`, is interpolated.

    **Values of the previous substep**: The values of the ramp of the previous
    substep are taken from the substeps which were already evaluated with this
    plugin. Hence, the first substep of a job can only be subdivided if the ramped
    items (or boundaries) were already updated by a previous job with this plugin.
    This is usually not required, because the first value of a ramp typically equals
    the initial state, e.g. ``fem.math.linsteps([0, 1], num=5)``. A step without a
    ramp can not be subdivided.

    ..  note::

        A cutback may be combined with a :class:`~felupe.LinesearchPlugin`,
        ``Job(steps, plugins=[LinesearchPlugin(), CutbackPlugin()])``. Then, a failed
        line search leads to a cutback.

    Examples
    --------
    A coarse cube is compressed by 70% in a single substep. The first Newton
    increment leads to inadmissible deformation gradients and to NaN values in the
    stresses.

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
    >>> cutback = fem.CutbackPlugin()
    >>> job = fem.Job(steps=[step], plugins=[cutback]).evaluate(verbose=0)

    The second substep is subdivided into increments. The load factors of the
    converged increments are stored in the plugin, one list for each substep.

    >>> cutback.load_factors
    [[1.0], [0.5, 1.0]]

    >>> cutback.cutbacks
    [0, 1]

    See Also
    --------
    felupe.Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    felupe.SubstepState : A class to keep track of the state of a substep.
    felupe.LinesearchPlugin : A backtracking line search for the Newton-Raphson method.
    felupe.SolidBody.checkpoint : Return a checkpoint of the solid body.
    """

    # note: a plugin must not be callable, otherwise it is dispatched as simple
    # callable plugin in the ``after_substep`` hook (see ``EventDispatcher``).

    def __init__(
        self,
        factor=0.5,
        max_cutbacks=5,
        growth=1.0,
        exceptions=(ValueError, ArithmeticError),
    ):
        self.factor = float(factor)
        self.max_cutbacks = int(max_cutbacks)
        self.growth = float(growth)

        if not 0 < self.factor < 1:
            raise ValueError("The cutback factor must be in the interval (0, 1).")

        if self.max_cutbacks < 0:
            raise ValueError("The maximum number of cutbacks must not be negative.")

        if not self.growth >= 1:
            raise ValueError("The growth factor must be greater or equal one.")

        if isinstance(exceptions, type):
            exceptions = (exceptions,)

        self.exceptions = tuple(exceptions)

        self.load_factors = []
        self.cutbacks = []

        # values of the ramp of the last converged substep (for each ramped item)
        self._values = {}

        # values of the ramp of the current substep (not converged yet)
        self._pending = None

        # the checkpoint of the current substep
        self._checkpoint = None

    @staticmethod
    def _stateful(context):
        """Return a list of unique items and ramped items (of the step) with the
        methods ``checkpoint()`` and ``restore(checkpoint)``."""

        ramp = getattr(context.step, "ramp", None) or {}
        objects = []

        for obj in [*(context.items or []), *ramp.keys()]:
            has_checkpoint = callable(getattr(obj, "checkpoint", None))
            has_restore = callable(getattr(obj, "restore", None))
            is_new = not any(obj is other for other in objects)

            if has_checkpoint and has_restore and is_new:
                objects.append(obj)

        return objects

    def checkpoint(self, context):
        """Return a checkpoint of the unknowns and of the items of a substep.

        Parameters
        ----------
        context : felupe.Context
            The context of a substep with the unknowns ``x0``, the items and the step.

        Returns
        -------
        dict
            A dict with the copied values of the unknowns ``"x0"`` and a list of the
            items (and ramped items) with their checkpoints ``"items"``.
        """

        return {
            "x0": [field.values.copy() for field in context.x0.fields],
            "items": [(obj, obj.checkpoint()) for obj in self._stateful(context)],
        }

    def restore(self, context, checkpoint):
        """Restore a checkpoint of the unknowns and of the items of a substep.

        Parameters
        ----------
        context : felupe.Context
            The context of a substep with the unknowns ``x0``, the items and the step.
        checkpoint : dict
            A dict with the copied values of the unknowns ``"x0"`` and a list of the
            items (and ramped items) with their checkpoints ``"items"``.

        Notes
        -----
        The checkpoints of the items are restored first. Then, the values of the
        unknowns are restored in-place and the fields of the items are linked to the
        unknowns.
        """

        for obj, obj_checkpoint in checkpoint["items"]:
            obj.restore(obj_checkpoint)

        x0 = context.x0

        for field, values in zip(x0.fields, checkpoint["x0"]):
            field.values[:] = values

        for item in context.items or []:
            item.field.link(x0)

    def _commit(self):
        "Commit the values of the ramp of the last converged substep."
        if self._pending is not None:
            self._values.update(self._pending)
            self._pending = None

    def before_substep(self, context, state):
        "Create a checkpoint of the unknowns and of the items."

        # the previous substep has converged
        self._commit()
        self._pending = dict(state.values)

        self._checkpoint = self.checkpoint(context)

        self.load_factors.append([1.0])
        self.cutbacks.append(0)

    def after_substep(self, context, state):
        "Commit the values of the ramp of the converged substep."
        self._commit()

    def _failed(self, state, message, cause):
        "Replace the error of the substep by a descriptive error."

        error = ValueError(
            f"Cutback failed: substep {1 + state.substepnumber} {message}"
        )
        error.__cause__ = cause
        state.error = error

    def after_failed_substep(self, context, state):
        """Subdivide the failed substep into increments with reduced load factors and
        hand the result of the last increment back to the step, see
        :class:`~felupe.SubstepState`."""

        # the substep is already recovered by another plugin
        if state.result is not None:
            return

        # the substep is not converged
        self._pending = None
        self.load_factors[-1] = []

        # errors of other types are raised without a cutback
        if not isinstance(state.error, self.exceptions):
            return

        checkpoint = self._checkpoint
        self.restore(context, checkpoint)

        values1 = state.values
        missing = [key for key in values1.keys() if key not in self._values]

        if len(values1) == 0 or len(missing) > 0:
            reason = "the step has no ramp"
            if len(missing) > 0:
                reason = (
                    "the values of the ramp of the previous substep are unknown (they "
                    "are only known if the ramped items or boundaries were updated in "
                    "a previous substep with this plugin)"
                )

            self._failed(
                state, f"can not be subdivided, because {reason}.", state.error
            )
            return

        values0 = {key: self._values[key] for key in values1.keys()}

        t = 0.0  # load factor of the last converged increment
        dt = 1.0  # the increment of the failed attempt
        error = state.error  # the error of the last failed attempt
        values_converged = None  # values of the last converged increment
        load_factors = []
        cutbacks = 0

        while cutbacks < self.max_cutbacks:

            # a failed attempt: reduce the increment
            cutbacks += 1
            dt *= self.factor

            # evaluate increments until the next failure or the end of the substep
            while True:
                t_new = t + dt

                # the last increment uses the values of the substep
                if t_new > 1.0 - 1e-12:
                    t_new = 1.0

                values = interpolate(values0, values1, t_new)

                try:
                    res = context.solve(values)

                except self.exceptions as attempt_error:
                    error = attempt_error
                    self.restore(context, checkpoint)
                    break

                # the increment has converged
                context.x0.link(res.x)
                load_factors.append(t_new)
                values_converged = values
                t = t_new

                if t == 1.0:
                    state.result = res
                    state.load_factors = load_factors

                    self._pending = dict(values1)
                    self.load_factors[-1] = load_factors
                    self.cutbacks[-1] = cutbacks
                    return

                checkpoint = self.checkpoint(context)
                dt = min(self.growth * dt, 1.0 - t)

        # the substep is not recovered, the checkpoint of the last converged increment
        # is already restored
        if values_converged is not None:
            self._values.update(values_converged)

        state.load_factors = load_factors
        self.load_factors[-1] = load_factors
        self.cutbacks[-1] = cutbacks

        self._failed(
            state,
            f"was not recovered after {self.max_cutbacks} cutbacks (last converged "
            f"load factor {t:.4g}, last increment {dt:1.3e}). The unknowns and "
            "the items are restored to the last converged increment.",
            error,
        )
