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

from ..dof import apply, partition
from ..tools import Context, newtonraphson
from ._job import JobState


class SubstepState(JobState):
    r"""A class to keep track of the state of a substep during evaluation.

    Parameters
    ----------
    stepnumber : int or None, optional
        The (zero-based) index of the step within the job (default is None).
    substepnumber : int or None, optional
        The (zero-based) index of the substep within the step (default is None).
    time : int or None, optional
        The (zero-based) index of the substep within the job, e.g. the time of the
        XDMF result file (default is None).
    error : Exception or None, optional
        The error which was raised by the Newton-Raphson method of the substep
        (default is None).
    values : dict or None, optional
        The values of the ramp of the substep, i.e. a dict with the ramped
        :class:`~felupe.Boundary` conditions or items as keys (default is None).
    result : felupe.tools.NewtonResult or None, optional
        The result of the Newton-Raphson method of the substep (default is None).
    load_factors : list of float or None, optional
        The load factors :math:`t \in (0, 1]` of the converged increments of a
        subdivided substep, e.g. of a :class:`~felupe.CutbackPlugin` (default is None).
        None means that the substep is not subdivided.

    Notes
    -----
    One state is created for each substep by :meth:`~felupe.Step.generate_states` and
    its attributes are updated in-place. The same state is passed to the hooks
    ``before_substep``, ``after_failed_substep`` and ``after_substep``. As a
    :class:`~felupe.JobState`, it holds the step number, the substep number and the
    time.

    ..  list-table:: Attributes of the state in the hooks of a substep.
        :header-rows: 1

        * - Hook
          - ``result``
          - ``error``
        * - ``before_substep``
          - None
          - None
        * - ``after_failed_substep``
          - None
          - the raised error
        * - ``after_substep``
          - the result
          - None or the error of a recovered substep

    If the Newton-Raphson method of a substep raises an error (which is an instance of
    :class:`Exception`), the hook ``after_failed_substep`` is triggered. A plugin may
    recover the substep in this hook, e.g. by the callable ``solve`` of the
    :class:`~felupe.tools.Context`. Then, the plugin has to set the result of the
    substep ``state.result``, i.e. the :class:`~felupe.tools.NewtonResult` for the
    values of the ramp of the substep. The unknowns ``x0`` must be linked to the result,
    which is done by ``solve``. Otherwise, if the result is still None after all
    plugins are called, the error ``state.error`` is raised. A plugin may replace the
    error, e.g. by a more descriptive error.

    See Also
    --------
    felupe.Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    felupe.JobState : A class to keep track of the state of a Job during evaluation.
    felupe.Plugin : Base class for plugins.
    felupe.CutbackPlugin : A cutback of the increment of failed substeps.
    """

    def __init__(
        self,
        stepnumber=None,
        substepnumber=None,
        time=None,
        error=None,
        values=None,
        result=None,
        load_factors=None,
    ):
        super().__init__(
            stepnumber=stepnumber, substepnumber=substepnumber, time=time, error=error
        )
        self.values = values
        self.result = result
        self.load_factors = load_factors


class Step:
    """A Step with multiple substeps, subsequently depending on the solution
    of the previous substep.

    Parameters
    ----------
    items : list of SolidBody, SolidBodyNearlyIncompressible, SolidBodyPressure, SolidBodyGravity, PointLoad, MultiPointConstraint or MultiPointContact
        A list of items with methods for the assembly of sparse vectors/matrices.
    ramp : dict, optional
        A dict with :class:`~felupe.Boundary` or ``item``-keys which holds the array of
        values to ramp (default is None). If None, only one substep is evaluated.
    boundaries : dict of Boundary, optional
        A dict with :class:`~felupe.Boundary` conditions (default is None).

    Notes
    -----
    For each substep, the ramped items (and boundaries) are updated with the values of
    the ramp and the Newton-Raphson method is evaluated, see :meth:`generate_states`.
    If the Newton-Raphson method of a substep raises an error, plugins may recover the
    substep, e.g. a :class:`~felupe.CutbackPlugin` subdivides the substep into smaller
    increments.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>>
        >>> boundaries = fem.dof.symmetry(field[0])
        >>> boundaries["clamped"] = fem.Boundary(field[0], fx=1, skip=(True, False, False))
        >>> boundaries["move"] = fem.Boundary(field[0], fx=1, skip=(False, True, True))
        >>>
        >>> umat = fem.NeoHooke(mu=1, bulk=2)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> move = fem.math.linsteps([0, 1], num=5)
        >>> step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
        >>>
        >>> job = fem.Job(steps=[step]).evaluate()
        >>> ax = solid.plot("Principal Values of Cauchy Stress").show()

    See Also
    --------
    Job : A job with a list of steps and a method to evaluate them.
    CharacteristicCurve : A job with a list of steps and a method to evaluate them.
        Force-displacement curve data is tracked during evaluation for a given
        :class:`~felupe.Boundary`.
    SubstepState : A class to keep track of the state of a substep during evaluation.
    CutbackPlugin : A cutback of the increment of failed substeps.
    """

    def __init__(self, items, ramp=None, boundaries=None):
        self.items = items

        if ramp is None:
            self.ramp = {}
            self.nsubsteps = 1
        else:
            self.ramp = dict(ramp)
            self.nsubsteps = len(list(self.ramp.values())[0])

        if boundaries is None:
            boundaries = {}

        self.boundaries = boundaries

    def generate(self, **kwargs):
        """Yield the results of all generated substeps.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for :meth:`generate_states`.

        Yields
        ------
        felupe.tools.NewtonResult
            The result of the Newton-Raphson method for each substep.

        See Also
        --------
        felupe.Step.generate_states : Yield the states of all generated substeps.
        """

        for state in self.generate_states(**kwargs):
            yield state.result

    def generate_states(self, stepnumber=None, time=None, **kwargs):
        """Yield the states of all generated substeps.

        Parameters
        ----------
        stepnumber : int or None, optional
            The (zero-based) index of the step within the job (default is None).
        time : int or None, optional
            The (zero-based) index of the first substep of the step within the job
            (default is None).
        **kwargs : dict
            Keyword arguments for :func:`~felupe.newtonraphson`. The keyword argument
            ``x0``, the field container with the unknowns, is required. An optional
            :class:`~felupe.EventDispatcher` ``dispatcher`` is used to trigger the
            hooks of the plugins.

        Yields
        ------
        felupe.SubstepState
            The state of each completed substep with the result of the
            Newton-Raphson method ``state.result``.

        Notes
        -----
        The unknowns ``x0`` are linked to the result of each completed substep, i.e.
        they are the starting point of the next substep.

        If a ``dispatcher`` is given, the hook ``before_substep`` is triggered before
        each substep. If the Newton-Raphson method of a substep raises an error, the
        hook ``after_failed_substep`` is triggered and plugins may recover the substep,
        see :class:`~felupe.SubstepState`. If the substep is not recovered, the error
        is raised. The :class:`~felupe.tools.Context` of these hooks holds the step,
        the items, the unknowns ``x0`` and the callable ``res = solve(values)``, which
        updates the ramped items with given values, evaluates the Newton-Raphson
        method and links the unknowns ``x0`` to the result.
        """

        field = kwargs["x0"]
        dispatcher = kwargs.get("dispatcher")

        def solve(values):
            "Update the ramped items with given values and solve the load case."

            # update items
            for item, value in values.items():
                item.update(value)

            # update load case
            dof0, dof1 = partition(field, self.boundaries)
            ext0 = apply(field, self.boundaries, dof0)

            # run newton-raphson iterations
            res = newtonraphson(
                items=self.items,
                dof0=dof0,
                dof1=dof1,
                ext0=ext0,
                **kwargs,
            )

            # the converged result is the starting point of the next evaluation
            if res.success:
                field.link(res.x)

            return res

        def trigger(hook, context, state):
            if dispatcher is not None:
                dispatcher.trigger(hook, context, state)

        context = Context(step=self, items=self.items, x0=field, solve=solve)

        for substep in range(self.nsubsteps):
            values = {item: value[substep] for item, value in self.ramp.items()}
            state = SubstepState(
                stepnumber=stepnumber,
                substepnumber=substep,
                time=None if time is None else time + substep,
                values=values,
            )

            trigger("before_substep", context, state)

            try:
                state.result = solve(values)
            except Exception as error:
                state.error = error

            if state.error is not None:
                # plugins may recover the substep (outside of the except-clause to
                # avoid chained errors of the plugins)
                trigger("after_failed_substep", context, state)

                if state.result is None:
                    raise state.error

            if not state.result.success:
                break

            yield state
