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

import warnings

from ..plugins import ProgressPlugin, XDMFWriterPlugin
from ..region import RegionVertex
from ..tools import Context, EventDispatcher


class JobState:
    "A class to keep track of the state of a Job during evaluation."

    def __init__(self, stepnumber=None, substepnumber=None, time=None):
        self.stepnumber = stepnumber
        self.substepnumber = substepnumber
        self.time = time


class Job:
    r"""A job with a list of steps and a method to evaluate them.

    Parameters
    ----------
    steps : list of Step
        A list with steps, where each step subsequently depends on the solution of the
        previous step.
    callback : callable, optional
        A callable which is called after each completed substep. Function signature must
        be ``lambda stepnumber, substepnumber, substep, **kwargs: None``, where
        ``substep`` is an instance of :class:`~felupe.tools.NewtonResult`. The field
        container of the completed substep is available as ``substep.x``. Default
        is ``callback=lambda stepnumber, substepnumber, substep, **kwargs: None``.
    plugins : list or None, optional
        A list of plugins with hooks to be used during evaluation. Available hooks are
        ``before_job``, ``after_job``, ``before_step``, ``after_step`` and
        ``after_substep``. Each hook takes the job and the current state as arguments.
        All hooks are optional. Default is None, which is equivalent to an empty list.
        Simple callable plugins are dispatched at the ``after_substep`` hook.
    **kwargs : dict, optional
        Optional keyword-arguments for the ``callback`` function.

    Attributes
    ----------
    nsteps : int
        The number of steps.
    timetrack : list of int
        A list with times at which the results are written to the XDMF result file.
    fnorms : list of list of float
        List with norms of the objective function for each completed substep of each
        step. See also class:`~felupe.tools.NewtonResult`.

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
        >>> solid.plot("Principal Values of Cauchy Stress").show()

    See Also
    --------
    Step : A Step with multiple substeps, subsequently depending on the solution
        of the previous substep.
    CharacteristicCurve : A job with a list of steps and a method to evaluate them.
        Force-displacement curve data is tracked during evaluation for a given
        :class:`~felupe.Boundary`.
    tools.NewtonResult : A data class which represents the result found by
        Newton's method.

    """

    def __init__(
        self,
        steps,
        callback=lambda stepnumber, substepnumber, substep, **kwargs: None,
        plugins=None,
        **kwargs,
    ):
        self.steps = steps
        self.nsteps = len(steps)
        self.callback = callback

        self.timetrack = []
        self.fnorms = []
        self.kwargs = kwargs

        self._progress_plugin = ProgressPlugin()
        self._writer_plugin = XDMFWriterPlugin()

        self.dispatcher = EventDispatcher(plugins=plugins)
        self.dispatcher.add_plugin(self._progress_plugin)
        self.dispatcher.add_plugin(self._writer_plugin)

    def evaluate(
        self,
        filename=None,
        mesh=None,
        point_data=None,
        cell_data=None,
        point_data_default=True,
        cell_data_default=True,
        verbose=None,
        parallel=False,
        tqdm="tqdm",
        **kwargs,
    ):
        """Evaluate the steps.

        Parameters
        ----------
        filename : str or None, optional
            The filename of the XDMF result file. Must include the file extension
            ``my_result.xdmf``. If None, no result file is written during evaluation.
            Default is None.
        mesh : Mesh or None, optional
            A mesh which is used for the XDMF time series writer. If None, it is taken
            from the field of the first item of the first step if no keyword argument
            ``x0`` is given. If None and ``x0=field``, the mesh is taken from the ``x0``
            field container. Default is None.
        point_data : dict or None, optional
            Additional dict of point-data for the meshio XDMF time series writer.
        cell_data : dict or None, optional
            Additional dict of cell-data for the meshio XDMF time series writer.
        point_data_default : bool, optional
            Flag to write default point-data to the XDMF result file. This includes
            ``"Displacement"``. Default is True.
        cell_data_default : bool, optional
            Flag to write default cell-data to the XDMF result file. This includes
            ``"Principal Values of Logarithmic Strain"``, ``"Logarithmic Strain"`` and
            ``"Deformation Gradient"``. Default is True.
        verbose : bool or int or None, optional
            Verbosity level to control how messages are printed during evaluation. If
            1 or True and ``tqdm`` is installed, a progress bar is shown. If ``tqdm`` is
            missing or verbose is 2, more detailed text-based messages are printed.
            Default is None. If None, verbosity is set to True. If None and the
            environmental variable FELUPE_VERBOSE is set and its value is not ``true``,
            then logging is turned off.
        parallel : bool, optional
            Flag to use a threaded version of :func:`numpy.einsum` during assembly.
            Requires ``einsumt``. This may add additional overhead to small-sized
            problems. Default is False.
        tqdm : str, optional
            If verbose is True, choose a backend for ``tqdm`` ("tqdm", ``"auto"`` or
            ``"notebook"``. Default is ``"tqdm"``.
        **kwargs : dict
            Optional keyword arguments for :meth:`~felupe.Step.generate`. If
            ``parallel=True``, it is added as ``kwargs["parallel"] = True`` to the dict
            of additional keyword arguments. If ``x0`` is present in ``kwargs.keys()``,
            it is used as the mesh for the XDMF time series writer.

        Returns
        -------
        Job
            The job object.

        Notes
        -----
        Requires ``meshio`` and ``h5py`` if ``filename`` is not None. Also requires
        ``tqdm`` for an interactive progress bar if ``verbose=True``.

        See Also
        --------
        Step : A Step with multiple substeps, subsequently depending on the solution
            of the previous substep.
        CharacteristicCurve : A job with a list of steps and a method to evaluate them.
            Force-displacement curve data is tracked during evaluation for a given
            :class:`~felupe.Boundary` condition.
        tools.NewtonResult : A data class which represents the result found by
            Newton's method.
        """

        # configure plugins
        self._progress_plugin.configure(
            verbose=verbose,
            tqdm=tqdm,
        )
        self._xdmf_writer_plugin.configure(
            filename=filename,
            mesh=mesh,
            point_data=point_data,
            cell_data=cell_data,
            point_data_default=point_data_default,
            cell_data_default=cell_data_default,
            kwargs=kwargs,
        )

        context = Context(job=self)
        state = JobState()
        self.dispatcher.trigger("before_job", context, state)

        if parallel:
            if "kwargs" not in kwargs.keys():
                kwargs["kwargs"] = {}
            kwargs["kwargs"]["parallel"] = True

        time = 0

        for j, step in enumerate(self.steps):

            context = Context(job=self, step=step)
            state = JobState(stepnumber=j, time=time)
            self.dispatcher.trigger("before_step", context, state)

            substeps = step.generate(dispatcher=self.dispatcher, **kwargs)

            for i, substep in enumerate(substeps):
                self.fnorms.append(substep.fnorms)

                self.callback(j, i, substep, **self.kwargs)

                # update x0 after each completed substep
                if "x0" in kwargs.keys():
                    kwargs["x0"].link(substep.x)

                context = Context(job=self, step=step, substep=substep)
                state = JobState(stepnumber=j, substepnumber=i, time=time)
                self.dispatcher.trigger("after_substep", context, state)

                self.timetrack.append(time)
                time += 1

            context = Context(job=self, step=step)
            state = JobState(stepnumber=j, time=time)
            self.dispatcher.trigger("after_step", context, state)

        context = Context(job=self)
        state = JobState()
        self.dispatcher.trigger("after_job", context, state)

        return self
