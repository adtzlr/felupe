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
import os

from ..math import deformation_gradient as defgrad
from ..math import displacement as disp
from ..tools._misc import logo, runs_on


def displacement(field, substep=None):
    "Return the displacement vvectors."
    return disp(field, dim=3)


def deformation_gradient(field, substep=None):
    "Return the Deformation Gradient tensors."
    return [defgrad(field).mean(-2).transpose([2, 0, 1])]


def log_strain_principal(field, substep=None):
    "Return principal values of logarithmic strain tensors."
    return [field.evaluate.log_strain(tensor=False)[::-1].mean(-2).T]


def log_strain(field, substep=None):
    "Return Lagrangian logarithmic strain tensors."
    return [field.evaluate.log_strain(tensor=True, asvoigt=True).mean(-2).T]


def print_header():
    print("\n".join([logo(), runs_on(), "", "Run Job", "======="]))


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
    **kwargs : dict, optional
        Optional keyword-arguments for the ``callback`` function.

    Attributes
    ----------
    steps : list of Step
        A list with steps, where each step subsequently depends on the solution of the
        previous step.
    nsteps : int
        The number of steps.
    callback : callable
        A callable which is called after each completed substep. Function signature must
        be ``lambda stepnumber, substepnumber, substep: None``, where ``substep`` is an
        instance of :class:`~felupe.tools.NewtonResult`. THe field container of the
        completed substep is available as ``substep.x``.
    timetrack : list of int
        A list with times at which the results are written to the XDMF result file.
    fnorms : list of list of float
        List with norms of the objective function for each completed substep of each
        step. See also class:`~felupe.tools.NewtonResult`.
    kwargs : dict
        Optional keyword-arguments for the ``callback`` function.

    Examples
    --------
    ..  pyvista-plot::

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
        **kwargs,
    ):
        self.steps = steps
        self.nsteps = len(steps)
        self.callback = callback
        self.timetrack = []
        self.fnorms = []
        self.kwargs = kwargs

    def _write(self, writer, time, substep, point_data, cell_data):
        field = substep.x
        kwargs = dict(field=field, substep=substep)
        writer.write_data(
            time,
            point_data={key: value(**kwargs) for key, value in point_data.items()},
            cell_data={key: value(**kwargs) for key, value in cell_data.items()},
        )

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
        **kwargs,
    ):
        """Evaluate the steps.

        Parameters
        ----------
        filename : str or None, optional
            The filename of the XDMF result file. Must include the file extension
            ``my_result.xdmf``. If None, no result file is writte during evaluation.
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
        if verbose is None:
            FELUPE_VERBOSE = os.environ.get("FELUPE_VERBOSE")
            if FELUPE_VERBOSE is None:
                verbose = True
            else:
                verbose = FELUPE_VERBOSE == "true"

        if verbose:
            try:
                from tqdm import tqdm
            except ModuleNotFoundError:
                verbose = 2

        if verbose == 2:
            print_header()

        if parallel:
            if "kwargs" not in kwargs.keys():
                kwargs["kwargs"] = {}
            kwargs["kwargs"]["parallel"] = True

        time = 0

        if filename is not None:
            from meshio.xdmf import TimeSeriesWriter

            if mesh is None:
                if "x0" in kwargs.keys():
                    mesh = kwargs["x0"].region.mesh.as_meshio()
                else:
                    mesh = self.steps[0].items[0].field.region.mesh.as_meshio()

            pdata = point_data_default if point_data_default is True else {}
            cdata = cell_data_default if cell_data_default is not None else {}

            pdata = {}
            cdata = {}

            if point_data_default is True:
                pdata = {"Displacement": displacement}

            if cell_data_default is True:
                cdata = {
                    "Principal Values of Logarithmic Strain": log_strain_principal,
                    "Logarithmic Strain": log_strain,
                    "Deformation Gradient": deformation_gradient,
                }

            if point_data is None:
                point_data = {}

            if cell_data is None:
                cell_data = {}

        else:  # fake a mesh and a TimeSeriesWriter
            from contextlib import nullcontext

            TimeSeriesWriter = nullcontext

        with TimeSeriesWriter(filename) as writer:
            if filename is not None:
                writer.write_points_cells(mesh.points, mesh.cells)

            if verbose == 1:
                total = sum([step.nsubsteps for step in self.steps])
                progress_bar = tqdm(total=total, unit="substep")

            for j, step in enumerate(self.steps):
                newton_verbose = False
                if verbose == 2:
                    print(f"Begin Evaluation of Step {j + 1}.")
                    newton_verbose = True

                substeps = step.generate(verbose=newton_verbose, **kwargs)
                for i, substep in enumerate(substeps):
                    self.fnorms.append(substep.fnorms)
                    if verbose == 2:
                        _substep = f"Substep {i + 1}/{step.nsubsteps}"
                        _step = f"Step {j + 1}/{self.nsteps}"

                        print(f"{_substep} of {_step} successful.")

                    self.callback(j, i, substep, **self.kwargs)

                    # update x0 after each completed substep
                    if "x0" in kwargs.keys():
                        kwargs["x0"].link(substep.x)

                    self.timetrack.append(time)

                    if filename is not None:
                        self._write(
                            writer=writer,
                            time=time,
                            substep=substep,
                            point_data={**pdata, **point_data},
                            cell_data={**cdata, **cell_data},
                        )

                    time += 1

                    if verbose == 1:
                        progress_bar.update(1)

            if verbose == 1:
                progress_bar.close()

        return self
