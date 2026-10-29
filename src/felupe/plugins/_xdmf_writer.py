# SPDX-License-Identifier: GPL-3.0-or-later

import warnings

from ..math import deformation_gradient as defgrad
from ..math import displacement as disp
from ..mesh import Mesh
from ..region import RegionVertex
from ._plugin import Plugin


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


class XDMFWriterPlugin(Plugin):
    r"""A job plugin to write XDMF files during job evaluation.

    Parameters
    ----------
    filename : str or None, optional
        The filename of the XDMF result file, including the file extension, e.g.
        ``"result.xdmf"``. If None, no result file is written (default is None).
    mesh : Mesh or None, optional
        A mesh which is used for the XDMF time series writer. If None, it is taken from
        the keyword-argument ``x0`` or from the field of the first item of the first
        step (default is None).
    point_data : dict or None, optional
        Additional dict of point-data, i.e. functions ``fun(field, substep)``
        (default is None).
    cell_data : dict or None, optional
        Additional dict of cell-data, i.e. functions ``fun(field, substep)`` (default is
        None).
    point_data_default : bool, optional
        Flag to write default point-data, i.e. the displacements (default is False).
    cell_data_default : bool, optional
        Flag to write default cell-data, i.e. the (principal values of the)
        logarithmic strains and the deformation gradients (default is False).
    kwargs : dict or None, optional
        The keyword-arguments of the evaluation of the job, e.g. ``x0`` (default is
        None).
    time : TimeStep or None, optional
        A time step item. If given, its (absolute) time is the time of the result
        file. If None, the (zero-based) index of the substep within the job is the time
        of the result file (default is None).

    Notes
    -----
    Requires ``meshio`` and ``h5py``.

    The time of the result file is the index of the substep within the job, i.e.
    ``0, 1, 2, ...``. With a time step item ``time``, the (absolute) time of the
    substeps is written instead, e.g. ``Job(steps).evaluate(filename="result.xdmf",
    time=time)`` for the time step item ``time = TimeStep(items=[solid])`` in the ramp
    of the steps. Readers of XDMF result files, e.g. ParaView, require increasing
    times. If the time of a substep is not greater than the time of the previous
    substep, e.g. in a step without a ramp of the time, a warning is emitted (once)
    and the substep is written with the (repeated) time.

    See Also
    --------
    felupe.Job : A job with a list of steps and a method to evaluate them.
    felupe.TimeStep : A time step item, which updates the time increment of other
        items.
    """

    def __init__(
        self,
        filename=None,
        mesh=None,
        point_data=None,
        cell_data=None,
        point_data_default=False,
        cell_data_default=False,
        kwargs=None,
        time=None,
    ):
        self.configure(
            filename=filename,
            mesh=mesh,
            point_data=point_data,
            cell_data=cell_data,
            point_data_default=point_data_default,
            cell_data_default=cell_data_default,
            kwargs=kwargs,
            time=time,
        )

    def configure(
        self,
        filename,
        mesh,
        point_data,
        cell_data,
        point_data_default,
        cell_data_default,
        kwargs,
        time=None,
    ):
        self.filename = filename
        self.mesh = mesh
        self.kwargs = {} if kwargs is None else kwargs
        self.point_data = point_data
        self.cell_data = cell_data
        self.point_data_default = point_data_default
        self.cell_data_default = cell_data_default
        self.time = time

    def _time(self, state):
        """Return the time of the result file for a substep, i.e. the (absolute) time
        of the time step item or the index of the substep within the job."""

        if self.time is None:
            return state.time

        # the time step item is updated before the substep, i.e. its old time is the
        # time of the substep
        time = self.time.time_old

        if self._time_last is not None and time <= self._time_last:
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"The time {time:g} of the XDMF result file is not greater than "
                    f"the time {self._time_last:g} of the previous substep, e.g. in a "
                    "step without a ramp of the time. Readers of XDMF result files, "
                    "e.g. ParaView, require increasing times. Use increasing times in "
                    "the ramps of the time step item or write the index of the "
                    "substeps as time, i.e. `time=None`."
                )

        self._time_last = time

        return time

    def _write(self, writer, time, substep, point_data, cell_data):
        field = substep.x
        kwargs = dict(field=field, substep=substep)
        writer.write_data(
            time,
            point_data={key: value(**kwargs) for key, value in point_data.items()},
            cell_data={key: value(**kwargs) for key, value in cell_data.items()},
        )

    def before_job(self, context, state):

        # the time of the previous substep and the flag of the warning (once per job)
        self._time_last = None
        self._warned = False

        if self.filename is not None:
            from meshio.xdmf import TimeSeriesWriter

            if self.mesh is None:
                if "x0" in self.kwargs.keys():
                    self.mesh = self.kwargs["x0"].region.mesh.as_meshio()
                else:
                    self.mesh = (
                        context.job.steps[0].items[0].field.region.mesh.as_meshio()
                    )

            if isinstance(self.mesh, Mesh):
                self.mesh = self.mesh.as_meshio()

            self.pdata = {}
            self.cdata = {}

            if self.point_data_default is True:
                self.pdata = {"Displacement": displacement}

            if self.cell_data_default is True:
                self.cdata = {
                    "Principal Values of Logarithmic Strain": log_strain_principal,
                    "Logarithmic Strain": log_strain,
                    "Deformation Gradient": deformation_gradient,
                }
                if "x0" in self.kwargs.keys():
                    if isinstance(self.kwargs["x0"].region, RegionVertex):
                        # strains and deformation gradient can't be evaluated due to
                        # missing shape function gradient w.r.t. the undeformed
                        # coordinates in RegionVertex
                        self.cdata = {}
                        warnings.warn(
                            " ".join(
                                [
                                    "RegionVertex detected as region of the global",
                                    "field, result-file will only contain points and",
                                    "default point-data.",
                                ]
                            )
                        )

            if self.point_data is None:
                self.point_data = {}

            if self.cell_data is None:
                self.cell_data = {}

        else:  # fake a mesh and a TimeSeriesWriter
            from contextlib import nullcontext

            TimeSeriesWriter = nullcontext

        self.TimeSeriesWriter = TimeSeriesWriter
        self.writer = self.TimeSeriesWriter(self.filename)
        self.writer.__enter__()

        if self.filename is not None:
            self.writer.write_points_cells(self.mesh.points, self.mesh.cells)

    def after_substep(self, context, state):
        if self.filename is not None:
            self._write(
                writer=self.writer,
                time=self._time(state),
                substep=context.substep,
                point_data={**self.pdata, **self.point_data},
                cell_data={**self.cdata, **self.cell_data},
            )

    def after_job(self, context, state):
        self.writer.__exit__(None, None, None)
