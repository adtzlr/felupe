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

from ..math import deformation_gradient as defgrad
from ..math import displacement as disp
from ..region import RegionVertex


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


class XDMFWriterPlugin:
    r"""A job plugin to write XDMF files during job evaluation.

    Parameters
    ----------

    Notes
    -----
    Requires ``h5py``.

    See Also
    --------
    Job : A job with a list of steps and a method to evaluate them.
    """

    def __init__(self):
        self.filename = None
        self.mesh = None
        self.kwargs = None
        self.point_data = None
        self.cell_data = None
        self.point_data_default = None
        self.cell_data_default = None

    def configure(
        self,
        filename=None,
        mesh=None,
        point_data=None,
        cell_data=None,
        point_data_default=False,
        cell_data_default=False,
        kwargs=None,
    ):
        self.filename = filename
        self.mesh = mesh
        self.kwargs = kwargs
        self.point_data = point_data
        self.cell_data = cell_data
        self.point_data_default = point_data_default
        self.cell_data_default = cell_data_default

    def _write(self, writer, time, substep, point_data, cell_data):
        field = substep.x
        kwargs = dict(field=field, substep=substep)
        writer.write_data(
            time,
            point_data={key: value(**kwargs) for key, value in point_data.items()},
            cell_data={key: value(**kwargs) for key, value in cell_data.items()},
        )

    def before_job(self, context, state):

        if self.filename is not None:
            from meshio.xdmf import TimeSeriesWriter

            if self.mesh is None:
                if "x0" in self.kwargs.keys():
                    self.mesh = self.kwargs["x0"].region.mesh.as_meshio()
                else:
                    self.mesh = (
                        context.job.steps[0].items[0].field.region.mesh.as_meshio()
                    )

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

    def after_iteration(self, context, state):
        if state.error:
            self.writer.__exit__(None, None, None)  # pragma: no cover

    def after_substep(self, context, state):
        if self.filename is not None:
            self._write(
                writer=self.writer,
                time=state.time,
                substep=context.substep,
                point_data={**self.pdata, **self.point_data},
                cell_data={**self.cdata, **self.cell_data},
            )

    def after_job(self, context, state):
        self.writer.__exit__(None, None, None)
