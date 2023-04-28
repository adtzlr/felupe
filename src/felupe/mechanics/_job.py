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

from ..math import dot, eigh, eigvalsh, tovoigt, transpose
from ..tools._misc import logo, runs_on


def displacement(field, substep=None):
    "Displacement Vector"
    u = field[0].values
    return np.pad(u, ((0, 0), (0, 3 - u.shape[1])))


def deformation_gradient(field, substep=None):
    "Deformation Gradient"
    F = field[0].extract()
    return [F.mean(-2).transpose([2, 0, 1])]


def log_strain_principal(field, substep=None):
    "Principal Values of Logarithmic Strain"
    u = field[0]
    F = u.extract()
    stretch = np.sqrt(eigvalsh(dot(transpose(F), F)))[::-1]
    strain = np.log(stretch).mean(-2)
    return [strain.T]


def log_strain(field, substep=None):
    "Lagrangian Logarithmic Strain Tensor"
    u = field[0]
    F = u.extract()
    w, v = eigh(dot(transpose(F), F))
    stretch = np.sqrt(w)
    strain = np.einsum("a...,ia...,ja...->ij...", np.log(stretch), v, v)
    return [tovoigt(strain.mean(-2), True).T]


def print_header():
    print("\n".join([logo(), runs_on(), "", "Run Job", "======="]))


class Job:
    "A job with a list of steps."

    def __init__(
        self,
        steps,
        callback=lambda stepnumber, substepnumber, substep: None,
        filename=None,
    ):
        self.steps = steps
        self.nsteps = len(steps)
        self.callback = callback
        self.timetrack = []

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
        verbose=True,
        parallel=False,
        **kwargs,
    ):
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

        increment = 0

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

                    if verbose == 2:
                        _substep = f"Substep {i + 1}/{step.nsubsteps}"
                        _step = f"Step {j + 1}/{self.nsteps}"

                        print(f"{_substep} of {_step} successful.")

                    self.callback(j, i, substep)

                    # update x0 after each completed substep
                    if "x0" in kwargs.keys():
                        kwargs["x0"].link(substep.x)

                    if filename is not None:
                        self.timetrack.append(increment)
                        self._write(
                            writer=writer,
                            time=increment,
                            substep=substep,
                            point_data={**pdata, **point_data},
                            cell_data={**cdata, **cell_data},
                        )

                    increment += 1

                    if verbose == 1:
                        progress_bar.update(1)

            if verbose == 1:
                progress_bar.close()
