# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
from .. import __version__ as version

from ..math import dot, transpose, eigvalsh, eigh, tovoigt


def displacement(substep):
    "Displacement Vector"
    u = substep.x[0].values
    return np.pad(u, ((0, 0), (0, 3 - u.shape[1])))


def deformation_gradient(substep):
    "Deformation Gradient"
    F = substep.x[0].extract()
    return [F.mean(-2).transpose([2, 0, 1])]


def log_strain_principal(substep):
    "Principal Values of Logarithmic Strain"
    u = substep.x[0]
    F = u.extract()
    stretch = np.sqrt(eigvalsh(dot(transpose(F), F)))[::-1]
    strain = np.log(stretch).mean(-2)
    return [strain.T]


def log_strain(substep):
    "Lagrangian Logarithmic Strain Tensor"
    u = substep.x[0]
    F = u.extract()
    w, v = eigh(dot(transpose(F), F))
    stretch = np.sqrt(w)
    strain = np.einsum("a...,ia...,ja...->ij...", np.log(stretch), v, v)
    return [tovoigt(strain.mean(-2), True).T]


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

    def _write(self, writer, time, substep, point_data, cell_data):
        writer.write_data(
            time,
            point_data={key: value(substep) for key, value in point_data.items()},
            cell_data={key: value(substep) for key, value in cell_data.items()},
        )

    def evaluate(
        self,
        filename=None,
        mesh=None,
        point_data={"Displacement": displacement},
        cell_data={
            "Principal Values of Logarithmic Strain": log_strain_principal,
            "Logarithmic Strain": log_strain,
            "Deformation Gradient": deformation_gradient,
        },
        verbose=True,
        **kwargs,
    ):

        if verbose:
            print(
                f"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|
FElupe Version {version}

"""
            )

            print("Run Job")
            print("=======\n")

        if filename is not None:
            from meshio.xdmf import TimeSeriesWriter

            if mesh is None:
                if "x0" in kwargs.keys():
                    mesh = kwargs["x0"].region.mesh.as_meshio()
                else:
                    mesh = self.steps[0].items[0].field.region.mesh.as_meshio()

            increment = 0

        else:  # fake a mesh and a TimeSeriesWriter
            from contextlib import nullcontext

            TimeSeriesWriter = nullcontext

        with TimeSeriesWriter(filename) as writer:

            if filename is not None:
                writer.write_points_cells(mesh.points, mesh.cells)

            for j, step in enumerate(self.steps):

                if verbose:
                    print(f"Begin Evaluation of Step {j + 1}.")

                substeps = step.generate(verbose=verbose, **kwargs)
                for i, substep in enumerate(substeps):

                    if verbose:
                        _substep = f"Substep {i}/{step.nsubsteps - 1}"
                        _step = f"Step {j + 1}/{self.nsteps}"

                        print(f"{_substep} of {_step} successful.")

                    self.callback(j, i, substep)

                    # update x0 after each completed substep
                    if "x0" in kwargs.keys():
                        kwargs["x0"].link(substep.x)

                    if filename is not None:
                        self._write(writer, increment, substep, point_data, cell_data)
                        increment += 1
