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

from types import SimpleNamespace

import numpy as np

from ..mechanics._job import (
    deformation_gradient,
    displacement,
    log_strain,
    log_strain_principal,
)


class Plot:
    def plot(
        self,
        scalars,
        component=0,
        label=None,
        show_edges=True,
        show_undeformed=True,
        time=0,
        cmap="turbo",
        cpos=None,
        theme="document",
        scalar_bar_vertical=True,
        add_axes=True,
        off_screen=False,
        plotter=None,
        **kwargs,
    ):
        "Create or append to a given plotter and return the plotter."

        import pyvista as pv

        if cpos is None and np.allclose(self.mesh.points[:, 2], 0):
            cpos = "xy"

        if theme is not None:
            pv.set_plot_theme(theme)

        if plotter is None:
            plotter = pv.Plotter(off_screen=off_screen)

        if scalars in self.mesh.point_data.keys():
            data = self.mesh.point_data[scalars]
        else:
            data = self.mesh.cell_data[scalars]

        dim = data.shape[1]

        if label is None:
            data_label = scalars

            component_labels_dict = {
                2: ["X", "Y"],
                3: ["X", "Y", "Z"],
                6: ["XX", "YY", "ZZ", "XY", "YZ", "XZ"],
                9: [
                    "XX",
                    "XY",
                    "XZ",
                    "YX",
                    "YY",
                    "YZ",
                    "ZX",
                    "ZY",
                    "ZZ",
                ],
            }

            if "Principal Values of " in scalars:
                component_labels_dict[2] = [
                    "\n (Max. Principal)",
                    "\n (Min. Principal)",
                ]
                component_labels_dict[3] = [
                    "\n (Max. Principal)",
                    "\n (Int. Principal)",
                    "\n (Min. Principal)",
                ]
                data_label = data_label[20:]

            component_labels = np.arange(dim)
            if dim in component_labels_dict.keys():
                component_labels = component_labels_dict[dim]

            component_label = component_labels[component]
            label = f"{data_label} {component_label}"

        if show_undeformed:
            plotter.add_mesh(self.mesh, show_edges=False, opacity=0.2)

        plotter.add_mesh(
            mesh=self.mesh.warp_by_vector("Displacement"),
            scalars=scalars,
            component=component,
            show_edges=show_edges,
            cmap=cmap,
            scalar_bar_args={
                "title": label,
                "interactive": True,
                "vertical": scalar_bar_vertical,
            },
            **kwargs,
        )
        plotter.camera_position = cpos

        if add_axes:
            plotter.add_axes()

        return plotter


class ResultXdmf(Plot):
    def __init__(
        self,
        filename,
        time=0,
    ):
        "XDMF Result file reader and plotter."

        self.filename = filename
        self.mesh = self._read(time)

    def _read(self, time):
        "Read the file and obtain the mesh."

        import pyvista as pv

        self.file = pv.XdmfReader(self.filename)
        self.file.set_active_time_value(time)

        mesh = self.file.read()[0]

        return mesh

    def set_active_time_value(self, time):
        "Set new active time value and re-read the mesh."

        self.mesh = self._read(time)


class Result(Plot):
    def __init__(self, mesh, field=None, point_data=None, cell_data=None):

        import pyvista as pv

        points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.points.shape[1])))
        cells = np.pad(
            mesh.cells, ((0, 0), (1, 0)), constant_values=mesh.cells.shape[1]
        )

        meshio_to_pyvista_cell_types = {
            "line": pv.CellType.LINE,
            "triangle": pv.CellType.TRIANGLE,
            "triangle6": pv.CellType.QUADRATIC_TRIANGLE,
            "tetra": pv.CellType.TETRA,
            "tetra10": pv.CellType.QUADRATIC_TETRA,
            "quad": pv.CellType.QUAD,
            "quad8": pv.CellType.QUADRATIC_QUAD,
            "quad9": pv.CellType.BIQUADRATIC_QUAD,
            "hexahedron": pv.CellType.HEXAHEDRON,
            "hexahedron20": pv.CellType.QUADRATIC_HEXAHEDRON,
            "hexahedron27": pv.CellType.TRIQUADRATIC_HEXAHEDRON,
        }

        cell_types = meshio_to_pyvista_cell_types[mesh.cell_type] * np.ones(
            mesh.ncells, dtype=int
        )

        self.mesh = pv.UnstructuredGrid(cells, cell_types, points)

        point_data_from_field = {}
        cell_data_from_field = {}

        if field is not None:
            substep = SimpleNamespace(x=field)

            point_data_from_field["Displacement"] = displacement(substep)

            cell_data_from_field["Deformation Gradient"] = deformation_gradient(
                substep
            )[0]
            cell_data_from_field["Logarithmic Strain"] = log_strain(substep)[0]
            cell_data_from_field[
                "Principal Values of Logarithmic Strain"
            ] = log_strain_principal(substep)[0]

        if point_data is None:
            point_data = {}

        if cell_data is None:
            cell_data = {}

        pdata = {**point_data_from_field, **point_data}
        cdata = {**cell_data_from_field, **point_data}

        for label, data in pdata.items():
            self.mesh.point_data[label] = data

        for label, data in cdata.items():
            self.mesh.cell_data[label] = data

        self.mesh.set_active_scalars(None)
        self.mesh.set_active_vectors(None)
        self.mesh.set_active_tensors(None)
