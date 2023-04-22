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


class XdmfReader:
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

    def plot(
        self,
        scalars,
        component=0,
        label=None,
        show_edges=True,
        show_undeformed=True,
        time=0,
        cmap="turbo",
        cpos="xy",
        theme="document",
        scalar_bar_vertical=True,
        add_axes=True,
        off_screen=False,
        plotter=None,
        **kwargs,
    ):
        "Create or append to a given plotter and return the plotter."

        import pyvista as pv
        
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

            if "Principal Values of " in scalars:
                component_labels = [
                    "\n (Max. Principal)",
                    "\n (Int. Principal)",
                    "\n (Min. Principal)",
                ]
                data_label = data_label[20:]

            elif dim == 3:
                component_labels = ["X", "Y", "Z"]

            elif dim == 6:
                component_labels = ["XX", "YY", "ZZ", "XY", "YZ", "XZ"]

            elif dim == 9:
                component_labels = [
                    "XX",
                    "XY",
                    "XZ",
                    "YX",
                    "YY",
                    "YZ",
                    "ZX",
                    "ZY",
                    "ZZ",
                ]

            else:
                component_labels = np.arange(dim)

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
