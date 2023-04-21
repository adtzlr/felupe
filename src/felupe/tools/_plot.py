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


class ResultFile:
    def __init__(
        self,
        filename,
        scalars="Principal Values of Logarithmic Strain",
        component=0,
        label=None,
        show_edges=True,
        cmap="turbo",
        cpos="xy",
        time=0,
    ):

        from pyvista import XdmfReader

        self.filename = filename
        self.scalars = scalars
        self.component = component
        self.label = label
        self.show_edges = show_edges

        self.file = XdmfReader(self.filename)
        self.file.set_active_time_value(time)

        self.cmap = cmap
        self.cpos = cpos

    def init(self, off_screen=False):
        "Init a plotter, read the file, add its mesh and return the plotter."

        from pyvista import Plotter

        plotter = Plotter(off_screen=off_screen)

        if self.label is None:
            if "Principal Values of " in self.scalars:
                comp = ["Maximum", "Intermediate", "Minimum"]
                self.label = f"{self.scalars[20:]} ({comp[self.component]})"

        mesh = self.file.read()[0]

        plotter.add_mesh(
            mesh=mesh.warp_by_vector("Displacement"),
            scalars=self.scalars,
            component=self.component,
            show_edges=self.show_edges,
            cmap=self.cmap,
            scalar_bar_args={"title": self.label},
        )
        plotter.add_axes()

        return plotter

    def as_png(self, filename=None):
        "Take a screenshot of the scene."

        if filename is None:
            name = ".".join(self.filename.split(".")[:-1])
            filename = f"{name}.png"

        plotter = self.init(off_screen=True)
        plotter.show(cpos=self.cpos, screenshot=filename)

    def show(self):
        "Show the scene."

        plotter = self.init(off_screen=False)
        plotter.show(cpos=self.cpos)
