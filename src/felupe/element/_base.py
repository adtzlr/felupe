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


class Element:
    def __init__(self, shape):
        self.shape = shape
        self.dim = self.shape[1]

    def view(self, point_data=None, cell_data=None, cell_type=None):
        """View the element with optional given dicts of point- and cell-data items.

        Parameters
        ----------
        point_data : dict or None, optional
            Additional point-data dict (default is None).
        cell_data : dict or None, optional
            Additional cell-data dict (default is None).
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None).

        Returns
        -------
        felupe.ViewMesh
            A object which provides visualization methods for
            :class:`felupe.element.Element`.

        See Also
        --------
        felupe.ViewMesh : Visualization methods for :class:`felupe.Mesh`.
        """

        from ..mesh import Mesh

        mesh = Mesh(points=self.points, cells=self.cells, cell_type=self.cell_type)

        return mesh.view(
            point_data=point_data,
            cell_data=cell_data,
            cell_type=cell_type,
        )

    def plot(
        self,
        *args,
        style="wireframe",
        color="black",
        add_axes_at_origin=True,
        add_point_labels=True,
        show_points=True,
        font_size=26,
        **kwargs,
    ):
        """Plot the element.

        See Also
        --------
        felupe.Scene.plot: Plot method of a scene.
        """

        view = self.view()
        if self.points.shape[1] > 1:
            view.mesh = view.mesh.extract_surface().extract_feature_edges()

        plotter = view.plot(
            *args,
            show_undeformed=False,
            opacity=0.8,
            add_axes=False,
            show_edges=False,
            style=style,
            color=color,
            **kwargs,
        )
        if add_point_labels:
            plotter.add_point_labels(
                points=np.pad(self.points, ((0, 0), (0, 3 - self.shape[1]))),
                labels=[f"{a}" for a in np.arange(len(self.points))],
                font_size=font_size,
                show_points=show_points,
                point_size=20,
                point_color="black",
                shape=None,
                fill_shape=False,
                render_points_as_spheres=True,
                always_visible=True,
            )

        if add_axes_at_origin:
            actor = plotter.add_axes_at_origin(xlabel="r", ylabel="s", zlabel="t")
            actor.SetNormalizedShaftLength((0.9, 0.9, 0.9))
            actor.SetNormalizedTipLength((0.1, 0.1, 0.1))

            if self.shape[1] == 3:
                actor.SetTotalLength([1.3, 1.3, 1.3])
            elif self.shape[1] == 2:
                actor.SetZAxisLabelText("")
                actor.SetTotalLength([1.3, 1.3, 0])
            elif self.shape[1] == 1:
                actor.SetYAxisLabelText("")
                actor.SetZAxisLabelText("")
                actor.SetTotalLength([1.3, 0, 0])

            plotter.camera.zoom(0.7)

        if self.shape[1] == 3:
            plotter.camera.azimuth = -17

        return plotter

    def screenshot(
        self,
        *args,
        filename=None,
        transparent_background=None,
        scale=None,
        **kwargs,
    ):
        """Take a screenshot of the element.

        See Also
        --------
        pyvista.Plotter.screenshot: Take a screenshot of a PyVista plotter.
        """

        if filename is None:
            filename = f"{self.cell_type}.png"

        return self.plot(*args, off_screen=True, **kwargs).screenshot(
            filename=filename,
            transparent_background=transparent_background,
            scale=scale,
        )
