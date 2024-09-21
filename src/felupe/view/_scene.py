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


class Scene:
    r"""Base class for plotting a static scene.

    Attributes
    ----------
    mesh : pyvista.UnstructuredGrid
        A generalized Dataset with the mesh as well as point- and cell-data. This is
        not an instance of :class:`felupe.Mesh`.

    Examples
    --------
    ..  pyvista-plot::
        :force_static:

        >>> import numpy as np
        >>> import felupe as fem
        >>>
        >>> scene = fem.view.Scene()
        >>> scene.mesh = fem.Cube(n=3).as_unstructured_grid()
        >>> scene.mesh.point_data["Displacement"] = np.arange(81).reshape(27, 3) / 300
        >>> scene.mesh.set_active_scalars(None)
        >>>
        >>> scene.plot("Displacement", component=None).show()

    See Also
    --------
    felupe.ViewMesh : Provide Visualization methods for a mesh with optional given
        dicts of point- and cell-data items.
    felupe.ViewField : Provide Visualization methods for a field container.
    felupe.ViewSolid : Provide Visualization methods for a field container or a
        solid body.
    """

    def plot(
        self,
        name=None,
        component=0,
        label=None,
        factor=1.0,
        show_edges=True,
        show_undeformed=True,
        cmap="turbo",
        view="default",
        theme="default",
        scalar_bar_args=None,
        scalar_bar_vertical=False,
        add_axes=True,
        off_screen=False,
        plotter=None,
        notebook=False,
        extract_surface=False,
        nonlinear_subdivision=1,
        smooth_shading=True,
        split_sharp_edges=True,
        edge_color="black",
        line_width=1.0,
        **kwargs,
    ):
        """Plot scalars, selected by name and component.

        Parameters
        ----------
        name: str or None, optional
            Name of array of scalars to plot (default is None).
        component : int, optional
            Component of vector-valued scalars to plot (default is 0).
        label : str or None, optional
            A custom label which is shown in the scalar bar. If no label is given, it is
            created by the name including the component. For vector-valued scalars, the
            component number is replaced by ``"X"``, etc. For 3d-tensors in
            full or reduced vector storage (Voigt-Notation) the component number is
            replaced by ``"XY"``, etc. If ``"Principal Values of"`` is in the name, the
            component number is replaced by ``"\n (Max. Principal)"``, assuming that the
            principal values are sorted in descending order.
        factor : float, optional
            Factor for the scaling of the warped (deformed) mesh (default is 1.0).
        show_edges : bool, optional
            Show the edges of the cells (default is True).
        show_undeformed : bool, optional
            Show the undeformed model (default is True).
        cmap : str, optional
            The color map (default is "turbo").
        view : str or None, optional
            The camera position, e.g. "xy" or "iso" (default is "default"). If not
            specified, this is None for 3d-meshes and "xy" for 2d-meshes.
        theme : str or None, optional
            The theme used for plotting, e.g. "default" or "document" (default is None).
        scalar_bar_vertical : bool, optional
            A flag to show the interactive scalar bar in vertical orientation on the
            right side (default is True).
        add_axes : bool, optional
            Add the axes, i.e. the coordinate system to the scene (default is True).
        off_screen : bool, optional
            Initialize the plotter off-screen and don't open a window on plotting. For
            screenshots, it is necessary to set ``off_screen=True`` (default is False).
        plotter : pyvista.Plotter or None, optional
            Use a given Plotter instead of creating a new instance (default is None).
        notebook : bool, optional
            When True, the resulting plot is placed inline a jupyter notebook. Assumes a
            jupyter console is active. Automatically enables off_screen (default is
            False).
        extract_surface : bool, optional
            Extract the surface mesh. Required to hide internal edges of quadratic
            cells (default is False). If True and ``show_edges=True``, the feature edges
            of a separated mesh are plotted.
        nonlinear_subdivision : int, optional
            Number of subdivisions to generate a smooth surface based on the mid-edge
            points (default is 1, no subdivision). If greater than 1, the surface of
            the mesh is extracted.
        smooth_shading : bool, optional
            A flag to enable smooth shading (default is True). Only considered if
            number of subdivisions is greater than 1.
        split_sharp_edges : bool, optional
            A flag to split sharp edges (default is True). Use this flag in combination
            with smooth shading. Only considered if number of subdivisions is greater
            than 1.
        edge_color : str, optional
            The color of the edge lines (default is "black").
        line_width : float, optional
            The line-width of the edge lines (default is 1.0).


        Returns
        -------
        plotter : pyvista.Plotter
            A Plotter object with methods ``plot()``, ``screenshot()``, etc.

        See Also
        --------
        pyvista.Plotter : Plotting object to display vtk meshes or numpy arrays.
        """

        import pyvista as pv

        if theme is not None:
            pv.set_plot_theme(theme)

        if plotter is None:
            plotter_kwargs = dict()
            if off_screen:
                plotter_kwargs["off_screen"] = off_screen
            if notebook:
                plotter_kwargs["notebook"] = notebook
            plotter = pv.Plotter(**plotter_kwargs)

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if name is not None:
            if component is not None:
                if name in self.mesh.point_data.keys():
                    data = self.mesh.point_data[name]
                else:
                    data = self.mesh.cell_data[name]

        if name is not None and label is None:
            data_label = name

            if component is not None:
                if name in self.mesh.point_data.keys():
                    data = self.mesh.point_data[name]
                else:
                    data = self.mesh.cell_data[name]

                dim = 1

                if len(data.shape) == 2:
                    dim = data.shape[1]

                component_labels_dict = {
                    1: [""],
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

                if "Principal Values of " in data_label:
                    data[:] = np.flip(np.sort(data, axis=-1), axis=-1)
                    component_labels_dict[2] = [
                        "(Max. Principal)",
                        "(Min. Principal)",
                    ]
                    component_labels_dict[3] = [
                        "(Max. Principal)",
                        "(Int. Principal)",
                        "(Min. Principal)",
                    ]
                    data_label = data_label[20:]

                if "Equivalent of " in data_label:
                    component_labels_dict[1] = [""]

                component_labels = np.arange(dim)
                if dim in component_labels_dict.keys():
                    component_labels = component_labels_dict[dim]

                component_label = component_labels[component]

            else:
                component_label = "Magnitude"

            label = f"{data_label} {component_label}"

        if show_undeformed:
            plotter.add_mesh(
                self.mesh, show_edges=False, opacity=0.2, line_width=line_width
            )

        mesh = self.mesh
        if "Displacement" in self.mesh.point_data.keys():
            mesh = mesh.warp_by_vector("Displacement", factor=factor)

        surface = mesh
        show_edges_surface = show_edges
        kwargs_with_line_width = {**kwargs}

        if mesh.number_of_cells > 0:
            if extract_surface or nonlinear_subdivision > 1:
                surface = surface.extract_surface(
                    nonlinear_subdivision=nonlinear_subdivision
                )
                show_edges_surface = False
            else:
                kwargs_with_line_width["line_width"] = line_width

        # disable surface-related arguments if the mesh contains no cells
        if mesh.number_of_cells == 0 or nonlinear_subdivision == 1:
            smooth_shading = None
            split_sharp_edges = None

        # don't show edges for the base (surface) mesh to hide internal edges of
        # quadratic / Lagrange cell-types
        plotter.add_mesh(
            mesh=surface,
            scalars=name,
            component=component,
            show_edges=show_edges_surface,
            cmap=cmap,
            scalar_bar_args={
                "title": label,
                "interactive": True,
                "vertical": scalar_bar_vertical,
                **scalar_bar_args,
            },
            smooth_shading=smooth_shading,
            split_sharp_edges=split_sharp_edges,
            **kwargs_with_line_width,
        )

        # extract the feature edges (without cell-internal edges)
        if (
            mesh.number_of_cells > 0
            and show_edges
            and (extract_surface or nonlinear_subdivision > 1)
        ):
            edges = (
                mesh.separate_cells()
                .extract_surface(nonlinear_subdivision=nonlinear_subdivision)
                .extract_feature_edges()
            )
            actor = plotter.add_mesh(edges, color=edge_color, line_width=line_width)
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()

        if view == "default":
            if np.allclose(self.mesh.points[:, 2], 0):
                view = "xy"
                plotter.enable_parallel_projection()

            else:
                view = None
                plotter.camera.elevation = -15
                plotter.camera.azimuth = -100

        plotter.camera_position = view
        # pv.set_plot_theme(theme)

        if add_axes:
            plotter.add_axes()

        return plotter
