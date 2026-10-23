# SPDX-License-Identifier: GPL-3.0-or-later

"""The plotting backend of the scenes by PyVista."""

import numpy as np

from .._scalars import is_principal, scalars_label, sort_principal_values


def _default_view(plotter, points):
    """Prepare the default view of the camera and return the camera position. Planar
    meshes are viewed from the top ("xy") with a parallel projection. For all other
    meshes, the camera is rotated (the camera position is None)."""

    if np.allclose(points[:, 2], 0):
        plotter.enable_parallel_projection()
        return "xy"

    plotter.camera.elevation = -15
    plotter.camera.azimuth = -100

    return None


class PyVistaBackend:
    """The plotting backend of the scenes by PyVista. The plotter is a
    :class:`pyvista.Plotter`.

    See Also
    --------
    felupe.view.Scene : Base class for plotting a static scene.
    """

    def as_unstructured_grid(self, scene, cell_type=None):
        """Return the mesh with the point- and cell-data of a scene as
        :class:`pyvista.UnstructuredGrid`. The active scalars, vectors and tensors are
        not set.

        Parameters
        ----------
        scene : felupe.ViewMesh
            A scene with the attributes ``points``, ``cells``, ``cell_type``,
            ``point_data`` and ``cell_data``.
        cell_type : pyvista.CellType or None, optional
            Cell-type of PyVista (default is None). If None, it is taken from the cell
            type of the scene.

        Returns
        -------
        pyvista.UnstructuredGrid
            The mesh with the point- and cell-data.
        """

        from ...mesh._convert import as_unstructured_grid

        grid = as_unstructured_grid(
            scene.points, scene.cells, scene.cell_type, pyvista_cell_type=cell_type
        )

        for label, data in scene.point_data.items():
            grid.point_data[label] = data

        for label, data in scene.cell_data.items():
            grid.cell_data[label] = data

        grid.set_active_scalars(None)
        grid.set_active_vectors(None)
        grid.set_active_tensors(None)

        return grid

    def prepare_plotter(
        self, plotter=None, theme="default", off_screen=False, notebook=False
    ):
        """Set the theme and return the given plotter or a new plotter.

        Parameters
        ----------
        plotter : pyvista.Plotter or None, optional
            Use a given Plotter instead of creating a new instance (default is None).
        theme : str or None, optional
            The theme used for plotting, e.g. "default" or "document" (default is
            "default"). If None, the theme is not changed.
        off_screen : bool, optional
            Initialize the plotter off-screen (default is False).
        notebook : bool, optional
            Place the plot inline a jupyter notebook (default is False).

        Returns
        -------
        pyvista.Plotter
            The given or a new plotter.
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

        return plotter

    def add_view(
        self,
        plotter,
        scene,
        name=None,
        component=0,
        label=None,
        factor=1.0,
        show_edges=True,
        show_undeformed=True,
        cmap="turbo",
        view="default",
        scalar_bar_args=None,
        scalar_bar_vertical=False,
        add_axes=True,
        extract_surface=False,
        nonlinear_subdivision=1,
        smooth_shading=True,
        split_sharp_edges=True,
        edge_color="black",
        line_width=1.0,
        show_logo=False,
        **kwargs,
    ):
        """Add the mesh of a scene with scalars, selected by name and component, to a
        plotter. The arguments are documented in :meth:`felupe.view.Scene.plot`.

        Returns
        -------
        pyvista.Plotter
            The plotter.
        """

        if show_logo:
            from importlib.resources import as_file, files

            with as_file(files("felupe.view") / "logo.png") as logo_path:
                plotter.add_logo_widget(
                    logo=logo_path,
                    position=(0.75, 0.85),
                    size=(0.2, 0.2),
                )

        if scalar_bar_args is None:
            scalar_bar_args = {}

        if name is not None and component is not None:
            # raises a KeyError if the scalars are not available
            if name in scene.mesh.point_data.keys():
                data = scene.mesh.point_data[name]
            else:
                data = scene.mesh.cell_data[name]

        if name is not None and label is None:
            dim = 1

            if component is not None:
                if len(data.shape) == 2:
                    dim = data.shape[1]

                if is_principal(name):
                    data[:] = sort_principal_values(data)

            label = scalars_label(name, component=component, dim=dim)

        if show_undeformed:
            plotter.add_mesh(
                scene.mesh, show_edges=False, opacity=0.2, line_width=line_width
            )

        mesh = scene.mesh
        if "Displacement" in scene.mesh.point_data.keys():
            mesh = mesh.warp_by_vector("Displacement", factor=factor)

        surface = mesh
        show_edges_surface = show_edges
        kwargs_with_line_width = {**kwargs}

        if mesh.number_of_cells > 0:
            if extract_surface or nonlinear_subdivision > 1:
                surface = surface.extract_surface(
                    algorithm=None,
                    nonlinear_subdivision=nonlinear_subdivision,
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
                .extract_surface(
                    algorithm=None,
                    nonlinear_subdivision=nonlinear_subdivision,
                )
                .extract_feature_edges()
            )
            actor = plotter.add_mesh(edges, color=edge_color, line_width=line_width)
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()

        if view == "default":
            view = _default_view(plotter, scene.mesh.points)

        plotter.camera_position = view

        if add_axes:
            plotter.add_axes()

        return plotter
