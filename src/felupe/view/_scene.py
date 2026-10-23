# SPDX-License-Identifier: GPL-3.0-or-later

from ._backends import backend_of


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
        >>> _ = scene.mesh.set_active_scalars(None)
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
        show_logo=False,
        **kwargs,
    ):
        """Plot scalars, selected by name and component.

        The scene is drawn by the plotting backend of a given plotter or, without a
        given plotter, by the current plotting backend (default is ``"pyvista"``), see
        :func:`~felupe.set_plotting_backend`. The arguments are those of the plotting
        backend of PyVista.

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
            A given plotter of another plotting backend is drawn by its plotting
            backend.
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
        show_logo : bool, optional
            A flag to show the FElupe logo in the top right corner (default is False).
        **kwargs : dict, optional
            Additional keyword arguments are passed to the
            :meth:`~pyvista.Plotter.add_mesh` method of the :class:`~pyvista.Plotter`.

        Returns
        -------
        plotter : pyvista.Plotter
            A Plotter object with methods ``plot()``, ``screenshot()``, etc. Another
            plotting backend returns its own plotter.

        See Also
        --------
        pyvista.Plotter : Plotting object to display vtk meshes or numpy arrays.
        felupe.set_plotting_backend : Set the current plotting backend by its name.
        """

        backend = backend_of(plotter)

        plotter = backend.prepare_plotter(
            plotter=plotter, theme=theme, off_screen=off_screen, notebook=notebook
        )

        return backend.add_view(
            plotter,
            self,
            name=name,
            component=component,
            label=label,
            factor=factor,
            show_edges=show_edges,
            show_undeformed=show_undeformed,
            cmap=cmap,
            view=view,
            scalar_bar_args=scalar_bar_args,
            scalar_bar_vertical=scalar_bar_vertical,
            add_axes=add_axes,
            extract_surface=extract_surface,
            nonlinear_subdivision=nonlinear_subdivision,
            smooth_shading=smooth_shading,
            split_sharp_edges=split_sharp_edges,
            edge_color=edge_color,
            line_width=line_width,
            show_logo=show_logo,
            **kwargs,
        )
