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

from ._scene import _default_view


def _on_click(interactor, button, callback, tolerance=6):
    """Call ``callback(x, y)`` on a click of ``button`` ("Left" or "Right"), i.e.
    press and release (almost) at the same position. A drag is ignored, so
    rotating and zooming keep working as usual."""
    press = []

    def on_press(obj, _):
        press[:] = obj.GetEventPosition()

    def on_release(obj, _):
        if press:
            x, y = obj.GetEventPosition()
            moved = abs(x - press[0]) + abs(y - press[1])
            press.clear()
            if moved <= tolerance:
                callback(x, y)

    for event, func in [("Press", on_press), ("Release", on_release)]:
        tag = interactor.AddObserver(f"{button}Button{event}Event", func)
        # Passive observers are notified of every event, independent of the
        # interactor style. The style grabs the focus on a button press (for
        # rotate / zoom) and would otherwise swallow the release event.
        interactor.GetCommand(tag).SetPassiveObserver(True)


def _polygons(dataset, nonlinear_subdivision=1):
    """Return the surface (polygons) of a dataset. Nonlinear faces are triangulated
    and subdivided ``nonlinear_subdivision - 1`` times."""
    kwargs = dict(
        pass_pointid=False,
        pass_cellid=False,
        nonlinear_subdivision=nonlinear_subdivision,
    )
    try:
        return dataset.extract_surface(**kwargs, algorithm=None)
    except TypeError:  # pragma: no cover (older PyVista without ``algorithm``)
        return dataset.extract_surface(**kwargs)


def _extract_surface(mesh, nonlinear_subdivision=1):
    """Return the boundary surface (polygons) and the boundary faces of a mesh.

    The faces are the cells of an unstructured grid with the original point ids as point
    data ``"point_ids"``, i.e. the nonlinear faces of quadratic (or Lagrange) cells are
    kept. The surface contains the polygons of the triangulated (and subdivided) faces
    with the ids of their faces as cell data ``"face_ids"``.

    The subdivision of nonlinear faces creates new points, which are not points of the
    mesh. PyVista interpolates the point data on these new points, i.e. their point ids
    are wrong (but look valid). Hence, the point ids of the surface are only included
    for ``nonlinear_subdivision=1``. The point ids of the polygons are given by the
    points of their faces.
    """
    from pyvista import wrap
    from vtkmodules.vtkFiltersGeometry import vtkUnstructuredGridGeometryFilter

    # pass the point ids as point data, ``vtkOriginalPointIds`` of the extracted
    # surface are wrong for quadratic cells (e.g. hexahedron20 or tetra10)
    grid = mesh.as_unstructured_grid()
    grid.point_data["point_ids"] = np.arange(grid.n_points)

    geometry = vtkUnstructuredGridGeometryFilter()
    geometry.SetInputData(grid)
    geometry.Update()

    faces = wrap(geometry.GetOutput())
    faces.cell_data["face_ids"] = np.arange(faces.n_cells)

    surface = _polygons(faces, nonlinear_subdivision)

    if surface.GetNumberOfPolys() != surface.n_cells:
        raise ValueError("The extracted surface must only contain polygons.")

    if nonlinear_subdivision > 1:
        surface.point_data.remove("point_ids")

    return surface, faces


def _face_point_ids(faces, mask):
    "Return the sorted point ids (of the mesh) of the masked faces."
    cells = faces.extract_cells(np.flatnonzero(mask))
    return np.unique(cells.point_data.get("point_ids", np.array([], dtype=int)))


def _face_edges(faces, nonlinear_subdivision):
    "Return the (subdivided) edges of the faces, i.e. the boundary edges per face."
    polygons = _polygons(faces.separate_cells(), nonlinear_subdivision)
    return polygons.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )


def _surface_patches(surface, angle):
    """Split the surface at all edges with a kink angle above ``angle`` (in degrees).
    Return the patch label per face (the connected regions of the split surface) and
    the borders of the patches (the boundary edges of the split surface)."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkConnectivityFilter

    # consistent orientation, so that the angle between neighbouring normals
    # is the kink angle of the surface (the order of the faces is not changed)
    split = surface.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=True,
        feature_angle=angle,
        consistent_normals=True,
        auto_orient_normals=False,
    )

    connectivity = vtkConnectivityFilter()
    connectivity.SetInputData(split)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    regions = connectivity.GetOutput().GetCellData().GetArray("RegionId")

    borders = split.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return vtk_to_numpy(regions), borders


# the status of a patch
UNSELECTED, SELECTED, EXCLUDED = 0, 1, 2


def _patch_status(labels, seeds):
    """Return the status of the patches per item (face or edge) with the patch labels
    per item. The seeds are the clicked items with the status of their patches
    ``{item: status}``. A patch with selected and excluded seeds, e.g. merged by an
    increased angle, is excluded (excluded patches have priority)."""
    status = np.full(len(labels), UNSELECTED, dtype=np.uint8)
    for value in [SELECTED, EXCLUDED]:
        items = [item for item, seed in seeds.items() if seed == value]
        status[np.isin(labels, labels[items])] = value
    return status


def _cycle(seeds, labels, item):
    """Cycle the status of the patch of the clicked item, from unselected to selected,
    from selected to excluded and from excluded to unselected. Return the new seeds,
    i.e. the other seeds on this patch are replaced by the clicked item."""
    status = int(_patch_status(labels, seeds)[item])
    seeds = {s: seed for s, seed in seeds.items() if labels[s] != labels[item]}
    if status != EXCLUDED:
        seeds[item] = status + 1
    return seeds


def _initial_seeds(dataset, dim, selected, excluded):
    """Return the initial seeds ``{cell: status}``, i.e. the closest cells (faces or
    edges) of the dataset to the selected and excluded points (with ``dim``
    coordinates). Excluded points have priority."""
    seeds = {}
    for points, status in [(selected, SELECTED), (excluded, EXCLUDED)]:
        if points is None or dataset.n_cells == 0:
            continue
        points = np.asarray(points, dtype=float).reshape(-1, dim)
        points = np.pad(points, ((0, 0), (0, 3 - dim)))
        for cell in np.atleast_1d(dataset.find_closest_cell(points)):
            seeds[int(cell)] = status
    return seeds


def _patch_info(labels, status):
    "Return the number of selected and excluded patches as text."
    n_selected = len(np.unique(labels[status == SELECTED]))
    n_excluded = len(np.unique(labels[status == EXCLUDED]))
    return f"{n_selected} selected, {n_excluded} excluded"


def _colors(*colors):
    """Return the colors of the patches as hex strings. A color of None is the default
    color of the global theme."""
    import pyvista as pv

    # the colormap is created by matplotlib, which doesn't know all PyVista color
    # names (e.g. "light_blue" of the default theme) -> use hex strings instead
    return [
        pv.Color(pv.global_theme.color if color is None else color).hex_rgb
        for color in colors
    ]


def _show(
    plotter, points, name, toggle, clear, set_angle, angle, slider, action="toggle"
):
    """Add the controls of an interactive selection of patches to the plotter, show it
    and wait until the selection is finished. The camera looks at the (3d-) points with
    the default view of :meth:`~felupe.view.Scene.plot`.

    * **Left click**: ``toggle(x, y)`` with the display coordinates of the click, the
      ``action`` is shown in the help text.
    * **Right click**: finish, same as ``q`` or closing the window.
    * **Button** or ``c``: ``clear()``.
    * **Slider**: ``set_angle(value)``, also called once with the initial ``angle``.
    """

    plotter.add_text(
        f"Left click: {action} {name} patch",
        position="lower_left",
        font_size=10,
    )
    plotter.add_text(
        "Right click: done",
        position="lower_right",
        font_size=10,
    )

    def finish(*_):
        plotter.add_text(
            "selection finished - closing window ...",
            position="upper_left",
            font_size=10,
            name="info",
        )
        plotter.render()
        interactor = plotter.iren.interactor
        interactor.ExitCallback()  # same as the window's close button or "q"
        interactor.SetDone(True)  # PyVista >= 0.48 polls this flag on Windows

    def on_clear_button(_):
        clear_button.GetRepresentation().SetState(0)  # behave like a push button
        clear()

    _on_click(plotter.iren.interactor, "Left", toggle)
    _on_click(plotter.iren.interactor, "Right", finish)
    plotter.add_key_event("c", clear)

    clear_button = plotter.add_checkbox_button_widget(
        on_clear_button,
        value=False,
        position=(10, 40),
        size=24,
        border_size=2,
        color_on="grey",
        color_off="lightgrey",
    )
    plotter.add_text("Clear selection", position=(42, 44), font_size=10)

    if slider:
        plotter.add_slider_widget(
            set_angle,
            rng=[0.0, 180.0],
            value=float(angle),
            pointa=(0.7, 0.9),
            pointb=(0.9, 0.9),
            title="Angle in deg",
            fmt="%.0f",
            interaction_event="end",
        )

    set_angle(angle)

    # same default view and axes as ``Scene.plot()``
    plotter.camera_position = _default_view(plotter, points)
    plotter.add_axes()

    plotter.show()


def select_surface_points(
    mesh,
    angle=30.0,
    slider=True,
    selected=None,
    excluded=None,
    color="lightgrey",
    selected_color=None,
    excluded_color="darkred",
    show_edges=False,
    nonlinear_subdivision=1,
    **kwargs,
):
    """Interactively select (and exclude) smooth surface patches and return their point
    ids.

    Parameters
    ----------
    mesh : Mesh
        The mesh from which to select surface points.
    angle : float, optional
        Max. angle in degrees between the normals of two neighbouring faces
        to be treated as one connected patch (default is 30).
    slider : bool, optional
        Show a slider to change the angle interactively (default is True).
    selected : array_like or None, optional
        Coordinates of points to initially select the surface patches of their closest
        faces, e.g. ``[(1.0, 0.5, 0.5)]``, same as a left click on these patches.
        Default is None.
    excluded : array_like or None, optional
        Coordinates of points to initially exclude the surface patches of their
        closest faces, same as two left clicks on these patches. Default is None.
    color : str, optional
        Color of unselected surface patches (default is "lightgrey").
    selected_color : str or None, optional
        Color of selected surface patches. Default is None, which lets PyVista choose
        the default color based on the global theme.
    excluded_color : str, optional
        Color of excluded surface patches (default is "darkred").
    show_edges : bool, optional
        Whether to show the edges of the mesh (default is False). The patches are
        separated by their borders anyway.
    nonlinear_subdivision : int, optional
        Number of subdivisions to generate a smooth surface based on the mid-edge
        points of quadratic (or Lagrange) cells, same as in
        :meth:`~felupe.Mesh.plot` (default is 1, no subdivision). If greater than 1,
        the surface is plotted with smooth shading and the shown edges are the
        (curved) edges of the faces. The patches are evaluated on the subdivided
        surface.
    **kwargs : optional
        Additional keyword arguments to pass to the PyVista plotter.

    Returns
    -------
    numpy.ndarray
        Sorted point ids (of ``mesh``) of all faces on the selected patches, without
        the points of all faces on the excluded patches. A face is on a patch if one of
        its triangles (on the subdivided surface) is on the patch.

    Notes
    -----
    The selection is controlled by the mouse and the keyboard.

    * **Left click**: cycle the patch under the cursor from unselected to selected,
      from selected to excluded and from excluded back to unselected (a drag rotates
      as usual).
    * **Right click**: finish (a drag zooms as usual), same as ``q`` or closing the
      window.
    * **Button** or ``c``: clear the selection, i.e. all patches are unselected.

    Excluded patches have priority over selected patches: the points shared by a
    selected and an excluded patch, e.g. the points on their common border, are not
    selected. If both a selected and an excluded patch are merged into one patch by an
    increased angle, the merged patch is excluded.

    The subdivision of nonlinear faces creates new points, which are only used for the
    visualization. The returned point ids are always the point ids of the faces of the
    mesh, e.g. all eight points of a face of a quadratic hexahedron (including the
    mid-edge points), independent of ``nonlinear_subdivision``.

    Examples
    --------
    The surface patch at :math:`x=1` of a cube is selected and the surface patch at
    :math:`z=1` is excluded. Here, the patches are given by points next to them, which
    is the same as a left click on the patch at :math:`x=1` and two left clicks on the
    patch at :math:`z=1`. The selection is finished by a right click (not required for
    the static images of the documentation). The selected points are plotted on the
    mesh, the points on the common edge of both patches are not selected.

    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>> import pyvista as pv
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> point_ids = mesh.select_surface_points(
        ...     selected=[(1.0, 0.5, 0.5)], excluded=[(0.5, 0.5, 1.0)]
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> points = mesh.points[point_ids]
        >>> actor = plotter.add_points(points, color="red", point_size=10)
        >>> mesh.plot(plotter=plotter).show()

    See Also
    --------
    felupe.view.select_edge_points : Interactively select smooth edge patches and
        return their point ids.
    """
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCellPicker

    subdivided = nonlinear_subdivision > 1
    surface, faces = _extract_surface(mesh, nonlinear_subdivision)
    points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.dim)))

    # clicked polygons with the status of their patches, re-evaluated if the angle
    # changes (the polygons of the subdivided faces, if nonlinear_subdivision > 1)
    state = dict(seeds=_initial_seeds(surface, mesh.dim, selected, excluded))
    surface.cell_data["status"] = np.full(surface.n_cells, UNSELECTED, dtype=np.uint8)

    def patch_status():
        return _patch_status(state["labels"], state["seeds"])

    def point_ids(polygons):
        """Return the sorted point ids (of ``mesh``) of the faces of the polygons. The
        point ids of the (subdivided) polygons themselves are not used."""
        mask = np.zeros(faces.n_cells, dtype=bool)
        mask[surface.cell_data["face_ids"][polygons]] = True
        return _face_point_ids(faces, mask)

    def selected_points(status):
        "Return the point ids of selected faces without the points of excluded faces."
        return np.setdiff1d(
            point_ids(status == SELECTED), point_ids(status == EXCLUDED)
        )

    color, selected_color, excluded_color = _colors(
        color, selected_color, excluded_color
    )

    # the plotted surface, smooth shading for a subdivided surface: the points are split
    # at sharp edges, the order of the polygons is not changed (same as the picked ids)
    plotted = surface
    if subdivided:
        plotted = surface.compute_normals(
            cell_normals=False,
            point_normals=True,
            split_vertices=True,
            feature_angle=pv.global_theme.sharp_edges_feature_angle,
            auto_orient_normals=False,
        )

    # always use a native, blocking window
    plotter = pv.Plotter(notebook=False, **kwargs)
    actor = plotter.add_mesh(
        plotted,
        scalars="status",
        cmap=[color, selected_color, excluded_color],
        clim=[UNSELECTED, EXCLUDED],
        n_colors=3,
        show_scalar_bar=False,
        show_edges=show_edges and not subdivided,
        edge_color="grey",
    )

    if subdivided:
        actor.GetProperty().SetInterpolationToPhong()

        # hide the edges of the subdivided polygons, show the edges of the faces
        if show_edges:
            edges = plotter.add_mesh(
                _face_edges(faces, nonlinear_subdivision),
                color="grey",
                pickable=False,
            )
            edges.mapper.SetResolveCoincidentTopologyToPolygonOffset()

    def update():
        status = patch_status()
        plotted.cell_data["status"] = status
        selected = selected_points(status)
        if len(selected) > 0:
            plotter.add_points(
                points[selected],
                color=selected_color,
                point_size=8,
                name="selected_points",
                pickable=False,
            )
        else:
            plotter.remove_actor("selected_points")
        plotter.add_text(
            f"Angle: {state['angle']:.0f} deg   "
            f"Surface patches: {_patch_info(state['labels'], status)}   "
            f"Points: {len(selected)}",
            position="upper_left",
            font_size=10,
            name="info",
        )
        plotter.render()

    picker = vtkCellPicker()
    picker.SetTolerance(1e-4)
    picker.PickFromListOn()
    picker.AddPickList(actor)

    def toggle_patch(x, y):
        picker.Pick(x, y, 0, plotter.renderer)
        face = picker.GetCellId()
        if face < 0:
            return
        state["seeds"] = _cycle(state["seeds"], state["labels"], face)
        update()

    def clear():
        state["seeds"] = {}
        update()

    def set_angle(value):
        state["angle"] = float(value)
        state["labels"], borders = _surface_patches(surface, value)
        plotter.remove_actor("patch_borders")
        if borders.n_cells > 0:
            plotter.add_mesh(
                borders,
                color="black",
                line_width=3,
                name="patch_borders",
                pickable=False,
            )
        update()

    _show(
        plotter,
        points,
        "surface",
        toggle_patch,
        clear,
        set_angle,
        angle,
        slider,
        action="select / exclude / unselect",
    )

    return selected_points(patch_status())
