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


def _extract_surface(mesh):
    "Return the boundary surface of a mesh with the original point ids."

    # pass the point ids as point data, ``vtkOriginalPointIds`` of the extracted
    # surface are wrong for quadratic cells (e.g. hexahedron20 or tetra10)
    grid = mesh.as_unstructured_grid()
    grid.point_data["point_ids"] = np.arange(grid.n_points)
    try:
        surface = grid.extract_surface(
            pass_pointid=False, pass_cellid=False, algorithm=None
        )
    except TypeError:  # pragma: no cover (older PyVista without ``algorithm``)
        surface = grid.extract_surface(pass_pointid=False, pass_cellid=False)

    if surface.GetNumberOfPolys() != surface.n_cells:
        raise ValueError("The extracted surface must only contain polygons.")

    return surface


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


def _show(plotter, name, toggle, clear, set_angle, angle, slider, action="toggle"):
    """Add the controls of an interactive selection of patches to the plotter, show it
    and wait until the selection is finished.

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

    # look at planar meshes from the top, same as ``Scene.plot()``
    if np.allclose(plotter.bounds[4:], 0):
        plotter.view_xy()
        plotter.enable_parallel_projection()

    plotter.show()


def select_surface_points(
    mesh,
    angle=30.0,
    slider=True,
    color="lightgrey",
    selected_color=None,
    excluded_color="darkred",
    show_edges=True,
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
    color : str, optional
        Color of unselected surface patches (default is "lightgrey").
    selected_color : str or None, optional
        Color of selected surface patches. Default is None, which lets PyVista choose
        the default color based on the global theme.
    excluded_color : str, optional
        Color of excluded surface patches (default is "darkred").
    show_edges : bool, optional
        Whether to show mesh edges (default is True).
    **kwargs : optional
        Additional keyword arguments to pass to the PyVista plotter.

    Returns
    -------
    numpy.ndarray
        Sorted point ids (of ``mesh``) of all faces on the selected patches, without
        the points of all faces on the excluded patches.

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

    See Also
    --------
    felupe.view.select_edge_points : Interactively select smooth edge patches and
        return their point ids.
    """
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCellPicker

    UNSELECTED, SELECTED, EXCLUDED = 0, 1, 2

    surface = _extract_surface(mesh)
    points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.dim)))

    # clicked faces with the status of their patches, re-evaluated if the angle changes
    state = dict(seeds={})
    surface.cell_data["status"] = np.full(surface.n_cells, UNSELECTED, dtype=np.uint8)

    def patch_status():
        "Return the status of the patches per face, excluded patches have priority."
        labels = state["labels"]
        status = np.full(surface.n_cells, UNSELECTED, dtype=np.uint8)
        for value in [SELECTED, EXCLUDED]:
            seeds = [face for face, seed in state["seeds"].items() if seed == value]
            status[np.isin(labels, labels[seeds])] = value
        return status

    def point_ids(faces):
        "Return the sorted point ids (of ``mesh``) of the faces."
        cells = surface.extract_cells(np.flatnonzero(faces))
        return np.unique(cells.point_data.get("point_ids", np.array([], dtype=int)))

    def selected_points(status):
        "Return the point ids of selected faces without the points of excluded faces."
        return np.setdiff1d(
            point_ids(status == SELECTED), point_ids(status == EXCLUDED)
        )

    color, selected_color, excluded_color = _colors(
        color, selected_color, excluded_color
    )

    # always use a native, blocking window
    plotter = pv.Plotter(notebook=False, **kwargs)
    actor = plotter.add_mesh(
        surface,
        scalars="status",
        cmap=[color, selected_color, excluded_color],
        clim=[UNSELECTED, EXCLUDED],
        n_colors=3,
        show_scalar_bar=False,
        show_edges=show_edges,
        edge_color="grey",
    )

    def update():
        status = patch_status()
        surface.cell_data["status"] = status
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
        labels = state["labels"]
        n_selected = len(np.unique(labels[status == SELECTED]))
        n_excluded = len(np.unique(labels[status == EXCLUDED]))
        plotter.add_text(
            f"Angle: {state['angle']:.0f} deg   "
            f"Surface patches: {n_selected} selected, {n_excluded} excluded   "
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
        labels = state["labels"]
        status = int(patch_status()[face])
        seeds = {
            s: seed for s, seed in state["seeds"].items() if labels[s] != labels[face]
        }
        if status != EXCLUDED:
            seeds[face] = status + 1  # unselected -> selected -> excluded
        state["seeds"] = seeds  # excluded -> unselected
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
        "surface",
        toggle_patch,
        clear,
        set_angle,
        angle,
        slider,
        action="select / exclude / unselect",
    )

    return selected_points(patch_status())
