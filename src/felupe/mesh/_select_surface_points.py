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
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


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
    "Return the boundary surface with cell normals and original point ids."
    import pyvista as pv

    mesh = pv.wrap(mesh)

    # pass the point ids as point data, ``vtkOriginalPointIds`` of the extracted
    # surface are wrong for quadratic cells (e.g. hexahedron20 or tetra10)
    mesh.point_data["point_ids"] = np.arange(mesh.n_points)
    try:
        surface = mesh.extract_surface(
            pass_pointid=False, pass_cellid=False, algorithm=None
        )
    except TypeError:  # pragma: no cover (older PyVista without ``algorithm``)
        surface = mesh.extract_surface(pass_pointid=False, pass_cellid=False)

    # check before the normals are computed (PyVista raises a TypeError otherwise)
    if surface.GetNumberOfPolys() != surface.n_cells:
        raise ValueError("The extracted surface must only contain polygons.")

    # consistent orientation, so that the angle between neighbouring normals
    # is the kink angle of the surface (no point splitting -> ids unchanged)
    surface = surface.compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
    )
    return surface


def _topology(surface):
    from vtkmodules.util.numpy_support import vtk_to_numpy

    "Face connectivity, edges and all pairs of faces which share an edge."
    polys = surface.GetPolys()
    offsets = vtk_to_numpy(polys.GetOffsetsArray()).astype(np.int64)
    conn = vtk_to_numpy(polys.GetConnectivityArray()).astype(np.int64)
    n_faces = len(offsets) - 1

    # face id of each connectivity entry and the edge to the next vertex
    face = np.repeat(np.arange(n_faces), np.diff(offsets))
    nxt = np.arange(1, len(conn) + 1)
    nxt[offsets[1:] - 1] = offsets[:-1]
    edges = np.sort(np.column_stack([conn, conn[nxt]]), axis=1)

    unique_edges, edge_id = np.unique(edges, axis=0, return_inverse=True)
    edge_id = edge_id.ravel()
    order = np.argsort(edge_id, kind="stable")
    edge_faces = face[order]  # faces grouped by edge
    counts = np.bincount(edge_id, minlength=len(unique_edges))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    # all pairs of faces per edge (also for non-manifold edges)
    pairs = [np.zeros((0, 2), dtype=np.int64)]
    for k in np.unique(counts[counts > 1]):
        idx = starts[counts == k][:, None] + np.arange(k)
        f = edge_faces[idx]
        i, j = np.triu_indices(k, 1)
        pairs.append(np.column_stack([f[:, i].ravel(), f[:, j].ravel()]))
    pairs = np.concatenate(pairs)

    return dict(
        n_faces=n_faces,
        face=face,
        conn=conn,
        pairs=pairs,
        edges=unique_edges,
        edge_faces=edge_faces,
        counts=counts,
        starts=starts,
    )


def surface_patches(surface, topology, angle):
    "Patch label per face: connected faces with normal angles below ``angle``."

    normals = surface.cell_data["Normals"]
    a, b = topology["pairs"].T
    cos = np.einsum("ij,ij->i", normals[a], normals[b])
    keep = cos >= np.cos(np.deg2rad(angle))
    n = topology["n_faces"]
    graph = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _feature_edges(surface, topology, labels):
    "Lines along patch borders and open boundaries."
    import pyvista as pv

    lab = labels[topology["edge_faces"]]
    lo = np.minimum.reduceat(lab, topology["starts"])
    hi = np.maximum.reduceat(lab, topology["starts"])
    edges = topology["edges"][(topology["counts"] == 1) | (lo != hi)]
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel()
    return pv.PolyData(surface.points, lines=lines)


def select_surface_points(
    mesh,
    angle=30.0,
    slider=True,
    color="lightgrey",
    selected_color=None,
    show_edges=True,
    **kwargs,
):
    """Interactively select smooth surface patches and return their point ids.

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
    show_edges : bool, optional
        Whether to show mesh edges (default is True).
    **kwargs : optional
        Additional keyword arguments to pass to the PyVista plotter.

    Returns
    -------
    numpy.ndarray
        Sorted point ids (of ``mesh``) of all faces on the selected patches.

    Controls
    --------
    Left click    toggle the patch under the cursor (a drag rotates as usual)
    Right click   finish (a drag zooms as usual), same as q or closing the window
    Button / c    clear the selection
    """
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCellPicker

    surface = _extract_surface(mesh.as_unstructured_grid())
    topology = _topology(surface)
    point_ids = np.asarray(surface.point_data["point_ids"])

    state = dict(
        angle=float(angle),
        labels=surface_patches(surface, topology, angle),
        seeds=[],  # clicked faces, patches are re-evaluated if the angle changes
    )
    surface.cell_data["selected"] = np.zeros(surface.n_cells, dtype=np.uint8)

    def selected_faces():
        if not state["seeds"]:
            return np.zeros(topology["n_faces"], dtype=bool)
        return np.isin(state["labels"], state["labels"][state["seeds"]])

    def selected_surface_points():
        mask = selected_faces()[topology["face"]]
        return np.unique(topology["conn"][mask])

    if selected_color is None:
        selected_color = pv.global_theme.color

    # the colormap is created by matplotlib, which doesn't know all PyVista color
    # names (e.g. "light_blue" of the default theme) -> use hex strings instead
    color = pv.Color(color).hex_rgb
    selected_color = pv.Color(selected_color).hex_rgb

    # always use a native, blocking window
    plotter = pv.Plotter(notebook=False, **kwargs)
    actor = plotter.add_mesh(
        surface,
        scalars="selected",
        cmap=[color, selected_color],
        clim=[0, 1],
        n_colors=2,
        show_scalar_bar=False,
        show_edges=show_edges,
        edge_color="grey",
    )
    plotter.add_text(
        "Left click: toggle surface patch",
        position="lower_left",
        font_size=10,
    )
    plotter.add_text(
        "Right click: done",
        position="lower_right",
        font_size=10,
    )

    def update():
        faces = selected_faces()
        surface.cell_data["selected"] = faces.astype(np.uint8)
        plotter.add_mesh(
            _feature_edges(surface, topology, state["labels"]),
            color="black",
            line_width=3,
            name="feature_edges",
            pickable=False,
        )
        points = selected_surface_points()
        if len(points) > 0:
            plotter.add_points(
                surface.points[points],
                color=selected_color,
                point_size=8,
                name="selected_points",
                pickable=False,
            )
        else:
            plotter.remove_actor("selected_points")
        n_patches = len(np.unique(state["labels"][state["seeds"]]))
        plotter.add_text(
            f"Angle: {state['angle']:.0f} deg   "
            f"Surface patches: {n_patches}   Points: {len(points)}",
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
        label = state["labels"][face]
        seeds = [s for s in state["seeds"] if state["labels"][s] != label]
        if len(seeds) == len(state["seeds"]):
            seeds.append(face)  # not selected yet -> select
        state["seeds"] = seeds
        update()

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

    def clear():
        state["seeds"] = []
        update()

    def on_clear_button(_):
        clear_button.GetRepresentation().SetState(0)  # behave like a push button
        clear()

    def set_angle(value):
        state["angle"] = np.round(float(value))
        state["labels"] = surface_patches(surface, topology, value)
        update()

    _on_click(plotter.iren.interactor, "Left", toggle_patch)
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

    update()
    plotter.show()

    return np.unique(point_ids[selected_surface_points()])
