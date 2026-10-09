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

from ._mesh_select_surface_points import (
    EXCLUDED,
    SELECTED,
    UNSELECTED,
    _colors,
    _cycle,
    _extract_surface,
    _initial_seeds,
    _patch_info,
    _patch_status,
    _show,
    _surface_patches,
)


def _lines(points, edges):
    "Return the edges (pairs of point ids) as lines."
    import pyvista as pv

    return pv.PolyData(points, lines=np.pad(edges, ((0, 0), (1, 0)), constant_values=2))


def _edges(borders):
    """Return the borders of the surface patches as sorted pairs of point ids, without
    duplicates. The border between two patches is a boundary edge of both patches."""

    if borders.n_cells == 0:
        return np.zeros((0, 2), dtype=int)

    point_ids = np.asarray(borders.point_data["point_ids"])
    lines = borders.lines.reshape(-1, 3)[:, 1:]  # two points per line

    return np.unique(np.sort(point_ids[lines], axis=1), axis=0)


def _edge_patches(points, edges, angle):
    """Split the edges at all points with a kink angle above ``angle`` (in degrees)
    and at all points without exactly two edges (ends and junctions). Return the
    patch label per edge (the connected chains of the split edges)."""
    import pyvista as pv
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from vtkmodules.vtkFiltersCore import vtkConnectivityFilter

    if len(edges) == 0:
        return np.zeros(0, dtype=int)

    # the edge-ends, the other end of edge-end ``i`` is ``i ^ 1``
    ends = edges.ravel()
    order = np.argsort(ends, kind="stable")
    point_ids, first, count = np.unique(
        ends[order], return_index=True, return_counts=True
    )

    # the kink angle at all points with two edge-ends (a, b)
    two = count == 2
    a, b, p = order[first[two]], order[first[two] + 1], point_ids[two]
    u = points[p] - points[ends[a ^ 1]]
    v = points[ends[b ^ 1]] - points[p]
    cos = np.sum(u * v, axis=1) / np.linalg.norm(u, axis=1) / np.linalg.norm(v, axis=1)

    smooth = np.zeros(len(points), dtype=bool)
    smooth[p[cos >= np.cos(np.radians(angle))]] = True

    # split all other points: a new point for each edge-end
    nodes = np.where(smooth[ends], ends, len(points) + np.arange(len(ends)))
    split = pv.PolyData(
        np.vstack([points, points[ends]]),
        lines=np.pad(nodes.reshape(-1, 2), ((0, 0), (1, 0)), constant_values=2),
    )

    connectivity = vtkConnectivityFilter()
    connectivity.SetInputData(split)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    regions = connectivity.GetOutput().GetCellData().GetArray("RegionId")

    return vtk_to_numpy(regions)


def select_edge_points(
    mesh,
    angle=30.0,
    slider=True,
    selected=None,
    excluded=None,
    color="black",
    selected_color=None,
    excluded_color="red",
    show_edges=True,
    **kwargs,
):
    """Interactively select (and exclude) smooth edge patches and return their point
    ids.

    The edges are the borders of the surface patches (see
    :func:`~felupe.view.select_surface_points`), i.e. the edges between neighbouring
    faces with a kink angle above ``angle`` and the boundary edges of open surfaces,
    e.g. of 2d-meshes. An edge patch is a chain of connected edges. The chains are
    split at points with a kink angle above ``angle`` and at points with more than two
    edges.

    Parameters
    ----------
    mesh : Mesh
        The mesh from which to select edge points.
    angle : float, optional
        Max. angle in degrees between the normals of two neighbouring faces or between
        two neighbouring edges to be treated as one connected patch (default is 30).
    slider : bool, optional
        Show a slider to change the angle interactively (default is True).
    selected : array_like or None, optional
        Coordinates of points to initially select the edge patches of their closest
        edges, e.g. ``[(1.0, 1.0, 0.5)]``, same as a left click on these patches.
        Default is None.
    excluded : array_like or None, optional
        Coordinates of points to initially exclude the edge patches of their closest
        edges, same as two left clicks on these patches. Default is None.
    color : str, optional
        Color of unselected edge patches (default is "black").
    selected_color : str or None, optional
        Color of selected edge patches. Default is None, which lets PyVista choose
        the default color based on the global theme.
    excluded_color : str, optional
        Color of excluded edge patches (default is "red").
    show_edges : bool, optional
        Whether to show the edges of the mesh on the surface (default is True).
    **kwargs : optional
        Additional keyword arguments to pass to the PyVista plotter.

    Returns
    -------
    numpy.ndarray
        Sorted point ids (of ``mesh``) of all edges on the selected patches, without
        the points of all edges on the excluded patches.

    Notes
    -----
    The selection is controlled by the mouse and the keyboard.

    * **Left click**: cycle the patch next to the cursor from unselected to selected,
      from selected to excluded and from excluded back to unselected (a drag rotates
      as usual). Edges which are hidden behind the surface are ignored.
    * **Right click**: finish (a drag zooms as usual), same as ``q`` or closing the
      window.
    * **Button** or ``c``: clear the selection, i.e. all patches are unselected.

    Excluded patches have priority over selected patches: the points shared by a
    selected and an excluded patch, e.g. a corner point between two edge patches, are
    not selected. If both a selected and an excluded patch are merged into one patch by
    an increased angle, the merged patch is excluded.

    Examples
    --------
    The edge patch at :math:`x=z=1` of a cube is selected and the edge patch at
    :math:`x=1, y=0` is excluded. Here, the patches are given by points next to them,
    which is the same as a left click on the edge at :math:`x=z=1` and two left clicks
    on the edge at :math:`x=1, y=0`. The selection is finished by a right click (not
    required for the static images of the documentation). The selected points are
    plotted on the mesh, the common corner point of both patches is not selected.

    ..  pyvista-plot::
        :force_static:

        >>> import felupe as fem
        >>> import pyvista as pv
        >>>
        >>> mesh = fem.Cube(n=6)
        >>> point_ids = mesh.select_edge_points(
        ...     selected=[(1.0, 0.5, 1.0)], excluded=[(1.0, 0.0, 0.5)]
        ... )
        >>>
        >>> plotter = pv.Plotter()
        >>> points = mesh.points[point_ids]
        >>> actor = plotter.add_points(points, color="red", point_size=10)
        >>> mesh.plot(plotter=plotter).show()

    See Also
    --------
    felupe.view.select_surface_points : Interactively select smooth surface patches
        and return their point ids.
    """
    import pyvista as pv
    from vtkmodules.vtkRenderingCore import vtkCellPicker

    surface, _ = _extract_surface(mesh)
    points = np.pad(mesh.points, ((0, 0), (0, 3 - mesh.dim)))

    # clicked edges as pairs of point ids with the status of their patches,
    # re-evaluated if the angle changes (the edges themselves depend on the angle)
    edges = _edges(_surface_patches(surface, angle)[1])
    seeds = _initial_seeds(_lines(points, edges), mesh.dim, selected, excluded)
    state = dict(seeds={tuple(edges[i].tolist()): seed for i, seed in seeds.items()})

    def current_seeds():
        "Return the seeds of the current edges by their index."
        index = state["index"]
        return {index[s]: seed for s, seed in state["seeds"].items() if s in index}

    def edge_status():
        return _patch_status(state["labels"], current_seeds())

    def selected_points(status):
        "Return the point ids of selected edges without the points of excluded edges."
        edges = state["edges"]
        return np.setdiff1d(edges[status == SELECTED], edges[status == EXCLUDED])

    color, selected_color, excluded_color = _colors(
        color, selected_color, excluded_color
    )

    # always use a native, blocking window
    plotter = pv.Plotter(notebook=False, **kwargs)
    surface_actor = plotter.add_mesh(
        surface,
        color="lightgrey",
        show_edges=show_edges,
        edge_color="grey",
    )

    def update():
        status = edge_status()
        if "edge_patches" in plotter.actors:
            state["lines"].cell_data["status"] = status
        point_ids = selected_points(status)
        if len(point_ids) > 0:
            plotter.add_points(
                points[point_ids],
                color=selected_color,
                point_size=8,
                name="selected_points",
                pickable=False,
            )
        else:
            plotter.remove_actor("selected_points")
        plotter.add_text(
            f"Angle: {state['angle']:.0f} deg   "
            f"Edge patches: {_patch_info(state['labels'], status)}   "
            f"Points: {len(point_ids)}",
            position="upper_left",
            font_size=10,
            name="info",
        )
        plotter.render()

    # pick edges next to the cursor, tolerance as fraction of the window diagonal
    edge_picker = vtkCellPicker()
    edge_picker.SetTolerance(5e-3)
    edge_picker.PickFromListOn()

    surface_picker = vtkCellPicker()
    surface_picker.SetTolerance(1e-4)
    surface_picker.PickFromListOn()
    surface_picker.AddPickList(surface_actor)

    def visible(point):
        """Return True if the point is not hidden behind the surface, i.e. if the
        surface is not intersected in front of the point by its line of sight."""
        renderer = plotter.renderer
        renderer.SetWorldPoint(*point, 1.0)
        renderer.WorldToDisplay()
        x, y = renderer.GetDisplayPoint()[:2]

        if not surface_picker.Pick(x, y, 0, renderer):
            return True

        # the line of sight starts at the near clipping plane
        renderer.SetDisplayPoint(x, y, 0.0)
        renderer.DisplayToWorld()
        near = np.array(renderer.GetWorldPoint())
        near = near[:3] / near[3]

        depth = np.linalg.norm(point - near)
        depth_surface = np.linalg.norm(surface_picker.GetPickPosition() - near)

        return depth - depth_surface < 1e-3 * surface.length

    def toggle_patch(x, y):
        if len(state["edges"]) == 0:
            return
        edge_picker.Pick(x, y, 0, plotter.renderer)
        edge = edge_picker.GetCellId()
        if edge < 0 or not visible(np.array(edge_picker.GetPickPosition())):
            return
        # keep the seeds of edges which don't exist for the current angle
        index = state["index"]
        seeds = {s: seed for s, seed in state["seeds"].items() if s not in index}
        for i, seed in _cycle(current_seeds(), state["labels"], edge).items():
            seeds[tuple(state["edges"][i].tolist())] = seed
        state["seeds"] = seeds
        update()

    def clear():
        state["seeds"] = {}
        update()

    def set_angle(value):
        _, borders = _surface_patches(surface, value)
        edges = _edges(borders)

        state["angle"] = float(value)
        state["edges"] = edges
        state["labels"] = _edge_patches(points, edges, value)
        state["index"] = {edge: i for i, edge in enumerate(map(tuple, edges.tolist()))}

        plotter.remove_actor("edge_patches")
        edge_picker.InitializePickList()

        if len(edges) > 0:
            state["lines"] = _lines(points, edges)
            state["lines"].cell_data["status"] = np.full(
                len(edges), UNSELECTED, dtype=np.uint8
            )
            actor = plotter.add_mesh(
                state["lines"],
                scalars="status",
                cmap=[color, selected_color, excluded_color],
                clim=[UNSELECTED, EXCLUDED],
                n_colors=3,
                show_scalar_bar=False,
                line_width=4,
                name="edge_patches",
            )
            actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
            edge_picker.AddPickList(actor)

        update()

    _show(
        plotter,
        points,
        "edge",
        toggle_patch,
        clear,
        set_angle,
        angle,
        slider,
        action="select / exclude / unselect",
    )

    return selected_points(edge_status())
