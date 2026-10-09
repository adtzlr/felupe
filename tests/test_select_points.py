# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest

import felupe as fem


@contextmanager
def interact(*actions):
    """Run an interactive selection off-screen. Instead of starting the event loop,
    ``Plotter.show()`` applies the given actions to the plotter, i.e. it simulates
    the clicks, key presses and widget interactions of a user.

    Nothing is rendered, because the CI runners have no (virtual) display. The cell
    picker works on the geometry and the camera only. The interactor isn't enabled
    without rendering, hence the events are invoked directly. The interactor style
    (rotate, zoom, ...) and the widgets must not process the mouse events (they
    would crash without a render window).
    """

    pv = pytest.importorskip("pyvista")

    def show(plotter, *args, **kwargs):
        plotter.view_isometric()  # camera looks at the faces x=1, y=1 and z=1
        plotter.iren.interactor.SetInteractorStyle(None)
        for widget in [
            *widgets(plotter).button_widgets,
            *widgets(plotter).slider_widgets,
        ]:
            widget.ProcessEventsOff()
        for action in actions:
            action(plotter)

    with mock.patch.object(pv, "OFF_SCREEN", True):
        with mock.patch.object(pv.Plotter, "show", show):
            yield


def click(point, button="Left", drag=0, offset=(0, 0)):
    """Click on a point given in world (x, y, z) or display (x, y) coordinates, moved
    by ``offset`` pixels. The button is released ``drag`` pixels away from the position
    where it was pressed.
    """

    def action(plotter):
        if len(point) == 3:
            plotter.renderer.SetWorldPoint(*point, 1.0)
            plotter.renderer.WorldToDisplay()
            x, y = plotter.renderer.GetDisplayPoint()[:2]
        else:
            x, y = point

        x, y = int(round(x + offset[0])), int(round(y + offset[1]))
        interactor = plotter.iren.interactor
        interactor.SetEventPosition(x, y)
        interactor.InvokeEvent(f"{button}ButtonPressEvent")
        interactor.SetEventPosition(x + drag, y + drag)
        interactor.InvokeEvent(f"{button}ButtonReleaseEvent")

    return action


def press_key(key):
    def action(plotter):
        interactor = plotter.iren.interactor
        interactor.SetKeySym(key)
        interactor.InvokeEvent("KeyPressEvent")

    return action


def widgets(plotter):
    "Return the widgets of the plotter (moved to ``Plotter.widgets`` in PyVista 0.49)."
    return getattr(plotter, "widgets", plotter)


def push_clear_button(plotter):
    button = widgets(plotter).button_widgets[0]
    button.GetRepresentation().SetState(1)
    button.InvokeEvent("StateChangedEvent")


def move_slider(value):
    def action(plotter):
        slider = widgets(plotter).slider_widgets[0]
        slider.GetRepresentation().SetValue(value)
        slider.InvokeEvent("EndInteractionEvent")

    return action


def parallel_projection(expected):
    "Check the camera projection (it is not changed by the isometric view)."

    def action(plotter):
        camera = plotter.renderer.GetActiveCamera()
        assert bool(camera.GetParallelProjection()) == expected

    return action


finish = click((1, 1), button="Right")


def point_ids(mask):
    return np.arange(len(mask))[mask]


def test_select_surface_points():
    meshes = [
        fem.Cube(n=3),
        fem.Cube(n=3).triangulate(),
        fem.Cube(n=3).add_midpoints_edges(),
    ]

    for mesh in meshes:
        x, y, z = mesh.points.T

        with interact(
            click((1.0, 0.5, 0.5)),  # select the patch x=1
            click((0.5, 1.0, 0.5)),  # select the patch y=1
            click((0.5, 1.0, 0.5)),  # exclude the patch y=1
            click((0.5, 1.0, 0.5)),  # unselect the patch y=1
            click((0.5, 0.5, 1.0), drag=50),  # a drag rotates, no selection
            click((2, 2)),  # click on the background, no selection
            finish,
        ):
            selected = fem.view.select_surface_points(mesh)

        assert np.array_equal(selected, point_ids(np.isclose(x, 1)))

        with interact(
            click((1.0, 0.5, 0.5)),
            click((0.5, 0.5, 1.0)),
            finish,
        ):
            selected = mesh.select_surface_points(slider=False, color="grey")

        assert np.array_equal(selected, point_ids(np.isclose(x, 1) | np.isclose(z, 1)))


def test_select_surface_points_clear():
    mesh = fem.Cube(n=3)
    x, y, z = mesh.points.T

    with interact(
        click((1.0, 0.5, 0.5)),
        press_key("c"),  # clear the selection
        click((0.5, 1.0, 0.5)),
        push_clear_button,  # clear the selection
        click((0.5, 0.5, 1.0)),
        finish,
    ):
        selected = mesh.select_surface_points(selected_color="red")

    assert np.array_equal(selected, point_ids(np.isclose(z, 1)))

    # no selection
    with interact(finish):
        selected = mesh.select_surface_points()

    assert len(selected) == 0


def test_select_surface_points_exclude():
    meshes = [
        fem.Cube(n=3),
        fem.Cube(n=3).triangulate(),
        fem.Cube(n=3).add_midpoints_edges(),
    ]

    for mesh in meshes:
        x, y, z = mesh.points.T

        # excluded patches have priority at shared points
        with interact(
            click((1.0, 0.5, 0.5)),  # select the patch x=1
            click((0.5, 1.0, 0.5)),  # select the patch y=1
            click((0.5, 1.0, 0.5)),  # exclude the patch y=1
            finish,
        ):
            selected = mesh.select_surface_points(excluded_color="black")

        assert np.array_equal(selected, point_ids(np.isclose(x, 1) & ~np.isclose(y, 1)))

        # an excluded patch alone has no points
        with interact(click((0.5, 1.0, 0.5)), click((0.5, 1.0, 0.5)), finish):
            selected = mesh.select_surface_points()

        assert len(selected) == 0

    mesh = fem.Cube(n=3)
    x, y, z = mesh.points.T

    # three clicks: select, exclude and unselect the patch x=1
    with interact(
        click((1.0, 0.5, 0.5)),
        click((1.0, 0.5, 0.5)),
        click((1.0, 0.5, 0.5)),
        click((0.5, 0.5, 1.0)),
        finish,
    ):
        selected = mesh.select_surface_points()

    assert np.array_equal(selected, point_ids(np.isclose(z, 1)))

    # clear the excluded patches
    with interact(
        click((1.0, 0.5, 0.5)),
        click((1.0, 0.5, 0.5)),  # exclude the patch x=1
        press_key("c"),
        click((0.5, 1.0, 0.5)),
        finish,
    ):
        selected = mesh.select_surface_points()

    assert np.array_equal(selected, point_ids(np.isclose(y, 1)))

    # a merged patch of a selected and an excluded patch is excluded
    with interact(
        click((1.0, 0.5, 0.5)),  # select the patch x=1
        click((0.5, 1.0, 0.5)),
        click((0.5, 1.0, 0.5)),  # exclude the patch y=1
        move_slider(120),
        finish,
    ):
        selected = mesh.select_surface_points()

    assert len(selected) == 0

    # the next click on the merged patch unselects it
    with interact(
        click((1.0, 0.5, 0.5)),
        click((0.5, 1.0, 0.5)),
        click((0.5, 1.0, 0.5)),
        move_slider(120),
        click((0.5, 0.5, 1.0)),  # unselect the merged patch
        move_slider(30),
        click((0.5, 0.5, 1.0)),  # select the patch z=1
        finish,
    ):
        selected = mesh.select_surface_points()

    assert np.array_equal(selected, point_ids(np.isclose(z, 1)))


def test_select_surface_points_angle():
    mesh = fem.Cube(n=3)
    x = mesh.points[:, 0]
    surface = point_ids(
        np.any(np.isclose(mesh.points, 0) | np.isclose(mesh.points, 1), axis=1)
    )

    # all faces of the cube are on one patch for angles above 90 degrees
    with interact(click((1.0, 0.5, 0.5)), move_slider(120), finish):
        selected = mesh.select_surface_points()

    assert np.array_equal(selected, surface)

    with interact(click((1.0, 0.5, 0.5)), finish):
        selected = mesh.select_surface_points(angle=120)

    assert np.array_equal(selected, surface)

    # back to one patch per side of the cube
    with interact(move_slider(120), click((1.0, 0.5, 0.5)), move_slider(30), finish):
        selected = mesh.select_surface_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 1)))


def test_select_surface_points_curved():
    # a ring with inner radius 1 and outer radius 2, revolved around the x-axis
    mesh = fem.Rectangle(a=(0, 1), b=(1, 2), n=3).revolve(n=37, phi=360)
    x, y, z = mesh.points.T
    radius = np.hypot(y, z)

    # the outer mantle is a smooth patch (10 degrees between neighbouring faces)
    with interact(click((0.25, np.sqrt(2), np.sqrt(2))), finish):
        selected = mesh.select_surface_points(angle=30)

    assert np.array_equal(selected, point_ids(np.isclose(radius, 2)))

    # one face (the faces in axial direction are coplanar)
    with interact(click((0.25, np.sqrt(2), np.sqrt(2))), finish):
        selected = mesh.select_surface_points(angle=5)

    assert len(selected) == 6
    assert np.allclose(radius[selected], 2)


def test_select_surface_points_without_polygons():
    pytest.importorskip("pyvista")

    with pytest.raises(ValueError):
        fem.mesh.Line(n=3).select_surface_points()


def test_select_edge_points():
    meshes = [
        fem.Cube(n=3),
        fem.Cube(n=3).triangulate(),
        fem.Cube(n=3).add_midpoints_edges(),
    ]

    for mesh in meshes:
        x, y, z = mesh.points.T

        with interact(
            parallel_projection(False),
            click((1.0, 1.0, 0.25)),  # select the edge x=1, y=1
            click((0.5, 1.0, 1.0)),  # select the edge y=1, z=1
            click((0.5, 1.0, 1.0)),  # exclude the edge y=1, z=1
            click((0.5, 1.0, 1.0)),  # unselect the edge y=1, z=1
            click((1.0, 0.5, 0.5)),  # the edge behind the face x=1 is hidden
            click((0.5, 0.5, 1.0)),  # the edge behind the face z=1 is hidden
            click((0.5, 1.0, 1.0), drag=50),  # a drag rotates, no selection
            click((2, 2)),  # click on the background, no selection
            finish,
        ):
            selected = fem.view.select_edge_points(mesh)

        assert np.array_equal(selected, point_ids(np.isclose(x, 1) & np.isclose(y, 1)))

        # click next to the edges, outside and inside of the silhouette
        with interact(
            click((1.0, 0.5, 0.0), offset=(3, -3)),
            click((0.0, 0.5, 1.0), offset=(3, -3)),
            finish,
        ):
            selected = mesh.select_edge_points(slider=False, color="grey")

        assert np.array_equal(
            selected,
            point_ids(
                (np.isclose(x, 1) & np.isclose(z, 0))
                | (np.isclose(x, 0) & np.isclose(z, 1))
            ),
        )


def test_select_edge_points_clear():
    mesh = fem.Cube(n=3)
    x, y, z = mesh.points.T

    with interact(
        click((1.0, 1.0, 0.25)),
        press_key("c"),  # clear the selection
        click((0.5, 1.0, 1.0)),
        push_clear_button,  # clear the selection
        click((1.0, 0.5, 1.0)),
        finish,
    ):
        selected = mesh.select_edge_points(selected_color="red")

    assert np.array_equal(selected, point_ids(np.isclose(x, 1) & np.isclose(z, 1)))

    # no selection
    with interact(finish):
        selected = mesh.select_edge_points()

    assert len(selected) == 0


def test_select_edge_points_exclude():
    meshes = [
        fem.Cube(n=3),
        fem.Cube(n=3).triangulate(),
        fem.Cube(n=3).add_midpoints_edges(),
    ]

    for mesh in meshes:
        x, y, z = mesh.points.T

        # excluded patches have priority at shared points (the corner x=y=z=1)
        with interact(
            click((1.0, 1.0, 0.25)),  # select the edge x=1, y=1
            click((0.5, 1.0, 1.0)),  # select the edge y=1, z=1
            click((0.5, 1.0, 1.0)),  # exclude the edge y=1, z=1
            finish,
        ):
            selected = mesh.select_edge_points(excluded_color="orange")

        assert np.array_equal(
            selected,
            point_ids(np.isclose(x, 1) & np.isclose(y, 1) & ~np.isclose(z, 1)),
        )

        # an excluded patch alone has no points
        with interact(click((0.5, 1.0, 1.0)), click((0.5, 1.0, 1.0)), finish):
            selected = mesh.select_edge_points()

        assert len(selected) == 0

    mesh = fem.Cube(n=3)
    x, y, z = mesh.points.T

    # three clicks: select, exclude and unselect the edge x=1, y=1
    with interact(
        click((1.0, 1.0, 0.25)),
        click((1.0, 1.0, 0.25)),
        click((1.0, 1.0, 0.25)),
        click((1.0, 0.5, 1.0)),
        finish,
    ):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 1) & np.isclose(z, 1)))

    # clear the excluded patches
    with interact(
        click((1.0, 1.0, 0.25)),
        click((1.0, 1.0, 0.25)),  # exclude the edge x=1, y=1
        press_key("c"),
        click((0.5, 1.0, 1.0)),
        finish,
    ):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(y, 1) & np.isclose(z, 1)))


def test_select_edge_points_exclude_planar():
    mesh = fem.Rectangle(n=4)
    x, y = mesh.points.T

    # a merged patch of a selected and an excluded patch is excluded
    with interact(
        click((1.0, 0.5, 0.0)),  # select the edge x=1
        click((0.5, 1.0, 0.0)),
        click((0.5, 1.0, 0.0)),  # exclude the edge y=1
        finish,
    ):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 1) & ~np.isclose(y, 1)))

    with interact(
        click((1.0, 0.5, 0.0)),
        click((0.5, 1.0, 0.0)),
        click((0.5, 1.0, 0.0)),
        move_slider(120),
        finish,
    ):
        selected = mesh.select_edge_points()

    assert len(selected) == 0

    # the next click on the merged patch unselects it
    with interact(
        click((1.0, 0.5, 0.0)),
        click((0.5, 1.0, 0.0)),
        click((0.5, 1.0, 0.0)),
        move_slider(120),
        click((0.0, 0.5, 0.0)),  # unselect the merged patch
        move_slider(30),
        click((0.0, 0.5, 0.0)),  # select the edge x=0
        finish,
    ):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 0)))


def test_select_edge_points_exclude_curved():
    # a ring with inner radius 1 and outer radius 2, revolved around the x-axis
    mesh = fem.Rectangle(a=(0, 1), b=(1, 2), n=3).revolve(n=37, phi=360)
    x, y, z = mesh.points.T
    radius = np.hypot(y, z)
    circle = np.isclose(x, 1) & np.isclose(radius, 2)

    # the excluded axial edge on the outer mantle doesn't exist for an angle of 30
    # degrees, but it is kept (and excluded again for an angle of 5 degrees)
    with interact(
        click((0.25, 2.0, 0.0)),
        click((0.25, 2.0, 0.0)),  # exclude the axial edge at y=2, z=0
        move_slider(30),
        click((1.0, np.sqrt(2), np.sqrt(2))),  # select the outer circle
        finish,
    ):
        selected = mesh.select_edge_points(angle=5)

    assert np.array_equal(selected, point_ids(circle))

    # only the clicked edge of the outer circle (from 0 to 10 degrees) is selected
    # for an angle of 5 degrees, without the point of the excluded axial edge
    phi = np.radians(5)
    with interact(
        click((0.25, 2.0, 0.0)),
        click((0.25, 2.0, 0.0)),
        move_slider(30),
        click((1.0, 2 * np.cos(phi), 2 * np.sin(phi))),
        move_slider(5),
        finish,
    ):
        selected = mesh.select_edge_points(angle=5)

    assert len(selected) == 1
    assert np.allclose(
        mesh.points[selected], [1, 2 * np.cos(2 * phi), 2 * np.sin(2 * phi)]
    )


def test_select_edge_points_angle():
    mesh = fem.Cube(n=3)
    x, y, z = mesh.points.T

    # there are no edges for angles above 90 degrees
    with interact(click((1.0, 1.0, 0.25)), move_slider(120), finish):
        selected = mesh.select_edge_points()

    assert len(selected) == 0

    with interact(click((1.0, 1.0, 0.25)), finish):
        selected = mesh.select_edge_points(angle=120)

    assert len(selected) == 0

    # the clicked edge is selected again
    with interact(click((1.0, 1.0, 0.25)), move_slider(120), move_slider(30), finish):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 1) & np.isclose(y, 1)))


def test_select_edge_points_planar():
    mesh = fem.Rectangle(n=4)
    x, y = mesh.points.T

    with interact(parallel_projection(True), click((1.0, 0.5, 0.0)), finish):
        selected = mesh.select_edge_points()

    assert np.array_equal(selected, point_ids(np.isclose(x, 1)))

    # all edges of the boundary are on one patch for angles above 90 degrees
    with interact(click((1.0, 0.5, 0.0)), move_slider(120), finish):
        selected = mesh.select_edge_points()

    boundary = np.isclose(x, 0) | np.isclose(x, 1) | np.isclose(y, 0) | np.isclose(y, 1)
    assert np.array_equal(selected, point_ids(boundary))


def test_select_edge_points_curved():
    # a ring with inner radius 1 and outer radius 2, revolved around the x-axis
    mesh = fem.Rectangle(a=(0, 1), b=(1, 2), n=3).revolve(n=37, phi=360)
    x, y, z = mesh.points.T
    radius = np.hypot(y, z)

    # the outer circle is a smooth patch (10 degrees between neighbouring edges)
    with interact(click((1.0, np.sqrt(2), np.sqrt(2))), finish):
        selected = mesh.select_edge_points(angle=30)

    assert np.array_equal(selected, point_ids(np.isclose(x, 1) & np.isclose(radius, 2)))

    # one edge
    with interact(click((1.0, np.sqrt(2), np.sqrt(2))), finish):
        selected = mesh.select_edge_points(angle=5)

    assert len(selected) == 2
    assert np.allclose(x[selected], 1)
    assert np.allclose(radius[selected], 2)


def test_select_edge_points_without_polygons():
    pytest.importorskip("pyvista")

    with pytest.raises(ValueError):
        fem.mesh.Line(n=3).select_edge_points()


def test_boundary_select_points():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    with interact(click((1.0, 0.5, 0.5)), finish):
        boundary = fem.Boundary(
            field[0], select_points="surfaces", skip=(0, 1, 1), value=0.2
        )

    expected = fem.Boundary(field[0], fx=1, skip=(0, 1, 1), value=0.2)

    assert np.array_equal(boundary.mask, expected.mask)
    assert np.array_equal(boundary.points, expected.points)
    assert np.array_equal(boundary.dof, expected.dof)

    with interact(click((1.0, 1.0, 0.25)), finish):
        boundary = fem.Boundary(
            field[0], select_points="edges", skip=(0, 1, 1), value=0.2
        )

    expected = fem.Boundary(field[0], fx=1, fy=1, mode="and", skip=(0, 1, 1))

    assert np.array_equal(boundary.mask, expected.mask)
    assert np.array_equal(boundary.points, expected.points)
    assert np.array_equal(boundary.dof, expected.dof)

    with pytest.raises(KeyError):
        fem.Boundary(field[0], select_points="points")


def test_boundary_select_points_planar():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=2)])

    with interact(click((0.0, 0.5, 0.0)), finish):
        boundary = fem.Boundary(field[0], select_points="edges")

    expected = fem.Boundary(field[0], fx=0)

    assert np.array_equal(boundary.mask, expected.mask)
    assert np.array_equal(boundary.points, expected.points)
    assert np.array_equal(boundary.dof, expected.dof)


if __name__ == "__main__":
    test_select_surface_points()
    test_select_surface_points_clear()
    test_select_surface_points_exclude()
    test_select_surface_points_angle()
    test_select_surface_points_curved()
    test_select_surface_points_without_polygons()
    test_select_edge_points()
    test_select_edge_points_clear()
    test_select_edge_points_exclude()
    test_select_edge_points_exclude_planar()
    test_select_edge_points_exclude_curved()
    test_select_edge_points_angle()
    test_select_edge_points_planar()
    test_select_edge_points_curved()
    test_select_edge_points_without_polygons()
    test_boundary_select_points()
    test_boundary_select_points_planar()
