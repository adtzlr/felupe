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
    the clicks, key presses and widget interactions of a user."""

    pv = pytest.importorskip("pyvista")
    show = pv.Plotter.show

    def show_and_interact(plotter, *args, **kwargs):
        plotter.view_isometric()  # camera looks at the faces x=1, y=1 and z=1
        plotter.renderer.GetRenderWindow().Render()
        for action in actions:
            action(plotter)
        return show(plotter, *args, **kwargs)

    with mock.patch.object(pv, "OFF_SCREEN", True):
        with mock.patch.object(pv.Plotter, "show", show_and_interact):
            yield


def click(point, button="Left", drag=0):
    """Click on a point given in world (x, y, z) or display (x, y) coordinates. The
    button is released ``drag`` pixels away from the position where it was pressed.
    """

    def action(plotter):
        if len(point) == 3:
            plotter.renderer.SetWorldPoint(*point, 1.0)
            plotter.renderer.WorldToDisplay()
            x, y = plotter.renderer.GetDisplayPoint()[:2]
        else:
            x, y = point

        x, y = int(round(x)), int(round(y))
        interactor = plotter.iren.interactor
        interactor.SetEventPosition(x, y)
        getattr(interactor, f"{button}ButtonPressEvent")()
        interactor.SetEventPosition(x + drag, y + drag)
        getattr(interactor, f"{button}ButtonReleaseEvent")()

    return action


def press_key(key):
    def action(plotter):
        interactor = plotter.iren.interactor
        interactor.SetKeySym(key)
        interactor.KeyPressEvent()

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
            click((0.5, 1.0, 0.5)),  # deselect the patch y=1
            click((0.5, 0.5, 1.0), drag=50),  # a drag rotates, no selection
            click((2, 2)),  # click on the background, no selection
            finish,
        ):
            selected = fem.mesh.select_surface_points(mesh)

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


def test_boundary_select():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    with interact(click((1.0, 0.5, 0.5)), finish):
        boundary = fem.Boundary(field[0], select=True, skip=(0, 1, 1), value=0.2)

    expected = fem.Boundary(field[0], fx=1, skip=(0, 1, 1), value=0.2)

    assert np.array_equal(boundary.mask, expected.mask)
    assert np.array_equal(boundary.points, expected.points)
    assert np.array_equal(boundary.dof, expected.dof)


if __name__ == "__main__":
    test_select_surface_points()
    test_select_surface_points_clear()
    test_select_surface_points_angle()
    test_select_surface_points_curved()
    test_select_surface_points_without_polygons()
    test_boundary_select()
