# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from unittest import mock

import numpy as np
import pytest

import felupe as fem
from felupe.view._backends import backend_of


class RecordingPlotter(list):
    "A plotter which records the added scenes and primitives as (kind, data, options)."

    def kinds(self):
        return [kind for kind, data, options in self]


class SceneBackend(fem.view.PlottingBackend):
    "A plotting backend with scenes only (without primitives and screenshots)."

    def is_plotter(self, plotter):
        return isinstance(plotter, RecordingPlotter)

    def prepare_plotter(self, plotter=None, **kwargs):
        if plotter is None:
            plotter = RecordingPlotter()
        return plotter

    def add_view(self, plotter, scene, **kwargs):
        plotter.append(("view", scene, kwargs))
        return plotter


class RecordingBackend(fem.view.PlottingBackend):
    "A plotting backend which records the calls (without screenshots)."

    def __init__(self):
        self.prepared = []

    def is_plotter(self, plotter):
        return isinstance(plotter, RecordingPlotter)

    def prepare_plotter(self, plotter=None, **kwargs):
        self.prepared.append(kwargs)
        if plotter is None:
            plotter = RecordingPlotter()
        return plotter

    def add_view(self, plotter, scene, **kwargs):
        plotter.append(("view", scene, kwargs))
        return plotter

    def add_points(self, plotter, points, color=None, point_size=None, **kwargs):
        options = dict(color=color, point_size=point_size, **kwargs)
        plotter.append(("points", np.asarray(points), options))
        return plotter

    def add_lines(self, plotter, lines, color=None, line_width=None, **kwargs):
        options = dict(color=color, line_width=line_width, **kwargs)
        plotter.append(("lines", np.asarray(lines), options))
        return plotter

    def add_arrows(self, plotter, points, directions, scale=1.0, color=None, **kwargs):
        data = (np.asarray(points), np.asarray(directions))
        plotter.append(("arrows", data, dict(scale=scale, color=color, **kwargs)))
        return plotter


@pytest.fixture
def recording():
    "Register and activate a recording plotting backend, restore the default after."

    backend = RecordingBackend()
    fem.view.register_plotting_backend("recording", backend)
    fem.set_plotting_backend("recording")

    yield backend

    fem.set_plotting_backend("pyvista")


def test_default_backend():
    assert fem.get_plotting_backend() == "pyvista"
    assert fem.view.get_plotting_backend is fem.get_plotting_backend
    assert backend_of().name == "pyvista"
    assert isinstance(backend_of(), fem.view.PlottingBackend)


def test_set_unknown_backend():
    with pytest.raises(ValueError, match='"unknown"'):
        fem.set_plotting_backend("unknown")

    assert fem.get_plotting_backend() == "pyvista"


def test_register_invalid_backend():
    with pytest.raises(TypeError):
        fem.view.register_plotting_backend("invalid", object())

    with pytest.raises(TypeError):
        fem.view.register_plotting_backend("invalid", RecordingBackend)

    # abstract methods must be implemented
    with pytest.raises(TypeError):
        fem.view.PlottingBackend()


def test_plot_by_current_backend(recording):
    assert fem.get_plotting_backend() == "recording"
    assert recording.name == "recording"

    mesh = fem.Cube(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)

    plotter = solid.plot("Principal Values of Cauchy Stress", factor=2.0)
    plotter = field.plot("Displacement", component=None, plotter=plotter)
    plotter = mesh.plot(plotter=plotter, theme=None)

    assert isinstance(plotter, RecordingPlotter)
    assert len(plotter) == 3

    # the scenes are views with the data of the model
    assert plotter.kinds() == ["view", "view", "view"]
    kind, scene, options = plotter[0]
    assert isinstance(scene, fem.ViewSolid)
    assert "Principal Values of Cauchy Stress" in scene.cell_data
    assert options["name"] == "Principal Values of Cauchy Stress"
    assert options["factor"] == 2.0

    assert isinstance(plotter[1][1], fem.ViewField)
    assert plotter[1][2]["component"] is None
    assert plotter[2][2]["show_undeformed"] is False

    # the options of the plotter
    assert recording.prepared[0] == dict(
        theme="default", off_screen=False, notebook=False
    )
    assert recording.prepared[2]["theme"] is None


def test_plot_by_backend_of_given_plotter(recording):
    pv = pytest.importorskip("pyvista")

    mesh = fem.Cube(n=3)

    # a pyvista plotter is drawn by the pyvista backend
    plotter = pv.Plotter(off_screen=True)
    assert backend_of(plotter).name == "pyvista"
    assert mesh.plot(plotter=plotter) is plotter
    assert len(plotter.actors) > 0

    # a recording plotter is drawn by the recording backend
    fem.set_plotting_backend("pyvista")
    plotter = RecordingPlotter()
    assert backend_of(plotter) is recording
    assert mesh.plot(plotter=plotter) is plotter
    assert len(plotter) == 1

    # an unknown plotter is drawn by the current backend
    assert backend_of(object()).name == "pyvista"


def test_decorations(recording):
    mesh = fem.Cube(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    right = np.arange(mesh.npoints)[mesh.x == 1]

    # a point load with arrows for the non-zero components (scaled by the extent)
    load = fem.PointLoad(field, points=right, values=[1, 0, -2])
    plotter = load.plot(color="blue")
    assert plotter.kinds() == ["view", "arrows", "arrows"]

    (points, directions), options = plotter[1][1:]
    assert np.allclose(points, mesh.points[right])
    assert np.allclose(directions, [1, 0, 0])
    assert options == dict(scale=0.125, color="blue")
    assert np.allclose(plotter[2][1][1], [0, 0, -1])

    # a boundary with points and lines of its prescribed directions
    boundary = fem.Boundary(field[0], fx=1, skip=(0, 1, 0))
    plotter = boundary.plot(width=4)
    assert plotter.kinds() == ["view", "points", "lines", "lines"]

    kind, points, options = plotter[1]
    assert np.allclose(points, mesh.points[right])
    assert options == dict(color="black", point_size=10, label="Boundary Condition")

    kind, lines, options = plotter[3]
    assert lines.shape == (len(right), 2, 3)
    assert np.allclose(lines[:, 0], mesh.points[right])
    assert np.allclose(lines[:, 1] - lines[:, 0], [0, 0, 0.125])
    assert options == dict(color="black", line_width=4)

    # the quadrature points
    plotter = fem.GaussLegendre(order=1, dim=2).plot(weighted=True)
    assert plotter.kinds() == ["view"] + ["points"] * 4
    assert all(points.shape == (1, 2) for kind, points, options in plotter[1:])
    assert plotter[1][2] == dict(color="grey", point_size=20)

    # a constraint with its points and the reference axis on a new plotter
    yz = mesh.points[:, 1:] - 0.5
    left = (mesh.x == 0) & (np.linalg.norm(yz, axis=1) > 0.1)
    sleeve = fem.CylindricalConstraint(
        field, points=left, axis=(1, 0, 0), center=(0, 0.5, 0.5)
    )
    plotter = sleeve.plot(point_size=5)
    assert recording.prepared[-1] == dict(theme=None)
    assert plotter.kinds() == ["points", "lines"]
    assert plotter[0][1].shape == (np.sum(left), 3)
    assert plotter[0][2] == dict(color="black", point_size=5)

    kind, lines, options = plotter[1]
    assert lines.shape == (1, 2, 3)
    assert np.allclose(np.cross(lines[0, 1] - lines[0, 0], [1, 0, 0]), 0)
    assert options == dict(color="black", line_width=2)

    # a multi-point constraint with lines from the center-point on a new plotter
    mesh = fem.Cube(n=3)
    mesh.update(points=np.vstack([mesh.points, [2.0, 0.5, 0.5]]))
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    right = np.arange(mesh.npoints)[mesh.x == 1]
    mpc = fem.MultiPointConstraint(field, points=right, centerpoint=-1)

    plotter = mpc.plot(color="red", opacity=0.5)
    assert recording.prepared[-1] == dict(theme=None)
    assert plotter.kinds() == ["lines"]

    kind, lines, options = plotter[0]
    assert np.allclose(lines[:, 0], [2.0, 0.5, 0.5])
    assert np.allclose(lines[:, 1], mesh.points[right])
    assert options == dict(color="red", line_width=None, opacity=0.5)

    # a multi-point constraint without points adds no lines
    mpc = fem.MultiPointConstraint(field, points=[], centerpoint=-1)
    assert mpc.plot(plotter=RecordingPlotter()) == []


def test_decorations_not_supported():
    fem.view.register_plotting_backend("scenes", SceneBackend())
    fem.set_plotting_backend("scenes")

    try:
        mesh = fem.Cube(n=3)
        field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])

        with pytest.raises(
            NotImplementedError, match='"scenes" does not support arrows'
        ):
            fem.PointLoad(field, points=[0]).plot()

        with pytest.raises(NotImplementedError, match="does not support points"):
            fem.Boundary(field[0], fx=0).plot()

        with pytest.raises(NotImplementedError, match="does not support lines"):
            fem.Boundary(field[0], fx=0).plot(show_points=False)

    finally:
        fem.set_plotting_backend("pyvista")


def test_pyvista_plotter_without_pyvista():
    with mock.patch.dict(sys.modules, {"pyvista": None}):
        assert not backend_of().is_plotter(object())


def test_screenshot_not_supported(recording):
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)
    boundaries = fem.dof.uniaxial(field, clamped=True)
    container = fem.MeshContainer([mesh, mesh.copy()])

    screenshots = [
        mesh.screenshot,
        field.screenshot,
        solid.screenshot,
        container.screenshot,
        boundaries.screenshot,
        region.screenshot,
        fem.Hexahedron().screenshot,
        fem.GaussLegendre(order=1, dim=3).screenshot,
    ]

    for screenshot in screenshots:
        with pytest.raises(NotImplementedError, match='"recording"'):
            screenshot()

    # nothing was plotted before the error
    assert recording.prepared == []

    # the images of matplotlib are taken by the screenshots
    plt = pytest.importorskip("matplotlib.pyplot")

    for imshow in [mesh.imshow, field.imshow, solid.imshow, container.imshow]:
        with pytest.raises(NotImplementedError, match='"recording"'):
            imshow()

    with pytest.raises(NotImplementedError, match='"recording"'):
        boundaries.imshow()

    assert recording.prepared == []
    plt.close("all")


def test_screenshot_by_backend_of_given_plotter(recording):
    pv = pytest.importorskip("pyvista")

    mesh = fem.Cube(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)

    # a given pyvista plotter takes the screenshot
    plotter = mesh.plot(plotter=pv.Plotter(off_screen=True), theme=None)
    image = boundaries.screenshot(filename=None, plotter=plotter)
    assert image.ndim == 3

    with pytest.raises(NotImplementedError):
        boundaries.screenshot(filename=None)


if __name__ == "__main__":
    test_default_backend()
    test_set_unknown_backend()
    test_register_invalid_backend()
    test_pyvista_plotter_without_pyvista()
