# SPDX-License-Identifier: GPL-3.0-or-later

import sys
from unittest import mock

import pytest

import felupe as fem
from felupe.view._backends import backend_of


class RecordingPlotter(list):
    "A plotter which records the added scenes with their options."


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
        plotter.append((scene, kwargs))
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
    scene, options = plotter[0]
    assert isinstance(scene, fem.ViewSolid)
    assert "Principal Values of Cauchy Stress" in scene.cell_data
    assert options["name"] == "Principal Values of Cauchy Stress"
    assert options["factor"] == 2.0

    assert isinstance(plotter[1][0], fem.ViewField)
    assert plotter[1][1]["component"] is None
    assert plotter[2][1]["show_undeformed"] is False

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
