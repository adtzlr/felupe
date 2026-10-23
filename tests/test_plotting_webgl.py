# SPDX-License-Identifier: GPL-3.0-or-later

import json
import sys
from unittest import mock

import numpy as np
import pytest

import felupe as fem
from felupe.view._backends._webgl import WebGLPlotter, _backend
from felupe.view._backends._webgl._page import decode, encode, in_kernel
from felupe.view._backends._webgl._widget import widget_class
from felupe.view._surface import triangulate_surface


@pytest.fixture
def webgl():
    "Activate the plotting backend webgl (and forget the warned options)."

    _backend._WARNED.clear()
    fem.set_plotting_backend("webgl")

    yield

    fem.set_plotting_backend("pyvista")


def solved():
    "A solid body of a cube after a uniaxial loadcase."

    mesh = fem.Cube(n=3)
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)
    move = fem.math.linsteps([0, 0.3], num=1)
    step = fem.Step([solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate()

    return mesh, field, boundaries, solid


def test_view(webgl):
    mesh, field, boundaries, solid = solved()

    plotter = solid.plot("Principal Values of Cauchy Stress", factor=2.0)
    assert isinstance(plotter, WebGLPlotter)
    assert plotter.view == "default" and plotter.axes

    # the undeformed model (ghost) and the deformed model
    ghost, model = plotter.objects
    assert ghost["ghost"] and ghost["opacity"] == 0.2
    assert "displacement" not in ghost and "edges" not in ghost

    surface = triangulate_surface(mesh)
    positions = decode(model["positions"]).reshape(-1, 3)
    displacement = decode(model["displacement"]).reshape(-1, 3)
    assert np.allclose(positions, surface.interpolate(mesh.points))
    assert np.allclose(displacement, surface.interpolate(field[0].values), atol=1e-6)
    assert model["factor"] == 2.0
    assert np.array_equal(decode(model["triangles"]), surface.triangles.ravel())
    assert np.array_equal(decode(model["edges"]), surface.edges.ravel())
    assert np.array_equal(decode(model["faceIds"]), surface.faces)
    assert decode(model["lut"]).shape == (768,)

    # the cell-data of the triangles: the largest principal values
    stress = solid.evaluate.cauchy_stress(field)
    principal = np.linalg.eigvalsh(stress.T).mean(axis=1).max(axis=1)
    values = decode(model["values"])
    assert model["valuesAt"] == "cell"
    assert np.allclose(values, principal[surface.owners], atol=1e-6)
    assert np.allclose(model["range"], [principal.min(), principal.max()])
    assert model["label"] == "Cauchy Stress (Max. Principal)"
    assert model["scalarBar"] and not model["verticalBar"]

    # the magnitude of the point-data
    plotter = field.plot("Displacement", component=None, show_undeformed=False)
    (model,) = plotter.objects
    norm = np.linalg.norm(field[0].values, axis=1)
    assert model["valuesAt"] == "point"
    assert np.allclose(decode(model["values"]), surface.interpolate(norm), atol=1e-6)
    assert model["label"] == "Displacement Magnitude"

    # a given label, a component and other options
    plotter = field.plot(
        "Logarithmic Strain",
        component=3,
        label="strain",
        show_edges=False,
        show_undeformed=False,
        view="xy",
        add_axes=False,
        scalar_bar_vertical=True,
        color="red",
        opacity=0.5,
        show_scalar_bar=False,
    )
    (model,) = plotter.objects
    assert model["label"] == "strain" and "edges" not in model
    assert model["verticalBar"] and not model["scalarBar"]
    assert model["color"] == "#ff0000" and model["opacity"] == 0.5
    assert plotter.view == "xy" and not plotter.axes

    # the wireframe of a mesh, a second view keeps the camera of the first view
    plotter = mesh.plot(style="wireframe", plotter=plotter)
    assert not plotter.objects[-1]["faces"] and "edges" in plotter.objects[-1]
    assert plotter.view == "xy"

    with pytest.raises(KeyError):
        field.plot("Unknown")


def test_view_subdivisions(webgl):
    mesh = fem.mesh.convert(fem.Cube(n=3), order=2)

    # the edges of quadratic cells are subdivided into order * 2 ** (k - 1) segments
    for k, subdivisions in [(1, 2), (2, 4), (3, 8)]:
        (model,) = mesh.plot(nonlinear_subdivision=k).objects
        surface = triangulate_surface(mesh, subdivisions=subdivisions)
        assert model["triangles"]["size"] == surface.triangles.size
        assert model["smooth"] == (k > 1)
        assert "faceIds" not in model

    # linear cells are not subdivided
    (model,) = fem.Cube(n=3).plot(nonlinear_subdivision=3, smooth_shading=False).objects
    assert model["triangles"]["size"] == 6 * 4 * 2 * 3
    assert not model["smooth"] and "faceIds" in model


def test_view_options_not_supported(webgl):
    mesh = fem.Rectangle(n=3)

    for options in [
        dict(theme="document"),
        dict(scalar_bar_args={"title": "x"}),
        dict(show_logo=True),
        dict(specular=0.5),
        dict(style="points"),
    ]:
        with pytest.warns(UserWarning, match='plotting backend "webgl"'):
            mesh.plot(**options)

    # each option is warned once
    with warnings_as_errors():
        mesh.plot(theme="document", specular=0.5)

    # scenes without points and cells, e.g. of ViewXdmf
    scene = fem.view.Scene()
    scene.mesh = None
    with pytest.raises(NotImplementedError, match="without points and cells"):
        scene.plot()

    with pytest.raises(NotImplementedError, match="screenshots"):
        mesh.screenshot()


class warnings_as_errors:
    "A context which turns warnings into errors."

    def __enter__(self):
        import warnings

        self.catcher = warnings.catch_warnings()
        self.catcher.__enter__()
        warnings.simplefilter("error")

    def __exit__(self, *args):
        self.catcher.__exit__(*args)


def test_decorations(webgl):
    mesh, field, boundaries, solid = solved()

    # boundaries with points (labels of the legend) and lines
    plotter = boundaries.plot()
    kinds = [obj["type"] for obj in plotter.objects]
    assert kinds.count("points") == 5 and kinds.count("lines") > 0
    assert [entry["label"] for entry in plotter.legend] == list(boundaries.keys())
    assert plotter.legend[0]["color"] == "#1f77b4"

    # arrows of a point load by triangles of glyphs
    right = np.arange(mesh.npoints)[mesh.x == 1]
    load = fem.PointLoad(field, points=right, values=[1, 0, 0])
    plotter = load.plot(plotter=fem.Cube(n=3).plot(), color="green")
    arrows = plotter.objects[-1]
    glyph, glyph_triangles = _backend._arrow()
    assert arrows["triangles"]["size"] == len(right) * glyph_triangles.size
    assert arrows["color"] == "#008000"

    # the length of the arrows is relative to the extent of the deformed model
    tips = decode(arrows["positions"]).reshape(len(right), -1, 3)
    deformed = mesh.points + field[0].values
    length = 0.125 * np.ptp(deformed, axis=0).max()
    assert np.allclose(tips[:, :, 0].max(axis=1) - deformed[right, 0], length)

    # a rigid plane with its outline, a center-point and a normal
    center = fem.Cube(n=3)
    center.update(points=np.vstack([center.points, [0.5, 0.5, 1.5]]))
    fc = fem.FieldContainer([fem.Field(fem.RegionHexahedron(center), dim=3)])
    top = np.arange(center.npoints)[center.z == 1]
    plane = fem.ContactRigidPlane(fc, points=top, centerpoint=-1, normal=[0, 0, -1])
    plotter = plane.plot(line_width=2, opacity=0.4)
    shape, point, normal = plotter.objects
    assert shape["opacity"] == 0.4 and shape["lineWidth"] == 2
    assert decode(shape["edges"]).size == 2 * 4
    assert point["type"] == "points" and point["size"] == 12
    assert normal["type"] == "lines"

    # without edges or fully clipped
    plotter = plane.plot(show_edges=False)
    assert "edges" not in plotter.objects[0]
    plotter = plane.plot(offset=-2, sym=(False, False, True), show_point=False)
    assert [obj["type"] for obj in plotter.objects] == ["lines"]

    # the default and given colors of the meshes of a container
    container = fem.MeshContainer([fem.Cube(n=2), fem.Cube(n=2).translate(1, axis=0)])
    plotter = container.plot(labels=["a", "b"])
    assert plotter.objects[0]["color"] == _backend.DEFAULT_COLOR
    assert [entry["label"] for entry in plotter.legend] == ["a", "b"]

    # options of the primitives which are not supported
    backend = fem.view._backends.backend_of(plotter)
    for add in [
        lambda: backend.add_points(plotter, [[0, 0, 0]], render_points_as_spheres=True),
        lambda: backend.add_lines(plotter, [[[0, 0], [1, 1]]], lighting=False),
    ]:
        with warnings_as_errors():
            add()

    for add in [
        lambda: backend.add_points(plotter, [[0, 0, 0]], specular=1),
        lambda: backend.add_lines(plotter, [[[0, 0], [1, 1]]], specular=2),
        lambda: backend.add_arrows(plotter, [[0, 0]], [[1, 0]], specular=3),
        lambda: backend.add_shape(plotter, plane_shape(), specular=4),
        lambda: backend.add_legend(plotter, [("a", "red")], specular=5),
    ]:
        _backend._WARNED.clear()
        with pytest.warns(UserWarning, match='"specular"'):
            add()

    # zero-length arrows are not shown
    plotter = backend.add_arrows(fem.view._backends.WebGLPlotter(), [[0, 0]], [[0, 0]])
    assert plotter.objects[0]["triangles"]["size"] == 0


def plane_shape():
    from felupe.view._shapes import Plane

    return Plane(center=(0, 0, 0), normal=(0, 0, 1), size=1.0)


def test_colors():
    assert _backend._color(None, "#000000") == "#000000"
    assert _backend._color("red") == "#ff0000"
    assert _backend._color("tab:blue") == "#1f77b4"
    assert _backend._color("no-color") == "no-color"
    assert _backend._lut("viridis").shape == (256, 3)
    assert _backend._lut("unknown") is None

    with mock.patch.dict(sys.modules, {"matplotlib": None, "matplotlib.colors": None}):
        assert _backend._color("red") == "red"
        assert _backend._lut("viridis") is None


def test_page(webgl, tmp_path):
    plotter = fem.Rectangle(n=3).plot()
    page = plotter.html

    # the scene is inline JSON and the module is inline (without a closing tag)
    start = page.index('type="application/json">') + len('type="application/json">')
    scene = json.loads(page[start : page.index("</script>", start)])
    assert scene["view"] == "default" and len(scene["objects"]) == 1
    module = page[page.index('<script type="module">') :]
    assert module.count("</script") == 1
    assert "export function createViewer" in module

    # the file and the rich representation
    path = plotter.export_html(tmp_path / "plots" / "rectangle.html")
    assert path.read_text(encoding="utf-8") == page
    assert plotter._repr_html_().startswith('<iframe class="felupe-webgl"')

    # show the page in the web browser (in a temporary file or in a given file)
    with mock.patch("webbrowser.open") as open_browser:
        path = plotter.show()
        assert path.suffix == ".html" and path.exists()
        open_browser.assert_called_once_with(path.resolve().as_uri())
        path.unlink()

        path = plotter.show(filename=tmp_path / "show.html")
        assert path == tmp_path / "show.html"

    # in a notebook, the viewer is shown in the output of the cell as a widget
    display = pytest.importorskip("IPython.display")
    pytest.importorskip("anywidget")

    with (
        mock.patch.object(_backend, "in_kernel", return_value=True),
        mock.patch.object(display, "display") as show,
    ):
        assert plotter.show() is None
        (widget,), options = show.call_args
        assert widget.height == "480px" and options == {}

    # without anywidget, the page is shown in an iframe
    widget_class.cache_clear()
    with (
        mock.patch.dict(sys.modules, {"anywidget": None}),
        mock.patch.object(_backend, "in_kernel", return_value=True),
        mock.patch.object(display, "display") as show,
    ):
        assert plotter.show() is None
        (bundle,), options = show.call_args
        assert bundle["text/html"].startswith("<iframe") and options == {"raw": True}

        with pytest.raises(ImportError):
            plotter.widget()

    widget_class.cache_clear()
    assert not in_kernel()


def test_widget(webgl):
    pytest.importorskip("anywidget")

    mesh, field, boundaries, solid = solved()
    plotter = solid.plot("Displacement")

    # the arrays of the scene are binary buffers
    widget = plotter.widget(height="300px")
    assert widget.height == "300px"
    assert "export function createViewer" in widget._esm
    assert "export default" in widget._esm
    assert repr(widget).startswith("FElupe WebGL viewer")

    for obj, synced in zip(plotter.objects, widget.scene["objects"]):
        for key in ["positions", "triangles", "values"]:
            if key in obj:
                assert isinstance(synced[key]["data"], bytes)
                assert np.array_equal(decode(synced[key]), decode(obj[key]))

    assert widget.scene["view"] == plotter.view
    assert widget.scene["legend"] is None

    # the same scene in a page with base64-encoded arrays
    scene = encode(plotter.scene())
    assert isinstance(scene["objects"][0]["positions"]["data"], str)
    assert np.array_equal(
        decode(scene["objects"][0]["positions"]),
        decode(plotter.objects[0]["positions"]),
    )


def test_lines_and_points_only(webgl):
    # a mesh of lines has no faces and no undeformed model
    plotter = fem.mesh.Line(n=4).plot()
    (model,) = plotter.objects
    assert model["triangles"]["size"] == 0 and model["edges"]["size"] == 2 * 3

    # the points of a quadrature scheme
    plotter = fem.GaussLegendre(order=1, dim=2).plot()
    assert [obj["type"] for obj in plotter.objects] == ["surface"] + ["points"] * 4
