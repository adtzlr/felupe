# SPDX-License-Identifier: GPL-3.0-or-later

import sys
import warnings
from unittest import mock

import numpy as np
import pytest

import felupe as fem
from felupe.view._backends import _matplotlib as backend
from felupe.view._backends import backend_of
from felupe.view._backends._matplotlib import MatplotlibPlotter
from felupe.view._surface import triangulate_surface

plt = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture
def mpl():
    "Activate the plotting backend matplotlib (and forget the warned options)."

    plt.switch_backend("Agg")
    backend._WARNED.clear()
    fem.set_plotting_backend("matplotlib")

    yield

    fem.set_plotting_backend("pyvista")
    plt.close("all")


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


def paths(plotter):
    "Return the paths of the drawn model of a plotter."

    plotter.render()
    return plotter._collection.get_paths()


def test_view(mpl):
    mesh, field, boundaries, solid = solved()

    plotter = solid.plot("Principal Values of Cauchy Stress", factor=2.0)
    assert isinstance(plotter, MatplotlibPlotter)
    assert plotter.view == "default" and plotter.axes

    # the undeformed model (ghost) and the deformed model
    ghost, model = plotter.objects
    assert ghost["ghost"] and ghost["opacity"] == 0.2 and not ghost["stroke"]

    surface = triangulate_surface(mesh)
    undeformed = surface.interpolate(mesh.points)
    deformed = undeformed + 2.0 * surface.interpolate(field[0].values)
    assert np.allclose(ghost["positions"], undeformed)
    assert np.allclose(model["positions"], deformed)

    # the faces of linear cells are quads (without triangles)
    (triangles, *_), (quads, faces, colors) = model["polygons"]
    assert len(triangles) == 0 and quads.shape == (24, 4)
    assert model["stroke"] and model["cull"] and model["outlines"] is None

    # the cell-data of the quads: the largest principal values
    stress = solid.evaluate.cauchy_stress(field)
    principal = np.linalg.eigvalsh(stress.T).mean(axis=1).max(axis=1)
    assert np.allclose(model["values"], principal[surface.owners])

    bar = model["scalar_bar"]
    assert bar["label"] == "Cauchy Stress (Max. Principal)"
    assert np.allclose(bar["range"], [principal.min(), principal.max()])
    assert not bar["vertical"] and bar["n_labels"] == 5

    # the ghost (24 faces) and the visible faces of the deformed cube (back-face
    # culling)
    nquads = len(paths(plotter)) - 24
    assert 0 < nquads < 24

    # the magnitude of the point-data, colored by the mean value of the points of the
    # quads
    plotter = field.plot("Displacement", component=None, show_undeformed=False)
    (model,) = plotter.objects
    norm = surface.interpolate(np.linalg.norm(field[0].values, axis=1))
    assert np.allclose(model["values"], norm)
    assert model["scalar_bar"]["label"] == "Displacement Magnitude"

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
        lighting=False,
    )
    (model,) = plotter.objects
    assert model["scalar_bar"]["label"] == "strain"
    assert model["scalar_bar"]["vertical"]
    assert not model["stroke"] and not model["cull"] and model["opacity"] == 0.5
    assert plotter.view == "xy" and not plotter.axes

    # the colors of the translucent faces are not lit and they have no edges
    collection = plotter.render()._collection
    assert np.allclose(collection.get_facecolors()[:, 3], 0.5)
    assert np.allclose(collection.get_linewidths(), 0)

    # a mesh without scalars and without a scalar bar
    plotter = field.plot("Displacement", show_scalar_bar=False, show_undeformed=False)
    assert plotter.objects[0]["scalar_bar"] is None

    # the wireframe of a mesh, a second view keeps the camera of the first view
    plotter = mesh.plot(style="wireframe", plotter=plotter)
    wireframe = plotter.objects[-1]
    assert not wireframe["fill"] and wireframe["stroke"] and not wireframe["cull"]
    assert plotter.view == "default"

    # the faces of the wireframe are not filled, subdivided faces by their outlines
    faces = plotter.render()._collection.get_facecolors()
    assert np.sum(faces[:, 3] == 0) == 24

    plotter = mesh.plot(style="wireframe", nonlinear_subdivision=2)
    assert len(paths(plotter)) == 24

    with pytest.raises(KeyError):
        field.plot("Unknown")


def test_view_subdivisions(mpl):
    mesh = fem.mesh.convert(fem.Cube(n=3), order=2)

    # the edges are subdivided into order * 2 ** (k - 1) segments (also of linear
    # cells), the faces are drawn by quads and by the outlines of the faces
    for cube, k, n in [(mesh, 1, 2), (mesh, 2, 4), (fem.Cube(n=3), 3, 4)]:
        (model,) = cube.plot(nonlinear_subdivision=k).objects
        (triangles, *_), (quads, faces, colors) = model["polygons"]
        assert len(triangles) == 0 and len(quads) == 24 * n**2
        assert not model["stroke"]

        segments, owners = model["outlines"]
        assert len(segments) == 24 * 4 * n
        assert np.array_equal(np.bincount(owners), np.full(24, 4 * n))

    # one outline (path) per visible face with 4 * 2 line segments
    drawn = paths(mesh.plot())
    outlines = [path for path in drawn if len(path) == 16]
    assert 0 < len(outlines) < 24

    # the faces of triangles are kept
    tetra = fem.Cube(n=3).triangulate()
    (model,) = tetra.plot().objects
    (triangles, *_), (quads, *_) = model["polygons"]
    assert len(quads) == 0 and len(triangles) > 0


def test_polygons():
    # two triangles of a quad face are merged, other triangles are kept
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [7, 8, 9], [7, 9, 8]])
    faces = np.array([0, 0, 1, 2, 2])
    (single, single_faces, first), (quads, quad_faces, quad_first) = backend._polygons(
        triangles, faces, 1
    )
    assert np.array_equal(quads, [[0, 1, 2, 3]]) and np.array_equal(quad_faces, [0])
    assert np.array_equal(single, triangles[[2, 3, 4]])
    assert np.array_equal(first, [2, 3, 4]) and np.array_equal(quad_first, [0])

    # no triangles
    ((empty, *_),) = backend._polygons(np.zeros((0, 3), dtype=int), [], 1)
    assert empty.shape == (0, 3)

    # the outlines of the faces
    segments, owners = backend._face_outlines(triangles[:2], faces[:2])
    assert len(segments) == 4 and np.all(owners == 0)


def test_camera(mpl):
    # a planar model is viewed from the top by a parallel projection
    plotter = fem.Rectangle(n=3).plot()
    camera = plotter._camera()
    assert camera.parallel and np.allclose(camera.basis, np.eye(3))

    # a three-dimensional model by a perspective projection, optionally parallel
    plotter = fem.Cube(n=3).plot()
    camera = plotter._camera()
    assert not camera.parallel
    assert np.allclose(camera.basis, backend.DEFAULT_VIEW)

    assert plotter.enable_parallel_projection() is plotter
    assert plotter._camera().parallel

    # the named views
    for view in ["xy", "yx", "xz", "zx", "yz", "zy", "iso"]:
        plotter = fem.Cube(n=3).plot(view=view)
        basis = plotter._camera().basis
        assert np.allclose(basis @ basis.T, np.eye(3))
        assert np.isclose(np.linalg.det(basis), 1)

    # the projection keeps the order of the depths
    camera = backend._Camera(np.eye(3), np.zeros(3), 1.0, parallel=False)
    uv, depth = camera.project([[0, 0, 0], [0, 0, 1], [0.5, 0, 0]])
    assert depth[1] < depth[0] and uv[2, 0] > 0

    # faces towards the camera are drawn in their colors
    colors = np.array([[0.2, 0.4, 0.6], [0.2, 0.4, 0.6]])
    shaded = camera.shade(colors, np.array([[0, 0, 2.0], [1.0, 0, 0]]))
    assert np.allclose(shaded[0], colors[0])
    assert np.all(shaded[1] < colors[1])

    # an empty scene
    camera = MatplotlibPlotter(off_screen=True)._camera()
    assert not camera.parallel


def test_screenshot(mpl, tmp_path):
    mesh, field, boundaries, solid = solved()
    figures = plt.get_fignums()

    # off-screen screenshots, without a figure of pyplot
    image = solid.screenshot("Principal Values of Cauchy Stress", filename=None)
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8
    assert plt.get_fignums() == figures

    image = mesh.screenshot(transparent_background=True, scale=2, filename=None)
    assert image.shape == (960, 1280, 4)
    assert image[0, 0, 3] == 0

    # raster and vector formats
    for suffix in ["png", "pdf", "svg", "eps"]:
        path = tmp_path / "shots" / f"solid.{suffix}"
        solid.screenshot(filename=path)
        assert path.exists() and path.stat().st_size > 0

    # the model in a vector format: one path per visible face and a path per edge of
    # the faces
    svg = (tmp_path / "shots" / "solid.svg").read_text(encoding="utf-8")
    assert svg.count("<path") > 24

    # the screenshots of the other plot-methods
    assert boundaries.screenshot(filename=None).shape == (480, 640, 3)
    container = fem.MeshContainer([fem.Cube(n=2), fem.Cube(n=2).translate(1, axis=0)])
    assert container.screenshot(filename=None).shape == (480, 640, 3)
    quadrature = fem.GaussLegendre(order=1, dim=2)
    assert quadrature.screenshot(filename=tmp_path / "quadrature.png").ndim == 3
    assert fem.Rectangle(n=3).screenshot(filename=tmp_path / "rectangle.png").ndim == 3

    # the image data in the axes of matplotlib
    ax = field.imshow("Displacement")
    assert len(ax.images) == 1

    # the plots of elements and regions require PyVista
    with pytest.raises(NotImplementedError, match="finite elements"):
        fem.Hexahedron().plot()


def test_given_axes(mpl, tmp_path):
    fig, ax = plt.subplots(1, 2)
    ax[0].set_title("cube")

    # a scene on given axes is drawn whenever an object is added
    plotter = fem.Cube(n=3).plot(plotter=ax[0])
    assert isinstance(plotter, MatplotlibPlotter) and plotter._ax is ax[0]
    assert len(ax[0].collections) == 1 and not plotter._stale

    # the same plotter for the same axes, the scene is drawn again
    assert fem.Cube(n=3).translate(1, axis=0).plot(plotter=ax[0]) is plotter
    assert len(plotter.objects) == 2 and len(ax[0].collections) == 1
    assert ax[0].get_title() == "cube"

    # decorations on the axes, e.g. of boundary conditions
    mesh, field, boundaries, solid = solved()
    solid.plot("Displacement", plotter=ax[1], show_undeformed=False)
    boundaries.plot(plotter=ax[1])
    assert [entry[0] for entry in _PLOTTER(ax[1]).legend] == list(boundaries.keys())

    path = tmp_path / "axes.pdf"
    fig.savefig(path)
    assert path.stat().st_size > 0

    # an explicit plotter of given axes
    fig, ax = plt.subplots()
    plotter = MatplotlibPlotter(ax=ax)
    assert fem.Rectangle(n=3).plot(plotter=plotter) is plotter
    assert plotter.ax is ax and plotter.figure is fig
    assert plotter.enable_parallel_projection()._camera().parallel

    # three-dimensional axes and other objects are not plotters of matplotlib
    ax3d = plt.figure().add_subplot(projection="3d")
    mpl_backend = backend_of(plotter)
    assert not mpl_backend.is_plotter(ax3d)
    assert not mpl_backend.is_plotter(object())

    with pytest.raises(TypeError, match="plotter of matplotlib"):
        mpl_backend.prepare_plotter(plotter=object())

    # without an imported matplotlib, there are no axes of matplotlib
    with mock.patch.dict(sys.modules, {"matplotlib.axes": None}):
        assert not mpl_backend.is_plotter(ax)


def _PLOTTER(ax):
    "Return the plotter of given axes."

    return backend._PLOTTERS[ax]


def test_scalar_bars(mpl):
    mesh, field, boundaries, solid = solved()

    # the options of the scalar bar
    plotter = solid.plot(
        "Equivalent of Cauchy Stress",
        scalar_bar_args={"title": "von Mises", "n_labels": 3, "fmt": "%.1e"},
    )
    plotter.render()
    cax = plotter._artists[1]
    assert cax.get_title() == "von Mises"
    labels = [tick.get_text() for tick in cax.get_xticklabels()]
    assert len(labels) == 3 and all("e" in label for label in labels)

    # a vertical scalar bar
    plotter = solid.plot("Displacement", scalar_bar_args={"vertical": True})
    plotter.render()
    cax = plotter._artists[1]
    assert len(cax.get_yticklabels()) == 5

    # a uniform range is drawn by one tick
    zero = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    plotter = zero.plot("Displacement", show_undeformed=False)
    plotter.render()
    cax = plotter._artists[1]
    assert [tick.get_text() for tick in cax.get_xticklabels()] == ["0"]

    # one scalar bar per label, side by side
    plotter = solid.plot("Displacement", component=0)
    plotter = solid.plot("Displacement", component=1, plotter=plotter)
    plotter = solid.plot("Displacement", component=1, plotter=plotter)
    assert len(plotter._scalar_bars()) == 2
    plotter = solid.plot("Displacement", component=0, scalar_bar_vertical=True)
    plotter = solid.plot("Displacement", component=1, plotter=plotter)
    plotter.render()

    # labels of ticks
    assert backend._format_number(0.0) == "0"
    assert backend._format_number(0.1234) == "0.123"
    assert backend._format_number(1234.5) == "1230"
    assert backend._format_number(1e5) == "1.00e+05"
    assert backend._format_number(np.nan) == "nan"

    # values which are not finite are grey
    cmap = backend._colormap("viridis")
    colors = backend._map_colors([0, np.nan, 1], cmap, [0, 1])
    assert np.allclose(colors[1], 0.6)
    assert np.allclose(backend._map_colors([1, 2], cmap, None), cmap(0.5)[:3])


def test_view_options_not_supported(mpl):
    mesh = fem.Rectangle(n=3)

    for options in [
        dict(theme="document"),
        dict(scalar_bar_args={"position_x": 0.2}),
        dict(show_logo=True),
        dict(specular=0.5),
        dict(style="points"),
    ]:
        with pytest.warns(UserWarning, match='plotting backend "matplotlib"'):
            mesh.plot(**options)

    # each option is warned once
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh.plot(theme="document", specular=0.5)

    # scenes without points and cells, e.g. of ViewXdmf
    scene = fem.view.Scene()
    scene.mesh = None
    with pytest.raises(NotImplementedError, match="without points and cells"):
        scene.plot()


def test_decorations(mpl):
    mesh, field, boundaries, solid = solved()

    # boundaries with points (labels of the legend) and lines
    plotter = boundaries.plot()
    kinds = [obj["type"] for obj in plotter.objects]
    assert kinds.count("points") == 5 and kinds.count("lines") > 0
    assert [label for label, color in plotter.legend] == list(boundaries.keys())
    assert np.allclose(plotter.legend[0][1], backend._rgba("tab:blue"))
    plotter.render()
    assert plotter._ax.get_legend() is not None

    # arrows of a point load by triangles of glyphs
    right = np.arange(mesh.npoints)[mesh.x == 1]
    load = fem.PointLoad(field, points=right, values=[1, 0, 0])
    plotter = load.plot(plotter=fem.Cube(n=3).plot(), color="green")
    arrows = plotter.objects[-1]
    ((triangles, faces, colors), _) = arrows["polygons"]
    assert len(triangles) > 0 and np.allclose(colors, backend._rgba("green")[:3])

    # a rigid plane (one quad) with its outline, a center-point and a normal
    center = fem.Cube(n=3)
    center.update(points=np.vstack([center.points, [0.5, 0.5, 1.5]]))
    fc = fem.FieldContainer([fem.Field(fem.RegionHexahedron(center), dim=3)])
    top = np.arange(center.npoints)[center.z == 1]
    plane = fem.ContactRigidPlane(fc, points=top, centerpoint=-1, normal=[0, 0, -1])
    plotter = plane.plot(line_width=2, opacity=0.4)
    shape, point, normal = plotter.objects
    (_, (quads, *_)) = shape["polygons"]
    assert len(quads) == 1 and shape["opacity"] == 0.4 and shape["line_width"] == 2
    assert len(shape["outlines"][0]) == 4
    assert point["type"] == "points" and point["size"] == 12
    assert normal["type"] == "lines"
    plotter.render()

    # without edges or fully clipped
    plotter = plane.plot(show_edges=False)
    assert plotter.objects[0]["outlines"] is None
    plotter = plane.plot(offset=-2, sym=(False, False, True), show_point=False)
    assert [obj["type"] for obj in plotter.objects] == ["lines"]

    # the default and given colors of the meshes of a container
    container = fem.MeshContainer([fem.Cube(n=2), fem.Cube(n=2).translate(1, axis=0)])
    plotter = container.plot(labels=["a", "b"])
    color = plotter.objects[0]["polygons"][1][2][0]
    assert np.allclose(color, backend._rgba(backend.DEFAULT_COLOR)[:3])
    assert [label for label, color in plotter.legend] == ["a", "b"]
    assert plotter.objects[0]["cull"]

    # options of the primitives which are not supported
    mpl_backend = backend_of(plotter)
    for add in [
        lambda: mpl_backend.add_points(
            plotter, [[0, 0, 0]], render_points_as_spheres=1
        ),
        lambda: mpl_backend.add_lines(plotter, [[[0, 0], [1, 1]]], lighting=False),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            add()

    for add in [
        lambda: mpl_backend.add_points(plotter, [[0, 0, 0]], specular=1),
        lambda: mpl_backend.add_lines(plotter, [[[0, 0], [1, 1]]], specular=2),
        lambda: mpl_backend.add_arrows(plotter, [[0, 0]], [[1, 0]], specular=3),
        lambda: mpl_backend.add_shape(plotter, plane_shape(), specular=4),
        lambda: mpl_backend.add_legend(plotter, [("a", "red")], specular=5),
    ]:
        backend._WARNED.clear()
        with pytest.warns(UserWarning, match='"specular"'):
            add()

    # zero-length arrows are not shown
    plotter = mpl_backend.add_arrows(
        MatplotlibPlotter(off_screen=True), [[0, 0]], [[0, 0]]
    )
    assert len(plotter.objects[0]["polygons"][0][0]) == 0
    assert len(paths(plotter)) == 0


def plane_shape():
    from felupe.view._shapes import Plane

    return Plane(center=(0, 0, 0), normal=(0, 0, 1), size=1.0)


def test_show(mpl):
    # a figure of pyplot is shown by pyplot
    plotter = fem.Rectangle(n=3).plot()
    assert plotter.figure.canvas.manager is not None

    with mock.patch.object(plt, "show") as show:
        assert plotter.show() is None
        show.assert_called_once()

    plotter.close()
    assert plotter.figure.number not in plt.get_fignums()

    # an off-screen figure is not shown outside of a notebook
    plotter = fem.Rectangle(n=3).plot(off_screen=True)
    assert plotter.figure.canvas.manager is None

    with mock.patch.object(plt, "show") as show:
        assert plotter.show() is None
        show.assert_not_called()

    plotter.close()

    # in a notebook, the image of the figure is shown in the output of the cell
    display = pytest.importorskip("IPython.display")

    with (
        mock.patch.object(backend, "in_kernel", return_value=True),
        mock.patch.object(display, "display") as show,
    ):
        plotter = fem.Rectangle(n=3).plot()
        assert plotter.figure.canvas.manager is None
        assert plotter.show() is None
        (image,), options = show.call_args
        assert image.data.startswith(b"\x89PNG")

    # the figure is saved by the plotter
    with mock.patch.object(plotter.figure, "savefig") as savefig:
        plotter.savefig("model.pdf")
        savefig.assert_called_once_with("model.pdf")

    # a changed view of a drawn scene is drawn by a screenshot
    plotter = fem.Cube(n=3).plot(off_screen=True)
    plotter.ax.set_title("cube")
    plotter.view = "xy"
    plotter.screenshot()
    assert np.allclose(plotter._camera().basis, np.eye(3))
    assert plotter.ax.get_title() == "cube"


def test_lines_and_points_only(mpl):
    # a mesh of lines has no faces and no undeformed model
    plotter = fem.mesh.Line(n=4).plot()
    (model,) = plotter.objects
    assert len(model["polygons"][0][0]) == 0 and len(model["lines"][0]) == 3
    assert len(paths(plotter)) == 3

    # the colors of the lines by point-data
    line = fem.mesh.Line(n=4)
    view = fem.ViewMesh(line, point_data={"x": line.points[:, 0]})
    (model,) = view.plot("x", show_undeformed=False).objects
    lines, colors = model["lines"]
    assert not np.allclose(colors[0], colors[-1])

    # the points of a quadrature scheme
    plotter = fem.GaussLegendre(order=1, dim=2).plot()
    assert [obj["type"] for obj in plotter.objects] == ["surface"] + ["points"] * 4
    assert len(paths(plotter)) == 4

    # the points are drawn after the faces of a surface (at equal depth)
    plotter = fem.Rectangle(n=2).plot()
    plotter = backend_of(plotter).add_points(plotter, [[0.5, 0.5]], color="red")
    drawn = plotter.render()._collection
    assert np.allclose(drawn.get_facecolors()[-1], backend._rgba("red"))
