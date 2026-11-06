# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import felupe as fem


def view_of(mesh):
    "Return a view of a mesh with the ids of the points and of the cells as data."

    return fem.ViewMesh(
        mesh,
        point_data={"Point ID": np.arange(mesh.npoints, dtype=float)},
        cell_data={"Cell ID": np.arange(mesh.ncells)},
    )


def test_clip_crinkle():
    mesh = fem.Cube(n=6)  # points at 0.0, 0.2, ..., 1.0
    view = view_of(mesh)

    # default: normal "x", origin at the center (0.5), keep the part behind the plane
    clipped = view.clip()
    assert isinstance(clipped, fem.ViewMesh)
    assert len(clipped.cells) == 3 * 25
    assert np.isclose(clipped.points[:, 0].max(), 0.6)

    # cells are kept as a whole, also if they are cut by the plane
    assert len(view.clip("x", origin=(0.3, 0, 0)).cells) == 2 * 25
    assert len(view.clip("x", origin=(0.3, 0, 0), invert=False).cells) == 4 * 25

    # a negative normal is equal to the inverted positive normal
    for origin in [(0.3, 0.7, 0.1), (0.4, 0.4, 0.4)]:
        for invert in [True, False]:
            a = view.clip("-y", origin=origin, invert=invert)
            b = view.clip("y", origin=origin, invert=not invert)
            assert np.array_equal(a.cell_data["Cell ID"], b.cell_data["Cell ID"])

    # cells which only touch the plane (at x=0.4) are not kept on both sides
    assert len(view.clip("x", origin=(0.4, 0, 0), invert=True).cells) == 2 * 25
    assert len(view.clip("x", origin=(0.4, 0, 0), invert=False).cells) == 3 * 25

    # the value shifts the plane along the normalized normal vector
    a = view.clip((0, 0, 2), origin=(0, 0, 0.5), value=-0.2)
    b = view.clip("z", origin=(0, 0, 0.3))
    assert np.array_equal(a.cell_data["Cell ID"], b.cell_data["Cell ID"])

    # unused points are removed and the point-data is taken by the kept points
    clipped = view.clip((1, 1, 0), origin=(0.45, 0.35, 0), invert=False)
    ids = clipped.point_data["Point ID"].astype(int)
    assert np.allclose(clipped.points, mesh.points[ids])
    assert np.array_equal(
        clipped.points[clipped.cells],
        mesh.points[mesh.cells[clipped.cell_data["Cell ID"]]],
    )
    assert np.array_equal(np.unique(clipped.cells), np.arange(len(clipped.points)))

    # the scene is not modified
    assert len(view.cells) == mesh.ncells and len(view.points) == mesh.npoints

    # everything or nothing is clipped
    assert len(view.clip("x", origin=(2, 0, 0)).cells) == mesh.ncells
    nothing = view.clip("x", origin=(-1, 0, 0))
    assert len(nothing.cells) == 0 and len(nothing.points) == 0


def test_clip_planar():
    mesh = fem.Rectangle(n=5).add_midpoints_edges()
    clipped = view_of(mesh).clip("-x", origin=(0.3, 0.0))
    assert clipped.cells.shape == (3 * 4, 8)
    assert clipped.points.shape[1] == 2


def test_clip_field_and_solid():
    mesh = fem.Cube(n=4)
    field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
    solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)

    for view in [field.view(), solid.view(project=fem.project)]:
        clipped = view.clip("-z", origin=(0, 0, 0.6))
        assert type(clipped) is type(view)
        assert len(clipped.cells) == 2 * 9

        for data in clipped.point_data.values():
            assert len(data) == len(clipped.points)

        for data in clipped.cell_data.values():
            assert len(data) == len(clipped.cells)


@pytest.mark.parametrize("backend", ["webgl", "matplotlib"])
def test_clip_plot(backend):
    try:
        fem.set_plotting_backend(backend)
        mesh = fem.Cube(n=4)
        field = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh), dim=3)])
        view = field.view().clip("y", origin=(0, 0.4, 0), invert=False)
        plotter = view.plot("Principal Values of Logarithmic Strain", off_screen=True)

        # a crinkled clip is supported, an exact clip requires PyVista
        assert plotter is not None

        if backend == "webgl":
            pytest.importorskip("pyvista")
            with pytest.raises(NotImplementedError):
                field.view().clip("y", crinkle=False).plot()

    finally:
        fem.set_plotting_backend(None)


def test_clip_errors():
    view = view_of(fem.Cube(n=3))

    with pytest.raises(ValueError, match="must be one of"):
        view.clip("w")

    with pytest.raises(ValueError, match="non-zero vector"):
        view.clip((0, 0, 0))

    # vectors with more than three components or with more than one axis
    for normal in [(1, 0, 0, 0), [[1, 0, 0]], ()]:
        with pytest.raises(ValueError, match="normal must be a vector"):
            view.clip(normal)

    for origin in [(0, 0, 0, 0), [[0, 0, 0]], 0.5]:
        with pytest.raises(ValueError, match="origin must be a vector"):
            view.clip("x", origin=origin)


def test_clip_pyvista():
    pytest.importorskip("pyvista")

    meshes = [
        fem.Cube(n=6),
        fem.Cube(n=4).triangulate(),
        fem.Cube(n=3).add_midpoints_edges(),
        fem.Rectangle(n=5),
    ]

    # planes which don't pass through points (VTK keeps the cells which touch the
    # plane only for invert=False)
    for mesh in meshes:
        view = view_of(mesh)
        for normal in ["x", "-y", "z", (1, 1, 0), (0, -2, 1)]:
            for origin in [(0.53, 0.46, 0.51), (0.37, 0.41, 0.43)]:
                for invert in [True, False]:
                    kwargs = dict(normal=normal, origin=origin, invert=invert)
                    ids = view.clip(**kwargs).cell_data["Cell ID"]
                    reference = view.mesh.clip(crinkle=True, **kwargs)
                    if reference.n_cells == 0:
                        assert len(ids) == 0
                        continue
                    assert np.array_equal(
                        np.sort(ids), np.sort(reference.cell_data["Cell ID"])
                    )

    # the exact clip by PyVista
    exact = view_of(fem.Cube(n=6)).clip("x", origin=(0.3, 0, 0), crinkle=False)
    assert type(exact) is fem.view.Scene
    assert np.isclose(exact.mesh.bounds[1], 0.3)
