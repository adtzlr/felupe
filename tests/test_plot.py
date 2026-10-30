# SPDX-License-Identifier: GPL-3.0-or-later

import contextlib
import pathlib
import sys
import tempfile
from unittest import mock

import numpy as np
import pytest

import felupe as fem


def pre(n=3, filename=None):
    mesh = fem.Cube(n=n)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    umat = fem.NeoHooke(mu=1)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    move = fem.math.linsteps([0, 1], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(filename=filename, verbose=False)

    return mesh, field, solid


def pre_2d(n=3, filename=None):
    mesh = fem.Rectangle(n=n)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    umat = fem.NeoHooke(mu=1)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    move = fem.math.linsteps([0, 1], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(filename=filename, verbose=False)

    return mesh, field, solid


def test_xdmf_cell_data(tmp_path):
    try:
        filename = str(tmp_path / "result.xdmf")
        with contextlib.chdir(tmp_path):  # meshio writes the h5-file to the cwd
            pre(n=3, filename=filename)

        view = fem.ViewXdmf(filename, time=3)
        plotter = view.plot(
            "Principal Values of Logarithmic Strain",
            off_screen=True,
        )
        # plotter.show(screenshot="cube-xdmf.png")

    except ModuleNotFoundError:
        pass


def test_xdmf_point_data(tmp_path):
    try:
        filename = str(tmp_path / "result.xdmf")
        with contextlib.chdir(tmp_path):  # meshio writes the h5-file to the cwd
            pre_2d(n=3, filename=filename)

        view = fem.ViewXdmf(filename, time=3)
        plotter = view.plot(
            "Displacement",
            off_screen=True,
            scalar_bar_vertical=True,
        )
        # plotter.show(screenshot="rectangle-xdmf.png")

    except ModuleNotFoundError:
        pass


def test_cell_data():
    try:
        import pyvista

        mesh, field, solid = pre(n=3)
        view = fem.View(
            field,
            point_data={"My Displacements": field[0].values},
            cell_data={"Cell Volume": field.region.dV.sum(0).ravel()},
            cell_type=pyvista.CellType.HEXAHEDRON,
        )
        plotter = view.plot(
            "Principal Values of Logarithmic Strain",
            off_screen=True,
            theme="document",
        )
        plotter = view.plot(
            "Cell Volume",
            off_screen=True,
            show_logo=True,
            component=None,
        )
        # plotter.show(screenshot="cube.png")

    except ModuleNotFoundError:
        pass


def test_point_data():
    try:
        mesh, field, solid = pre_2d(n=3)
        view = fem.View(field)
        plotter = view.plot(
            "Displacement",
            off_screen=True,
            scalar_bar_vertical=False,
        )
        # plotter.show(screenshot="rectangle.png")

    except ModuleNotFoundError:
        pass


def test_mesh():
    try:
        mesh, field, solid = pre(n=3)
        view = fem.ViewMesh(mesh)
        plotter = view.plot(
            off_screen=True,
        )
        # plotter.show(screenshot="cube.png")

    except ModuleNotFoundError:
        pass


def test_model():
    try:
        mesh, field, solid = pre(n=3)
        with pytest.raises(TypeError):
            fem.ViewField(field, project=True)
        with pytest.raises(TypeError):
            fem.View(field, solid=solid, project=True)
        view = fem.View(field, solid=solid, project=fem.project)
        plotter = view.plot(
            "Equivalent of Cauchy Stress",
            off_screen=True,
            notebook=True,
        )
        # plotter.show(screenshot="undeformed.png")

    except ModuleNotFoundError:
        pass


def test_logo_path():
    from importlib.resources import as_file, files

    with as_file(files("felupe.view") / "logo.png") as logo_path:
        assert logo_path.is_file()


def test_view_without_pyvista():
    mesh, field, solid = pre(n=3)

    # the views keep the arrays, PyVista is only required for the grid and plotting
    with mock.patch.dict(sys.modules, {"pyvista": None}):
        view = fem.ViewSolid(field, solid=solid, point_data={"Values": mesh.points})

        assert view.points is mesh.points
        assert view.cells is mesh.cells
        assert view.cell_type == "hexahedron"
        assert view.point_data["Values"] is mesh.points
        assert "Displacement" in view.point_data
        assert "Principal Values of Cauchy Stress" in view.cell_data

        with pytest.raises(ImportError):
            view.mesh

        with pytest.raises(ImportError):
            view.plot()


def test_view_mesh_unstructured_grid():
    pv = pytest.importorskip("pyvista")

    mesh, field, solid = pre(n=3)
    view = fem.ViewSolid(field, solid=solid)

    # the grid is created on first access and kept
    grid = view.mesh
    assert isinstance(grid, pv.UnstructuredGrid)
    assert view.mesh is grid
    assert grid.n_cells == mesh.ncells
    assert set(grid.point_data.keys()) == set(view.point_data.keys())
    assert set(grid.cell_data.keys()) == set(view.cell_data.keys())
    assert grid.active_scalars_name is None

    # the principal values are sorted in descending order for plotting
    plotter = view.plot("Principal Values of Cauchy Stress", off_screen=True)
    assert "Cauchy Stress (Max. Principal)" in plotter.scalar_bars.keys()
    values = grid.cell_data["Principal Values of Cauchy Stress"]
    assert np.all(np.diff(values, axis=1) <= 0)

    # the grid may be replaced
    view.mesh = grid.extract_surface(algorithm=None)
    assert isinstance(view.mesh, pv.PolyData)
    plotter = view.plot("Displacement", off_screen=True, plotter=plotter)

    # a pyvista cell type is used for the grid
    view = fem.ViewMesh(mesh, cell_type=pv.CellType.HEXAHEDRON)
    assert view.mesh.celltypes[0] == pv.CellType.HEXAHEDRON


def test_scalars_label():
    from felupe.view._scalars import component_labels, scalars_label

    assert scalars_label("Displacement", component=0, dim=3) == "Displacement X"
    assert scalars_label("Displacement", component=None) == "Displacement Magnitude"
    assert scalars_label("Cauchy Stress", component=3, dim=6) == "Cauchy Stress XY"
    assert scalars_label("Cauchy Stress", component=7, dim=9) == "Cauchy Stress ZY"
    assert scalars_label("Cell Volume", component=0, dim=1) == "Cell Volume "
    assert scalars_label("Values", component=2, dim=4) == "Values 2"

    name = "Principal Values of Logarithmic Strain"
    assert scalars_label(name, 0, 3) == "Logarithmic Strain (Max. Principal)"
    assert scalars_label(name, 1, 3) == "Logarithmic Strain (Int. Principal)"
    assert scalars_label(name, 1, 2) == "Logarithmic Strain (Min. Principal)"
    assert scalars_label(name, None, 3) == f"{name} Magnitude"

    assert component_labels("Displacement", dim=2) == ["X", "Y"]
    assert np.all(component_labels("Values", dim=4) == np.arange(4))


if __name__ == "__main__":
    for test in [test_xdmf_cell_data, test_xdmf_point_data]:
        with tempfile.TemporaryDirectory() as tmp:
            test(tmp_path=pathlib.Path(tmp))

    test_cell_data()
    test_point_data()
    test_mesh()
    test_model()
    test_logo_path()
    test_view_without_pyvista()
    test_view_mesh_unstructured_grid()
    test_scalars_label()
