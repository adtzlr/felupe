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

import pytest

import felupe as fem


def pre(n=3):
    mesh = fem.Cube(n=n)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True)[0]
    umat = fem.NeoHooke(mu=1)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    move = fem.math.linsteps([0, 1], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(filename="result.xdmf", verbose=False)

    return mesh, field, solid


def pre_2d(n=3):
    mesh = fem.Rectangle(n=n)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    boundaries = fem.dof.uniaxial(field, clamped=True)[0]
    umat = fem.NeoHooke(mu=1)
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    move = fem.math.linsteps([0, 1], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    fem.Job(steps=[step]).evaluate(filename="result.xdmf", verbose=False)

    return mesh, field, solid


def test_xdmf_cell_data():
    try:
        import pyvista

        pre(n=3)

        # TODO: Remove if pyvista has a release with ``XdmfReader`` included.
        major, minor = pyvista.__version__.split(".")[:2]
        if int(major) == 0 and int(minor) < 39:
            print("Xdmf-Reader not available. Requires ``pyvista >= 0.39``.")
        else:
            view = fem.ViewXdmf("result.xdmf", time=3)
            plotter = view.plot(
                "Principal Values of Logarithmic Strain",
                off_screen=True,
            )
            # plotter.show(screenshot="cube-xdmf.png")

    except ModuleNotFoundError:
        pass


def test_xdmf_point_data():
    try:
        import pyvista

        pre_2d(n=3)

        # TODO: Remove if pyvista has a release with ``XdmfReader`` included.
        major, minor = pyvista.__version__.split(".")[:2]
        if int(major) == 0 and int(minor) < 39:
            print("Xdmf-Reader not available. Requires ``pyvista >= 0.39``.")
        else:
            view = fem.ViewXdmf("result.xdmf", time=3)
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


if __name__ == "__main__":
    test_xdmf_cell_data()
    test_xdmf_point_data()
    test_cell_data()
    test_point_data()
    test_mesh()
    test_model()
