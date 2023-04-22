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


def test_xdmf_cell_data():

    pre(n=3)
    result = fem.XdmfReader("result.xdmf", time=3)
    plotter = result.plot(
        scalars="Principal Values of Logarithmic Strain",
        cpos="iso",
        off_screen=True,
    )
    plotter.show(screenshot="cube.png")


def test_xdmf_point_data():

    pre(n=3)
    result = fem.XdmfReader("result.xdmf", time=3)
    plotter = result.plot(
        scalars="Displacement",
        cpos="iso",
        off_screen=True,
    )
    plotter.show(screenshot="cube.png")


if __name__ == "__main__":
    test_xdmf_cell_data()
    test_xdmf_point_data()
