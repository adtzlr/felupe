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

import numpy as np

import felupe as fem


def test_free_vibration():
    mesh = fem.Cube(a=(0, 0, 5), b=(50, 100, 30), n=(3, 6, 4))
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    umat = fem.NeoHooke(mu=1, bulk=2)
    solid = fem.SolidBody(umat=umat, field=field, density=1.5e-9)

    job = fem.FreeVibration([solid]).evaluate()
    new_field, frequency = job.extract(n=-1, inplace=False)


def test_free_vibration_mixed():
    meshes = [
        fem.Cube(a=(0, 0, 30), b=(50, 100, 35), n=(3, 6, 2)),
        fem.Cube(a=(0, 0, 5), b=(50, 100, 30), n=(3, 6, 4)),
        fem.Cube(a=(0, 0, 0), b=(50, 100, 5), n=(3, 6, 2)),
    ]
    container = fem.MeshContainer(meshes, merge=True)
    mesh = container.stack()

    regions = [fem.RegionHexahedron(m) for m in container.meshes]
    fields = [
        fem.FieldsMixed(regions[0], n=1),
        fem.FieldsMixed(regions[1], n=3),
        fem.FieldsMixed(regions[2], n=1),
    ]

    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3), *fields[1][1:]])

    boundaries = dict(left=fem.Boundary(field[0], fx=0))
    rubber = fem.ThreeFieldVariation(fem.NeoHooke(mu=1, bulk=5000))
    steel = fem.LinearElasticLargeStrain(2.1e5, 0.3)
    solids = [
        fem.SolidBody(umat=steel, field=fields[0], density=7.85e-9),
        fem.SolidBody(umat=rubber, field=fields[1], density=1.5e-9),
        fem.SolidBody(umat=steel, field=fields[2], density=7.85e-9),
    ]

    job = fem.FreeVibration(solids, boundaries).evaluate(x0=field)
    new_field, frequency = job.extract(x0=field, n=-1, inplace=False)


if __name__ == "__main__":
    test_free_vibration()
    test_free_vibration_mixed()
