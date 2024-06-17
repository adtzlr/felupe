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


def test_dtype():
    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh).astype(np.float32)
    displacement = fem.Field(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


def test_dtype_axi():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh).astype(np.float32)
    displacement = fem.FieldAxisymmetric(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


def test_dtype_planestrain():
    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh).astype(np.float32)
    displacement = fem.FieldPlaneStrain(region, dtype=np.float32)
    field = fem.FieldContainer([displacement])

    assert field.extract()[0].dtype == np.float32
    assert field.extract(grad=False)[0].dtype == np.float32


if __name__ == "__main__":
    test_dtype()
    test_dtype_axi()
    test_dtype_planestrain()
