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
import pytest

import felupe as fem


def pre(dim):
    m = fem.Cube(n=3)
    r = fem.RegionHexahedron(m)
    u = fem.FieldContainer([fem.Field(r, dim=dim)])
    return r, u


def pre_constant(dim):
    m = fem.Cube(n=3)
    r = fem.RegionConstantHexahedron(m)
    u = fem.FieldContainer([fem.Field(r, dim=dim)])
    return r, u


def test_basis():
    for parallel in [False, True]:
        r, u = pre(dim=3)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert not np.any(b[0].basis.grad == None)

        r, u = pre(dim=1)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert not np.any(b[0].basis.grad == None)

        r, u = pre_constant(dim=3)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert np.all(b[0].basis.grad == None)

        r, u = pre_constant(dim=1)
        b = fem.assembly.expression.Basis(u, parallel=parallel)

        assert np.all(b[0].basis.grad == None)


if __name__ == "__main__":
    test_basis()
