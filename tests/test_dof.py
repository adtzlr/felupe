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
import numpy as np
import felupe as fe


def pre():
    m = fe.mesh.Rectangle()
    e = fe.element.Quad()
    q = fe.quadrature.GaussLegendre(1, 2)
    r = fe.Region(m, e, q)
    u = fe.Field(r, dim=2)
    return u


def test_loadcase():
    u = pre()
    v = fe.FieldMixed((u, u))

    ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=False)
    assert len(ux) == 4

    ux = fe.dof.uniaxial(u, right=1.0, move=0.2, clamped=True)
    assert len(ux) == 4
    assert "right" in ux[0]

    ux = fe.dof.uniaxial(v, right=1.0, move=0.2, clamped=True)
    assert len(ux) == 5
    assert "right" in ux[0]


if __name__ == "__main__":
    test_loadcase()
