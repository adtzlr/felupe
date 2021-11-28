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
from copy import deepcopy
import numpy as np
import felupe as fe


def test_meshes():

    m = fe.mesh.Line(n=5)
    assert m.points.shape == (5, 1)
    assert m.cells.shape == (4, 2)

    m = fe.Rectangle(a=(-1.2, -2), b=(2, 3.1), n=(4, 9))
    assert m.points.shape == (4 * 9, 2)
    assert m.cells.shape == (3 * 8, 4)

    fe.mesh.convert(m, order=0)
    fe.mesh.convert(m, order=0, calc_points=True)
    fe.mesh.convert(m, order=2)
    fe.mesh.convert(m, order=2, calc_midfaces=True)

    mm = deepcopy(m)
    mm.cell_type = "fancy"

    with pytest.raises(NotImplementedError):
        fe.mesh.convert(mm, order=2)

    with pytest.raises(NotImplementedError):
        fe.mesh.convert(m, order=1)

    fe.mesh.revolve(m, n=11, phi=180, axis=0)
    fe.mesh.revolve((m.points, m.cells), n=11, phi=180, axis=0)
    fe.mesh.revolve((m.points, m.cells), n=11, phi=360, axis=0)

    with pytest.raises(ValueError):
        fe.mesh.revolve((m.points, m.cells), n=11, phi=361, axis=0)

    fe.mesh.expand((m.points, m.cells))
    fe.mesh.expand(m)

    m = fe.Cube(a=(-1, -2, -0.5), b=(2, 3.1, 1), n=(4, 9, 5))
    assert m.points.shape == (4 * 9 * 5, 3)
    assert m.cells.shape == (3 * 8 * 4, 8)

    fe.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    with pytest.raises(ValueError):
        fe.mesh.expand((m.points, m.cells))

    with pytest.raises(ValueError):
        fe.mesh.revolve((m.points, m.cells))

    fe.mesh.convert(m, order=2, calc_midfaces=True, calc_midvolumes=True)

    fe.mesh.rotate(m, angle_deg=10, axis=0, center=None)
    fe.mesh.rotate((m.points, m.cells), angle_deg=10, axis=0, center=None)
    fe.mesh.rotate(m, angle_deg=10, axis=1, center=[0, 0, 0])

    fe.mesh.CubeArbitraryOrderHexahedron()
    fe.mesh.RectangleArbitraryOrderQuad()

    m = fe.Rectangle(n=5)
    m.points = np.vstack((m.points, [10, 10]))
    assert m.points.shape == (26, 2)
    assert m.cells.shape == (16, 4)

    m_dg = m.disconnect()
    assert m_dg.dim == m.dim
    assert m_dg.npoints == m.cells.size

    fe.mesh.sweep(m)
    fe.mesh.sweep((m.points, m.cells), decimals=4)

    m.save()

    m.cell_type = None
    with pytest.raises(TypeError):
        m.save()


if __name__ == "__main__":
    test_meshes()
