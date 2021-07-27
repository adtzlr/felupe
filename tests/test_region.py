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
import felupe as fe


def test_region_cube_hex():

    m = fe.mesh.Cube(n=4)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(1, 3)

    r = fe.Region(m, e, q)

    assert np.isclose(r.dV.sum(), 1)
    assert np.isclose(r.volume().sum(), 1)
    assert r.dhdX.shape == (e.nbasis, m.ndim, q.npoints, m.ncells)
    assert r.h.shape == (e.nbasis, q.npoints)


def test_region_cube_aol():

    order = 1

    m = fe.mesh.CubeArbitraryOderHexahedron(order=order)
    e = fe.element.ArbitraryOrderLagrange(order, ndim=3)
    q = fe.quadrature.GaussLegendre(order, 3)

    r = fe.Region(m, e, q)

    # assert np.isclose(r.dV.sum(), 1)
    # assert np.isclose(r.volume().sum(), 1)
    assert r.dhdX.shape == (e.nbasis, m.ndim, q.npoints, m.ncells)
    assert r.h.shape == (e.nbasis, q.npoints)


if __name__ == "__main__":
    test_region_cube_hex()
    test_region_cube_aol()
