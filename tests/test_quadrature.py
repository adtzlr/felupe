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


def test_gausslegendre():
    q11 = fe.quadrature.GaussLegendre(order=1, dim=1)
    assert q11.points.shape == (2, 1)
    assert q11.weights.sum() == 2

    q12 = fe.quadrature.GaussLegendre(order=1, dim=2)
    assert q12.points.shape == (4, 2)
    assert q12.weights.sum() == 4

    q13 = fe.quadrature.GaussLegendre(order=1, dim=3)
    assert q13.points.shape == (8, 3)
    assert q13.weights.sum() == 8

    with pytest.raises(ValueError):
        fe.quadrature.GaussLegendre(order=1, dim=4)


def test_triangle():
    q = fe.quadrature.Triangle(order=1)
    assert q.points.shape == (1, 2)
    assert q.weights.sum() == 1 / 2


def test_tetra():
    q = fe.quadrature.Tetrahedron(order=1)
    assert q.points.shape == (1, 3)
    assert q.weights.sum() == 1 / 6


if __name__ == "__main__":
    test_gausslegendre()
    test_triangle()
    test_tetra()
