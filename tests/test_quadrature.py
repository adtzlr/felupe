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


def test_gausslegendre():
    q11 = fem.GaussLegendre(order=1, dim=1)
    assert q11.points.shape == (2, 1)
    assert q11.weights.sum() == 2

    for permute in [False, True]:
        q12 = fem.GaussLegendre(order=1, dim=2, permute=permute)
        assert q12.points.shape == (4, 2)
        assert np.isclose(q12.weights.sum(), 4)

        q13 = fem.GaussLegendre(order=1, dim=3, permute=permute)
        assert q13.points.shape == (8, 3)
        assert np.isclose(q13.weights.sum(), 8)

        q23 = fem.GaussLegendre(order=2, dim=3, permute=permute)
        assert q23.points.shape == (27, 3)
        assert np.isclose(q23.weights.sum(), 8)

    q23 = fem.GaussLegendre(order=2, dim=3)
    q23.plot(off_screen=True)
    q23.plot(off_screen=True, weighted=True)

    assert q23.inv().points.shape == q23.points.shape
    assert np.allclose(q23.weights, q23.inv().weights)

    with pytest.raises(ValueError):
        fem.GaussLegendre(order=1, dim=4)


def test_gausslegendre_boundary():
    with pytest.raises(ValueError):
        q11 = fem.GaussLegendreBoundary(order=1, dim=1)
        assert q11.points.shape == (2, 1)
        assert q11.weights.sum() == 2

    for permute in [False, True]:
        q12 = fem.GaussLegendreBoundary(order=1, dim=2, permute=permute)
        assert q12.points.shape == (2, 2)
        assert np.isclose(q12.weights.sum(), 2)

        q13 = fem.GaussLegendreBoundary(order=1, dim=3, permute=permute)
        assert q13.points.shape == (4, 3)
        assert np.isclose(q13.weights.sum(), 4)

        q23 = fem.GaussLegendreBoundary(order=2, dim=3, permute=permute)
        assert q23.points.shape == (9, 3)
        assert np.isclose(q23.weights.sum(), 4)

    q23 = fem.GaussLegendreBoundary(order=2, dim=3)
    assert q23.inv().points.shape == q23.points.shape
    assert np.allclose(q23.weights, q23.inv().weights)

    with pytest.raises(ValueError):
        fem.GaussLegendreBoundary(order=1, dim=4)


def test_triangle():
    q = fem.TriangleQuadrature(order=1)
    assert q.points.shape == (1, 2)
    assert q.weights.sum() == 1 / 2

    q = fem.TriangleQuadrature(order=2)
    assert q.points.shape == (3, 2)
    assert q.weights.sum() == 1 / 2

    q = fem.TriangleQuadrature(order=3)
    assert q.points.shape == (4, 2)
    assert q.weights.sum() == 1 / 2

    with pytest.raises(NotImplementedError):
        fem.TriangleQuadrature(order=4)


def test_tetra():
    q = fem.TetrahedronQuadrature(order=1)
    assert q.points.shape == (1, 3)
    assert q.weights.sum() == 1 / 6

    q = fem.TetrahedronQuadrature(order=2)
    assert q.points.shape == (4, 3)
    assert q.weights.sum() == 1 / 6

    q = fem.TetrahedronQuadrature(order=3)
    assert q.points.shape == (5, 3)
    assert q.weights.sum() == 1 / 6

    with pytest.raises(NotImplementedError):
        fem.TetrahedronQuadrature(order=4)


def test_sphere():
    q = fem.BazantOh(n=21)
    assert q.points.shape == (21, 3)
    assert np.isclose(q.weights.sum(), 1)


if __name__ == "__main__":
    test_gausslegendre()
    test_gausslegendre_boundary()
    test_triangle()
    test_tetra()
    test_sphere()
