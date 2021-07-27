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
import felupe as fe


def test_meshes():
    
    m = fe.mesh.Line(n=5)
    assert m.points.shape == (5, 1)
    assert m.cells.shape == (4, 2)
    
    m = fe.mesh.Rectangle(a=(-1.2, -2), b=(2, 3.1), n=(4, 9))
    assert m.points.shape == (4 * 9, 2)
    assert m.cells.shape == (3 * 8, 4)

    m = fe.mesh.Cube(a=(-1, -2, -0.5), b=(2, 3.1, 1), n=(4, 9, 5))
    assert m.points.shape == (4 * 9 * 5, 3)
    assert m.cells.shape == (3 * 8 * 4, 8)
    
    m = fe.mesh.Cylinder(n = (3,9,3), phi=180)
    assert m.points.shape == (3 * 9 * 3, 3)
    assert m.cells.shape == (2 * 8 * 2, 8)

if __name__ == "__main__":
    test_meshes()
