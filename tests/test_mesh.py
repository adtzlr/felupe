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


def test_mesh():
    mr1 = fe.mesh.Rectangle()
    mr2 = fe.mesh.Rectangle(a=(-1.2, -2), b=(2, 3.1), n=(4, 9))
    mr3 = fe.mesh.Rectangle(n=7)

    mc1 = fe.mesh.Cube()
    mc2 = fe.mesh.Cube(a=(-1, -2, -0.5), b=(2, 3.1, 1), n=(4, 9, 5))
    mc3 = fe.mesh.Cube(n=7)


if __name__ == "__main__":
    test_mesh()
