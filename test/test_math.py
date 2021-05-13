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


def test_math():
    H = (np.random.rand(3, 3, 8, 200) - 0.5) / 10
    F = fe.math.identity(H) + H
    C = fe.math.dot(fe.math.transpose(F), F)
    A = np.random.rand(3, 3, 3, 3, 8, 200)

    fe.math.dot(C, C)
    fe.math.dot(C, A)
    fe.math.dot(A, C)
    fe.math.dot(A, A)

    fe.math.ddot(C, C)
    fe.math.ddot(C, A)
    fe.math.ddot(A, C)
    fe.math.ddot(A, A)

    fe.math.inv(C)
    fe.math.det(C)
    fe.math.cof(C)
    fe.math.dya(C, C)
    fe.math.cdya_ik(F, F)
    fe.math.cdya_il(F, F)
    fe.math.cdya(F, F)

    fe.math.tovoigt(C)
    fe.math.eigvals(C)


if __name__ == "__main__":
    test_math()
