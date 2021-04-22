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


def test_helpers():
    H = (np.random.rand(3, 3, 8, 200) - 0.5) / 10
    F = fe.helpers.identity(H) + H
    C = fe.helpers.dot(fe.helpers.transpose(F), F)
    A = np.random.rand(3, 3, 3, 3, 8, 200)

    fe.helpers.dot(C, C)
    fe.helpers.dot(C, A)
    fe.helpers.dot(A, C)
    fe.helpers.dot(A, A)

    fe.helpers.ddot(C, C)
    fe.helpers.ddot(C, A)
    fe.helpers.ddot(A, C)
    fe.helpers.ddot(A, A)

    fe.helpers.inv(C)
    fe.helpers.det(C)
    fe.helpers.cof(C)
    fe.helpers.dya(C, C)
    fe.helpers.cdya_ik(F, F)
    fe.helpers.cdya_il(F, F)
    fe.helpers.cdya(F, F)

    fe.helpers.tovoigt(C)
    fe.helpers.eigvals(C)


if __name__ == "__main__":
    test_helpers()
