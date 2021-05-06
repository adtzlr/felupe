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


def test_quadrature():
    q03 = fe.quadrature.Constant()
    q01 = fe.quadrature.Constant(dim=1)
    q02 = fe.quadrature.Constant(dim=2)
    q03 = fe.quadrature.Constant(dim=3)

    q13 = fe.quadrature.Linear()
    q11 = fe.quadrature.Linear(dim=1)
    q12 = fe.quadrature.Linear(dim=2)
    q13 = fe.quadrature.Linear(dim=3)

    q23 = fe.quadrature.Quadratic()
    q21 = fe.quadrature.Quadratic(dim=1)
    q22 = fe.quadrature.Quadratic(dim=2)
    q23 = fe.quadrature.Quadratic(dim=3)

    q33 = fe.quadrature.Cubic()
    q31 = fe.quadrature.Cubic(dim=1)
    q32 = fe.quadrature.Cubic(dim=2)
    q33 = fe.quadrature.Cubic(dim=3)


if __name__ == "__main__":
    test_quadrature()
