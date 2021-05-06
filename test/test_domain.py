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


def test_domain():
    element = fe.element.Hex1()
    mesh = fe.mesh.Cube(n=5)
    quadrature = fe.quadrature.Linear(dim=3)

    domain = fe.domain.Domain(element, mesh, quadrature)

    # elemental volumes and total domain volume
    Ve = domain.volume()
    Vd = Ve.sum()

    if not np.isclose(Vd, 1):
        raise ValueError("Error in Domain Volume calculation.")

    u = domain.zeros()
    x = domain.zeros((1, 7))
    y = domain.fill(value=-12.3, dim=5)
    z = domain.empty(dim=(6, 3, 7))

    if not x.shape == (domain.nnodes, 1, 7):
        raise ValueError("Error in domain.zeros() function.")

    return domain


if __name__ == "__main__":
    domain = test_domain()
