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

import matplotlib.pyplot as plt

tol = 1e-6

move = np.linspace(0, 1, 5)
a = -5
b = 10

H = 25
mesh = fe.mesh.ScaledCube(
    symmetry=(True, False, False), n=(16, 61, 13), L=100, B=200, H=H, dL=0, dB=0
)
# mesh = fe.mesh.Cylinder(D=120, H=26, n=(16, 10), dD=10)
mesh0 = fe.mesh.convert(mesh, order=0)

region = fe.Region(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3))
region0 = fe.Region(mesh0, fe.element.Hex0(), fe.quadrature.Linear(dim=3))

# u at nodes
u = fe.Field(region, 3)
p = fe.Field(region0, 1)
J = fe.Field(region0, 1, values=1)
fields = (u, p, J)

# load constitutive material formulation
mat = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)

# boundaries
f0 = lambda x: np.isclose(x, -H / 2)
f1 = lambda x: np.isclose(x, H / 2)
bounds = fe.doftools.symmetry(u, (1, 0, 0))
bounds["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds["topx"] = fe.Boundary(u, skip=(0, 1, 1), fz=f1)
bounds["topy"] = fe.Boundary(u, skip=(1, 0, 1), fz=f1, value=0.1)
bounds["move"] = fe.Boundary(u, skip=(1, 1, 0), fz=f1)

results = fe.utils.incsolve(fields, region, mat.f, mat.A, bounds, a * move, tol=tol)

fe.utils.savehistory(region, results)

# reaction force calculation
force_move = fe.utils.reactionforce(results, bounds)

xy1, xxyy1 = fe.utils.curve(a * move, 2 * force_move[:, 2])
plt.plot(*xy1, "o")
plt.plot(*xxyy1, "C0--")

xy2, xxyy2 = fe.utils.curve(a * move, 2 * force_move[:, 1])
plt.figure()
plt.plot(*xy2, "o")
plt.plot(*xxyy2, "C0--")
