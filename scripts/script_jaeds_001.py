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
mesh = fe.mesh.CubeAdvanced(
    symmetry=(True, False, False), n=(31, 31, 13), L=200, B=100, H=H
)
mesh0 = fe.mesh.convert(mesh, order=0)

region  = fe.Region(mesh,  fe.element.Hex1(), fe.quadrature.Linear(dim=3))
region0 = fe.Region(mesh0, fe.element.Hex0(), fe.quadrature.Linear(dim=3))

u = fe.Field(region,  3)
p = fe.Field(region0, 1)
J = fe.Field(region0, 1, values=1)
fields = (u, p, J)

mat = fe.constitution.NeoHooke(mu=0.6, bulk=3000.0)

f0 = lambda x: np.isclose(x, -H / 2)
f1 = lambda x: np.isclose(x,  H / 2)

bounds = fe.doftools.symmetry(u, (1, 0, 0))
bounds["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds["top"]    = fe.Boundary(u, skip=(0, 0, 1), fz=f1)
bounds["move"]   = fe.Boundary(u, skip=(1, 1, 0), fz=f1)

bounds2 = fe.doftools.symmetry(u, (1, 0, 0))
bounds2["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds2["top"]    = fe.Boundary(u, skip=(0, 1, 1), fz=f1)
bounds2["fix"]    = fe.Boundary(u, skip=(1, 1, 0), fz=f1, value=a * move[-1])
bounds2["move"]   = fe.Boundary(u, skip=(1, 0, 1), fz=f1)

results1 = fe.utils.incsolve(fields, region, mat.f, mat.A, bounds, a * move, tol=tol)

results2 = fe.utils.incsolve(
    results1[-1].fields, region, mat.f, mat.A, bounds2, b * move, tol=tol
)

fe.utils.savehistory(region, [*results1, *results2], filename="script_001")

force_move  = fe.utils.force(results1, bounds["move"])
force_move2 = fe.utils.force(results2, bounds["move"])

xy1, xxyy1 = fe.utils.curve(a * move, 2 * force_move[:, 2])
plt.plot(*xy1, "o")
plt.plot(*xxyy1, "C0--")

xy2, xxyy2 = fe.utils.curve(b * move, 2 * force_move2[:, 1])
plt.figure()
plt.plot(*xy2, "o")
plt.plot(*xxyy2, "C0--")