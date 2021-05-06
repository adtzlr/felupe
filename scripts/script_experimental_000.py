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

tol = 1e-6

move = np.linspace(0, 1, 3)
a = -4
b = 5

mesh = fe.mesh.ScaledCube(
    a=(-1, 0, -1), b=(1, 1, 1), n=(16, 16, 16), L=95, B=220, H=26, dL=10, dB=10
)

region = fe.Region(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3))
region0 = fe.Region(mesh, fe.element.Hex0(), fe.quadrature.Linear(dim=3))

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
bounds = fe.doftools.symmetry(u, (0, 1, 0))
bounds["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds["top"] = fe.Boundary(u, skip=(0, 0, 1), fz=f1)
bounds["move"] = fe.Boundary(u, skip=(1, 1, 0), fz=f1)

bounds2 = fe.doftools.symmetry(u, (0, 1, 0))
bounds2["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds2["top"] = fe.Boundary(u, skip=(1, 0, 1), fz=f1)
bounds2["fix"] = fe.Boundary(u, skip=(1, 1, 0), fz=f1, value=a * move[-1])
bounds2["move"] = fe.Boundary(u, skip=(0, 1, 1), fz=f1)

results1 = fe.utils.incsolve(fields, region, mat.f, mat.A, bounds, a * move, tol=tol)

results2 = fe.utils.incsolve(
    results1[-1].fields, region, mat.f, mat.A, bounds2, b * move, tol=tol
)

fe.utils.savehistory(region, [*results1, *results2])

# experimental reaction force calculation
force_move = results1[-1].r.reshape(-1,3)[bounds["move"].nodes].sum(axis=1)

force_z = np.array([res.r[bounds["move"].dof].sum() for res in results1])
force_x = np.array([res.r[bounds2["move"].dof].sum() for res in results2])

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

f_z = interp1d(a * move[: len(force_z)], force_z, kind="quadratic")
z = np.linspace(0, a * move[: len(force_z)][-1])
plt.plot(a * move[: len(force_z)], 2 * force_z, "o")
plt.plot(z, 2 * f_z(z), "C0--")

f_x = interp1d(b * move[: len(force_x)], force_x, kind="quadratic")
x = np.linspace(0, b * move[: len(force_x)][-1])
plt.figure()
plt.plot(b * move[: len(force_x)], 2 * force_x, "o")
plt.plot(x, 2 * f_x(x), "C0--")

print("c_Z0 = ", (np.diff(2 * f_z(z)) / np.diff(z))[0])
print("c_X0 = ", (np.diff(2 * f_x(x)) / np.diff(x))[0])

print("c_Z = ", (np.diff(2 * f_z(z)) / np.diff(z))[-1])
print("c_X = ", (np.diff(2 * f_x(x)) / np.diff(x))[-1])
print("V = ", region.volume().sum())
