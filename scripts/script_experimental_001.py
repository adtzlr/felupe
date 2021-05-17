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
a = 1
b = 5

H = 26
mesh = fe.mesh.CubeAdvanced(
    symmetry=(False, True, False), n=(16, 9, 12), L=95, B=220, H=H, dL=10, dB=10
)
# mesh = fe.mesh.Cube(a=(-13,0,-13),b=(13,26,13),n=5)

# mesh = fe.mesh.CylinderAdvanced(D=120, H=26, n=(16, 10, 10), dD=10)
mesh0 = fe.mesh.convert(mesh, order=0)

region = fe.Region(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3))
region0 = fe.Region(mesh0, fe.element.Hex0(), fe.quadrature.Linear(dim=3))

# u at nodes
u = fe.Field(region, 3)
p = fe.Field(region0, 1)
J = fe.Field(region0, 1, values=1)
fields = (u, p, J)

from felupe.math import det, transpose, inv, identity, cdya_ik, dya, cdya_il, ddot


def P(F, param):
    mu, bulk = param

    detF = det(F)
    iFT = transpose(inv(F))

    p = bulk * (detF - 1)

    Pdev = mu * (F - ddot(F, F) / 3 * iFT) * detF ** (-2 / 3)
    Pvol = p * detF * iFT

    return Pdev + Pvol


def A(F, param):
    mu, bulk = param

    detF = det(F)
    iFT = transpose(inv(F))
    eye = identity(F)

    A4_dev = (
        mu
        * (
            cdya_ik(eye, eye)
            - 2 / 3 * dya(F, iFT)
            - 2 / 3 * dya(iFT, F)
            + 2 / 9 * ddot(F, F) * dya(iFT, iFT)
            + 1 / 3 * ddot(F, F) * cdya_il(iFT, iFT)
        )
        * detF ** (-2 / 3)
    )

    p = bulk * (detF - 1)
    q = p + bulk * detF

    A4_vol = detF * (q * dya(iFT, iFT) - p * cdya_il(iFT, iFT))

    return A4_dev + A4_vol


nh = fe.constitution.NeoHooke(1.0, 5000.0)
mat = fe.constitution.GeneralizedMixedField(P, A, (1.0, 5000.0))

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

results1 = fe.utils.incsolve(
    fields, region, mat.f, mat.A, bounds, a * move, tol=tol, parallel=True
)

# u.values = (np.random.rand(*u.values.shape)-1) * 0.1
# F = u.grad() + identity(u.grad())
# pp = p.interpolate()
# JJ = J.interpolate()

# f1 = nh.f(F,pp,JJ)
# f2 = mat.f(F,pp,JJ)

# for i in range(len(f1)):
#     print(np.allclose(f1[i], f2[i]))

# A1 = nh.A(F,pp,JJ)
# A2 = mat.A(F,pp,JJ)

# for i in range(len(A1)):
#     print(np.allclose(A1[i], A2[i]))
