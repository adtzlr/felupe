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

from felupe.math import sym, grad

H = 26
mesh = fe.mesh.ScaledCube(
    symmetry=(False, True, False), n=21, L=95, B=220, H=H, dL=10, dB=10
)

region = fe.Region(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3))

# u at nodes
u = fe.Field(region, 3)
fields = (u,)

# load constitutive material formulation
mat = fe.constitution.LinearElastic(E=210000, nu=0.48)

strain = sym(grad(u))
stress = mat.stress(strain)
elasticity = mat.elasticity(strain)

r = fe.IntegralForm(stress, v=u, dV=region.dV, grad_v=True).assemble().toarray()[:, 0]
K = fe.IntegralForm(
    elasticity, v=u, u=u, dV=region.dV, grad_v=True, grad_u=True
).assemble()

# boundaries
f0 = lambda x: np.isclose(x, -H / 2)
f1 = lambda x: np.isclose(x, H / 2)
bounds = fe.doftools.symmetry(u, (0, 1, 0))
bounds["bottom"] = fe.Boundary(u, skip=(0, 0, 0), fz=f0)
bounds["top"] = fe.Boundary(u, skip=(0, 0, 1), fz=f1)
bounds["move"] = fe.Boundary(u, skip=(1, 1, 0), fz=f1, value=-5)

dof0, dof1, _ = fe.doftools.partition(u, bounds)
u0ext = fe.doftools.apply(u, bounds, dof0)
system = fe.solve.partition(u, K, dof1, dof0)
du = fe.solve.solve(*system, u0ext)

u += du

fe.utils.save(region, u)
