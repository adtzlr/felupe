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

tol = 1e-8
move = -np.arange(0.1, 0.8, 0.05)

mesh    = fe.mesh.Cube(n=8)
domain  = fe.Domain(mesh, fe.element.Hex1(), fe.quadrature.Linear(dim=3)) 
domain0 = fe.Domain(mesh, fe.element.Hex0(), fe.quadrature.Linear(dim=3)) 

# u at nodes
uh = domain.zeros()
ph = domain0.zeros(dim=1)
Jh = domain0.zeros(dim=1)

H = domain.grad(uh)
F = fe.helpers.identity(H) + H
p = domain0.interpolate(ph)
J = domain0.interpolate(Jh)

# load constitutive material formulation
NH = fe.constitution.NeoHooke(mu=1.0, bulk=5.0)

def fu(F, p, J):
    return F + p

def fp(F, p, J):
    return fe.helpers.det(F)-J

def fJ(F, p, J):
    return 5000 * (J - 1) - p

ru_aie = domain.integrate(fu(F,p,J))
rp_ae = fe.domain._integrate0(domain.h, fp(F,p,J), domain.Jw)
rJ_ae = fe.domain._integrate0(domain.h, fJ(F,p,J), domain.Jw)

#result = fe.utils.incsolve(u, domain, bounds, move, NH.P, NH.A)