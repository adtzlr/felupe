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
from felupe.helpers import (identity, dya, det, cof, inv, ddot,
                            transpose, dot, eigvals, cdya_il, dya, cdya_ik)

tol = 1e-8
move = 0.5

element = fe.element.Hex1()
mesh = fe.mesh.Cube(n=11) 
quadrature = fe.quadrature.Linear(dim=3)
domain = fe.Domain(mesh=mesh, 
                   element=element, 
                   quadrature=quadrature) 

# u at nodes
u = domain.zeros()

# boundaries
f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

symx = fe.Boundary(domain.dof, mesh, "sym-x", skip=(0, 1, 1), fx=f0)
symy = fe.Boundary(domain.dof, mesh, "sym-y", skip=(1, 0, 1), fy=f0)
symz = fe.Boundary(domain.dof, mesh, "sym-z", skip=(1, 1, 0), fz=f0)
movz = fe.Boundary(domain.dof, mesh, "movez", skip=(1, 1, 0), fz=f1, value=move)
bounds = [symx, symy, symz, movz]

symx = fe.Boundary(domain.dof, mesh, "sym-x", skip=(0, 1, 1), fx=f0)
symy = fe.Boundary(domain.dof, mesh, "sym-y", skip=(1, 0, 1), fy=f0)
fixb = fe.Boundary(domain.dof, mesh, "sym-z", skip=(1, 1, 0), fz=f0)
fixt = fe.Boundary(domain.dof, mesh, "fix-t", skip=(0, 0, 1), fz=f1)
movt = fe.Boundary(domain.dof, mesh, "mov-t", skip=(1, 1, 0), fz=f1, value = move)
bounds = [symx, symy, fixb, fixt, movt]

# dofs to dismiss and to keep
dof0, dof1 = fe.doftools.partition(domain.dof, bounds)

# obtain external displacements for prescribed dofs
u0ext = fe.doftools.apply(u, domain.dof, bounds, dof0)

nrr_ = []
nr_ = []
nu_ = []

mu = 1.0 
bulk = 2.0

def P(F): 
    iFT = transpose(inv(F)) 
    J = det(F) 

    Pdev = mu * (F - ddot(F,F)/3*iFT) * J**(-2/3) 
    Pvol = bulk * (J - 1) * J * iFT 

    return Pdev + Pvol 

def A(F): 
    J = det(F) 
    iFT = transpose(inv(F)) 
    eye = identity(F) 

    A_dev = mu * (cdya_ik(eye, eye) 
                  - 2/3 * dya(F, iFT) 
                  - 2/3 * dya(iFT, F) 
                  + 2/9 * ddot(F, F) * dya(iFT, iFT) 
                  + 1/3 * ddot(F, F) * cdya_il(iFT, iFT) 
                 ) * J**(-2/3)
    
    A_vol = bulk * (J-1) * J * (dya(iFT,iFT) - cdya_il(iFT,iFT)) \
          + bulk * J**2 * dya(iFT,iFT) 

    return A_dev + A_vol 


for iteration in range(20):
    # deformation gradient at integration points
    F = identity(domain.grad(u)) + domain.grad(u)

    # residuals and tangent matrix
    r_aie = domain.integrate(P(F))
    r = domain.asmatrix(r_aie.copy())
    
    # reference force per dof
    rref = domain.asmatrix(abs(r_aie))

    K = domain.asmatrix(domain.integrate(A(F)))

    system = fe.solve.partition(u, r, K, dof1, dof0)
    du = fe.solve.solve(*system, u0ext)

    if np.any(np.isnan(du)):
        break
    else:
        rx = r.toarray().reshape(*mesh.nodes.shape)
        ry = rref.toarray().reshape(*mesh.nodes.shape)
        ry[ry == 0] = np.nan
        rxy = rx/ry
        rxy[mesh.elements_per_node == 1] = rx[mesh.elements_per_node == 1]
        norm_rr = max(rxy.ravel()[dof1])
        norm_r = np.linalg.norm(r.toarray()[dof1])
        
        norm_du = np.linalg.norm(du)
        nrr_.append(norm_rr)
        nr_.append(norm_r)
        nu_.append(norm_du)
        print(f"#{iteration+1:2d}: |f|={norm_r:1.3e} (|δu|={norm_du:1.3e})")

    u += du

    if norm_du < tol:
        break

# cauchy stress at integration points
s = dot(P(F), transpose(F)) / det(F)
sp = eigvals(s)

# shift stresses to nodes and average nodal values
cauchy = domain.tonodes(s, sym=True)
cauchyprinc = [domain.tonodes(sp_i, mode="scalar") for sp_i in sp]


import meshio

cells = {"hexahedron": mesh.connectivity}
mesh = meshio.Mesh(
    mesh.nodes,
    cells,
    # Optionally provide extra data on points, cells, etc.
    point_data={
        "Displacements": u,
        "CauchyStress": cauchy,
        "ReactionForce": r.todense().reshape(*mesh.nodes.shape),
        "MaxPrincipalCauchyStress": cauchyprinc[2],
        "IntPrincipalCauchyStress": cauchyprinc[1],
        "MinPrincipalCauchyStress": cauchyprinc[0],
    },
)
mesh.write("out.vtk")

import matplotlib.pyplot as plt

plt.semilogy(np.append(np.nan, nrr_[:]), "o", label="|r/r,ref|")
plt.semilogy(np.append(np.nan, nr_[:]), "o", label="|r|")
plt.semilogy(np.append(np.nan, nu_[:]), "o", label="|δu|")
plt.semilogy(np.logspace(1, -10, 10),'k--')
plt.xlim(1,8)
plt.xlabel("Iterations")
plt.ylabel("Norm of residuals")
plot.legend()
plt.savefig("convergence.png")
