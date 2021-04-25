# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:45:35 2021

@author: adutz
"""

import numpy as np

import felupe as fe
from felupe.helpers import (identity, dya, det, cof, 
                            transpose, dot, eigvals)

tol = 1e-10
move = -0.5

e = fe.element.Hex1()
m = fe.mesh.Cube(a=(0, 0, 0), b=(2, 2, 1), n=(11, 11, 6))
#m.nodes[:,:2] = 0.8*m.nodes[:,:2] + 0.2*m.nodes[:,:2] * (m.nodes[:,2]**7).reshape(-1,1)
q = fe.quadrature.Linear(dim=3)
c = fe.constitution.NeoHooke(mu=1, bulk=5000)

d = fe.Domain(e, m, q)

# undeformed element volumes
V = d.volume()

# u at nodes
u = d.zeros()
p = np.zeros(d.nelements)
J = np.ones(d.nelements)

# boundaries
f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

symx = fe.Boundary(d.dof, m, "sym-x", skip=(0, 1, 1), fx=f0)
symy = fe.Boundary(d.dof, m, "sym-y", skip=(1, 0, 1), fy=f0)
symz = fe.Boundary(d.dof, m, "sym-z", skip=(1, 1, 0), fz=f0)
movz = fe.Boundary(d.dof, m, "movez", skip=(1, 1, 0), fz=f1, value=move)
bounds = [symx, symy, symz, movz]

# symx = Boundary(d.dof, m, "sym-x", skip=(0, 1, 1), fx=f0)
# symy = Boundary(d.dof, m, "sym-y", skip=(1, 0, 1), fy=f0)
# fixb = Boundary(d.dof, m, "sym-z", skip=(1, 1, 0), fz=f0)
# fixt = Boundary(d.dof, m, "fix-t", skip=(0, 0, 1), fz=f1)
# movt = Boundary(d.dof, m, "mov-t", skip=(1, 1, 0), fz=f1, value = move)
# bounds = [symx, symy, fixb, fixt, movt]

# dofs to dismiss and to keep
dof0, dof1 = fe.doftools.partition(d.dof, bounds)

# obtain external displacements for prescribed dofs
u0ext = fe.doftools.apply(u, d.dof, bounds, dof0)

n_ = []

for iteration in range(16):
    # deformation gradient at integration points
    F = identity(d.grad(u)) + d.grad(u)

    # deformed element volumes
    v = d.volume(det(F))

    # additional integral over shape function
    H = d.integrate(cof(F)) / V

    # residuals and tangent matrix
    ru = d.asmatrix(d.integrate(c.P(F, p, J)))
    rp = d.asmatrix((v / V - J) * V * H * c.dUdJ(J))
    rJ = d.asmatrix((c.dUdJ(J) - p) * V * H)

    r = ru + rp + rJ
    Ku = d.integrate(c.A(F, p, J))
    K2 = c.d2UdJ2(J) * V * dya(H, H)
    
    G = d.integrate(c.A_vol(F, p, J))
    K3 = c.d2UdJ2(J) * (v/V - J) * G
    
    K = d.asmatrix(Ku + K2 + K3)

    system = fe.solve.partition(u, r, K, dof1, dof0)
    du = fe.solve.solve(*system, u0ext)
    dJ = np.einsum("aie,eai->e", H, du[m.connectivity])
    dp = dJ * c.d2UdJ2(J)

    if np.any(np.isnan(du)):
        break
    else:
        rref = np.linalg.norm(r[dof0].toarray()[:, 0])
        if rref == 0:
            norm_r = 1
        else:
            norm_r = np.linalg.norm(r[dof1].toarray()[:, 0]) / rref
        norm_du = np.linalg.norm(du)
        norm_dp = np.linalg.norm(dp)
        norm_dJ = np.linalg.norm(dJ)
        n_.append(norm_r * rref)
        print(
            f"#{iteration+1:2d}: |f|={norm_r:1.3e} (|δu|={norm_du:1.3e} |δp|={norm_dp:1.3e} |δJ|={norm_dJ:1.3e})"
        )

    u += du
    J += dJ
    p += dp

    if norm_r < tol:
        break

# cauchy stress at integration points
F = identity(d.grad(u)) + d.grad(u)

# cauchy stress at integration points
s = dot(c.P(F, p, J), transpose(F)) / det(F)
sp = eigvals(s)

# shift stresses to nodes and average nodal values
cauchy = d.tonodes(s, sym=True)
cauchyprinc = [d.tonodes(sp_i, mode="scalar") for sp_i in sp]


import meshio

cells = {"hexahedron": m.connectivity}
mesh = meshio.Mesh(
    m.nodes,
    cells,
    # Optionally provide extra data on points, cells, etc.
    point_data={
        "Displacements": u,
        "CauchyStress": cauchy,
        "ReactionForce": r.todense().reshape(*m.nodes.shape),
        "MaxPrincipalCauchyStress": cauchyprinc[2],
        "IntPrincipalCauchyStress": cauchyprinc[1],
        "MinPrincipalCauchyStress": cauchyprinc[0],
    },
    # Each item in cell data must match the cells array
    cell_data={
        "Pressure": [
            p,
        ],
        "Volume-Ratio": [
            J,
        ],
    },
)
mesh.write("out.vtk")

# FZ_move = np.sum(r[movt.dof])
import matplotlib.pyplot as plt

plt.semilogy(n_[1:], "o")
plt.semilogy(np.logspace(-1, -10, 10))
