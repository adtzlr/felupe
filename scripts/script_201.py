# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:45:35 2021

@author: adutz
"""

import numpy as np

import felupe as fe
from felupe.helpers import (identity, dya, det, cof, 
                            transpose, dot, eigvals)

tol = 5e-9
move = -0.1

e = fe.element.Quad1()
m = fe.mesh.Rectangle(a=(0, 0.15), b=(1, 2), n=(10,20))
#m.nodes[:,:2] = 0.8*m.nodes[:,:2] + 0.2*m.nodes[:,:2] * (m.nodes[:,2]**7).reshape(-1,1)
q = fe.quadrature.Linear(dim=2)
c = fe.constitution.NeoHooke(mu=1)

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

left  = fe.Boundary(d.dof, m, fx=f0)
right = fe.Boundary(d.dof, m, skip=(1,0), fx=f1)
move  = fe.Boundary(d.dof, m, skip=(0,1), fx=f1, value=move)

bounds = [left, right, move]

# dofs to dismiss and to keep
dof0, dof1 = fe.doftools.partition(d.dof, bounds)

# obtain external displacements for prescribed dofs
u0ext = fe.doftools.apply(u, d.dof, bounds, dof0)

n_ = []

for iteration in range(16):
    # deformation gradient at integration points
    H = d.grad(u)
    R = d.interpolate((m.nodes)[:,1])
    r = d.interpolate((m.nodes+u)[:,1])
    F = np.zeros((3,3,*H.shape[2:]))
    F[:2,:2] = identity(H) + H
    F[-1,-1] = r/R
    F = identity(H) + H

    # deformed element volumes
    v = d.volume(det(F))

    # additional integral over shape function
    H = d.integrate(cof(F)[:2,][:,:2]) / V

    # residuals and tangent matrix
    P = c.P(F, p, J)[:2,][:,:2]
    ru = d.asmatrix(d.integrate(P))
    rp = d.asmatrix((v / V - J) * V * H * c.dUdJ(J))
    rJ = d.asmatrix((c.dUdJ(J) - p) * V * H)
    
    # reference force per dof
    aru = d.asmatrix(abs(d.integrate(P)))
    arp = d.asmatrix(abs((v / V - J) * V * H * c.dUdJ(J)))
    arJ = d.asmatrix(abs((c.dUdJ(J) - p) * V * H))
    ar = aru + arp + arJ

    r = ru + rp + rJ
    
    A = c.A(F, p, J)[:2][:,:2][:,:,:2][:,:,:,:2]
    Ku = d.integrate(A)
    K2 = c.d2UdJ2(J) * V * dya(H, H)
    
    #G = d.integrate(c.A_vol(F, p, J))
    #K3 = (c.d2UdJ2(J) * (v/V - J) + c.dUdJ(J) - p) * G
    
    K = d.asmatrix(Ku + K2)# + K3)

    system = fe.solve.partition(u, r, K, dof1, dof0)
    du = fe.solve.solve(*system, u0ext)
    dJ = np.einsum("aie,eai->e", H, du[m.connectivity])
    dp = dJ * c.d2UdJ2(J)

    if np.any(np.isnan(du)):
        break
    else:
        #norm_r = np.linalg.norm(r[dof1]/ar[dof1])
        norm_r = np.max(r[dof1]/ar[dof1])
        norm_du = np.linalg.norm(du)
        norm_dp = np.linalg.norm(dp)
        norm_dJ = np.linalg.norm(dJ)
        n_.append(norm_r)
        print(
            f"#{iteration+1:2d}: |f|={norm_r:1.1e} (|δu|={norm_du:1.1e} |δp|={norm_dp:1.1e} |δJ|={norm_dJ:1.1e})"
        )

    u += du
    J += dJ
    p += dp

    if norm_r < tol:
        break

# deformation gradient at integration points
#F = identity(d.grad(u)) + d.grad(u)

# cauchy stress at integration points
s = dot(c.P(F, p, J), transpose(F)) / det(F)
sp = eigvals(s[:2][:,:2])

# shift stresses to nodes and average nodal values
cauchy = d.tonodes(s[:2][:,:2], sym=True)
cauchyprinc = [d.tonodes(sp_i, mode="scalar") for sp_i in sp]


import meshio

cells = {"quad": m.connectivity}
mesh = meshio.Mesh(
    m.nodes,
    cells,
    # Optionally provide extra data on points, cells, etc.
    point_data={
        "Displacements": u,
        "CauchyStress": cauchy,
        "ReactionForce": r.todense().reshape(*m.nodes.shape),
        "MaxPrincipalCauchyStress": cauchyprinc[1],
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
