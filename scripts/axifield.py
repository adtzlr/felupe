# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:41:18 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

me = fe.mesh.Rectangle(a=(0, 1), b=(1, 2), n=2)
qu = fe.quadrature.Linear(2)

el = fe.element.Quad1()
re = fe.Region(me, el, qu)
dV = re.dV

u = fe.Field(re, 2)
v = fe.Field(re, 1)
w = fe.Field(re, 3)
R = fe.Field(
    re,
    1,
    values=me.nodes[
        :,
        [
            1,
        ],
    ],
).interpolate()

f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

#b = fe.doftools.symmetry(w, (1, 0, 0))
b = {}
b["left"] = fe.Boundary(w, fx=f0, fz=f0, skip=(0, 1, 1))
b["right"] = fe.Boundary(w, fx=f1, fz=f0, skip=(1, 0, 1))
b["move"] = fe.Boundary(w, fx=f1, fz=f0, skip=(0, 1, 1), value=0.2)

mat = fe.constitution.NeoHooke(1, 2)

dof0, dof1, _ = fe.doftools.partition(w, b)
dof0 = [0,3,4,6,7,9]
dof1 = [1,2,3,5,8,10,11]
u0ext = np.zeros_like(dof0, dtype=float)
u0ext[1] = 0.2
u0ext[3] = 0.2
#dof1 = np.append(dof1, re.mesh.ndof + np.arange(re.nnodes))

for iteration in range(2):
    ri = (R + u.interpolate()[[1,]])
    F2d = u.grad() + fe.math.identity(u.grad())
    F = np.pad(F2d, ((0, 1), (0, 1), (0, 0), (0, 0)))
    F[2, 2] = ri / R

    P = mat.f_u(F)
    A = mat.A_uu(F)

    r = (
        fe.IntegralForm(P[:2, :2], u, dV, None, True)
        .assemble()
        .toarray()[:, 0]
    ).reshape(-1,2)
    
    rh = fe.IntegralForm(P[2, 2]/R, v, dV).assemble().toarray()
    
    K = fe.IntegralForm(A[:2, :2, :2, :2], u, dV, u, True, True).assemble()
    
    K2 = fe.IntegralForm(A[2, 2, 2, 2]/R**2, v, dV, v).assemble()
    K3 = fe.IntegralForm(A[2, 2, :2, :2]/R, v, dV, u, False, True).assemble()
    K4 = fe.IntegralForm(A[:2, :2, 2, 2]/R, u, dV, v, True, False).assemble()
    
    
    from scipy.sparse import block_diag, bmat
    r = np.hstack((r,rh)).ravel()
    #K = block_diag((K,K2)).tocsr()
    K = bmat([[K, K4],[K3,K2]]).tocsr()

    #u0ext = fe.doftools.apply(w, b, dof0)
    system = list(fe.solve.partition(w, K, dof1, dof0, r))
    system[0] = np.append(system[0], np.zeros(re.nnodes))
    du = fe.solve.solve(*system, u0ext).reshape(u.values.shape[0], -1)

    norm = np.linalg.norm(du)

    if norm < 1e-8 or np.isnan(norm):
        break
    else:
        print(iteration, norm)
        u += du[:,:2]

fe.utils.save(re, u, None, None, None)
