# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:41:18 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

me = fe.mesh.Rectangle(a=(0, 0.0), b=(1, 2), n=41)
qu = fe.quadrature.Linear(2)

el = fe.element.Quad1()
re = fe.Region(me, el, qu)
dV = re.dV

u = fe.Field(re, 2)
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

b = fe.doftools.symmetry(u, (0, 1, 0))
b = {}
b["left"] = fe.Boundary(u, fx=f0)
b["right"] = fe.Boundary(u, fx=f1, skip=(1, 0))
b["move"] = fe.Boundary(u, fx=f1, skip=(0, 1), value=-0.4)

mat = fe.constitution.NeoHooke(1, 2)

dof0, dof1, _ = fe.doftools.partition(u, b)

for iteration in range(16):
    r = (
        R
        + u.interpolate()[
            [
                1,
            ]
        ]
    )
    F = u.grad() + fe.math.identity(u.grad())
    F = np.pad(F, ((0, 1), (0, 1), (0, 0), (0, 0)))
    F[2, 2] = r / R

    P = mat.f_u(F)
    A = mat.A_uu(F)

    r = (
        fe.IntegralForm(P[:2, :2], u, dV, grad_v=True)
        .assemble(parallel=False)
        .toarray()[:, 0]
    )
    K = fe.IntegralForm(A[:2, :2, :2, :2], u, dV, u, True, True).assemble(
        parallel=False
    )

    u0ext = fe.doftools.apply(u, b, dof0)
    system = fe.solve.partition(u, K, dof1, dof0, r)
    du = fe.solve.solve(*system, u0ext).reshape(*u.values.shape)

    norm = np.linalg.norm(du)

    if norm < 1e-8 or np.isnan(norm):
        break
    else:
        print(iteration, norm)
        u += du

fe.utils.save(re, u, r, K, F[:2, :2])
