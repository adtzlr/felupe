# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:41:18 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

m = fe.mesh.Rectangle(a=(0, 0), b=(1, 1), n=8)
q = fe.quadrature.Linear(2)

e = fe.element.Quad1()
d = fe.Region(m, e, q)
dV = d.dV

e0 = fe.element.Quad0()
d0 = fe.Region(fe.mesh.convert(m, order=0), e0, q)

u = fe.Field(d, 2)
R = fe.Field(
    d,
    1,
    values=m.nodes[
        :,
        [
            1,
        ],
    ],
).interpolate()
p = fe.Field(d0, 1)
J = fe.Field(d0, 1)

fields = (u, p, J)

f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

b = fe.doftools.symmetry(u, (0, 1, 0))
b["left"] = fe.Boundary(u, fx=f0)
b["right"] = fe.Boundary(u, fx=f1, skip=(1, 0))
b["move"] = fe.Boundary(u, fx=f1, skip=(0, 1), value=0.5)

mat = fe.constitution.NeoHooke(1, 5000)

dof0, dof1, unstack = fe.doftools.partition(fields, b)

for iteration in range(4):
    ur = u.interpolate()[
        [
            1,
        ]
    ]
    F = u.grad() + fe.math.identity(u.grad())
    F = np.pad(F, ((0, 1), (0, 1), (0, 0), (0, 0)))
    F[2, 2] = (R + ur) / R

    P = mat.f(F, p.interpolate(), J.interpolate())
    A = mat.A(F, p.interpolate(), J.interpolate())
    P[0] = P[0][:2, :2]
    A[0] = A[0][:2, :2, :2, :2]
    A[1] = A[1][:2, :2]
    A[2] = A[1][:2, :2]

    r = fe.IntegralFormMixed(P, fields, dV).assemble().toarray()[:, 0]
    K = fe.IntegralFormMixed(A, fields, dV).assemble()

    u0ext = fe.doftools.apply(fields[0], b, dof0)
    system = fe.solve.partition(fields, K, dof1, dof0, r)
    dfields = np.split(fe.solve.solve(*system, u0ext), unstack)

    norm = np.linalg.norm(dfields[0])
    print(iteration, norm)

    if norm < 1e-8:
        break
    else:
        for field, dfield in zip(fields, dfields):
            field += dfield

fe.utils.save(d, fields, r, K, unstack=unstack)
