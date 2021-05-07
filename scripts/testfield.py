# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:41:18 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

m = fe.mesh.Cube(n=21)
q = fe.quadrature.Linear(3)

e1 = fe.element.Hex1()
r1 = fe.region.Region(m, e1, q)

dV = r1.dV

e0 = fe.element.Hex0()
r0 = fe.region.Region(m, e0, q)

uf = fe.Field(r1, 3, values=10)
pf = fe.Field(r0, 1, values=-1)
Jf = fe.Field(r0, 1, values=1)
fields = (uf, pf, Jf)

def convert(fields):
    u,p,J = fields
    return (
        u.grad() + fe.math.identity(u.grad()), 
        p.interpolate(), 
        J.interpolate()
    )

F, p, J = convert(fields)

def f(F, p, J):
    return p * fe.math.cof(F)

def g(F, p, J):
    return p * fe.math.dya(F,F)

r = fe.IntegralForm(f(F,p,J), uf, dV,    grad_v=True).assemble()
K = fe.IntegralForm(g(F,p,J), uf, dV, uf, grad_v=True, grad_u=True).assemble()

y = fe.IntegralForm(f(F,p,J), pf, dV, uf, grad_u=True).assemble()
z = fe.IntegralForm(f(F,p,J), uf, dV, pf, grad_v=True).assemble()