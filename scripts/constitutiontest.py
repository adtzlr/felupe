# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:41:18 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

m = fe.mesh.Cube(n=15)
q = fe.quadrature.Linear(3)

e = fe.element.Hex1()
d = fe.Region(m, e, q)
dV = d.dV

e0 = fe.element.Hex0()
d0 = fe.Region(fe.mesh.convert(m, order=0), e0, q)

u = fe.Field(d, 3)
p = fe.Field(d0, 1)
J = fe.Field(d0, 1, values=1)

fields = (u, p, J)

F = u.grad() + fe.math.identity(u.grad())

nh = fe.constitution.NeoHooke(mu=1)
um = fe.constitution.GeneralizedMixedField(nh.f_u, nh.A_uu, None)

z = um.f(F, p.interpolate(), J.interpolate())
Z = um.A(F, p.interpolate(), J.interpolate())
