# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 08:45:35 2021

@author: adutz
"""

import numpy as np

import felupe as fe
from felupe.helpers import (identity, dya, det, cof, 
                            transpose, dot, eigvals)

tol = 1e-5
move = -0.1

m = fe.mesh.Cube(n=4)
c = fe.constitution.NeoHooke(mu=1, bulk=5000)

e1 = fe.element.Hex1()
e0 = fe.element.Hex0()
q1 = fe.quadrature.Linear(dim=3)
q0 = fe.quadrature.Constant(dim=3)

du = fe.Domain(e1, m, q1)
dp = fe.Domain(e1, m, q0)
dJ = fe.Domain(e1, m, q0)
V = du.volume()

uh = du.zeros(3)
ph = dp.zeros(1)
Jh = dJ.ones(1)

F = identity(du.grad(uh)) + du.grad(uh)
p = dp.interpolate(ph)
J = dp.interpolate(Jh)