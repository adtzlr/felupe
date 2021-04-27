# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:17:05 2021

@author: adutz
"""

import numpy as np

import felupe as fe
from felupe.helpers import det, identity

e = fe.element.Quad1()
m = fe.mesh.Rectangle(n=2) 
q = fe.quadrature.Linear(dim=2)
d = fe.Domain(e,m,q) 

u = d.zeros()

H = d.grad(u)
R = d.interpolate((m.nodes)[:,1])
r = d.interpolate((m.nodes+u)[:,1])
F = np.zeros((3,3,*H.shape[2:]))
F[:2,:2] = identity(H) + H
F[-1,-1] = r/R

NH = fe.constitution.NeoHooke(mu=1)
P = NH.P(F,p=NH.bulk*(det(F)-1),J=0)