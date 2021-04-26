# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 21:17:05 2021

@author: adutz
"""

import felupe as fe

e20 = fe.element.Quad0()
m2 = fe.mesh.Rectangle(n=2) 
q20 = fe.quadrature.Constant(dim=2)
d2 = fe.Domain(e20,m2,q20) 

u2 = d2.zeros()

e31 = fe.element.Hex1()
m3 = fe.mesh.Cube(n=2) 
q31 = fe.quadrature.Linear(dim=3)
d3 = fe.Domain(e31,m3,q31) 

u3 = d3.zeros()
H3 = d3.grad(u3)