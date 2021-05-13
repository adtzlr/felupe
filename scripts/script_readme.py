# -*- coding: utf-8 -*-
"""
Created on Thu May 13 21:41:27 2021

@author: adutz
"""

import numpy as np
import felupe as fe

mesh = fe.mesh.Cube(n=9)
element = fe.element.Hex1()
quadrature = fe.quadrature.Linear(dim=3)

region = fe.Region(mesh, element, quadrature)

dV = region.dV
V = dV.sum()

displacement = fe.Field(region, dim=3)

u  = displacement.values
ui = displacement.interpolate()

dudX = displacement.grad()

# use math function for better readability
from felupe.math import grad, identity

dudX = grad(displacement)

F = identity(dudX) + dudX

umat = fe.constitution.NeoHooke(mu=1.0, bulk=2.0)

P = umat.f_u(F)
A = umat.A_uu(F)

f0 = lambda x: np.isclose(x, 0)
f1 = lambda x: np.isclose(x, 1)

boundaries = {}
boundaries["left"]  = fe.Boundary(displacement, fx=f0)
boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1,0,0))
boundaries["move"]  = fe.Boundary(displacement, fx=f1, skip=(0,1,1), value=0.5)

dof0, dof1, _ = fe.doftools.partition(displacement, boundaries)
u0ext = fe.doftools.apply(displacement, boundaries, dof0)

linearform = fe.IntegralForm(P, displacement, dV, grad_v=True)
bilinearform = fe.IntegralForm(A, displacement, dV, displacement, grad_v = True, grad_u=True)

r = linearform.assemble(parallel=True).toarray()[:,0]
K = bilinearform.assemble(parallel=True)

system = fe.solve.partition(displacement, K, dof1, dof0, r)
du = fe.solve.solve(*system, u0ext).reshape(*u.shape)
displacement += du

for iteration in range(8):
    dudX = grad(displacement)
    F = identity(dudX) + dudX
    P = umat.f_u(F)
    A = umat.A_uu(F)
    
    r = fe.IntegralForm(P, displacement, dV, grad_v=True).assemble().toarray()[:,0]
    K = fe.IntegralForm(A, displacement, dV, displacement, True, True).assemble()
    
    system = fe.solve.partition(displacement, K, dof1, dof0, r)
    du = fe.solve.solve(*system, u0ext).reshape(*u.shape)
    
    norm = np.linalg.norm(du)
    print(iteration, norm)
    displacement += du

    if norm < 1e-12:
        break
        
fe.utils.save(region, displacement, filename="result")