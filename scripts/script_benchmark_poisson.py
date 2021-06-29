# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:13:15 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

from timeit import timeit

n = 151

m = fe.mesh.Rectangle(n=(n, n))
e = fe.element.Quad1()
q = fe.quadrature.Linear(2)

def pre(n):
    m = fe.mesh.Rectangle(n=(n, n))
    e = fe.element.Quad1()
    q = fe.quadrature.Linear(2)
    return m, e, q

print('|   DOF   | Assembly | Linear solve |')
print('| ------- | -------- | ------------ |')

dof = np.array([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6])

for n in np.round(np.sqrt(dof/2)).astype(int):
    
    m, e, q = pre(n)
    region = fe.Region(m, e, q)
    
    field = fe.Field(region, dim=2, values=0)
    dudX = field.grad()
    
    eye = fe.math.identity(dudX)
    poisson = fe.math.cdya_ik(eye, eye)
    rhs = np.ones_like(field.interpolate())
    
    bilinearform = fe.IntegralForm(poisson, field, region.dV, field, 
                                   grad_v=True, grad_u=True)
    
    linearform = fe.IntegralForm(rhs, field, region.dV)
    
    A = bilinearform.assemble(parallel=True)
    b = linearform.assemble(parallel=True)
    
    def assembly(m, e, q):
        region = fe.Region(m, e, q)
    
        field = fe.Field(region, dim=2, values=0)
        dudX = field.grad()
        
        eye = fe.math.identity(dudX)
        poisson = fe.math.cdya_ik(eye, eye)
        rhs = np.ones_like(field.interpolate())
        
        bilinearform = fe.IntegralForm(poisson, field, region.dV, field, 
                                   grad_v=True, grad_u=True)
    
        linearform = fe.IntegralForm(rhs, field, region.dV)
        A = bilinearform.assemble(parallel=True)
        b = linearform.assemble(parallel=True)
        return A, b
    
    f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)
    
    boundaries = {}
    boundaries["left"  ] = fe.Boundary(field, fx=f0)
    boundaries["right" ] = fe.Boundary(field, fx=f1)
    boundaries["bottom"] = fe.Boundary(field, fy=f0)
    boundaries["top"   ] = fe.Boundary(field, fy=f1)
    
    dof0, dof1 = fe.doftools.partition(field, boundaries)
    unknowns0_ext = fe.doftools.apply(field, boundaries, dof0)
    
    def solver(field, A, dof1, dof0, b, unknowns0_ext):
        system = fe.solve.partition(field, A, dof1, dof0, -b)
        return fe.solve.solve(*system, unknowns0_ext)
    
    time_assembly = timeit(lambda: assembly(m, e, q), number=1)
    time_solve = timeit(lambda: solver(field, A, dof1, dof0, b, unknowns0_ext), number=1)

    print('| {:7d} | {:5.1f} s  | {:7.1f} s    |'.format(b.shape[0], time_assembly, time_solve))
    
    del m
    del e
    del q
    del region
    del field
    del dudX
    del eye
    del poisson
    del rhs
    del bilinearform
    del linearform
    del A
    del b
    del f0
    del f1
    del boundaries
    del dof0
    del dof1
    del unknowns0_ext