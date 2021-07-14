# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:13:15 2021

@author: Andreas
"""

import numpy as np
import felupe as fe

from timeit import timeit

n = 151

m = fe.mesh.Rectangle(n=n)
e = fe.element.Quad1()
q = fe.quadrature.GaussLegendre(order=1, dim=2)

def pre(n):
    m = fe.mesh.Rectangle(n=n)
    e = fe.element.Quad1()
    q = fe.quadrature.GaussLegendre(order=1, dim=2)
    return m, e, q

print('|   DOF   | Assembly | Linear solve |')
print('| ------- | -------- | ------------ |')

dof = np.array([1e4, 2e4, 4e4, 1e5, 2e5, 4e5, 1e6, 2e6])

for n in np.round(np.sqrt(dof/2)).astype(int):
    
    m, e, q = pre(n)
    region = fe.Region(m, e, q)
    
    field = fe.Field(region)
    
    unit_load = np.ones_like(field.interpolate())
    laplace = fe.math.laplace(field)

    bilinearform = fe.IntegralForm(
        fun=laplace, v=field, dV=region.dV, u=field, grad_v=True, grad_u=True
    )

    linearform = fe.IntegralForm(fun=unit_load, v=field, dV=region.dV)
    
    A = bilinearform.assemble(parallel=False)
    b = linearform.assemble(parallel=False)
    
    def assembly(m, e, q):
        region = fe.Region(m, e, q)
        
        field = fe.Field(region)
        
        unit_load = np.ones_like(field.interpolate())
        laplace = fe.math.laplace(field)

        bilinearform = fe.IntegralForm(
            fun=laplace, v=field, dV=region.dV, u=field, grad_v=True, grad_u=True
        )

        linearform = fe.IntegralForm(fun=unit_load, v=field, dV=region.dV)
        
        A = bilinearform.assemble(parallel=False)
        b = linearform.assemble(parallel=False)
        return A, b
    
    f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)
    
    boundaries = {}
    boundaries["left"  ] = fe.Boundary(field, fx=f0)
    boundaries["right" ] = fe.Boundary(field, fx=f1)
    boundaries["bottom"] = fe.Boundary(field, fy=f0)
    boundaries["top"   ] = fe.Boundary(field, fy=f1)
    
    dof0, dof1 = fe.doftools.partition(field, boundaries)
    
    def solver(field, A, dof1, dof0, b):
        system = fe.solve.partition(field, A, dof1, dof0, -b)
        return fe.solve.solve(*system)
    
    time_assembly = timeit(lambda: assembly(m, e, q), number=1)
    time_solve = timeit(lambda: solver(field, A, dof1, dof0, b), number=1)

    print('| {:7d} | {:5.1f} s  | {:7.1f} s    |'.format(b.shape[0], time_assembly, time_solve))
    
    del m
    del e
    del q
    del region
    del field
    del bilinearform
    del linearform
    del unit_load
    del laplace
    del A
    del b
    del f0
    del f1
    del boundaries
    del dof0
    del dof1