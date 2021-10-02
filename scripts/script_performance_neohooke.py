# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:13:15 2021

@author: adtzlr
"""

import numpy as np
import felupe as fe

from timeit import timeit

n = 2

m = mesh = fe.mesh.Cube(n=n)
e = fe.element.Hexahedron()
q = fe.quadrature.GaussLegendre(order=1, dim=3)


def pre(n):
    m = fe.mesh.Cube(n=n)
    e = fe.element.Hexahedron()
    q = fe.quadrature.GaussLegendre(order=1, dim=3)
    return m, e, q


print("|   DOF   | Assembly | Linear solve |")
print("| ------- | -------- | ------------ |")

dof = np.array([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 5e5])
dof = np.array([2e5])

for n in np.round((dof / 3) ** (1 / 3)).astype(int):

    m, e, q = pre(n)
    region = fe.Region(m, e, q)

    field = fe.Field(region, dim=3)
    dudX = field.grad()
    F = fe.math.identity(dudX) + dudX

    nh = fe.constitution.NeoHooke(mu=1.0, bulk=3.0)

    bilinearform = fe.IntegralForm(
        nh.A(F), field, region.dV, field, grad_v=True, grad_u=True
    )

    linearform = fe.IntegralForm(nh.P(F), field, region.dV, grad_v=True)

    A = bilinearform.assemble(parallel=True)
    b = linearform.assemble(parallel=True)

    def assembly(m, e, q, nh):
        region = fe.Region(m, e, q)

        field = fe.Field(region, dim=3)
        dudX = field.grad()
        F = fe.math.identity(dudX) + dudX

        bilinearform = fe.IntegralForm(
            nh.A(F), field, region.dV, field, grad_v=True, grad_u=True
        )

        linearform = fe.IntegralForm(nh.P(F), field, region.dV, grad_v=True)

        A = bilinearform.assemble(parallel=True)
        b = linearform.assemble(parallel=True)
        return A, b

    f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 1)

    f1 = lambda x: np.isclose(x, 1)

    boundaries = fe.doftools.symmetry(field)
    boundaries["right"] = fe.Boundary(field, fx=f1, skip=(1, 0, 0))
    boundaries["move"] = fe.Boundary(field, fx=f1, skip=(0, 1, 1), value=-0.1)

    dof0, dof1 = fe.doftools.partition(field, boundaries)
    u0_ext = fe.doftools.apply(field, boundaries, dof0)

    def solver(field, A, dof1, dof0, b, u0_ext):
        system = fe.solve.partition(field, A, dof1, dof0, -b)
        return fe.solve.solve(*system, u0_ext)

    time_assembly = timeit(lambda: assembly(m, e, q, nh), number=1)
    time_solve = timeit(lambda: solver(field, A, dof1, dof0, b, u0_ext), number=1)

    print(
        "| {:7d} | {:5.1f} s  | {:7.1f} s    |".format(
            b.shape[0], time_assembly, time_solve
        )
    )

    del m
    del e
    del q
    del region
    del field
    del dudX
    del F
    del nh
    del bilinearform
    del linearform
    del A
    del b
    del f0
    del f1
    del boundaries
    del dof0
    del dof1
    del u0_ext
