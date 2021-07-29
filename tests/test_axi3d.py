# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______ 
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___ 
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___ 
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import felupe as fe


def test_axi_to_3d():

    mesh = fe.mesh.Rectangle(a=(0, 0.5), b=(0.2, 1.5), n=(9, 16))

    element = fe.element.Quad()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=2)

    region = fe.Region(mesh, element, quadrature)

    u = fe.FieldAxisymmetric(region)

    f0, f1 = lambda x: np.isclose(x, 0), lambda x: np.isclose(x, 0.2)

    bounds = {}
    bounds["left"] = fe.Boundary(u, skip=(0, 0), fx=f0)
    bounds["right"] = fe.Boundary(u, skip=(1, 0), fx=f1)
    bounds["move"] = fe.Boundary(u, skip=(0, 1), fx=f1, value=-0.05)

    dof0, dof1 = fe.doftools.partition(u, bounds)

    region = fe.Region(mesh, element, quadrature)
    dA = region.dV

    umat = fe.constitution.models.NeoHookeCompressible(mu=1, bulk=5)

    for iteration in range(10):

        H = fe.math.grad(u)
        F = fe.math.identity(H) + H

        P = umat.P(F)
        A = umat.A(F)

        r = fe.IntegralFormAxisymmetric(P, u, dA).assemble().toarray()
        K = fe.IntegralFormAxisymmetric(A, u, dA, u).assemble()

        u0ext = fe.doftools.apply(u, bounds, dof0)
        system = fe.solve.partition(u, K, dof1, dof0, r)
        du = fe.solve.solve(*system, u0ext)
        u += du

        norm_u = np.linalg.norm(du)

        if norm_u < 1e-5:
            break

    fe.utils.save(region, u, r, F=F, f=P, filename="result_2d.vtk")
    ReactionForceZ_axi = r[bounds["move"].dof].sum()

    mesh_3d, element_3d, quadrature_3d, region_3d, u_3d = fe.utils.axito3d(
        mesh, element, quadrature, region, u, n=11, phi=180
    )
    dV = region_3d.dV

    bnds = fe.doftools.symmetry(u_3d, (0, 0, 1))
    bnds["left"] = fe.Boundary(u_3d, skip=(0, 0, 0), fx=f0)
    bnds["right"] = fe.Boundary(u_3d, skip=(1, 1, 0), fx=f1)
    bnds["move_axial"] = fe.Boundary(u_3d, skip=(0, 1, 1), fx=f1, value=-0.05)
    bnds["move_lateral"] = fe.Boundary(u_3d, skip=(1, 0, 1), fx=f1, value=0.0)

    dof0_3d, dof1_3d = fe.doftools.partition(u_3d, bnds)

    for iteration in range(8):

        H = u_3d.grad()
        F = fe.math.identity(H) + H

        P = umat.P(F)
        A = umat.A(F)

        r_3d = fe.IntegralForm(P, u_3d, dV, grad_v=True).assemble().toarray()
        K_3d = fe.IntegralForm(A, u_3d, dV, u_3d, True, True).assemble()

        u0ext_3d = fe.doftools.apply(u_3d, bnds, dof0_3d)
        system_3d = fe.solve.partition(u_3d, K_3d, dof1_3d, dof0_3d, r_3d)
        du_3d = fe.solve.solve(*system_3d, u0ext_3d)
        u_3d += du_3d

        norm_u = np.linalg.norm(du_3d)

        if norm_u < 1e-5:
            break

    fe.utils.save(region_3d, u_3d, r_3d, F=F, f=P, filename="result.vtk")
    ReactionForceZ_3d = 2 * r_3d[bnds["move_axial"].dof].sum()

    assert np.isclose(ReactionForceZ_axi, ReactionForceZ_3d, rtol=0.02)


if __name__ == "__main__":
    test_axi_to_3d()
