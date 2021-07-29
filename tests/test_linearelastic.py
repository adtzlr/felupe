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

import felupe as fe


def test_linearelastic():

    mesh = fe.mesh.Cube(n=3)
    element = fe.element.Hexahedron()
    quadrature = fe.quadrature.GaussLegendre(order=1, dim=3)
    region = fe.Region(mesh, element, quadrature)
    displacement = fe.Field(region, dim=3)
    umat = fe.constitution.LinearElastic(E=1, nu=0.3)

    boundaries, dof0, dof1, u0ext = fe.doftools.uniaxial(displacement, clamped=False)

    def fun(strain):
        stress = umat.stress(strain)
        linearform = fe.IntegralForm(stress, displacement, region.dV, grad_v=True)
        return linearform.assemble().toarray()[:, 0]

    def jac(strain):
        stress = umat.stress(strain)
        elasti = umat.elasticity(strain, stress)
        bilinearform = fe.IntegralForm(
            elasti, displacement, region.dV, displacement, grad_v=True, grad_u=True
        )
        return bilinearform.assemble()

    res = fe.tools.newtonrhapson(
        fun,
        displacement,
        jac,
        pre=fe.tools.strain,
        solve=fe.tools.solve,
        update=fe.tools.update,
        check=fe.tools.check,
        kwargs_solve={"field": displacement, "ext": u0ext, "dof0": dof0, "dof1": dof1},
        kwargs_check={
            "tol_f": 1e-3,
            "tol_x": 1e-3,
            "dof0": dof0,
            "dof1": dof1,
            "verbose": 0,
        },
    )

    fe.utils.save(region, res.x, filename="result.vtk")


if __name__ == "__main__":
    test_linearelastic()
