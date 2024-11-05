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

import felupe as fem


def test_visco_newton():
    import tensortrax as tr
    from tensortrax.math import trace
    from tensortrax.math.linalg import det, inv
    from tensortrax.math.special import from_triu_1d, triu_1d

    import felupe.constitution.tensortrax as mat

    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

    @mat.isochoric_volumetric_split
    def finite_strain_viscoelastic_newton(C, Cin, mu, eta, dtime):
        "Finite Strain Viscoelastic material formulation."

        def evolution(Ci, Cin, C, mu, eta, dtime):
            "Viscoelastic evolution equation."
            return mu / eta * C - (Ci - Cin) / dtime

        # update of state variables by evolution equation
        Cin = from_triu_1d(Cin, like=C)
        Ci = fem.newtonrhapson(
            x0=Cin,
            fun=tr.function(evolution, ntrax=C.ntrax),
            jac=tr.jacobian(evolution, ntrax=C.ntrax),
            solve=fem.math.solve_2d,
            args=(Cin, C.x, mu, eta, dtime),
            verbose=0,
        ).x
        Ci *= det(Ci) ** (-1 / 3)

        # first invariant of elastic part of right Cauchy-Green deformation tensor
        I1 = trace(C @ inv(Ci))

        # strain energy function and state variable
        return mu / 2 * (I1 - 3), triu_1d(Ci)

    umat = mat.Hyperelastic(
        finite_strain_viscoelastic_newton, mu=1.0, eta=1.0, dtime=0.1, nstatevars=6
    )
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

    move = fem.math.linsteps([0, 0.3], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"]).evaluate()
    assert np.all([norm[-1] < 1e-7 for norm in job.fnorms])


def test_visco_newton_jax():
    import jax
    from jax.numpy import trace
    from jax.numpy.linalg import det, inv
    from jax.scipy.optimize import minimize

    import felupe.constitution.jax as mat

    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

    @mat.isochoric_volumetric_split
    def finite_strain_viscoelastic_newton(C, statevars, mu, eta, dtime):
        "Finite Strain Viscoelastic material formulation."

        def evolution(Ci, Cin, C, mu, eta, dtime):
            "Viscoelastic evolution equation."
            Ci = Ci.reshape(3, 3)
            residuals = mu / eta * C - (Ci - Cin) / dtime
            return residuals.ravel() @ residuals.ravel()

        # update of state variables by evolution equation
        Cin = statevars[:9].reshape(3, 3)
        Ci = minimize(
            evolution,
            x0=Cin.ravel(),
            args=(Cin, jax.lax.stop_gradient(C), mu, eta, dtime),
            method="BFGS",
        ).x.reshape(3, 3)
        Ci *= det(Ci) ** (-1 / 3)

        # first invariant of elastic part of right Cauchy-Green deformation tensor
        I1 = trace(C @ inv(Ci))

        # strain energy function and state variable
        statevars_new = Ci.ravel()
        return mu / 2 * (I1 - 3), statevars_new

    umat = mat.Hyperelastic(
        finite_strain_viscoelastic_newton,
        mu=1.0,
        eta=1.0,
        dtime=0.1,
        nstatevars=9,
        jit=True,
    )
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

    move = fem.math.linsteps([0, 0.3], num=3)
    step = fem.Step(
        items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
    )
    job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
    job.evaluate(tol=1e-3)

    assert np.all([fnorm[-1] < 1e-3 for fnorm in job.fnorms])


if __name__ == "__main__":
    test_visco_newton()
    test_visco_newton_jax()
