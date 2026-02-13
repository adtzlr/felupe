r"""
Uniaxial loading/unloading of a viscoelastic material (VHB 4910)
----------------------------------------------------------------
This example shows how to implement a constitutive material model for
rubber viscoelastic materials using a strain energy density function coupled
with an ODE for an internal (state) variable [1]_. The ODE is discretized using a
backward-Euler scheme and the resulting nonlinear algebraic equations for the
internal variable are solved using Newton's method [2]_.
"""

# sphinx_gallery_thumbnail_number = -1
import tensortrax as tr
import tensortrax.math as tm

import felupe as fem


@fem.isochoric_volumetric_split
def viscoelastic_two_potential(C, ζn, dt, μ, α, m, a, η0, ηinf, β, K):
    # extract old state variables
    I = tm.base.eye(C)
    Cvn = tm.special.from_triu_1d(ζn, like=C) + I

    def invariants(Cv, C):
        "Invariants of the elastic and viscous parts of the deformation."
        Ce = C @ tm.linalg.inv(Cv)

        I1e = tm.trace(Ce)
        I1v = tm.trace(Cv)
        I2e = (I1e**2 - tm.trace(Ce @ Ce)) / 2

        return I1e, I1v, I2e

    def evolution(Cv, Cvn, C, ξ=1e-10):
        "Evolution equation for the internal (state) variable Cv."
        I1e, I1v, I2e = invariants(Cv, C)
        J2Neq = (I1e**2 / 3 - I2e) * sum(
            [3 ** (1 - a[r]) * m[r] * I1e ** (a[r] - 1) for r in [0, 1]]
        ) ** 2
        u = [3 ** (1 - a[r]) * m[r] * I1e ** (a[r] - 1) for r in [0, 1]]
        ηK = ηinf + (η0 - ηinf + K[0] * (I1v ** β[0] - 3 ** β[0])) / (
            1 + (K[1] * J2Neq + ξ) ** β[1]
        )
        G = sum(u) / ηK * tm.special.dev(C @ tm.linalg.inv(Cv)) @ Cv
        return G - (Cv - Cvn) / dt

    # update the state variables
    Cv = fem.newtonrhapson(
        x0=Cvn,
        fun=tr.function(evolution, ntrax=C.ntrax),
        jac=tr.jacobian(evolution, ntrax=C.ntrax),
        solve=fem.math.solve_2d,
        args=(Cvn, C.x),
        verbose=0,
    ).x

    # invariants of the (elastic and viscous) right Cauchy-Green deformation tensor
    I1 = tm.trace(C)
    I1e, I1v, I2e = invariants(Cv, C)

    # strain energy functions of the equilibrium and non-equilibrium parts
    p = [0, 1]
    ψEq = [3 ** (1 - α[r]) / (2 * α[r]) * μ[r] * (I1 ** α[r] - 3 ** α[r]) for r in p]
    ψNEq = [3 ** (1 - a[r]) / (2 * a[r]) * m[r] * (I1e ** a[r] - 3 ** a[r]) for r in p]

    return sum(ψEq) + sum(ψNEq), tm.special.triu_1d(Cv - I)


dt_vals = 4.0

umat = fem.Hyperelastic(
    viscoelastic_two_potential,
    nstatevars=6,
    dt=dt_vals,
    μ=[13.54, 1.08],
    α=[1.0, -2.474],
    m=[5.42, 20.78],
    a=[-10, 1.948],
    η0=7014.0,
    ηinf=0.1,
    β=[1.852, 0.26],
    K=[3507.0, 1.0],
    # parallel=True,
)

mesh = fem.Cube(n=2)
region = fem.RegionHexahedron(mesh)
field = fem.FieldContainer([fem.Field(region, dim=3)])
boundaries = fem.dof.uniaxial(field, clamped=False, return_loadcase=False)

solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

ldot = 0.05
lfinal = 3.0
tfinal = (lfinal - 1.0) / ldot
num_steps = tfinal / dt_vals
move = fem.math.linsteps([0.0, lfinal - 1.00, 0.358], num=int(num_steps) + 1)

step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate(tol=1e-2)

fig, ax = job.plot(
    xlabel=r"Displacement $u$ in mm $\longrightarrow$",
    ylabel=r"Normal Force $F$ in N $\longrightarrow$",
    marker="x",
)
ax.set_title("Uniaxial loading/unloading\n of a viscoelastic material (VHB 4910)")
solid.plot("Principal Values of Cauchy Stress").show()

# %%
# References
# ----------
# .. [1] A. Kumar and O. Lopez-Pamies, "On the two-potential constitutive modeling of
#    rubber viscoelastic materials", Comptes Rendus. Mécanique, vol. 344, no. 2. Cellule
#    MathDoc/Centre Mersenne, pp. 102–112, Jan. 26, 2016. doi:
#    `10.1016/j.crme.2015.11.004 <http://dx.doi.org/10.1016/j.crme.2015.11.004>`_.
# .. [2] B. Shrimali, K. Ghosh and O. Lopez-Pamies "The Nonlinear Viscoelastic Response of
#    Suspensions of Vacuous Bubbles in Rubber: I — Gaussian Rubber with Constant Viscosity",
#    Journal of Elasticity, vol. 153, pp. 479-508 (2023), Nov. 30, 2021. doi:
#    `10.1007/s10659-021-09868-y <https://doi.org/10.1007/s10659-021-09868-y>`_.
