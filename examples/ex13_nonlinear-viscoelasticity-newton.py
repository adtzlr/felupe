from tensortrax import function, jacobian
from tensortrax.math import trace
from tensortrax.math.linalg import inv
from tensortrax.math.special import dev, from_triu_1d, triu_1d
from matplotlib import pyplot as plt
import felupe as fem


@fem.constitution.isochoric_volumetric_split
def viscoelastic_two_potential(C, ζn, dt, μ, α, m, a, η0, ηinf, β, K):
    # extract old state variables
    Cvn = from_triu_1d(ζn, like=C)

    # first invariant of the right Cauchy-Green deformation tensor
    I1 = trace(C)

    def invariants(Cv, C):
        "Invariants of the elastic and viscous parts of the deformation."
        Ce = C @ inv(Cv)

        I1e = trace(Ce)
        I1v = trace(Cv)
        I2e = (I1e**2 - trace(Ce @ Ce)) / 2

        return I1e, I1v, I2e

    def evolution(Cv, Cvn, C):
        "Evolution equation for the internal (state) variable Cv."
        I1e, I1v, I2e = invariants(Cv, C)
        J2Neq = (I1e**2 / 3 - I2e) * sum(
            [3 ** (1 - a[r]) * m[r] * I1e ** (a[r] - 1) for r in [0, 1]]
        ) ** 2
        u = [3 ** (1 - a[r]) * m[r] * I1e ** (a[r] - 1) for r in [0, 1]]
        ηK = ηinf + (η0 - ηinf + K[0] * (I1v ** β[0] - 3 ** β[0])) / (
            1 + (K[1] * J2Neq + 1e-10) ** β[1]
        )
        G = sum(u) / ηK * dev(C @ inv(Cv)) @ Cv
        return G - (Cv - Cvn) / dt

    # update the state variables
    Cv = fem.newtonrhapson(
        x0=Cvn,
        fun=function(evolution, ntrax=C.ntrax),
        jac=jacobian(evolution, ntrax=C.ntrax),
        solve=fem.math.solve_2d,
        args=(Cvn, C.x),
        verbose=0,
        tol=1e-1,
    ).x

    I1e, I1v, I2e = invariants(Cv, C)

    # strain energy functions of the equilibrium and non-equilibrium parts
    p = [0, 1]
    ψEq = [3 ** (1 - α[r]) / (2 * α[r]) * μ[r] * (I1 ** α[r] - 3 ** α[r]) for r in p]
    ψNEq = [3 ** (1 - a[r]) / (2 * a[r]) * m[r] * (I1e ** a[r] - 3 ** a[r]) for r in p]

    return sum(ψEq) + sum(ψNEq), triu_1d(Cv)


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
boundaries, loadcase = fem.dof.uniaxial(field, clamped=False)

solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
solid.results.statevars[[0, 3, 5]] += 1.0

ldot = 0.05
lfinal = 3.0
tfinal = (lfinal - 1.0) / ldot
num_steps = tfinal / dt_vals
move = fem.math.linsteps([0.0, lfinal - 1.00, 0.358], num=int(num_steps) + 1)

step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)
job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate()
fig, ax = job.plot(
    xlabel=r"Displacement $u$ in mm $\longrightarrow$",
    ylabel=r"Normal Force $F$ in N $\longrightarrow$",
    marker="x",
)
ax.set_title("Uniaxial loading/unloading of a viscoelastic material (VHB 4910)")
img = solid.imshow("Principal Values of Cauchy Stress")
fig.tight_layout()
plt.show()
