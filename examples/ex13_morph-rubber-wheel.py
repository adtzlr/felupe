r"""
Rotating Rubber Wheel
---------------------
This example contains a simulation of a rotating rubber wheel in plane strain with the
`MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ material model formulation
[1]_. While the rotation is increased, a constant vertical compression is applied to the
rubber wheel by a frictionless contact on the bottom. The vertical reaction force is
then carried out for the rotation angles. The MORPH material model is implemented as a
first Piola-Kirchhoff stress-based formulation with automatic differentiation. The
Tresca invariant of the distortional part of the right Cauchy-Green deformation tensor
is used as internal state variable, see Eq. :eq:`morph-state`.

..  warning::
    While the `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_-material
    formulation captures the Mullins effect and quasi-static hysteresis effects of
    rubber mixtures very nicely, it has been observed to be unstable for medium- to
    highly-distorted states of deformation.

..  math::
    :label: morph-state
    
    \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}
    
    I_3 &= \det (\boldsymbol{C})
    
    \hat{\boldsymbol{C}} &= I_3^{-1/3} \boldsymbol{C}
    
    \hat{\lambda}^2_\alpha &= \text{eigvals}(\hat{\boldsymbol{C}})
    
    \hat{C}_T &= \max \left( \hat{\lambda}^2_\alpha - \hat{\lambda}^2_\beta \right)
    
    \hat{C}_T^S &= \max \left( \hat{C}_T, \hat{C}_{T,n}^S \right)

A sigmoid-function is used inside the deformation-dependent variables :math:`\alpha`,
:math:`\beta` and :math:`\gamma`, see Eq. :eq:`morph-sigmoid`.

..  math::
    :label: morph-sigmoid
    
    f(x) &= \frac{1}{\sqrt{1 + x^2}}
    
    \alpha &= p_1 + p_2 \ f(p_3\ C_T^S)
    
    \beta &= p_4\ f(p_3\ C_T^S)
    
    \gamma &= p_5\ C_T^S\ \left( 1 - f\left(\frac{C_T^S}{p_6}\right) \right)

The rate of deformation is described by the Lagrangian tensor and its Tresca-invariant,
see Eq. :eq:`morph-rate-of-deformation`.

..  note::
    It is important to evaluate the incremental right Cauchy-Green tensor by the
    difference of the final and the previous state of deformation, not by its variation
    with respect to the deformation gradient tensor.

..  math::
    :label: morph-rate-of-deformation
    
    \hat{\boldsymbol{L}} &= \text{sym}\left( 
            \text{dev}(\boldsymbol{C}^{-1} \Delta\boldsymbol{C}) 
        \right) \hat{\boldsymbol{C}}
    
    \lambda_{\hat{\boldsymbol{L}}, \alpha} &= \text{eigvals}(\hat{\boldsymbol{L}})
    
    \hat{L}_T &= \max \left(
        \lambda_{\hat{\boldsymbol{L}}, \alpha} - \lambda_{\hat{\boldsymbol{L}}, \beta}
    \right)
    
    \Delta\boldsymbol{C} &= \boldsymbol{C} - \boldsymbol{C}_n

The additional stresses evolve between the limiting stresses, see Eq.
:eq:`morph-stresses`. The additional deviatoric-enforcement terms [1]_ are neglected in
this example.

..  math::
    :label: morph-stresses
    
    \boldsymbol{S}_L &= \left(
        \gamma \exp \left(p_7 \frac{\hat{\boldsymbol{L}}}{\hat{L}_T}
            \frac{\hat{C}_T}{\hat{C}_T^S} \right) +
            p8 \frac{\hat{\boldsymbol{L}}}{\hat{L}_T}
    \right) \boldsymbol{C}^{-1}
    
    \boldsymbol{S}_A &= \frac{
        \boldsymbol{S}_{A,n} + \beta\ \hat{L}_T\ \boldsymbol{S}_L
    }{1 + \beta\ \hat{L}_T}
    
    \boldsymbol{S} &= 2 \alpha\ \text{dev}( \hat{\boldsymbol{C}} )
        \boldsymbol{C}^{-1} + \text{dev}\left( \boldsymbol{S}_A\ \boldsymbol{C} \right)
        \boldsymbol{C}^{-1}

..  note::
    Only the upper-triangle entries of the symmetric stress-tensor state
    variables are stored in the solid body. Hence, it is necessary to extract such
    variables with :func:`tm.special.from_triu_1d` and export them as
    :func:`tm.special.triu_1d`.
"""
# sphinx_gallery_thumbnail_number = -1
import numpy as np
import felupe as fem
import tensortrax.math as tm


def morph(F, statevars_old, p):
    "MORPH material model formulation."

    # right Cauchy-Green deformation tensor
    C = F.T @ F

    # extract old state variables
    CTSn = tm.array(statevars_old[0], like=C[0, 0])
    Cn = tm.special.from_triu_1d(statevars_old[1:7], like=C)
    SAn = tm.special.from_triu_1d(statevars_old[7:], like=C)

    # distortional part of right Cauchy-Green deformation tensor
    I3 = tm.linalg.det(C)
    CG = C * I3 ** (-1 / 3)

    # inverse of and incremental right Cauchy-Green deformation tensor
    invC = tm.linalg.inv(C)
    dC = C - Cn

    # eigenvalues of right Cauchy-Green deformation tensor (sorted in ascending order)
    λCG = tm.linalg.eigvalsh(CG)

    # Tresca invariant of distortional part of right Cauchy-Green deformation tensor
    CTG = λCG[-1] - λCG[0]

    # maximum Tresca invariant in load history
    CTS = tm.maximum(CTG, CTSn)

    def f(x):
        "Algebraic sigmoid function."
        return 1 / tm.sqrt(1 + x**2)

    # material parameters
    α = p[0] + p[1] * f(p[2] * CTS)
    β = p[3] * f(p[2] * CTS)
    γ = p[4] * CTS * (1 - f(CTS / p[5]))

    LG = tm.special.sym(tm.special.dev(invC @ dC)) @ CG
    λLG = tm.linalg.eigvalsh(LG)
    LTG = λLG[-1] - λLG[0]
    LG_LTG = tm.if_else(LTG > 0, LG / LTG, LG)

    # limiting stresses "L" and additional stresses "A"
    SL = (γ * tm.linalg.expm(p[6] * LG_LTG * CTG / CTS) + p[7] * LG_LTG) @ invC
    SA = (SAn + β * LTG * SL) / (1 + β * LTG)

    # second Piola-Kirchhoff stress tensor
    S = 2 * α * tm.special.dev(CG) @ invC + tm.special.dev(SA @ C) @ invC

    try:  # update state variables
        statevars_new = tm.stack([CTS, *tm.special.triu_1d(C), *tm.special.triu_1d(SA)])
    except:
        # not possible (and not necessary) during AD-based hessian evaluation
        statevars_new = statevars_old

    return F @ S, statevars_new


umat = fem.MaterialAD(
    morph,
    p=[0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244],
    nstatevars=13,
    # parallel=True,
)

# %%
# The force-stress curves are shown for uniaxial incompressible tension cycles.
ux = fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1, 2.5], num=(10, 10, 20, 20, 30, 30, 30))
ax = umat.plot(
    ux=ux,
    bx=None,
    ps=None,
    incompressible=True,
)

# %%
# A mesh is created for the wheel with :math:`r=0.4` and :math:`R=1`.
mesh = fem.mesh.Line(a=0.4, n=6).revolve(37, phi=360)
mesh.update(points=np.vstack([mesh.points, [0, -1.1]]))
x, y = mesh.points.T
mesh.plot().show()

# %%
# A quad-region and a plane-strain displacement field are created. Mesh-points at
# :math:`r` are added to the ``move``-boundary condition. The displacements due to the
# rotation of the wheel are evaluated for each rotation angle. The center-point of
# the bottom-edge is moved vertically upwards by ``0.2`` to enforce a vertical reaction
# force in the rubber wheel.
#
# .. note::
#    Try to simulate more rotation angles and obtain the vertical reaction force of the
#    wheel, e.g. run two full rotations of the wheel with
#    ``angles_deg = fem.math.linsteps([0, 360, 720], num=[18, 18])``.
region = fem.RegionQuad(mesh)
field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

mask = np.isclose(np.sqrt(x**2 + y**2), 0.4)
boundaries = {
    "move": fem.dof.Boundary(field[0], mask=mask),
    "bottom-x": fem.dof.Boundary(field[0], fy=-1.1, value=0.0, skip=(0, 1)),
    "bottom-y": fem.dof.Boundary(field[0], fy=-1.1, value=0.2, skip=(1, 0)),
}

angles_deg = fem.math.linsteps([0, 120], num=6)
move = []
for phi in angles_deg:
    center = mesh.points[boundaries["move"].points]
    center_rotated = fem.mesh.rotate(
        points=center,
        cells=None,
        cell_type=None,
        angle_deg=phi,
        axis=0,
        center=[0, 0],
    )[0]
    move.append((center_rotated - center).ravel())

# %%
# A nearly-incompressible solid body is created for the rubber. At the bottom, a
# frictionless contact edge is created. Both items are added to a step, which is further
# evaluated in a job. The reaction forces are plotted for the successive rotation angles
# of the wheel.
solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
bottom = fem.MultiPointContact(
    field,
    points=np.arange(mesh.npoints)[np.isclose(np.sqrt(x**2 + y**2), 1.0)],
    centerpoint=-1,
    skip=(1, 0),
)
step = fem.Step(
    items=[solid, bottom], ramp={boundaries["move"]: move}, boundaries=boundaries
)

job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
job.evaluate(tol=1e-1)

fig, ax = job.plot(
    x=angles_deg.reshape(-1, 1),
    yaxis=1,
    yscale=-1,
    xlabel=r"Rotation Angle $\theta$ in deg $\longrightarrow$",
    ylabel=r"Force $|F_y|$ in N $\longrightarrow$",
)
ax.set_xticks(angles_deg[::6])

# %%
# The resulting max. principal values of the Cauchy stresses are shown for the final
# rotation angle.
solid.plot(
    "Principal Values of Cauchy Stress",
    plotter=bottom.plot(color="black", line_width=5, opacity=1.0),
    project=fem.topoints,
).show()

# %%
# References
# ----------
# .. [1] D. Besdo and J. Ihlemann, "A phenomenological constitutive model for rubberlike
#    materials and its numerical applications", International Journal of Plasticity,
#    vol. 19, no. 7. Elsevier BV, pp. 1019–1036, Jul. 2003. doi:
#    `10.1016/s0749-6419(02)00091-8 <https://doi.org/10.1016/s0749-6419(02)00091-8>`_.
