r"""
Rotating Rubber Wheel
---------------------
This example contains a simulation of a rotating rubber wheel in plane strain with the
`MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_ material model formulation
[1]_. While the rotation is increased, a constant vertical compression is applied to the
rubber wheel by a frictionless contact on the bottom. The vertical reaction force is
then carried out for the rotation angles. The MORPH material model is implemented as a
second Piola-Kirchhoff stress-based formulation with automatic differentiation (JAX).
The Tresca invariant of the distortional part of the right Cauchy-Green deformation
tensor is used as internal state variable, see Eq. :eq:`morph-state`.

..  warning::
    While the `MORPH <https://doi.org/10.1016/s0749-6419(02)00091-8>`_-material
    formulation captures the Mullins effect and quasi-static hysteresis effects of
    rubber mixtures very nicely, it has been observed to be unstable for medium- to
    highly-distorted states of deformation. An alternative implementation by the method
    of `representative directions <https://nbn-resolving.org/urn:nbn:de:bsz:ch1-qucosa-114428>`_
    provides better stability but is computationally more costly [2]_, [3]_.

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
import felupe.constitution.jax as mat
from jax.scipy.linalg import expm
import jax.numpy as jnp


def morph(F, statevars, p):
    "MORPH material model formulation."

    # right Cauchy-Green deformation tensor
    C = F.T @ F

    # extract old state variables
    CTSn = statevars[0]
    from_triu = lambda C: C[jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])]
    Cn = from_triu(statevars[1:7])
    SAn = from_triu(statevars[7:13])

    # distortional part of right Cauchy-Green deformation tensor
    I3 = jnp.linalg.det(C)
    CG = C * I3 ** (-1 / 3)

    # inverse of and incremental right Cauchy-Green deformation tensor
    invC = jnp.linalg.inv(C)
    dC = C - Cn

    # eigenvalues of right Cauchy-Green deformation tensor (sorted in ascending order)
    ε = jnp.diag(jnp.array([1e-4, -1e-4, 0]))
    eigvalsh_ε = lambda C: jnp.linalg.eigvalsh(C + ε)
    λCG = eigvalsh_ε(CG)

    # Tresca invariant of distortional part of right Cauchy-Green deformation tensor
    CTG = λCG[-1] - λCG[0]

    # maximum Tresca invariant in load history
    CTS = jnp.maximum(CTG, CTSn)

    def sigmoid(x):
        "Algebraic sigmoid function."
        return 1 / jnp.sqrt(1 + x**2)

    # material parameters
    α = p[0] + p[1] * sigmoid(p[2] * CTS)
    β = p[3] * sigmoid(p[2] * CTS)
    γ = p[4] * CTS * (1 - sigmoid(CTS / p[5]))

    dev = lambda C: C - jnp.trace(C) / 3 * jnp.eye(3)
    sym = lambda C: (C + C.T) / 2

    LG = sym(dev(invC @ dC)) @ CG
    λLG = eigvalsh_ε(LG)
    LTG = λLG[-1] - λLG[0]

    # limiting stresses "L" and additional stresses "A"
    SL = (γ * expm(p[6] * LG / LTG * CTG / CTS) + p[7] * LG / LTG) @ invC
    SA = (SAn + β * LTG * SL) / (1 + β * LTG)

    # second Piola-Kirchhoff stress tensor
    S = 2 * α * dev(CG) @ invC + dev(SA @ C) @ invC

    # update the state variables
    i, j = jnp.triu_indices(3)
    to_triu = lambda C: C[i, j]
    statevars_new = jnp.concatenate([jnp.array([CTS]), to_triu(C), to_triu(SA)])

    return F @ S, statevars_new


umat = mat.Material(
    morph,
    p=[0.039, 0.371, 0.174, 2.41, 0.0094, 6.84, 5.65, 0.244],
    nstatevars=13,
)

# %%
# .. note::
#    The MORPH material model formulation is also available in FElupe, see
#    :class:`~felupe.constitution.jax.models.lagrange.morph`.
#
# The force-stress curves are shown for uniaxial incompressible tension cycles.
ux = fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1, 2.5], num=(5, 5, 10, 10, 15, 15, 15))
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

angles_deg = fem.math.linsteps([0, 180], num=9)
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
#
# .. [2] M. Freund, "Verallgemeinerung eindimensionaler Materialmodelle für die
#    Finite-Elemente-Methode", Dissertation, Technische Universität Chemnitz,
#    Chemnitz, 2013.
#
# .. [3] C. Miehe, S. Göktepe and F. Lulei, "A micro-macro approach to rubber-like
#    materials - Part I: the non-affine micro-sphere model of rubber elasticity",
#    Journal of the Mechanics and Physics of Solids, vol. 52, no. 11. Elsevier BV, pp.
#    2617–2660, Nov. 2004. doi:
#    `10.1016/j.jmps.2004.03.011 <https://doi.org/10.1016/j.jmps.2004.03.011>`_.
