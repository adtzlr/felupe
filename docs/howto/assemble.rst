Assemble Vectors and Matrices
-----------------------------

Integral (weak) forms are transformed into sparse vectors and matrices by calling the
assemble-method of an :class:`integral form<felupe.IntegralForm>`. The
:class:`Neo-Hookean <felupe.NeoHooke>` material model formulation is used to evaluate
both the variation and linerization of its strain energy density function.

..  plot::
    :context:

    import felupe as fem

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    umat = fem.NeoHooke(mu=1, bulk=2)
    field = fem.Field(region, dim=3).as_container()

    F = field.extract(grad=True, sym=False, add_identity=True)  # deformation-gradient
    P = umat.gradient(F)[:1]  # list with first Piola-Kirchhoff stress tensor
    A = umat.hessian(F)  # list with fourth-order elasticity tensor

The code for the variation of the total-potential energy, as given in Eq.
:eq:`total-potential-energy-variation`, is very close to the analytical expression.

..  math::
    :label: total-potential-energy-variation

    \delta \Pi = \int_V \boldsymbol{P} : \frac{
        \partial \delta \boldsymbol{u}
    }{\partial \boldsymbol{X}} \ dV

..  plot::
    :context:

    δΠ = fem.IntegralForm(  # variation of total potential energy
        fun=P,
        v=field,  # container for the test field
        dV=region.dV,  # differential volume
        grad_v=[True],  # use gradient of test field
    )
    vector = δΠ.assemble()

For the linearization, see Eq. eq:`total-potential-energy-linearization`.

..  math::
    :label: total-potential-energy-linearization

    \Delta \delta \Pi = \int_V \frac{
        \partial \delta \boldsymbol{u}
    }{\partial \boldsymbol{X}} : \mathbb{A} : \frac{
        \partial \Delta \boldsymbol{u}
    }{\partial \boldsymbol{X}} \ dV

..  plot::
    :context:

    ΔδΠ = fem.IntegralForm(  # linearization of total potential energy
        fun=A,
        v=field,  # container for the test field
        u=field,  # container for the trial field
        dV=region.dV,  # differential volume
        grad_v=[True],  # use gradient of test field
        grad_u=[True],  # use gradient of trial field
    )
    matrix = ΔδΠ.assemble()
