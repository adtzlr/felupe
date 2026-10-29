Define Time-Dependent Materials
-------------------------------

A time-dependent material, e.g. a viscoelastic material, takes the time increment
:math:`\Delta t` as an input, like the deformation gradient. The time increment is not a
material parameter, it is given by the time of the load path, e.g. by the ramp of a
:class:`~felupe.Step`.

Time-Dependent Materials
************************

A material with a given function, i.e. :class:`~felupe.Hyperelastic`,
:class:`~felupe.MaterialAD`, :class:`~felupe.Material`, :class:`~felupe.MaterialStrain`
or the materials of the JAX backend, is time-dependent if its function has an argument
``dtime`` without a default value, which is not given as keyword-argument of the
material. Then, the attribute ``time_dependent`` is True and the time increment is given
by the keyword-argument ``dtime`` of the methods ``gradient()`` and ``hessian()``.

..  code-block:: python

    import numpy as np
    import felupe as fem

    umat = fem.Hyperelastic(
        fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
    )
    assert umat.time_dependent

    F = np.diag([1.2, 1.0, 1.0]).reshape(3, 3, 1, 1)
    statevars = np.zeros((6, 1, 1))
    P, statevars_new = umat.gradient([F, statevars], dtime=0.1)

The built-in time-dependent materials are :func:`~felupe.finite_strain_viscoelastic`
for :class:`~felupe.Hyperelastic` and :func:`~felupe.linear_elastic_viscoelastic` for
:class:`~felupe.MaterialStrain`. Their initial state variables are zero. The materials
which wrap other materials, e.g. :class:`~felupe.NearlyIncompressible`,
:class:`~felupe.ThreeFieldVariation` or a composite material ``umat & other_umat``, pass
the time increment to their time-dependent materials.

..  code-block:: python

    umat = fem.Hyperelastic(
        fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
    ) & fem.Volumetric(bulk=5000)
    assert umat.time_dependent

..  note::

    A time increment which is given as keyword-argument of the material, e.g.
    ``Hyperelastic(finite_strain_viscoelastic, mu=1.0, eta=1.0, dtime=1.0,
    nstatevars=6)``, is a constant material parameter. Then, the material is not
    time-dependent.

Time Steps
**********

A solid body, i.e. :class:`~felupe.SolidBody`,
:class:`~felupe.SolidBodyNearlyIncompressible` or :class:`~felupe.TrussBody`, passes its
time increment ``time_step`` to its time-dependent material. The time increment is
updated by a :class:`~felupe.TimeStep`. Its (absolute) time is ramped in a
:class:`~felupe.Step`, like the boundary conditions. The time increment of a substep is
the difference to the time of the previous substep, the initial time is zero. Hence, the
first substep of a ramp which starts at zero has a time increment of zero, which is
handled by the time-dependent materials (no evolution).

..  code-block:: python

    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    umat = fem.Hyperelastic(
        fem.finite_strain_viscoelastic, mu=1.0, eta=1.0, nstatevars=6
    )
    solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    time = fem.TimeStep(items=[solid])

    # a loading with a constant stretch rate
    loading = fem.Step(
        items=[solid],
        ramp={
            boundaries["move"]: fem.math.linsteps([0, 1], num=10),
            time: fem.math.linsteps([0, 1], num=10),
        },
        boundaries=boundaries,
    )

    # a holding time: only the time is ramped, the displacements are kept constant
    holding = fem.Step(
        items=[solid],
        ramp={time: fem.math.linsteps([1, 11], num=10)},
        boundaries=boundaries,
    )

    job = fem.CharacteristicCurve(steps=[loading, holding], boundary=boundaries["move"])
    job.evaluate()

The same time step item is used in all steps, i.e. the time continues. The position of
the time step item in the list of items of a step doesn't matter, because all ramped
items are updated before the iterations of a substep. If a substep is subdivided by a
:class:`~felupe.CutbackPlugin`, the time is interpolated like the other values of the
ramp, i.e. each increment of the substep takes its own time increment.

Force-Stretch Curves and Material Parameters
********************************************

The force-stretch curves of a time-dependent material, e.g. by ``plot()``, require the
(absolute) times of the load cases. They are given as second row of a load case, e.g.
``ux=[stretches, times]``. Each load case starts at the initial time zero. A holding
time is given by repeated stretches.

..  code-block:: python

    stretches = fem.math.linsteps([1, 1.5, 1.5, 1], num=[5, 5, 5])
    times = fem.math.linsteps([0, 0.5, 5.5, 6], num=[5, 5, 5])

    ax = umat.plot(incompressible=True, ux=[stretches, times], ps=None, bx=None)

The material parameters of a time-dependent material are fitted to experimental data
with the times of the data points as third row, e.g. ``ux=[stretches, forces, times]``.
The time increment is not a material parameter, i.e. it is not fitted.

..  code-block:: python

    umat_new, res = umat.optimize(ux=[stretches, forces, times], incompressible=True)

Custom Time-Dependent Materials
*******************************

The function of a custom material takes the time increment as argument ``dtime`` (without
a default value), e.g. a strain energy density function of
:class:`~felupe.Hyperelastic`. A time increment of zero has to be supported, e.g. for
the first substep of a ramp which starts at zero.

..  code-block:: python

    import tensortrax.math as tm

    def viscoelastic(C, statevars, mu, eta, dtime):
        ...
        return ψ, statevars_new

    umat = fem.Hyperelastic(viscoelastic, mu=1.0, eta=1.0, nstatevars=6)

A subclass of :class:`~felupe.ConstitutiveMaterial` is time-dependent by the class
attribute ``time_dependent = True``. Its methods ``_gradient(x, dtime)`` and
``_hessian(x, dtime)`` take the time increment, which is the same for all chunks of
cells.

..  code-block:: python

    class Viscoelastic(fem.ConstitutiveMaterial):
        time_dependent = True

        def __init__(self, mu, eta):
            self.kwargs = {"mu": mu, "eta": eta}
            self.x = [np.eye(3), np.zeros(6)]

        def _gradient(self, x, dtime):
            F, statevars = x[0], x[-1]
            ...
            return [P, statevars_new]

        def _hessian(self, x, dtime):
            F, statevars = x[0], x[-1]
            ...
            return [A]

Materials of other packages, which are not derived from
:class:`~felupe.ConstitutiveMaterial`, are time-dependent if they have an attribute
``time_dependent`` which is True. Then, their methods ``gradient(x, dtime)`` and
``hessian(x, dtime)`` take the time increment as keyword-argument ``dtime``.

..  note::

    The materials of the JAX backend are just-in-time compiled, i.e. the time increment
    is a traced value. Instead of a Python condition, e.g. ``if dtime > 0``, use
    :func:`jax.numpy.where`, see :class:`felupe.constitution.jax.Hyperelastic`. The time
    increment is passed as an array of data type float, i.e. the compiled functions
    are re-used for all time increments.
