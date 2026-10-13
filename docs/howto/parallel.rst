Evaluate in parallel
--------------------

FElupe may evaluate two parts of a Newton-Raphson iteration by a pool of threads: the
assembly of the weak forms and the constitutive material. Both are evaluated in chunks of
cells and they are enabled independently. The pool of threads is shared and it uses one
thread per usable CPU of the process, e.g. the CPUs of a job on a shared HPC-node. For
Python 3.13+, the number of CPUs may be overridden by the environment variable
``PYTHON_CPU_COUNT``.

..  list-table::
    :header-rows: 1

    * - Part
      - Enabled by
      - Evaluated in parallel
    * - Assembly
      - ``job.evaluate(parallel=True)``, ``solid.assemble.vector(parallel=True)``,
        ``solid.assemble.matrix(parallel=True)`` or
        ``IntegralForm.assemble(parallel=True)``
      - The integration of the weak forms. For items on boundaries, e.g. a
        :class:`~felupe.SolidBodyPressure`, also the evaluation of the area change.
    * - Constitutive material
      - The material, e.g. ``NeoHooke(mu=1, bulk=5000, parallel=True)``
      - The gradient and the hessian of the strain energy density function.

The constitutive materials are evaluated in chunks of cells also without threads,
because the intermediate arrays of a chunk fit into the CPU caches. The number of
evaluation points (quadrature points times cells) per chunk is given by the attribute
``chunksize`` of the material (default is 8192). A material with ``parallel=True``
evaluates its chunks by the pool of threads.

..  code-block:: python

    import felupe as fem

    mesh = fem.Cube(n=31)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

    # evaluate the gradient and the hessian of the material by threads
    umat = fem.NeoHooke(mu=1, bulk=5000, parallel=True)
    solid = fem.SolidBody(umat=umat, field=field)

    move = fem.math.linsteps([0, 0.2], num=4)
    step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

    # evaluate the assembly by threads
    job = fem.Job(steps=[step]).evaluate(parallel=True)

Composed materials, like :class:`~felupe.ThreeFieldVariation`,
:class:`~felupe.NearlyIncompressible` or :class:`~felupe.OgdenRoxburgh`, take the flag
from their material by default, e.g. ``ThreeFieldVariation(NeoHooke(..., parallel=True))``
is evaluated in parallel. A composite material ``material & other_material`` is evaluated
in parallel if any of its materials is parallel.

..  note::
    Threads add an overhead to small-sized problems. A single chunk of cells is always
    evaluated without threads. With a multi-threaded BLAS-library, limit its number of
    threads to avoid an oversubscription of the CPUs.

..  warning::
    A user-defined material, which implements ``_gradient(x)`` and ``_hessian(x)`` of
    :class:`~felupe.ConstitutiveMaterial`, must be thread-safe if it is created or used
    with ``parallel=True``. Intermediate results must be local variables (not attributes
    of the material) and the material must not change its attributes during the
    evaluation. The result of a cell must not depend on other cells.

The JAX-based materials (:class:`felupe.constitution.jax.Hyperelastic` and
:class:`felupe.constitution.jax.Material`) are not evaluated in chunks of cells. For
these, ``parallel=True`` maps the quadrature points to multiple devices.
