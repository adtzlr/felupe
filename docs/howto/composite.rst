Multiple Solid Bodies
---------------------

This section demonstrates how to set up a problem with two regions, each associated to a
separated solid body. Different element formulations are used for the solid bodies.

..  pyvista-plot::
    :context:

    import felupe as fem

    inner = fem.Rectangle(a=(-1, -1), b=(1, 1), n=(5, 5)).triangulate()
    
    lower = fem.Rectangle(a=(-3, -3), b=(3, -1), n=(13, 5))
    upper = fem.Rectangle(a=(-3, 1), b=(3, 3), n=(13, 5))
    left = fem.Rectangle(a=(-3, -1), b=(-1, 1), n=(5, 5))
    right = fem.Rectangle(a=(1, -1), b=(3, 1), n=(5, 5))
    
    outer = fem.MeshContainer([lower, upper, left, right], merge=True).stack()
    
    container = fem.MeshContainer([inner, outer], merge=True)


A top-level (vertex) field, which contains all the unknowns, is required for the
definition of the boundary conditions as well as for the evaluation of the job.

..  note::
    Ensure to init the mesh container with ``merge=True``, otherwise the points-array of
    the container will be empty.
    
..  pyvista-plot::
    :context:
    :force_static:

    container = fem.MeshContainer([inner, outer], merge=True)
    field = fem.Field.from_mesh_container(container).as_container()
    
    field.plot(plotter=container.plot(), point_size=15, color="red").show()

The sub-meshes are available in the global mesh container, on which the sub-fields are
created.

..  pyvista-plot::
    :context:

    regions = [
        fem.RegionTriangle(container.meshes[0]),
        fem.RegionQuad(container.meshes[1]),
    ]
    fields = [
        fem.FieldContainer([fem.FieldPlaneStrain(regions[0], dim=2)]),
        fem.FieldContainer([fem.FieldPlaneStrain(regions[1], dim=2)]),
    ]
    
The displacement boundaries are created on the top-level field.

..  pyvista-plot::
    :context:

    boundaries = dict(
        fixed=fem.Boundary(field[0], fx=field.region.mesh.x.min()),
        move=fem.Boundary(field[0], fx=field.region.mesh.x.max()),
    )


The rubber is associated to a Neo-Hookean material formulation whereas the steel is
modeled by a linear elastic material formulation. Due to the large rotation, its
large-strain formulation is required. For each material a solid body is created.

..  pyvista-plot::
    :context:

    # two material model formulations
    linear_elastic = fem.LinearElasticLargeStrain(E=100, nu=0.3)
    neo_hooke = fem.NeoHooke(mu=1, bulk=1)
    
    # the solid bodies
    fiber = fem.SolidBody(umat=linear_elastic, field=fields[0])
    matrix = fem.SolidBody(umat=neo_hooke, field=fields[1])


A step is created and further added to a job. The global field must be passed as the
``x0`` argument during the evaluation of the job. Internally, all field values are
linked automatically, i.e. they share their ``values`` array.

..  pyvista-plot::
    :context:
    :force_static:

    # prepare a step with substeps
    move = fem.math.linsteps([0, 3], num=10)
    step = fem.Step(
        items=[matrix, fiber],
        ramp={boundaries["move"]: move}, 
        boundaries=boundaries,
    )
    
    # take care of the x0-argument
    job = fem.Job(steps=[step])
    job.evaluate(x0=field)

    fields[1].plot(
        "Principal Values of Logarithmic Strain", 
        show_undeformed=False, 
        plotter=fields[0].plot(
            "Principal Values of Logarithmic Strain", show_undeformed=False
        ),
    ).show()

Mixed-Field Formulations
~~~~~~~~~~~~~~~~~~~~~~~~

The solid bodies may also use different mixed-field formulations. Each field container
is created on the (unmerged) mesh of its body, e.g. a displacement field for a
linear-elastic body and mixed fields (displacement, pressure and volume ratio) for two
nearly-incompressible bodies. :func:`~felupe.field.merge` merges the points of the
displacement fields and stacks the points of the dual fields, i.e. the pressure and the
volume ratio are local to their body. No offsets or dual meshes have to be created
manually.

..  pyvista-plot::
    :context:
    :force_static:

    meshes = [
        fem.Rectangle(n=(4, 10)),
        fem.Rectangle(n=(6, 10)).translate(1, axis=0),
        fem.Rectangle(n=(10, 10)).translate(2, axis=0),
    ]
    fields = [
        fem.FieldContainer([fem.FieldAxisymmetric(fem.RegionQuad(meshes[0]), dim=2)]),
        fem.FieldsMixed(fem.RegionQuad(meshes[1]), n=3, axisymmetric=True),
        fem.FieldsMixed(fem.RegionQuad(meshes[2]), n=3, axisymmetric=True),
    ]
    field = fem.field.merge(fields)

The boundary conditions are created on the top-level field container. The fields of
the bodies are linked to the top-level field container, which is also used as the
``x0``-argument of the job by default.

..  pyvista-plot::
    :context:
    :force_static:

    boundaries = fem.dof.uniaxial(field, move=-0.8, clamped=True, return_loadcase=False)

    umats = [
        fem.LinearElasticLargeStrain(E=10, nu=0.3),
        fem.ThreeFieldVariation(fem.LinearElasticLargeStrain(E=3, nu=0.3)),
        fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=5000),
    ]
    solids = [fem.SolidBody(umat, f) for umat, f in zip(umats, fields)]

    step = fem.Step(items=solids, boundaries=boundaries)
    job = fem.Job(steps=[step]).evaluate()

    plotter = None
    for solid in solids:
        plotter = solid.plot(
            "Principal Values of Logarithmic Strain",
            show_undeformed=False,
            plotter=plotter,
        )
    plotter.show()

..  note::
    The first fields (displacements) of all bodies are merged. The other fields are
    grouped by their kind (continuous or local to a body), their dimension and their
    order in a field container. Hence, the bodies may use different mixed
    formulations, e.g. Q1/P0, Q2/P0 or Taylor-Hood elements, a displacement field only,
    or the displacement and the tensor-valued field of
    :class:`~felupe.ThirdMediumContactMixed` for a third medium next to a
    :class:`~felupe.ThreeFieldVariation` body. A continuous field which is only
    defined in some of the bodies, like the tensor-valued field of the third medium,
    only has the points of these bodies.
