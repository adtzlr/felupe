Combine Multiple Solid Bodies
-----------------------------

This section demonstrates how to set up a problem with two solid bodies, a stiff
inclusion and a rubber matrix. The bodies are created on meshes with different cell
types, i.e. different finite element formulations are used for the solid bodies.

..  pyvista-plot::
    :context:

    import felupe as fem

    inner = fem.Rectangle(a=(-1, -1), b=(1, 1), n=(5, 5)).triangulate()

    lower = fem.Rectangle(a=(-3, -3), b=(3, -1), n=(13, 5))
    upper = fem.Rectangle(a=(-3, 1), b=(3, 3), n=(13, 5))
    left = fem.Rectangle(a=(-3, -1), b=(-1, 1), n=(5, 5))
    right = fem.Rectangle(a=(1, -1), b=(3, 1), n=(5, 5))

    outer = fem.MeshContainer([lower, upper, left, right], merge=True).stack()

A field container is created on the mesh of each body.

..  pyvista-plot::
    :context:

    fields = [
        fem.FieldContainer([fem.FieldPlaneStrain(fem.RegionTriangle(inner), dim=2)]),
        fem.FieldContainer([fem.FieldPlaneStrain(fem.RegionQuad(outer), dim=2)]),
    ]

:func:`~felupe.field.merge` merges the field containers of the bodies into a top-level
field container, which contains all the unknowns. The points of the meshes are merged,
i.e. the bodies share the points at their interface. The field containers of the bodies
are modified in-place: their regions and fields are reloaded on the merged meshes and
the values of their fields are linked to (shared with) the top-level field container.
The top-level field container is required for the definition of the boundary conditions
and it is used as the ``x0``-argument of the job by default.

..  pyvista-plot::
    :context:
    :force_static:

    field = fem.field.merge(fields)

    field.plot(plotter=field.mesh_container.plot(), point_size=15, color="red").show()

..  note::
    The merged meshes are available in the mesh container ``field.mesh_container`` of the
    top-level field container. ``fem.FieldContainer(fields).merge()`` is equivalent to
    ``fem.field.merge(fields)``. Items with an internal state of the given fields, e.g.
    :class:`~felupe.SolidBodyNearlyIncompressible`, must be created after the merge.

The displacement boundaries are created on the top-level field container.

..  pyvista-plot::
    :context:

    boundaries = dict(
        fixed=fem.Boundary(field[0], fx=field.region.mesh.x.min()),
        move=fem.Boundary(field[0], fx=field.region.mesh.x.max()),
    )


The rubber matrix is associated to a Neo-Hookean material formulation whereas the stiff
inclusion is modeled by a linear elastic material formulation. Due to the large
rotation, its large-strain formulation is required. For each material a solid body is
created on the field container of its body.

..  pyvista-plot::
    :context:

    # two material model formulations
    linear_elastic = fem.LinearElasticLargeStrain(E=100, nu=0.3)
    neo_hooke = fem.NeoHooke(mu=1, bulk=1)

    # the solid bodies
    inclusion = fem.SolidBody(umat=linear_elastic, field=fields[0])
    matrix = fem.SolidBody(umat=neo_hooke, field=fields[1])


A step is created and further added to a job. The top-level field container is taken
from the field containers of the solid bodies, i.e. it doesn't have to be passed to
:meth:`~felupe.Job.evaluate`.

..  pyvista-plot::
    :context:
    :force_static:

    # prepare a step with substeps
    move = fem.math.linsteps([0, 3], num=10)
    step = fem.Step(
        items=[matrix, inclusion],
        ramp={boundaries["move"]: move},
        boundaries=boundaries,
    )

    job = fem.Job(steps=[step]).evaluate()

    fields[1].plot(
        "Principal Values of Logarithmic Strain",
        show_undeformed=False,
        plotter=fields[0].plot(
            "Principal Values of Logarithmic Strain", show_undeformed=False
        ),
    ).show()

Mixed-Field Formulations
~~~~~~~~~~~~~~~~~~~~~~~~

The solid bodies may also use different mixed-field formulations, which are merged in
the same way, e.g. a displacement field for a linear-elastic body and mixed fields
(displacement, pressure and volume ratio) for two nearly-incompressible bodies.
:func:`~felupe.field.merge` merges the points of the displacement fields and appends the
dual fields to the top-level field container, i.e. the pressure and the volume ratio are
local to their body. No offsets or dual meshes have to be created manually.

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

Like before, the boundary conditions are created on the top-level field container and
the solid bodies on the field containers of the bodies.

..  pyvista-plot::
    :context:
    :force_static:

    boundaries = fem.dof.uniaxial(field, move=-0.8, clamped=True)

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
    The first fields (displacements) of all bodies are merged. All other fields belong
    to their body, e.g. the pressure and the volume ratio of a mixed-field body or the
    tensor-valued field of :class:`~felupe.ThirdMediumContactMixed` of a third medium:
    they are not coupled across the bodies. Hence, the bodies may use different
    mixed formulations, e.g. Q1/P0, Q2/P0 or Taylor-Hood elements, a displacement field
    only, or third media next to :class:`~felupe.ThreeFieldVariation` bodies. A field
    which should be continuous across several bodies (other than the displacements) must
    be defined in a single field container.
