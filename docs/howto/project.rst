Project cell values to mesh-points
----------------------------------

This section demonstrates how to move cell-values, located at the quadrature points
of cells, to mesh-points. The results of :func:`~felupe.project`,
:func:`~felupe.topoints` and :func:`~felupe.tools.extrapolate` are compared for the
Cauchy stresses of a rectangular block under compression.

..  pyvista-plot::
    :context:

    import felupe as fem

    region = fem.RegionQuad(mesh=fem.Rectangle(b=(2, 1), n=(11, 6)))
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    
    boundaries = fem.dof.uniaxial(
        field, clamped=True, move=-0.3, axis=1, return_loadcase=False
    )
    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=5), field=field)

    job = fem.Job(steps=[fem.Step(items=[solid], boundaries=boundaries)]).evaluate()

Cell-based results, like Cauchy stresses, are not projected to mesh-points by default.
Different methods may be used to *move* the cell-data to the mesh-points.

..  pyvista-plot::
    :context:
    :force_static:

    solid.plot(
        name="Cauchy Stress", 
        label="Cauchy Stress YY",
        component=1,
        project=None,
    ).show()

..  pyvista-plot::
    :context:
    :force_static:

    solid.plot(
        name="Cauchy Stress", 
        label="Cauchy Stress YY (topoints)",
        component=1,
        project=fem.topoints,
    ).show()

..  pyvista-plot::
    :context:
    :force_static:

    solid.plot(
        name="Cauchy Stress", 
        label="Cauchy Stress YY (project)",
        component=1,
        project=fem.project,
    ).show()

..  pyvista-plot::
    :context:
    :force_static:

    solid.plot(
        name="Cauchy Stress", 
        label="Cauchy Stress YY (extrapolate)",
        component=1,
        project=fem.tools.extrapolate,
    ).show()