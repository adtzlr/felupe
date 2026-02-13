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

..  code-block::

    import pyvista as pv
    
    plotter = pv.Plotter(shape=(2, 2))
    kwargs = dict(name="Cauchy Stress", component=1, plotter=plotter)
    
    plotter.subplot(0, 0)
    kwargs_sbar = dict(interactive=False, title="Cauchy Stress YY (None)")
    solid.plot(project=None, **kwargs, scalar_bar_args=kwargs_sbar)
    
    plotter.subplot(0, 1)
    kwargs_sbar = dict(interactive=False, title="Cauchy Stress YY (topoints)")
    solid.plot(project=fem.topoints, **kwargs, scalar_bar_args=kwargs_sbar)
    
    plotter.subplot(1, 0)
    kwargs_sbar = dict(interactive=False, title="Cauchy Stress YY (project)")
    solid.plot(project=fem.project, **kwargs, scalar_bar_args=kwargs_sbar)
    
    plotter.subplot(1, 1)
    kwargs_sbar = dict(interactive=False, title="Cauchy Stress YY (extrapolate)")
    solid.plot(project=fem.tools.extrapolate, **kwargs, scalar_bar_args=kwargs_sbar)
    
    plotter.show()
