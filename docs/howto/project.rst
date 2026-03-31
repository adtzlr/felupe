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
The means of the cell-values are plotted if no projection method is specified. Different
methods may be used to *move* the cell-data to the mesh-points. With
:func:`~felupe.topoints`, the cell-values are translated to and averaged at the mesh-
points. With :func:`~felupe.project`, the cell-values are projected to the mesh-points
by solving a least-squares problem. With :func:`~felupe.tools.extrapolate`, the cell-
values are extrapolated to the mesh-points by evaluating the cell-values at the
quadrature points of the cells and extrapolating them to the mesh-points.

..  tab:: average

    ..  pyvista-plot::
        :context:
        :force_static:

        plotter = solid.plot(
            name="Cauchy Stress", 
            label="Cauchy Stress YY",
            component=1,
            project=None,
        )
        data = plotter.mesh.cell_data["Cauchy Stress"][..., 1]
        plotter.add_text(f"Range {data.min():.2f} ... {data.max():.2f} MPa")
        plotter.show()

..  tab:: shift to points

    ..  pyvista-plot::
        :context:
        :force_static:

        plotter = solid.plot(
            name="Cauchy Stress", 
            label="Cauchy Stress YY (topoints)",
            component=1,
            project=fem.topoints,
            clim=[-7.0, 0.0],
        )
        data = plotter.mesh.point_data["Cauchy Stress"][..., 1]
        plotter.add_text(f"Range {data.min():.2f} ... {data.max():.2f} MPa")
        plotter.show()

..  tab:: project to points

    ..  pyvista-plot::
        :context:
        :force_static:

        plotter = solid.plot(
            name="Cauchy Stress", 
            label="Cauchy Stress YY (project)",
            component=1,
            project=fem.project,
            clim=[-7.0, 0.0],
        )
        data = plotter.mesh.point_data["Cauchy Stress"][..., 1]
        plotter.add_text(f"Range {data.min():.2f} ... {data.max():.2f} MPa")
        plotter.show()

..  tab:: extrapolate to points

    ..  pyvista-plot::
        :context:
        :force_static:

        plotter = solid.plot(
            name="Cauchy Stress", 
            label="Cauchy Stress YY (extrapolate)",
            component=1,
            project=fem.tools.extrapolate,
            clim=[-7.0, 0.0],
        )
        data = plotter.mesh.point_data["Cauchy Stress"][..., 1]
        plotter.add_text(f"Range {data.min():.2f} ... {data.max():.2f} MPa")
        plotter.show()