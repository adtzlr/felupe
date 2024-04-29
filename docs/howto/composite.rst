Composite Regions with Solid Bodies
-----------------------------------

This section demonstrates how to set up a problem with two regions, each associated to a seperated solid body.

..  pyvista-plot::
    :context:

    import felupe as fem
    import numpy as np

    m = fem.Rectangle(n=21)


In a second step, sub-meshes are created.
    
..  pyvista-plot::
    :context:

    # take some points from the inside for the fiber-reinforced area
    eps = 1e-3
    mask = np.arange(m.npoints)[np.logical_and.reduce([
        m.points[:, 0] >= 0.3,
        m.points[:, 0] <= 0.7 + eps,
        m.points[:, 1] >= 0.3,
        m.points[:, 1] <= 0.7 + eps,
    ])]
    
    # copies of the mesh
    mesh = [m.copy(), m.copy()]
    
    # create sub-meshes (fiber, matrix)
    mesh[0].update(cells=m.cells[ np.all(np.isin(m.cells, mask), axis=1)])
    mesh[1].update(cells=m.cells[~np.all(np.isin(m.cells, mask), axis=1)])

This is followed by the creation of a global region/field and two sub-regions/sub-fields.

..  pyvista-plot::
    :context:
    
    # a global and two sub-regions
    region = fem.RegionQuad(m)
    regions = [fem.RegionQuad(me) for me in mesh]
    
    # a global and two sub-fields
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    fields = [
        fem.FieldContainer([fem.FieldPlaneStrain(regions[0], dim=2)]),
        fem.FieldContainer([fem.FieldPlaneStrain(regions[1], dim=2)]),
    ]
    
The displacement boundaries are created on the total field.

..  pyvista-plot::
    :context:

    boundaries = dict(
        fixed=fem.Boundary(field[0], fx=0),
        move=fem.Boundary(field[0], fx=1),
    )


The rubber is associated to a Neo-Hookean material formulation whereas the steel is modeled by a linear elastic material formulation. For each material a solid body is created.

..  pyvista-plot::
    :context:

    # two material model formulations
    neo_hooke = fem.NeoHooke(mu=1, bulk=1)
    linear_elastic = fem.LinearElastic(E=100, nu=0.3)
    
    # the solid bodies
    fiber = fem.SolidBody(umat=linear_elastic, field=fields[0])
    matrix = fem.SolidBody(umat=neo_hooke, field=fields[1])


A step is created and further added to a job. The global field must be passed to the ``x0`` argument during the evaluation of the job. Internally, all field values are linked automatically, i.e. they share their ``values`` attribute.

..  pyvista-plot::
    :context:
    :force_static:

    # prepare a step with substeps
    move = fem.math.linsteps([0, 0.5], num=10)
    step = fem.Step(
        items=[matrix, fiber],
        ramp={boundaries["move"]: move}, 
        boundaries=boundaries
    )
    
    # take care of the x0-argument
    job = fem.Job(steps=[step])
    job.evaluate(x0=field, filename="result.xdmf")

    field.plot("Principal Values of Logarithmic Strain").show()
