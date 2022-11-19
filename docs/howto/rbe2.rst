Multi-Point Constraints
-----------------------

This How-To demonstrates the usage of multi-point constraints (also called MPC or RBE2 rigid-body-elements) with an independent centerpoint and one or more dependent points. First, a centerpoint has to be added to the mesh. MPC objects are supported as items of the Newton-Rhapson procedure.

..  code-block:: python

    import numpy as np
    import felupe as fem

    # mesh with one additional rbe2-control point
    mesh = fem.Cube(n=11)
    mesh.points = np.vstack((mesh.points, [2.0, 0.0, 0.0]))
    mesh.update(mesh.cells)
    
    region = felupe.RegionHexahedron(mesh)
    displacement = felupe.Field(region, dim=3)
    field = felupe.FieldContainer([displacement])

An instance of :class:`felupe.MultiPointConstraint` defines the multi-point constraint. This instance provides two methods, :meth:`felupe.MultiPointConstraint.assemble.vector` and :meth:`felupe.MultiPointConstraint.assemble.matrix`.

..  code-block:: python

    MPC = fem.MultiPointConstraint(
        field=field, 
        points=np.arange(mesh.npoints)[mesh.points[:, 0] == 1], 
        centerpoint=mesh.npoints - 1, 
        skip=(0,1,1),
    )

Finally, add the results of these methods to the internal force vector or the stiffness matrix.

..  code-block:: python

    umat = felupe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    K = fem.IntegralForm(
        umat.hessian(field.extract()), field, region.dV, field
    ).assemble() + MPC.assemble.matrix()

    r = fem.IntegralForm(
        umat.gradient(field.extract()), field, region.dV
    ).assemble() + MPC.assemble.vector(field)