Multi-Point Constraints
-----------------------

This How-To demonstrates the usage of multi-point constraints (also called MPC or RBE2 rigid-body-elements) with an independent centerpoint and one or more dependent points. First, a centerpoint has to be added to the mesh.

..  code-block:: python

    import numpy as np
    import felupe as fe

    # mesh with one additional rbe2-control point
    mesh = fe.Cube(n=11)
    mesh.points = np.vstack((mesh.points, [2.0, 0.0, 0.0]))
    mesh.update(mesh.cells)

An instance of :class:`felupe.MultiPointConstraint` defines the multi-point constraint. This instance provides two methods, :meth:`felupe.MultiPointConstraint.stiffness` and :meth:`felupe.MultiPointConstraint.residuals`.

..  code-block:: python

    MPC = fe.MultiPointConstraint(
        mesh=mesh, 
        points=np.arange(mesh.npoints)[mesh.points[:, 0] == 1], 
        centerpoint=mesh.npoints - 1, 
        skip=(0,1,1),
    )

Finally, add the results of these methods to the internal force vector or the stiffness matrix.

..  code-block:: python

    region = felupe.RegionHexahedron(mesh)
    displacement = felupe.Field(region, dim=3)
    field = felupe.FieldContainer([displacement])
    umat = felupe.constitution.NeoHooke(mu=1.0, bulk=2.0)

    K = fe.IntegralForm(
        umat.hessian(field.extract()), field, region.dV, field
    ).assemble() + MPC.stiffness()

    r = fe.IntegralForm(
        umat.gradient(field.extract()), field, region.dV
    ).assemble() + MPC.residuals(field)