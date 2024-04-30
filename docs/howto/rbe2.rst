Multi-Point Constraints
-----------------------

This How-To demonstrates the usage of multi-point constraints (also called MPC or RBE2 rigid-body-elements) with an independent centerpoint and one or more dependent points. First, a centerpoint has to be added to the mesh. MPC objects are supported as ``items`` of a :class:`~felupe.Step` and within the :func:`~felupe.newtonrhapson` procedure.

..  pyvista-plot::
    :context:

    import numpy as np
    import felupe as fem

    # mesh with one additional rbe2-control point
    mesh = fem.Cube(n=11)
    mesh.update(points=np.vstack((mesh.points, [2.0, 0.0, 0.0])))
    
    # prevent the field-values at the center-point to be treated as dof0
    mesh.points_without_cells = mesh.points_without_cells[:0]
    
    region = fem.RegionHexahedron(mesh)
    displacement = fem.Field(region, dim=3)
    field = fem.FieldContainer([displacement])

An instance of :class:`~felupe.MultiPointConstraint` defines the multi-point constraint. This instance provides two methods, :meth:`MultiPointConstraint.assemble.vector() <felupe.MultiPointConstraint.assemble.vector>` and :meth:`MultiPointConstraint.assemble.matrix() <felupe.MultiPointConstraint.assemble.matrix>`.

..  pyvista-plot::
    :context:

    mpc = fem.MultiPointConstraint(
        field=field, 
        points=np.arange(mesh.npoints)[mesh.points[:, 0] == 1], 
        centerpoint=mesh.npoints - 1, 
        skip=(0,1,1),
    )

Finally, add the results of these methods to the internal force vector or the stiffness matrix.

..  pyvista-plot::
    :context:
    :force_static:

    umat = fem.NeoHooke(mu=1.0, bulk=2.0)
    body = fem.SolidBody(umat=umat, field=field)

    K = body.assemble.matrix() + mpc.assemble.matrix()
    r = body.assemble.vector(field) + mpc.assemble.vector(field)
    
    mesh.plot(plotter=mpc.plot()).show()
