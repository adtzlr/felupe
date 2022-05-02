Solid Body (Mechanics)
~~~~~~~~~~~~~~~~~~~~~~

The generation of internal force vectors or stiffness matrices of solid bodies are provided as assembly-methods of a :class:`felupe.SolidBody`. The correct integral form is chosen based on the :class:`felupe.Field`  (default, axisymmetric or mixed).

..  code-block:: python

    import felupe as fe

    neohooke = fe.NeoHooke(mu=1.0, bulk=5000.0)
    mesh = fe.Cube(n=6)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)
    
    body = fe.SolidBody(umat=neohooke, field=displacement)
    internal_force = body.assemble.vector(displacement, parallel=False, jit=False)
    stiffness_matrix = body.assemble.matrix(displacement, parallel=False, jit=False)


During assembly, several results are stored, e.g. the gradient of the strain energy density function per unit undeformed volume (first Piola-Kirchhoff stress tensor). Other results are the deformation gradient or the fourth-order elasticity tensor associated to the first Piola-Kirchhoff stress tensor.

..  code-block:: python
    
    F = body.results.kinematics[0]
    P = body.results.stress
    A = body.results.elasticity


The Cauchy stress tensor, as well as the gradient and the hessian of the strain energy density function per unit undeformed volume are obtained by evaluate-methods of the solid body.

..  code-block:: python
    
    P = body.evaluate.gradient(displacement)
    A = body.evaluate.hessian(displacement)
    s = body.evaluate.cauchy_stress(displacement)


Pressure Boundary on Solid Body (Mechanics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The generation of internal force vectors or stiffness matrices of pressure boundaries of solid bodies are provided as assembly-methods of a :class:`felupe.SolidBodyPressure`. The correct integral form is chosen based on the :class:`felupe.Field` (default or axisymmetric). If the internal field is a mixed field, the assembled vectors and matrices from the pressure contribution have to be resized to the dimensions of the internal force vector and the stiffness matrix.

..  code-block:: python
    
    region_pressure = fe.RegionHexahedronBoundary(
        mesh=mesh,
        only_surface=True, # select only faces on the outline
        mask=lambda x: x==0, # select a subset of faces on the surface
    )
    
    displacement_boundary = fe.Field(region_pressure)
    displacement_boundary.values = displacement.values # link field values
    
    body_pressure = fe.SolidBodyPressure(field=displacement)
    
    internal_force_pressure = body.assemble.vector(
        field=displacement, parallel=False, jit=False, resize=internal_force
    )
    
    stiffness_matrix_pressure = body.assemble.matrix(
        field=displacement, parallel=False, jit=False, resize=stiffness_matrix
    )