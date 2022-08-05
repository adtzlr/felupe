Solid Mechanics
~~~~~~~~~~~~~~~

The mechanics submodule contains classes for the generation of solid bodies.

Solid Body
----------

The generation of internal force vectors or stiffness matrices of solid bodies are provided as assembly-methods of a :class:`felupe.SolidBody`. The correct integral form is chosen based on the :class:`felupe.Field`  (default, axisymmetric or mixed).

..  code-block:: python

    import felupe as fe

    neohooke = fe.NeoHooke(mu=1.0, bulk=5000.0)
    mesh = fe.Cube(n=6)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)
    field = fe.FieldContainer([displacement])
    
    body = fe.SolidBody(umat=neohooke, field=field)
    internal_force = body.assemble.vector(field, parallel=False, jit=False)
    stiffness_matrix = body.assemble.matrix(field, parallel=False, jit=False)


During assembly, several results are stored, e.g. the gradient of the strain energy density function per unit undeformed volume (first Piola-Kirchhoff stress tensor). Other results are the deformation gradient or the fourth-order elasticity tensor associated to the first Piola-Kirchhoff stress tensor.

..  code-block:: python
    
    F = body.results.kinematics
    P = body.results.stress
    A = body.results.elasticity


The Cauchy stress tensor, as well as the gradient and the hessian of the strain energy density function per unit undeformed volume are obtained by evaluate-methods of the solid body.

..  code-block:: python
    
    P = body.evaluate.gradient(field)
    A = body.evaluate.hessian(field)
    s = body.evaluate.cauchy_stress(field)


Pressure Boundary on a Solid Body
---------------------------------

The generation of internal force vectors or stiffness matrices of pressure boundaries on solid bodies are provided as assembly-methods of a :class:`felupe.SolidBodyPressure`. The correct integral form is chosen based on the :class:`felupe.Field` (default or axisymmetric). If the internal field is a mixed field, the assembled vectors and matrices from the pressure contribution have to be resized to the dimensions of the internal force vector and the stiffness matrix.

..  code-block:: python
    
    region_pressure = fe.RegionHexahedronBoundary(
        mesh=mesh,
        only_surface=True, # select only faces on the outline
        mask=mesh.points[:, 0] == 0, # select a subset of faces on the surface
    )
    
    displacement_boundary = fe.Field(region_pressure, dim=3)
    field_boundary = fe.FieldContainer([displacement_boundary])
    displacement_boundary.values = displacement.values # link field values
    
    body_pressure = fe.SolidBodyPressure(field=field_boundary)
    
    internal_force_pressure = body_pressure.assemble.vector(
        field=field_boundary, parallel=False, jit=False, resize=internal_force
    )
    
    stiffness_matrix_pressure = body_pressure.assemble.matrix(
        field=field_boundary, parallel=False, jit=False, resize=stiffness_matrix
    )


For axisymmetric problems the boundary region has to be created with the attribute ``ensure_3d=True``.

..  code-block:: python
    
    mesh = fe.Rectangle(a=(0, 30), b=(20, 40), n=(21, 11))
    region = fe.RegionQuad(mesh)
    
    region_pressure = fe.RegionQuadBoundary(
        mesh=mesh,
        only_surface=True, # select only faces on the outline
        mask=mesh.points[:, 0] == 0, # select a subset of faces on the surface
        ensure_3d=True, # flag for axisymmetric boundary region
    )
    
    displacement = fe.FieldAxisymmetric(region)
    displacement_boundary = fe.FieldAxisymmetric(region_pressure)
    displacement_boundary.values = displacement.values # link field values