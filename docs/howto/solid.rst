Solid Bodies as Items for Assembly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mechanics submodule contains classes for the generation of solid bodies. Solid body objects are supported as items of a :class:`~felupe.Step` and the :func:`~felupe.newtonrhapson` procedure.

Solid Body
----------

The generation of internal force vectors and stiffness matrices of solid bodies are provided as assembly-methods of a :class:`~felupe.SolidBody` or a :class:`~felupe.SolidBodyNearlyIncompressible`.

..  code-block:: python

    import felupe as fem

    mesh = fem.Cube(n=6)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    
    # a solid body
    body = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=2), field=field)
    
    # a (nearly) incompressible solid body (to be used with quads and hexahedrons)
    body = fem.SolidBodyNearlyIncompressible(umat=fem.NeoHooke(mu=1), field=field, bulk=5000)
    
    internal_force = body.assemble.vector(field, parallel=False, jit=False)
    stiffness_matrix = body.assemble.matrix(field, parallel=False, jit=False)


During assembly, several results are stored, e.g. the gradient of the strain energy density function per unit undeformed volume w.r.t. the deformation gradient (first Piola-Kirchhoff stress tensor). Other results are the deformation gradient or the fourth-order elasticity tensor associated to the first Piola-Kirchhoff stress tensor.

..  math::

    \boldsymbol{F} &= \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}}

    \boldsymbol{P} &= \frac{\partial \psi(\boldsymbol{C}(\boldsymbol{F}))}{\partial \boldsymbol{F}}

    \mathbb{A} &= \frac{\partial^2 \psi(\boldsymbol{C}(\boldsymbol{F}))}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}

..  code-block:: python
    
    F = body.results.kinematics
    P = body.results.stress
    A = body.results.elasticity


The Cauchy stress tensor, as well as the gradient and the hessian of the strain energy density function per unit undeformed volume are obtained by evaluate-methods of the solid body.

..  math::

    \boldsymbol{P} &= \frac{\partial \psi(\boldsymbol{C}(\boldsymbol{F}))}{\partial \boldsymbol{F}}

    \mathbb{A} &= \frac{\partial^2 \psi(\boldsymbol{C}(\boldsymbol{F}))}{\partial \boldsymbol{F}\ \partial \boldsymbol{F}}

    \boldsymbol{\sigma} &= \frac{1}{J} \boldsymbol{P} \boldsymbol{F}^T


..  code-block:: python
    
    P = body.evaluate.gradient(field)
    A = body.evaluate.hessian(field)
    s = body.evaluate.cauchy_stress(field)


Body Force (Gravity) on a Solid Body
------------------------------------

The generation of external force vectors or stiffness matrices of body forces acting on solid bodies are provided as assembly-methods of a :class:`~felupe.SolidBodyGravity`.


..  math::

    \delta W_{ext} = \int_V \delta \boldsymbol{u} \cdot \rho \boldsymbol{g} \ dV


..  code-block:: python
    
    body = fem.SolidBodyGravity(field=field, gravity=[9810, 0, 0], density=7.85e-9)
    
    force_gravity = body.assemble.vector(field, parallel=False, jit=False)


Pressure Boundary on a Solid Body
---------------------------------

The generation of force vectors or stiffness matrices of pressure boundaries on solid bodies are provided as assembly-methods of a :class:`~felupe.SolidBodyPressure`.

..  math::

    \delta W_{ext} = \int_{\partial V} \delta \boldsymbol{u} \cdot p \ J \boldsymbol{F}^{-T} \ d\boldsymbol{A}

..  code-block:: python
    
    region_pressure = fem.RegionHexahedronBoundary(
        mesh=mesh,
        only_surface=True, # select only faces on the outline
        mask=mesh.points[:, 0] == 0, # select a subset of faces on the surface
    )
    
    displacement_boundary = 
    field_boundary = fem.FieldContainer([fem.Field(region_pressure, dim=3)])
    field_boundary.link(field)
    
    body_pressure = fem.SolidBodyPressure(field=field_boundary)
    
    force_pressure = body_pressure.assemble.vector(
        field=field_boundary, parallel=False, jit=False
    )
    
    stiffness_matrix_pressure = body_pressure.assemble.matrix(
        field=field_boundary, parallel=False, jit=False
    )


For axisymmetric problems the boundary region has to be created with the attribute ``ensure_3d=True``.

..  code-block:: python
    
    mesh = fem.Rectangle(a=(0, 30), b=(20, 40), n=(21, 11))
    region = fem.RegionQuad(mesh)
    
    region_pressure = fem.RegionQuadBoundary(
        mesh=mesh,
        only_surface=True, # select only faces on the outline
        mask=mesh.points[:, 0] == 0, # select a subset of faces on the surface
        ensure_3d=True, # flag for axisymmetric boundary region
    )
    
    field = fem.FieldContainer([fem.FieldAxisymmetric(region)])
    field_boundary = fem.FieldContainer([fem.FieldAxisymmetric(region_pressure)])
    field_boundary.link(field)
