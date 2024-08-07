Axisymmetric Problems
---------------------

For axisymmetric analyses an axisymmetric vector-valued field has to be created for the in-plane displacements.

..  code-block:: python

    import felupe as fem

    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    u = fem.FieldAxisymmetric(region, dim=2)
    field = fem.FieldContainer([u])

Now it gets important: The 3x3 deformation gradient for an axisymmetric problem is obtained with :meth:`~felupe.FieldAxisymmetric.grad` or :meth:`~felupe.FieldAxisymmetric.extract` methods. For a two-dimensional :class:`~felupe.FieldAxisymmetric` the gradient is modified to return a three-dimensional gradient.

..  code-block:: python

    F = field.extract(grad=True, sym=False, add_identity=True)

For simplicity, let's use the isotropic hyperelastic :class:`~felupe.NeoHooke` material model formulation.

..  code-block:: python

    umat = fem.NeoHooke(mu=1, bulk=5)

..  note::

    Internally, FElupe provides an adopted low-level :class:`~felupe.assembly.IntegralFormAxisymmetric` class for the integration and the sparse matrix assemblage of axisymmetric problems. It uses the additional information (e.g. radial coordinates at integration points) stored in :class:`~felupe.FieldAxisymmetric` to provide a consistent interface in comparison to :class:`~felupe.assembly.IntegralFormCartesian`. The top-level :class:`~felupe.IntegralForm` chooses the appropriate low-level integral form based on the kind of field inside the field container.

..  code-block:: python

    dA = region.dV

    r = fem.IntegralForm(umat.gradient(F), field, dA).assemble()
    K = fem.IntegralForm(umat.hessian(F), field, dA, field).assemble()

To sum up, for axisymmetric problems use :class:`~felupe.FieldAxisymmetric`. Of course, mixed-field formulations may also be used with axisymmetric (displacement) fields.
