Axisymmetric Problems
---------------------

For axisymmetric analyses an axisymmetric vector-valued field has to be created for the in-plane displacements.

..  code-block:: python

    import felupe as fem

    mesh = fem.Rectangle(n=3)
    region = fem.RegionQuad(mesh)
    u = fem.FieldAxisymmetric(region, dim=2)
    field = fem.FieldContainer([u])

Now it gets important: The 3x3 deformation gradient for an axisymmetric problem is obtained with :meth:`felupe.FieldAxisymmetric.grad` or :meth:`felupe.FieldAxisymmetric.extract` methods. For instances of :class:`felupe.FieldAxisymmetric` the gradient is modified to return a 3x3 gradient as described in :ref:`theory-axi`.

..  code-block:: python

    F = field.extract(grad=True, sym=False, add_identity=True)

For simplicity, let's assume a (built-in) Neo-Hookean material.

..  code-block:: python

    umat = fem.NeoHooke(mu=1, bulk=5)


FElupe provides an adopted :class:`felupe.IntegralFormAxisymmetric` class for the integration and the sparse matrix assemblage of axisymmetric problems under hood. It uses the additional information (e.g. radial coordinates at integration points) stored in :class:`felupe.FieldAxisymmetric` to provide a consistent interface in comparison to default IntegralForms.

..  code-block:: python

    dA = region.dV

    r = fem.IntegralForm(umat.gradient(F), field, dA).assemble()
    K = fem.IntegralForm(umat.hessian(F), field, dA, field).assemble()

To sum up, for axisymmetric problems use :class:`felupe.FieldAxisymmetric`. Of course, Mixed-field formulations may be applied on axisymmetric scenarios too.
