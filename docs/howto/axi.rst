Axisymmetric Problems
---------------------

For axisymmetric analyses an axisymmetric vector-valued field has to be created for the in-plane displacements.

..  code-block:: python

    import felupe as fe

    mesh = fe.Rectangle(n=3)
    region = fe.RegionQuad(mesh)
    u = fe.FieldAxisymmetric(region, dim=2)

Now it gets important: The 3x3 deformation gradient for an axisymmetric problem is obtained with :meth:`felupe.FieldAxisymmetric.grad` or :meth:`felupe.FieldAxisymmetric.extract` methods. For instances of :class:`felupe.FieldAxisymmetric` the gradient is modified to return a 3x3 gradient as described in :ref:`theory-axi`.

..  code-block:: python

    dudX = fe.math.grad(u)

or

..  code-block:: python

    F = u.extract(grad=True, sym=False, add_identity=True)

For simplicity, let's assume a (built-in) Neo-Hookean material.

..  code-block:: python

    umat = fe.NeoHooke(mu=1, bulk=5)


FElupe provides an adopted :class:`felupe.IntegralFormAxisymmetric` class for the integration and the sparse matrix assemblage of axisymmetric problems. It uses the additional information (e.g. radial coordinates at integration points) stored in :class:`felupe.FieldAxisymmetric` to provide a consistent interface in comparison to default IntegralForms.

..  code-block:: python

    F = fe.math.identity(dudX) + dudX

    r = fe.IntegralFormAxisymmetric(umat.gradient(F), u, dA).assemble()
    K = fe.IntegralFormAxisymmetric(umat.hessian(F), u, dA).assemble()

To sum up, for axisymmetric problems use :class:`felupe.FieldAxisymmetric` in conjunction with :class:`felupe.IntegralFormAxisymmetric`. Of course, Mixed-field formulations may be applied on axisymmetric scenarios too.
