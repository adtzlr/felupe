Two-dimensional Problems
------------------------

For plane-strain and axisymmetric problems a vector-valued field has to be created for
the two-dimensional in-plane displacement components.

..  tab:: Axisymmetric

    ..  code-block:: python

        import felupe as fem

        mesh = fem.Rectangle(n=3)
        region = fem.RegionQuad(mesh)
        displacement = fem.FieldAxisymmetric(region, dim=2)

..  tab:: Plane-Strain

    ..  code-block:: python

        import felupe as fem

        mesh = fem.Rectangle(n=3)
        region = fem.RegionQuad(mesh)
        displacement = fem.FieldPlaneStrain(region, dim=2)
        

The 3x3 deformation gradient for axisymmetric and plane-strain two-dimensional problems
is obtained by the :meth:`~felupe.FieldAxisymmetric.grad` or
:meth:`~felupe.FieldAxisymmetric.extract` methods (same for
:class:`~felupe.FieldPlaneStrain`). For these two-dimensional fields the gradient is
modified to return a three-dimensional gradient.

..  code-block:: python

    field = fem.FieldContainer([displacement])
    F = field.extract(grad=True, sym=False, add_identity=True)

For simplicity, let's use the isotropic hyperelastic :class:`~felupe.NeoHooke` material
model formulation.

..  code-block:: python

    umat = fem.NeoHooke(mu=1, bulk=5)

..  note::

    Internally, FElupe provides an adopted low-level
    :class:`~felupe.assembly.IntegralFormAxisymmetric` class for the integration and the
    sparse matrix assemblage of axisymmetric problems. It uses the additional
    information (e.g. radial coordinates at integration points) stored in
    :class:`~felupe.FieldAxisymmetric` to provide a consistent interface in comparison
    to :class:`~felupe.assembly.IntegralFormCartesian`. The top-level
    :class:`~felupe.IntegralForm` chooses the appropriate low-level integral form based
    on the kind of field inside the field container.

..  code-block:: python

    dA = region.dV

    r = fem.IntegralForm(umat.gradient(F), field, dA).assemble()
    K = fem.IntegralForm(umat.hessian(F), field, dA, field).assemble()

To sum up, for axisymmetric problems use :class:`~felupe.FieldAxisymmetric` and for
plane-strain problems use :class:`~felupe.FieldPlaneStrain`. Of course, mixed-field
formulations may also be used with axisymmetric or plane-strain (displacement) fields.
