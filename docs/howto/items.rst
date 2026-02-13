Create Custom Items
~~~~~~~~~~~~~~~~~~~
Items are supported to be used in a :class:`~felupe.Step` and in
:func:`~felupe.newtonrhapson`. They provide methods to assemble a vector and a sparse
matrix. An item has to be created as a class which takes a
:class:`~felupe.FieldContainer` as input argument. In its ``__init__(self, field)``
method, the helpers :class:`~felupe.mechanics.Assemble` and
:class:`~felupe.mechanics.Results` have to be evaluated and added as attributes.
Internal methods which assemble the sparse vector and matrix, optionally with an updated
:class:`~felupe.FieldContainer` provided by the `field` argument, have to be linked to
:class:`~felupe.mechanics.Assemble`. 

..  note::
    
    This is only a minimal working example for an item. For more details, see the
    sources of :class:`~felupe.SolidBody` or
    :class:`~felupe.SolidBodyNearlyIncompressible`.

..  pyvista-plot::
    :context:
    
    from scipy.sparse import csr_matrix
    import felupe as fem


    class MyItem:
        def __init__(self, field):
            self.field = field
            self.assemble = fem.mechanics.Assemble(vector=self._vector, matrix=self._matrix)
            self.results = fem.mechanics.Results()

        def _vector(self, field=None, **kwargs):
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

        def _matrix(self, field=None, **kwargs):
            return csr_matrix(([0.0], ([0], [0])), shape=(1, 1))

This item is now added to a basic script.

..  note::
    
    The vector- and matrix-shapes are automatically increased to match the shape of the
    other items if necessary. Hence, a sparse matrix with shape ``(1, 1)`` is a valid
    choice for a vector or a matrix filled with zeros.

..  pyvista-plot::
    :context:

    region = fem.RegionHexahedron(mesh=fem.Cube(n=3))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    boundaries = fem.dof.uniaxial(field, clamped=True, move=1.0, return_loadcase=False)

    solid = fem.SolidBody(umat=fem.NeoHooke(mu=1, bulk=2), field=field)
    my_item = MyItem(field=field)

    step = fem.Step(items=[solid, my_item], boundaries=boundaries)
    job = fem.Job(steps=[step]).evaluate()
