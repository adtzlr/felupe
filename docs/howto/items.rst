Create Custom Items
~~~~~~~~~~~~~~~~~~~
Items are supported to be used in a :class:`~felupe.Step` and in
:func:`~felupe.newtonraphson`. They provide methods to assemble a vector and a sparse
matrix. An item has to be created as a class which takes a
:class:`~felupe.FieldContainer` as input argument. In its ``__init__(self, field)``
method, the helpers :class:`~felupe.mechanics.Assemble` and
:class:`~felupe.mechanics.Results` have to be evaluated and added as attributes.
Internal methods which assemble the sparse vector and matrix, optionally with an updated
:class:`~felupe.FieldContainer` provided by the `field` argument, have to be linked to
:class:`~felupe.mechanics.Assemble`. 

..  note::
    
    This is a minimal working example for an item. For more details, see the sources of
    :class:`~felupe.SolidBody` or :class:`~felupe.SolidBodyNearlyIncompressible`. It is
    considered good practice to add the parameters of the item as attributes to the
    :class:`~felupe.mechanics.Results`, so that they are automatically available in the
    post-processing. To ramp parameters within a step, the item has to be updated with
    its ``update()`` method, which is called by the step at the beginning of each
    substep. If more parameters have to be updated, it is recommended to add a separate
    method for each parameter with  the naming convention ``_update_my_parameter()``.
    Then, :class:`~felupe.mechanics.UpdateItem` can be used to update the item with the
    syntax ``my_item["my_parameter"] = my_parameter_value`` in a step.


..  pyvista-plot::
    :context:
    
    from scipy.sparse import csr_matrix
    from felupe.mechanics import Assemble, Results, UpdateItem


    class MyItem:
        def __init__(self, field, my_parameter=1.0, my_other_parameter=2.0):
            self.field = field
            self.assemble = Assemble(vector=self._vector, matrix=self._matrix)
            self.results = Results()

            self.results.my_parameter = my_parameter
            self.results.my_other_parameter = my_other_parameter
        
        def update(self, my_parameter):
            self._update_my_parameter(my_parameter)
        
        def _update_my_parameter(self, my_parameter):
            self.results.my_parameter = my_parameter
        
        def _update_my_other_parameter(self, my_other_parameter):
            self.results.my_other_parameter = my_other_parameter

        def __getitem__(self, key):
            return UpdateItem(self, key)

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
    my_item = MyItem(field=field, my_parameter=1.0, my_other_parameter=2.0)

    table = fem.math.linsteps([0.0, 1.0], num=10)

    # ramp only one parameter, while the other one stays constant at its initial value
    ramp = {my_item: table}  # here, by default, "my_parameter" will be ramped

    # ramp one or more parameters simultaneously
    ramp = {
        my_item["my_parameter"]: table,
        my_item["my_other_parameter"]: 1.0 + table,
    }

    step = fem.Step(items=[solid, my_item], ramp=ramp, boundaries=boundaries)
    job = fem.Job(steps=[step]).evaluate()