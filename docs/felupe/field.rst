Field
=====

A :class:`~felupe.FieldContainer` with pre-defined fields is created with:

.. currentmodule:: felupe

.. autosummary::

   FieldsMixed

A :class:`~felupe.FieldContainer` is created with a list of one or more fields.

..  code-block:: python
    
    import felupe as fem
    
    region = fem.RegionHexahedron(fem.Cube())
    displacement = fem.Field(region)
    pressure = fem.FieldDual(region)
    
    field = fem.FieldContainer([displacement, pressure])
    
    # equivalent way to create a field container
    field = displacement & pressure

.. autosummary::

   FieldContainer

Available kinds of fields:

.. autosummary::

   Field
   FieldAxisymmetric
   FieldPlaneStrain
   FieldDual

**Detailed API Reference**

.. autoclass:: felupe.FieldsMixed
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.FieldContainer
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Field
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.FieldAxisymmetric
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.FieldPlaneStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.FieldDual
   :members:
   :undoc-members:
   :inherited-members:
