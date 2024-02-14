Small-Strain based Materials
----------------------------

User materials (Umat) based on the incremental small-strain tensor, e.g. suitable for linear elastic-plastic material formulations, are to be created with :class:`~felupe.MaterialStrain`. A user-defined function must be created with the arguments and must return:

+----------+---------------+---------------------------------------+
| **Kind** |  **Symbol**   | **Description**                       |
+==========+===============+=======================================+
| Argument |      dε       | strain increment                      |
+----------|---------------+---------------------------------------+
| Argument |      εn       | old strain tensor                     |
+----------|---------------+---------------------------------------+
| Argument |      σn       | old stress tensor                     |
+----------|---------------+---------------------------------------+
| Argument |      ζn       | list of old state variables           |
+----------|---------------+---------------------------------------+
| Return   |      σ        | tangent modulus                       |
+----------|---------------+---------------------------------------+
| Return   |      ζ        | list of new state variables           |
+----------|---------------+---------------------------------------+

..  code-block:: python

    def material(dε, εn, σn, ζn, **kwargs):
        return dσdε, σ, ζ

This function is further added as the ``material`` argument of :class:`~felupe.MaterialStrain`. If the material makes use of state variables, the shapes of these internal state variables must be provided.

..  code-block:: python
    
    import felupe as fem
    
    umat = fem.MaterialStrain(material=material, statevars=(0,), **kwargs)

FElupe contains two reference user materials, one for linear elastic materials and another one for linear elastic-plastic materials with isotropic hardening:

* :func:`~felupe.linear_elastic`
* :func:`~felupe.linear_elastic_plastic_isotropic_hardening`
