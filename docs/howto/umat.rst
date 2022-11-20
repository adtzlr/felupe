Small-Strain based User Materials
---------------------------------

User materials based on the incremental small-strain tensor, suitable for elastic-plastic material formulations, are to be created with :class:`felupe.UserMaterialStrain`. A user-defined function must be created with the arguments

+---------------+-----------------------------+
| **Arguments** | **Description**             |
+---------------+-----------------------------+
|      dε       | strain increment            |
+---------------+-----------------------------+
|      εn       | old strain tensor           |
+---------------+-----------------------------+
|      σn       | old stress tensor           |
+---------------+-----------------------------+
|      ζn       | list of old state variables |
+---------------+-----------------------------+

and must return:

+-------------+-----------------------------+
| **Returns** | **Description**             |
+-------------+-----------------------------+
|     dσdε    | tangent modulus             |
+-------------+-----------------------------+
|      σ      | new stress tensor           |
+-------------+-----------------------------+
|      ζ      | list of new state variables |
+-------------+-----------------------------+

..  code-block:: python

    def material(dε, εn, σn, ζn, **kwargs):
        return dσdε, σ, ζ

This function is further added as the ``material`` argument of :class:`felupe.UserMaterialStrain`. Optionally, the shapes of internal state variables may be passed.

..  code-block:: python
    
    import felupe as fem
    
    umat = fem.UserMaterialStrain(material=material, statevars=(0,), **kwargs)

FElupe contains two reference user materials, one for linear elastic materials and another one for linear elastic-plastic materials with isotropic hardening:

* :func:`felupe.constitution.linear_elastic`
* :func:`felupe.constitution.linear_elastic_plastic_isotropic_hardening`