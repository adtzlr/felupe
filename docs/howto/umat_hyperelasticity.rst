Isotropic Hyperelastic Materials
--------------------------------

User materials (Umat) based on the right Cauchy-Green deformation tensor, suitable for Total-Lagrangian isotropic hyperelastic material formulations, are to be created with :class:`~felupe.Hyperelastic`. Only the strain energy function must be defined. Both gradient and hessian are evaluated by forward-mode automatic differentiation. Therefore, only math-functions from :mod:`tensortrax.math` are supported. A user-defined function must be created with the argument and return values:

+----------+---------------+---------------------------------------+
| **Kind** |  **Symbol**   | **Description**                       |
+==========+===============+=======================================+
| Argument |       C       | right Cauchy-Green deformation tensor |
+----------+---------------+---------------------------------------+
| Return   |       W       | strain energy function                |
+----------+---------------+---------------------------------------+

..  code-block:: python

    import tensortrax.math as tm

    def strain_energy_function(C, **kwargs):
        return W

This function is further added as the ``fun`` argument of :class:`~felupe.Hyperelastic`.

..  code-block:: python
    
    import felupe as fem
    
    umat = fem.Hyperelastic(fun=strain_energy_function, **kwargs)

FElupe contains several reference implementations of hyperelastic user material
formulations, like

* :func:`~felupe.constitution.tensortrax.models.hyperelastic.neo_hooke`,
* :func:`~felupe.constitution.tensortrax.models.hyperelastic.mooney_rivlin`,
* :func:`~felupe.constitution.tensortrax.models.hyperelastic.yeoh` or
* :func:`~felupe.constitution.tensortrax.models.hyperelastic.ogden`.

A complete list of all available model formulations is available in the
:ref:`hyperelasticity <felupe-api-constitution-autodiff>` section of the API reference.
