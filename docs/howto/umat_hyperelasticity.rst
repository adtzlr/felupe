Isotropic Hyperelastic Materials
--------------------------------

User materials (Umat) based on the right Cauchy-Green deformation tensor, suitable for Total-Lagrangian isotropic hyperelastic material formulations, are to be created with :class:`~felupe.Hyperelastic`. Only the strain energy function must be defined. Both gradient and hessian are evaluated by forward-mode automatic differentiation. Therefore, only math-functions from `tensortrax.math <https://github.com/adtzlr/tensortrax>`_ are supported. A user-defined function must be created with the argument and return values:

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

FElupe contains several reference implementations of hyperelastic user material formulations

* :func:`~felupe.saint_venant_kirchhoff`
* :func:`~felupe.neo_hooke`
* :func:`~felupe.mooney_rivlin`
* :func:`~felupe.yeoh`
* :func:`~felupe.third_order_deformation`
* :func:`~felupe.ogden`
* :func:`~felupe.arruda_boyce`
* :func:`~felupe.extended_tube`
* :func:`~felupe.van_der_waals`

as well as a function decorator for the multiplicative isochoric-volumetric split of the Deformation Gradient.

* :func:`~felupe.isochoric_volumetric_split`
