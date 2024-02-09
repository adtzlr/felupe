Isotropic Hyperelastic Materials
--------------------------------

User materials (Umat) based on the right Cauchy-Green deformation tensor, suitable for Total-Lagrange isotropic hyperelastic material formulations, are to be created with :class:`~felupe.Hyperelastic`. Only the strain energy function must be defined. Both gradient and hessian are evaluated by forward-mode automatic differentiation. Therefore, only math-functions from `tensortrax.math <https://github.com/adtzlr/tensortrax>`_ are supported. A user-defined function must be created with the argument and return values:

+----------+---------------+---------------------------------------+
| **Kind** |  **Symbol**   | **Description**                       |
+==========+===============+=======================================+
| Argument |       C       | right Cauchy-Green deformation tensor |
+----------|---------------+---------------------------------------+
| Return   |       W       | strain energy function                |
+----------|---------------+---------------------------------------+

..  code-block:: python

    import tensortrax.math as tm

    def strain_energy_function(C):
        return W

This function is further added as the ``fun`` argument of :class:`~felupe.Hyperelastic`. Optionally, the evaluation may be performed in parallel (threaded).

..  code-block:: python
    
    import felupe as fem
    
    umat = fem.Hyperelastic(fun=strain_energy_function, parallel=False, **kwargs)

FElupe contains several reference implementations of hyperelastic user material formulations

* :func:`~felupe.constitution.saint_venant_kirchhoff`
* :func:`~felupe.constitution.neo_hooke`
* :func:`~felupe.constitution.mooney_rivlin`
* :func:`~felupe.constitution.yeoh`
* :func:`~felupe.constitution.third_order_deformation`
* :func:`~felupe.constitution.ogden`
* :func:`~felupe.constitution.arruda_boyce`
* :func:`~felupe.constitution.extended_tube`
* :func:`~felupe.constitution.van_der_waals`

as well as a function decorator for the multiplicative isochoric-volumetric split of the Deformation Gradient.

* :func:`~felupe.constitution.isochoric_volumetric_split`
