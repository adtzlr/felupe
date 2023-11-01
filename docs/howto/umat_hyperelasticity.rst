Isotropic Hyperelastic Materials
--------------------------------

User materials (Umat) based on the right Cauchy-Green deformation tensor, suitable for Total-Lagrange isotropic hyperelastic material formulations, are to be created with :class:`fem.Hyperelastic`. Only the strain energy function must be defined. Both gradient and hessian are evaluated by forward-mode automatic differentiation. Therefore, only math-functions from `tensortrax.math` are supported. A user-defined function must be created with the arguments

+---------------+---------------------------------------+
| **Arguments** | **Description**                       |
+===============+=======================================+
|       C       | right Cauchy-Green deformation tensor |
+---------------+---------------------------------------+

and must return:

+-------------+------------------------+
| **Returns** | **Description**        |
+=============+========================+
|      W      | strain energy function |
+-------------+------------------------+

..  code-block:: python

    import tensortrax.math as tm

    def strain_energy_function(C):
        return W

This function is further added as the ``fun`` argument of :class:`fem.Hyperelastic`. Optionally, the evaluation may be performed in parallel (threaded).

..  code-block:: python
    
    import felupe as fem
    
    umat = fem.Hyperelastic(fun=strain_energy_function, parallel=False, **kwargs)

FElupe contains several reference implementations of hyperelastic user material formulations

* :func:`fem.constitution.saint_venant_kirchhoff`
* :func:`fem.constitution.neo_hooke`
* :func:`fem.constitution.mooney_rivlin`
* :func:`fem.constitution.yeoh`
* :func:`fem.constitution.third_order_deformation`
* :func:`fem.constitution.ogden`
* :func:`fem.constitution.arruda_boyce`
* :func:`fem.constitution.extended_tube`
* :func:`fem.constitution.van_der_waals`

as well as a function decorator for the multiplicative isochoric-volumetric split of the Deformation Gradient.

* :func:`fem.constitution.isochoric_volumetric_split`
