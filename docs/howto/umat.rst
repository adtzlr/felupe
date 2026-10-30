Define Strain-Based Materials
-----------------------------

A user material (``umat``), based on the incremental strain tensor, e.g. suitable
for linear elastic-plastic material formulations, is provided by
:class:`~felupe.MaterialStrain`. A user-defined function must be created with the
arguments and must return:

+----------+---------------+---------------------------------------+
| **Kind** |  **Symbol**   | **Description**                       |
+==========+===============+=======================================+
| Argument |      dε       | strain increment                      |
+----------+---------------+---------------------------------------+
| Argument |      εn       | old strain tensor                     |
+----------+---------------+---------------------------------------+
| Argument |      σn       | old stress tensor                     |
+----------+---------------+---------------------------------------+
| Argument |      ζn       | list of old state variables           |
+----------+---------------+---------------------------------------+
| Return   |     dσdε      | tangent modulus                       |
+----------+---------------+---------------------------------------+
| Return   |      σ        | new stress tensor                     |
+----------+---------------+---------------------------------------+
| Return   |      ζ        | list of new state variables           |
+----------+---------------+---------------------------------------+

..  code-block:: python

    def material(dε, εn, σn, ζn, **kwargs):
        return dσdε, σ, ζ

This function is further added as the ``material`` argument of
:class:`~felupe.MaterialStrain`. If the material makes use of state variables, the
shapes of these internal state variables must be provided. By default, the small-strain
framework is used. Optionally, this may be changed to a Total-Lagrange, a
co-rotational or a Seth-Hill strain framework. The Seth-Hill framework requires the
strain exponent :math:`m` of the Seth-Hill strain tensors, e.g.
``framework="seth-hill", strain_exponent=0`` for the logarithmic strain tensor.

+------------------+--------------------------------+-----------------------------------+
| **Framework**    | **Strain** (argument)          | **Stress** (return)               |
+==================+================================+===================================+
| small-strain     | small-strain tensor            | Cauchy stress tensor              |
+------------------+--------------------------------+-----------------------------------+
| total-lagrange   | Green-Lagrange strain tensor   | second Piola-Kirchhoff stress     |
|                  | (Seth-Hill, :math:`m = 2`)     |                                   |
+------------------+--------------------------------+-----------------------------------+
| co-rotational    | Biot strain tensor             | Biot stress tensor                |
|                  | (Seth-Hill, :math:`m = 1`)     |                                   |
+------------------+--------------------------------+-----------------------------------+
| seth-hill        | Seth-Hill strain tensor        | work-conjugate stress tensor      |
|                  | (strain exponent :math:`m`)    |                                   |
+------------------+--------------------------------+-----------------------------------+

..  code-block:: python

    import felupe as fem

    umat = fem.MaterialStrain(
        material=material, 
        statevars=(0,), 
        framework="small-strain",  # "total-lagrange", "co-rotational", "seth-hill"
        **kwargs,
    )

FElupe contains three reference small-strain user materials, one for linear elastic
materials, one for linear elastic-plastic materials with isotropic hardening and one for
linear viscoelastic materials:

* :func:`~felupe.linear_elastic`
* :func:`~felupe.linear_elastic_plastic_isotropic_hardening`
* :func:`~felupe.linear_elastic_viscoelastic`

The linear viscoelastic material is time-dependent, i.e. it takes the time increment as
an input, see :doc:`umat_time`.
