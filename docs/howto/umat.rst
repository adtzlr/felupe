Small-Strain based User Materials
---------------------------------

User materials based on the incremental small-strain tensor, suitable for elastic-plastic material formulations, are to be created with :class:`felupe.UserMaterialStrain`. A function which takes the incremental strain tensor and returns both the stress tensor and the tangent modulus must be created with the following signature.

..  code-block:: python

    import felupe as fem

    def material(dε, εn, σn, ζn, **kwargs):
    """User-defined material formulation.

    Arguments
    ---------
    dε : ndarray
        Strain increment.
    εn : ndarray
        Old strain tensor.
    σn : ndarray
        Old stress tensor.
    ζn : list
        List of old state variables.
    """
    
    # evaluate the tangent modulus dσdε,
    # the stress tensor σ and the state variables ζ

    return dσdε, σ, ζ

This function is further added as the ``material`` argument of :class:`felupe.UserMaterialStrain`. Optionally, the shapes of internal state variables may be passed.

..  code-block:: python
    
    umat = fem.UserMaterialStrain(material=material, statevars=(0,), **kwargs)

FElupe consists two user materials, one for linear elastic materials and another one for linear elastic-plastic materials with isotropic hardening:

* :func:`felupe.constitution.linear_elastic`
* :func:`felupe.constitution.linear_elastic_plastic_isotropic_hardening`