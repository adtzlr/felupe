.. _felupe-api-constitution:

Constitution
~~~~~~~~~~~~

This module provides :class:`constitutive material <felupe.ConstitutiveMaterial>` formulations. In FElupe, a constitutive material definition, or so-called ``umat`` (user material), is a class with methods for evaluating gradients and hessians of the strain energy density function with respect to the defined fields in the field container, where the gradient of the first (displacement) field is passed as the deformation gradient. For all following fields, the field values (no gradients) are provided. An attribute ``x=[np.zeros(statevars_shape)]`` has to be added to the class to define the shape of optional state variables. For reasons of performance, FElupe passes the field gradients and values *all at once*, e.g. the deformation gradient is of shape ``(3, 3, q, c)``, where ``q`` refers to the number of quadrature points per cell and ``c`` to the number of cells. These last two axes are the so-called *trailing axes*. Math-functions from :ref:`felupe.math <felupe-api-math>` all support the operation on trailing axes. The constitutive material definition class should be inherited from :class:`~felupe.ConstitutiveMaterial` in order to provide force-stretch curves for elementary deformations. Take this code-block as a template for a two-field :math:`(\boldsymbol{u}, p)` formulation with the old displacement gradient as a state variable:

..  code-block:: python

    import numpy as np
    import felupe as fem

    # math-functions which support trailing axes
    from felupe.math import det, dya, identity, transpose, inv

    class MyMaterialFormulation(fem.ConstitutiveMaterial):

        def __init__(self):
            # provide the shape of state variables without trailing axes
            # values are ignored - state variables are always initiated with zeros
            self.x = [np.zeros((3, 3))]

        def gradient(self, x):
            "Gradients of the strain energy density function."

            # extract variables
            F, p, statevars = x[0], x[1], x[-1]

            # user code
            dWdF = None  # first Piola-Kirchhoff stress tensor
            dWdp = None

            # update state variables
            # example: the displacement gradient
            statevars_new = F - identity(F)

            return [dWdF, dWdp, statevars_new]

        def hessian(self, x, **kwargs):
            "Hessians of the strain energy density function."

            # extract variables
            F, p, statevars = x[0], x[1], x[-1]

            # user code
            d2WdFdF = None  # fourth-order elasticity tensor
            d2WdFdp = None
            d2Wdpdp = None

            # upper-triangle items of the hessian
            return [d2WdFdF, d2WdFdp, d2Wdpdp]

    umat = MyMaterialFormulation()

There are many different pre-defined constitutive material formulations available, including definitions for linear-elasticity, small-strain plasticity, hyperelasticity or pseudo-elasticity. The generation of user materials may be simplified when using frameworks for user-defined functions, like hyperelasticity (with automatic differentiation) or a small-strain based framework with state variables. The most general case is given by a framework with functions for the evaluation of stress and elasticity tensors in terms of the deformation gradient.

**View Force-Stretch Curves on Elementary Deformations**

.. currentmodule:: felupe

.. autosummary::

   ConstitutiveMaterial
   ViewMaterial
   ViewMaterialIncompressible

**Merge Constitutive Materials**

.. autosummary::

   CompositeMaterial

**Linear-Elasticity**

.. autosummary::

   LinearElastic
   LinearElasticPlaneStrain
   LinearElasticPlaneStress
   constitution.LinearElasticTensorNotation
   LinearElasticLargeStrain

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Hyperelasticity**

.. autosummary::

   NeoHooke
   NeoHookeCompressible
   Volumetric

**Hyperelastic Three-Field-Formulations** :math:`(\boldsymbol{u}, p, \bar{J})`

.. autosummary::

   ThreeFieldVariation
   NearlyIncompressible

**Pseudo-Elasticity (Isotropic Damage)**

.. autosummary::

   OgdenRoxburgh

**Hyperelastic User-Materials with Automatic Differentation**

.. autosummary::
   
   Hyperelastic
   MaterialAD

**Material Model Formulations (Strain Energy Functions) for** :class:`~felupe.Hyperelastic`

.. autosummary::

   saint_venant_kirchhoff
   neo_hooke
   mooney_rivlin
   yeoh
   third_order_deformation
   ogden
   arruda_boyce
   extended_tube
   van_der_waals
   alexander
   finite_strain_viscoelastic

**(Small) Strain-based User Materials**

.. autosummary::

   MaterialStrain
   linear_elastic
   linear_elastic_plastic_isotropic_hardening

**Deformation-Gradient-based User Materials**

.. autosummary::

   Material

**Kinematics**

.. autosummary::

   LineChange
   AreaChange
   VolumeChange

**Detailed API Reference**

.. autoclass:: felupe.ConstitutiveMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ViewMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ViewMaterialIncompressible
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.CompositeMaterial
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.NeoHooke
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.NeoHookeCompressible
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Volumetric
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.OgdenRoxburgh
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticLargeStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.LinearElasticTensorNotation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlaneStress
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlaneStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlasticIsotropicHardening
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ThreeFieldVariation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.NearlyIncompressible
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MaterialStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.linear_elastic

.. autofunction:: felupe.linear_elastic_plastic_isotropic_hardening

.. autoclass:: felupe.MaterialAD
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.saint_venant_kirchhoff

.. autofunction:: felupe.neo_hooke

.. autofunction:: felupe.mooney_rivlin

.. autofunction:: felupe.yeoh

.. autofunction:: felupe.third_order_deformation

.. autofunction:: felupe.ogden

.. autofunction:: felupe.arruda_boyce

.. autofunction:: felupe.extended_tube

.. autofunction:: felupe.van_der_waals

.. autofunction:: felupe.alexander

.. autofunction:: felupe.finite_strain_viscoelastic

.. autoclass:: felupe.LineChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.AreaChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.VolumeChange
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.lame_converter
