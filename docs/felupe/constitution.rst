Constitution
~~~~~~~~~~~~

This module provides constitutive material formulations.

**Linear-Elasticity**

.. currentmodule:: felupe

.. autosummary::

   LinearElastic
   LinearElasticPlaneStrain
   LinearElasticPlaneStress
   LinearElasticTensorNotation
   LinearElasticLargeStrain

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Hyperelasticity**

.. autosummary::

   NeoHooke
   ThreeFieldVariation

**Pseudo-Elasticity (Isotropic Damage)**

.. autosummary::

   OgdenRoxburgh

**Hyperelastic User-Materials with Automatic Differentation**

.. autosummary::
   
   Hyperelastic
   constitution.saint_venant_kirchhoff
   constitution.neo_hooke
   constitution.mooney_rivlin
   constitution.yeoh
   constitution.third_order_deformation
   constitution.ogden
   constitution.arruda_boyce
   constitution.extended_tube
   constitution.van_der_waals

**(Small) Strain-based User Materials**

.. autosummary::

   MaterialStrain
   constitution.linear_elastic
   constitution.linear_elastic_plastic_isotropic_hardening

**Deformation-Gradient-based User Materials**

.. autosummary::

   Material

**Kinematics**

.. autosummary::

   LineChange
   AreaChange
   VolumeChange

**Detailed API Reference**

.. autoclass:: fem.NeoHooke
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.OgdenRoxburgh
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.LinearElastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.LinearElasticLargeStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.constitution.LinearElasticTensorNotation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.LinearElasticPlaneStress
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.LinearElasticPlaneStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.LinearElasticPlasticIsotropicHardening
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.ThreeFieldVariation
   :members:
   :undoc-members:
   :inherited-members:

   .. autoclass:: fem.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.MaterialStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: fem.constitution.linear_elastic

.. autofunction:: fem.constitution.linear_elastic_plastic_isotropic_hardening

.. autoclass:: fem.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: fem.constitution.saint_venant_kirchhoff

.. autofunction:: fem.constitution.neo_hooke

.. autofunction:: fem.constitution.mooney_rivlin

.. autofunction:: fem.constitution.yeoh

.. autofunction:: fem.constitution.third_order_deformation

.. autofunction:: fem.constitution.ogden

.. autofunction:: fem.constitution.arruda_boyce

.. autofunction:: fem.constitution.extended_tube

.. autofunction:: fem.constitution.van_der_waals

.. autoclass:: fem.LineChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.AreaChange
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: fem.VolumeChange
   :members:
   :undoc-members:
   :inherited-members:
