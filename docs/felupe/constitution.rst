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

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Hyperelasticity**

.. autosummary::

   LinearElasticLargeStrain
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

.. autoclass:: felupe.NeoHooke
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

   .. autoclass:: felupe.Material
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MaterialStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.linear_elastic

.. autofunction:: felupe.constitution.linear_elastic_plastic_isotropic_hardening

.. autoclass:: felupe.Hyperelastic
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.constitution.saint_venant_kirchhoff

.. autofunction:: felupe.constitution.neo_hooke

.. autofunction:: felupe.constitution.mooney_rivlin

.. autofunction:: felupe.constitution.yeoh

.. autofunction:: felupe.constitution.third_order_deformation

.. autofunction:: felupe.constitution.ogden

.. autofunction:: felupe.constitution.arruda_boyce

.. autofunction:: felupe.constitution.extended_tube

.. autofunction:: felupe.constitution.van_der_waals

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
