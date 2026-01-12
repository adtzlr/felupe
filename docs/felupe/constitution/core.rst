.. _felupe-api-constitution-core:

Models
~~~~~~

This page contains the core (hard-coded) constitutive material model formulations (not using automatic differentiation) for linear-elasticitiy, small-strain plasticity, hyperelasticity and pseudo-elasticity.

.. currentmodule:: felupe

**Poisson Equation**

.. autosummary::

   Laplace

**Linear-Elasticity**

.. autosummary::

   LinearElastic
   LinearElastic1D
   LinearElasticPlaneStress
   constitution.LinearElasticPlaneStrain
   constitution.LinearElasticTensorNotation
   LinearElasticLargeStrain
   LinearElasticOrthotropic

**Plasticity**

.. autosummary::

   LinearElasticPlasticIsotropicHardening

**Hyperelasticity**

.. autosummary::

   NeoHooke
   NeoHookeCompressible
   OgdenRoxburgh

**Mixed-Field Formulations** :math:`(\boldsymbol{u}, p, \bar{J})`

.. autosummary::

   ThreeFieldVariation
   NearlyIncompressible

**Strain-based Materials**

.. autosummary::

   MaterialStrain
   linear_elastic
   linear_elastic_viscoelastic
   linear_elastic_plastic_isotropic_hardening

**Detailed API Reference**

.. autoclass:: felupe.Laplace
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElastic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElastic1D
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.linear_elastic

.. autoclass:: felupe.LinearElasticLargeStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.LinearElasticTensorNotation
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.constitution.LinearElasticPlaneStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlaneStress
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.LinearElasticPlasticIsotropicHardening
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.linear_elastic_plastic_isotropic_hardening

.. autoclass:: felupe.LinearElasticOrthotropic
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MaterialStrain
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.NearlyIncompressible
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

.. autoclass:: felupe.OgdenRoxburgh
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.ThreeFieldVariation
   :members:
   :undoc-members:
   :inherited-members:


