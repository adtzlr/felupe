.. _felupe-api-constitution-lagrange:

Total- & Updated-Lagrange
~~~~~~~~~~~~~~~~~~~~~~~~

This page contains Total- and Updated-Lagrange material formulations with automatic differentiation using :mod:`tensortrax.math`. The material model formulations are defined by the first Piola-Kirchhoff stress tensor. Function-decorators are available to use Total-Lagrange and Updated-Lagrange material formulations in :class:`~felupe.MaterialAD`.

.. figure:: /_static/logo_tensortrax.png
   :align: center

   Differentiable Tensors based on NumPy Arrays.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   MaterialAD
   total_lagrange
   updated_lagrange

**Material Models for** :class:`~felupe.MaterialAD`

.. autosummary::

   morph
   morph_representative_directions

**Detailed API Reference**

.. autoclass:: felupe.MaterialAD
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.morph

.. autofunction:: felupe.morph_representative_directions

.. autofunction:: felupe.total_lagrange

.. autofunction:: felupe.updated_lagrange
