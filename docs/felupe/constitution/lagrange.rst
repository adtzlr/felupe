.. _felupe-api-constitution-lagrange:

Total- & Updated-Lagrange
~~~~~~~~~~~~~~~~~~~~~~~~

This page contains Total- and Updated-Lagrange material formulations with automatic differentiation using `tensortrax <https://github.com/adtzlr/tensortrax>`_. The material model formulations are defined by the first Piola-Kirchhoff stress tensor. Function-decorators are available to use Total-Lagrange and Updated-Lagrange material formulations in :class:`~felupe.MaterialAD`.

**Frameworks**

.. currentmodule:: felupe

.. autosummary::
   
   MaterialAD
   total_lagrange
   updated_lagrange

**Material Models for** :class:`~felupe.MaterialAD`

.. autosummary::

   morph

**Detailed API Reference**

.. autoclass:: felupe.MaterialAD
   :members:
   :undoc-members:
   :inherited-members:

.. autofunction:: felupe.morph

.. autofunction:: felupe.total_lagrange

.. autofunction:: felupe.updated_lagrange
