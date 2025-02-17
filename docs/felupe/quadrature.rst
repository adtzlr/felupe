Quadrature
==========

This module contains quadrature (numeric integration) schemes for different finite element formulations. The integration points of a boundary-quadrature are located on the first edge for 2d elements and on the first face for 3d elements.

**Lines, Quads and Hexahedrons**

.. currentmodule:: felupe

.. autosummary::

   GaussLegendre
   GaussLegendreBoundary
   GaussLobatto
   GaussLobattoBoundary

**Triangles and Tetrahedrons**

.. autosummary::

   quadrature.Triangle
   quadrature.Tetrahedron
   TriangleQuadrature
   TetrahedronQuadrature

**Sphere**

.. autosummary::

   BazantOh

**Detailed API Reference**

.. autoclass:: felupe.quadrature.Scheme
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.GaussLegendre
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.GaussLegendreBoundary
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.GaussLobatto
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.GaussLobattoBoundary
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.quadrature.Triangle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.quadrature.Tetrahedron
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.BazantOh
   :members:
   :undoc-members:
   :show-inheritance:
