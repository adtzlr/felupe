Quadrature
==========

This module contains quadrature (numeric integration) schemes for different finite element formulations. The integration points of a boundary-quadrature are located on the first edge for 2d elements and on the first face for 3d elements.

**Lines, Quads and Hexahedrons**

.. currentmodule:: felupe

.. autosummary::

   GaussLegendre
   GaussLegendreBoundary

**Triangles and Tetrahedrons**

.. autosummary::

   TriangleQuadrature
   TetrahedronQuadrature
   quadrature.Triangle
   quadrature.Tetrahedron

**Detailed API Reference**

.. autoclass:: fem.GaussLegendre
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fem.GaussLegendreBoundary
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fem.quadrature.Triangle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: fem.quadrature.Tetrahedron
   :members:
   :undoc-members:
   :show-inheritance: