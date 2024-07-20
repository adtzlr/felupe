.. _felupe-api-mesh:

Mesh
====

This module contains meshing-related classes and functions. Standalone mesh-tools (functions) are also available as mesh-methods.

**Core**

.. currentmodule:: felupe

.. autosummary::

   Mesh
   MeshContainer

**Geometries**

.. autosummary::

   Point
   mesh.Line
   Rectangle
   Cube
   Grid
   Circle
   mesh.Triangle
   mesh.RectangleArbitraryOrderQuad
   mesh.CubeArbitraryOrderHexahedron

**Tools**

.. autosummary::

   mesh.expand
   mesh.translate
   mesh.rotate
   mesh.revolve
   mesh.sweep
   mesh.mirror
   mesh.merge_duplicate_points
   mesh.merge_duplicate_cells
   mesh.concatenate
   mesh.runouts
   mesh.triangulate
   mesh.convert
   mesh.collect_edges
   mesh.collect_faces
   mesh.collect_volumes
   mesh.add_midpoints_edges
   mesh.add_midpoints_faces
   mesh.add_midpoints_volumes
   mesh.flip
   mesh.fill_between
   mesh.dual
   mesh.stack
   mesh.read
   mesh.interpolate_line

**Detailed API Reference**

.. autoclass:: felupe.Mesh
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.MeshContainer
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: felupe.Point
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.mesh.Line
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.Rectangle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.Cube
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.Grid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.Circle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.mesh.Triangle
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.mesh.RectangleArbitraryOrderQuad
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: felupe.mesh.CubeArbitraryOrderHexahedron
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: felupe.mesh
   :members: expand, translate, rotate, revolve, sweep, mirror, concatenate, runouts, triangulate, convert, collect_edges, collect_faces, collect_volumes, add_midpoints_edges, add_midpoints_faces, add_midpoints_volumes, flip, fill_between, dual, stack, merge_duplicate_points, merge_duplicate_cells, read, interpolate_line
