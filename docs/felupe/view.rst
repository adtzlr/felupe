View
~~~~

Visualization methods for meshes, mesh and field containers as well as solid bodies. The
scenes are drawn by a plotting backend, by default by `PyVista <https://pyvista.org/>`_
(``"pyvista"``). The plotting backend ``"webgl"`` draws the scenes by a WebGL viewer in
the web browser (in a self-contained HTML page), without PyVista.

.. currentmodule:: felupe

.. autosummary::

   view.Scene
   ViewMesh
   ViewField
   ViewSolid
   ViewXdmf
   view.select_surface_points
   view.select_edge_points

**Plotting Backends**

.. autosummary::

   set_plotting_backend
   get_plotting_backend
   view.register_plotting_backend
   view.PlottingBackend

**Detailed API Reference**

.. autoclass:: felupe.view.Scene

.. autofunction:: felupe.ViewMesh

.. autofunction:: felupe.ViewField

.. autofunction:: felupe.ViewSolid

.. autofunction:: felupe.ViewXdmf

.. autofunction:: felupe.view.select_surface_points

.. autofunction:: felupe.view.select_edge_points

.. autofunction:: felupe.set_plotting_backend

.. autofunction:: felupe.get_plotting_backend

.. autofunction:: felupe.view.register_plotting_backend

.. autoclass:: felupe.view.PlottingBackend
   :members:
