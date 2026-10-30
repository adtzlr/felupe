View
~~~~

Visualization methods for meshes, mesh and field containers as well as solid bodies. The
scenes are drawn by a plotting backend, by default by `PyVista <https://pyvista.org/>`_
(``"pyvista"``). The plotting backend ``"webgl"`` draws the scenes by a WebGL viewer in
the web browser (in a self-contained HTML page), without PyVista. The plotting backend
``"matplotlib"`` draws the scenes on the axes of a figure of matplotlib, without PyVista,
e.g. for off-screen screenshots in vector formats (PDF or SVG).

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
   view.WebGLPlotter
   view.MatplotlibPlotter

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

.. autoclass:: felupe.view.WebGLPlotter
   :members: show, export_html, widget, html

.. autoclass:: felupe.view.MatplotlibPlotter
   :members: show, screenshot, savefig, render, enable_parallel_projection, close
