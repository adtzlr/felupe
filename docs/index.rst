FElupe documentation
====================

FElupe is a Python 3.8+ 🐍 finite element analysis package 📦 focussing on the formulation and numerical solution of nonlinear problems in continuum mechanics 🔧 of solid bodies 🚂. Its name is a combination of FE (finite element) and the german word *Lupe* 🔍 (magnifying glass) as a synonym for getting an insight 📖 how a finite element analysis code 🧮 looks like under the hood 🕳️.

.. grid::

   .. grid-item-card:: 🏃 Getting Started
      :link: tutorials
      :link-type: ref

      New to FElupe? The Beginner's Guide contains an introduction to the concept of FElupe.

   .. grid-item-card:: 📖 API reference
      :link: felupe-api
      :link-type: ref

      The reference guide contains a detailed description of the FElupe API. It describes how the methods work and which parameters can be used. Requires you to have an understanding of the key concepts.

.. grid::

   .. grid-item-card:: ☎ How-To
      :link: how-to
      :link-type: ref

      Step-by-step guides for specific tasks.

   .. grid-item-card:: 📚 Examples
      :link: examples
      :link-type: ref

      A gallery of examples.



.. admonition:: Highlights
   :class: admonition
   
   + high-level :ref:`finite-element-analysis API <felupe-api>`

   + flexible building blocks for :ref:`finite element assembly <felupe-api-assembly>`
   
   + hyperelastic :class:`integral (weak) forms <felupe.IntegralForm>`
   
   + straight-forward definition of :class:`mixed-fields <felupe.FieldsMixed>`
   
   + :class:`nearly-incompressible hyperelastic solid bodies <felupe.SolidBodyNearlyIncompressible>`

Installation
------------
Install Python, fire up 🔥 a terminal and run 🏃

.. image:: https://img.shields.io/pypi/v/felupe.svg
   :target: https://pypi.python.org/pypi/felupe/

.. code-block:: shell

   pip install felupe[all]

where ``[all]`` installs all optional dependencies. FElupe has minimal requirements, all available at PyPI supporting all platforms.

* `numpy <https://github.com/numpy/numpy>`_ for array operations
* `scipy <https://github.com/scipy/scipy>`_ for sparse matrices
* `tensortrax <https://github.com/adtzlr/tensortrax>`_ for automatic differentiation

In order to make use of all features of FElupe 💎💰💍👑💎, it is suggested to install all optional dependencies.

* `einsumt <https://github.com/mrkwjc/einsumt>`_ for parallel assembly
* `h5py <https://github.com/h5py/h5py>`_ for XDMF result files
* `matplotlib <https://github.com/matplotlib/matplotlib>`_ for plotting graphs
* `meshio <https://github.com/nschloe/meshio>`_ for mesh-related I/O
* `pyvista <https://github.com/pyvista/pyvista>`_ for interactive visualizations
* `tqdm <https://github.com/tqdm/tqdm>`_ for showing progress bars during job evaluations

Extension Packages
------------------

The capabilities of FElupe may be enhanced with extension packages created by the community.

+-----------------------------------------------------------+---------------------------------------------------------+
| Package                                                   | Description                                             |
+===========================================================+=========================================================+
| `hyperelastic <https://github.com/adtzlr/hyperelastic>`_  | Constitutive hyperelastic material formulations         |
+-----------------------------------------------------------+---------------------------------------------------------+
| `matadi <https://github.com/adtzlr/matadi>`_              | Material Definition with Automatic Differentiation (AD) |
+-----------------------------------------------------------+---------------------------------------------------------+
| `tensortrax <https://github.com/adtzlr/tensortrax>`_      | Math on Hyper-Dual Tensors with Trailing Axes           |
|                                                           | (bundled with FElupe)                                   |
+-----------------------------------------------------------+---------------------------------------------------------+
| `feplot <https://github.com/ZAARAOUI999/feplot>`_         | A visualization tool for FElupe                         |
+-----------------------------------------------------------+---------------------------------------------------------+

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   
   tutorial
   examples
   howto

.. toctree::
   :maxdepth: 1
   :caption: Reference:
   
   felupe

License
-------

FElupe - Finite Element Analysis (C) 2021-2024 Andreas Dutzler, Graz (Austria).

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see `<https://www.gnu.org/licenses/>`_.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
