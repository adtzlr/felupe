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

      The reference guide contains a detailed description of the FElupe API. It describes how the methods work and which parameters can be used. Requires an understanding of the key concepts.

.. grid::

   .. grid-item-card:: ☎ How-To
      :link: how-to
      :link-type: ref

      Step-by-step guides for specific tasks or problems with a focus on practical usability instead of completeness. Requires an understanding of the key concepts.

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

Getting Started
---------------
This tutorial covers the essential high-level parts of creating and solving problems with FElupe. As an introductory example 👨‍🏫, a quarter model of a solid cube with hyperelastic material behaviour is subjected to a uniaxial elongation applied at a clamped end-face.

First, let’s import FElupe and create a meshed cube out of hexahedron cells with a given number of points per axis. A numeric region, pre-defined for hexahedrons, is created on the mesh. A vector-valued displacement field is initiated on the region. Next, a field container is created on top of this field.

A uniaxial load case is applied on the displacement field stored inside the field container. This involves setting up symmetry planes as well as the absolute value of the prescribed displacement at the mesh-points on the right-end face of the cube. The right-end face is *clamped* 🛠️: only displacements in direction *x* are allowed. The dict of boundary conditions for this pre-defined load case are returned as ``boundaries`` and the partitioned degrees of freedom as well as the external displacements are stored within the returned dict ``loadcase``.

An isotropic pseudo-elastic Ogden-Roxburgh Mullins-softening model formulation in combination with an isotropic hyperelastic Neo-Hookean material formulation is applied on the displacement field of a nearly-incompressible solid body.

A step generates the consecutive substep-movements of a given boundary condition. The step is further added to a list of steps of a job 👩‍💻 (here, a characteristic-curve 📈 job is used). During evaluation ⏳, each substep of each step is solved by an iterative Newton-Rhapson procedure ⚖️. The solution is exported after each completed substep as a time-series ⌚ XDMF file. Finally, the result of the last completed substep is plotted.

..  code-block:: python

    python
   import felupe as fem

   mesh = fem.Cube(n=6)
   region = fem.RegionHexahedron(mesh)
   field = fem.FieldContainer([fem.Field(region, dim=3)])

   boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

   umat = fem.OgdenRoxburgh(material=fem.NeoHooke(mu=1), r=3, m=1, beta=0)
   solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)

   move = fem.math.linsteps([0, 1, 0, 1, 2, 1], num=5)
   step = fem.Step(items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries)

   job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
   job.evaluate(filename="result.xdmf")
   fig, ax = job.plot(
      xlabel="Displacement $u$ in mm $\longrightarrow$",
      ylabel="Normal Force $F$ in N $\longrightarrow$",
   )

   ax2 = solid.imshow("Principal Values of Cauchy Stress", theme="paraview")

..  image:: https://private-user-images.githubusercontent.com/5793153/299447037-2a236f27-53a5-42aa-a45e-85628ecddf72.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MDc1NzMxMzYsIm5iZiI6MTcwNzU3MjgzNiwicGF0aCI6Ii81NzkzMTUzLzI5OTQ0NzAzNy0yYTIzNmYyNy01M2E1LTQyYWEtYTQ1ZS04NTYyOGVjZGRmNzIucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI0MDIxMCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNDAyMTBUMTM0NzE2WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9M2VlOGRmYTg0NWNlYWQ2MTY3MDQ3NjlkMjc2Y2YxNDM5YWEzZjEyN2VhMDI0OWQ5NzY3MWMyN2FkNDYyZTlkMSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmYWN0b3JfaWQ9MCZrZXlfaWQ9MCZyZXBvX2lkPTAifQ.5u_2QMfyGBqDz1cqZeIHz4V-_pN5VEBEFGEPCrsr_vE

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

Performance
-----------
This is a simple benchmark to compare assembly times for linear elasticity and hyperelasticity on tetrahedrons.

.. tabs::

   .. tab:: Assembly Runtimes

      +---------+---------------------+-------------------+
      |   DOF   | Linear-Elastic in s | Hyperelastic in s |
      +=========+=====================+===================+
      |    1029 |                0.01 |              0.02 |
      +---------+---------------------+-------------------+
      |    2187 |                0.02 |              0.04 |
      +---------+---------------------+-------------------+
      |    6591 |                0.07 |              0.13 |
      +---------+---------------------+-------------------+
      |   14739 |                0.20 |              0.28 |
      +---------+---------------------+-------------------+
      |   41472 |                0.51 |              0.98 |
      +---------+---------------------+-------------------+
      |   98304 |                1.36 |              2.47 |
      +---------+---------------------+-------------------+

      +----------------+-------------------+
      | Analysis       |        DOF/s      |
      +================+===================+
      | Linear-Elastic |   84523 +/- 9541  |
      +----------------+-------------------+
      | Hyperelastic   |   49771 +/- 6781  |
      +----------------+-------------------+

      Tested on: KDE Neon (Ubuntu 22.04), Python 3.10, Intel® Core™ i7-6650U CPU @ 2.20GHz, 8GB RAM (Surface Pro 4).


   .. tab:: Source Code

      .. code-block:: python

         from timeit import timeit

         import matplotlib.pyplot as plt
         import numpy as np

         import felupe as fem


         def pre_linear_elastic(n, **kwargs):
            mesh = fem.Cube(n=n).triangulate()
            region = fem.RegionTetra(mesh)
            field = fem.FieldContainer([fem.Field(region, dim=3)])
            umat = fem.LinearElastic(E=1, nu=0.3)
            solid = fem.SolidBody(umat, field)
            return mesh, solid


         def pre_hyperelastic(n, **kwargs):
            mesh = fem.Cube(n=n).triangulate()
            region = fem.RegionTetra(mesh)
            field = fem.FieldContainer([fem.Field(region, dim=3)])
            umat = fem.NeoHooke(mu=1.0, bulk=2.0)
            solid = fem.SolidBody(umat, field)
            return mesh, solid


         print("# Assembly Runtimes")
         print("")
         print("|   DOF   | Linear-Elastic in s | Hyperelastic in s |")
         print("| ------- | ------------------- | ----------------- |")

         points_per_axis = np.round((np.logspace(3, 5, 6) / 3)**(1 / 3)).astype(int)

         number = 3
         parallel = False

         runtimes = np.zeros((len(points_per_axis), 2))

         for i, n in enumerate(points_per_axis):
            mesh, solid = pre_linear_elastic(n)
            matrix = solid.assemble.matrix(parallel=parallel)
            time_linear_elastic = (
               timeit(lambda: solid.assemble.matrix(parallel=parallel), number=number) / number
            )

            mesh, solid = pre_hyperelastic(n)
            matrix = solid.assemble.matrix(parallel=parallel)
            time_hyperelastic = (
               timeit(lambda: solid.assemble.matrix(parallel=parallel), number=number) / number
            )

            runtimes[i] = time_linear_elastic, time_hyperelastic

            print(
               f"| {mesh.points.size:7d} | {runtimes[i][0]:19.2f} | {runtimes[i][1]:17.2f} |"
            )

         dofs_le = points_per_axis ** 3 * 3 / runtimes[:, 0]
         dofs_he = points_per_axis ** 3 * 3 / runtimes[:, 1]

         print("")
         print("| Analysis       |        DOF/s      |")
         print("| -------------- | ----------------- |")
         print(
            f"| Linear-Elastic |   {np.mean(dofs_le):5.0f} +/-{np.std(dofs_le):5.0f}  |"
         )
         print(f"| Hyperelastic   |   {np.mean(dofs_he):5.0f} +/-{np.std(dofs_he):5.0f}  |")

         plt.figure()
         plt.loglog(
            points_per_axis ** 3 * 3,
            runtimes[:, 1],
            "C0",
            label=r"Stiffness Matrix (Hyperelastic)",
         )
         plt.loglog(
            points_per_axis ** 3 * 3,
            runtimes[:, 0],
            "C1--",
            label=r"Stiffness Matrix (Linear-Elastic)",
         )
         plt.xlabel(r"Number of degrees of freedom $\longrightarrow$")
         plt.ylabel(r"Runtime in s $\longrightarrow$")
         plt.legend()
         plt.tight_layout()
         plt.savefig("benchmark.png")



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
