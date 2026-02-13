.. _tutorials:

Beginner's Guide
================

This minimal code-block covers the essential high-level parts of creating and solving problems with FElupe. As an introductory example, a quarter model of a solid :class:`cube <felupe.Cube>` with hyperelastic material behaviour is subjected to a :func:`~felupe.dof.uniaxial` elongation applied at a clamped end-face.

First, let’s import FElupe and create a meshed :class:`cube <felupe.Cube>` out of :class:`hexahedron <felupe.Hexahedron>` cells with a given number of points per axis. A numeric :class:`region <felupe.RegionHexahedron>`, pre-defined for hexahedrons, is created on the mesh. A vector-valued displacement :class:`field <felupe.Field>` is initiated on the region. Next, a :class:`field container <felupe.FieldContainer>` is created on top of this field.

.. pyvista-plot::
   :context:

   import felupe as fem

   mesh = fem.Cube(n=6)
   region = fem.RegionHexahedron(mesh)
   field = fem.FieldContainer([fem.Field(region, dim=3)])

A :func:`~felupe.dof.uniaxial` load case is applied on the displacement :class:`field <felupe.Field>` stored inside the :class:`field container <felupe.FieldContainer>`. This involves setting up :func:`~felupe.dof.symmetry` planes as well as the absolute value of the prescribed displacement at the mesh-points on the right-end face of the cube. The right-end face is *clamped*: only displacements in direction *x* are allowed. The dict of :class:`boundary <felupe.Boundary>` conditions for this pre-defined load case are returned as ``boundaries``. Optionally, the partitioned degrees of freedom as well as the external displacements are stored within the returned dict ``loadcase``.

.. pyvista-plot::
   :context:

   boundaries, loadcase = fem.dof.uniaxial(
       field, clamped=True, return_loadcase=True
   )

An isotropic hyperelastic :class:`Neo-Hookean <felupe.NeoHooke>` material model formulation is applied on the displacement :class:`field <felupe.Field>` of a :class:`solid body <felupe.SolidBody>`.

.. pyvista-plot::
   :context:

   umat = fem.NeoHooke(mu=1, bulk=50)
   solid = fem.SolidBody(umat, field)

A :class:`step <felupe.Step>` generates the consecutive substep-movements of a given :class:`boundary <felupe.Boundary>` condition.

.. pyvista-plot::
   :context:

   move = fem.math.linsteps([0, 1], num=5)
   step = fem.Step(
       items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
   )

The :class:`step <felupe.Step>` is further added to a list of steps of a :class:`job <felupe.Job>` (here, a :class:`characteristic curve <felupe.CharacteristicCurve>` job is used). During :meth:`evaluation <felupe.Job.evaluate>`, each substep of each :class:`step <felupe.Step>` is solved by an iterative :func:`Newton-Rhapson <felupe.newtonrhapson>` procedure. The :func:`solution <felupe.tools.NewtonResult>` is exported after each completed substep as a time-series XDMF file.

.. pyvista-plot::
   :context:

   job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
   job.evaluate(filename="result.xdmf")

   fig, ax = job.plot(
       xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
       ylabel=r"Normal Force $F_1$ in N $\longrightarrow$",
   )

.. pyvista-plot::
   :include-source: False
   :context:
   :force_static:

   import pyvista as pv

   fig = ax.get_figure()
   chart = pv.ChartMPL(fig)
   chart.show()

Finally, the result of the last completed substep is plotted.

.. pyvista-plot::
   :context:

   solid.plot("Principal Values of Cauchy Stress").show()

Slightly modified code-blocks are provided for different kind of analyses

.. tab:: 3D

   and element formulations.

   .. tab:: Hexahedron

      .. code-block:: python

         import felupe as fem

         mesh = fem.Cube(n=6)
         region = fem.RegionHexahedron(mesh)
         field = fem.FieldContainer([fem.Field(region, dim=3)])

         boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

         umat = fem.NeoHooke(mu=1, bulk=50)
         solid = fem.SolidBody(umat, field)

         move = fem.math.linsteps([0, 1], num=5)
         step = fem.Step(
             items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
         )

         job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
         job.evaluate(filename="result.xdmf")
         fig, ax = job.plot(
             xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
             ylabel=r"Normal Force $F_1$ in N $\longrightarrow$",
         )
         solid.plot(
             "Principal Values of Cauchy Stress"
         ).show()
   
   .. tab:: Quadratic Hexahedron

      .. code-block:: python

         import felupe as fem

         mesh = fem.Cube(n=(9, 5, 5)).add_midpoints_edges()
         region = fem.RegionQuadraticHexahedron(mesh)
         field = fem.FieldContainer([fem.Field(region, dim=3)])

         boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

         umat = fem.NeoHooke(mu=1, bulk=50)
         solid = fem.SolidBody(umat, field)

         move = fem.math.linsteps([0, 1], num=5)
         step = fem.Step(
             items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
         )

         job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
         job.evaluate()
         fig, ax = job.plot(
             xlabel=r"Displacement $u$ in mm $\longrightarrow$",
             ylabel=r"Normal Force $F$ in N $\longrightarrow$",
         )
         solid.plot(
             "Principal Values of Cauchy Stress", nonlinear_subdivision=4
         ).show()

   .. tab:: Lagrange Hexahedron

      .. code-block:: python

         import felupe as fem

         mesh = fem.mesh.CubeArbitraryOrderHexahedron(order=3)
         region = fem.RegionLagrange(mesh, order=3, dim=3)
         field = fem.FieldContainer([fem.Field(region, dim=3)])

         boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

         umat = fem.NeoHooke(mu=1, bulk=50)
         solid = fem.SolidBody(umat, field)

         move = fem.math.linsteps([0, 1], num=5)
         step = fem.Step(
             items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
         )

         job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
         job.evaluate()
         fig, ax = job.plot(
             xlabel=r"Displacement $u$ in mm $\longrightarrow$",
             ylabel=r"Normal Force $F$ in N $\longrightarrow$",
         )
         solid.plot(
             "Principal Values of Cauchy Stress", project=fem.topoints, nonlinear_subdivision=4
         ).show()

.. tab:: Plane Strain

   and element formulations.

   .. tab:: Quad

      .. code-block:: python

         import felupe as fem

         mesh = fem.Rectangle(n=6)
         region = fem.RegionQuad(mesh)
         field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])

         boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

         umat = fem.NeoHooke(mu=1, bulk=50)
         solid = fem.SolidBody(umat, field)

         move = fem.math.linsteps([0, 1], num=5)
         step = fem.Step(
             items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
         )

         job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
         job.evaluate(filename="result.xdmf")
         fig, ax = job.plot(
             xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
             ylabel=r"Normal Force $F_1$ in N $\longrightarrow$",
         )
         solid.plot(
             "Principal Values of Cauchy Stress"
         ).show()

.. tab:: Axisymmetric

   and element formulations.

   .. tab:: Quad

      .. code-block:: python

         import felupe as fem

         mesh = fem.Rectangle(n=6)
         region = fem.RegionQuad(mesh)
         field = fem.FieldContainer([fem.FieldAxisymmetric(region, dim=2)])

         boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)

         umat = fem.NeoHooke(mu=1, bulk=50)
         solid = fem.SolidBody(umat, field)

         move = fem.math.linsteps([0, 1], num=5)
         step = fem.Step(
             items=[solid], ramp={boundaries["move"]: move}, boundaries=boundaries
         )

         job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
         job.evaluate(filename="result.xdmf")
         fig, ax = job.plot(
             xlabel=r"Displacement $d_1$ in mm $\longrightarrow$",
             ylabel=r"Normal Force $F_1$ in N $\longrightarrow$",
         )
         solid.plot(
             "Principal Values of Cauchy Stress"
         ).show()

Tutorials
---------

This section is all about learning. Each tutorial focuses on some lessons to learn.
