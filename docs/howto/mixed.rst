Mixed-Field Problems
~~~~~~~~~~~~~~~~~~~~

FElupe supports mixed-field formulations in a similar way it can handle (default) single-field formulations. The definition of a mixed-field formulation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements :math:`\boldsymbol{u}`, pressure :math:`p` and volume ratio :math:`J`. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient. The built-in Neo-Hookean material model is used as an argument of :class:`felupe.ThreeFieldVariation` for mixed-field problems.

..  code-block:: python

    import felupe as fem

    neohooke = fem.constitution.NeoHooke(mu=1.0, bulk=5000.0)
    umat = fem.constitution.ThreeFieldVariation(neohooke)

Next, let's create a meshed cube. A Taylor-Hood Q2/P1 hexahedron element formulation is created, where a tri-quadratic Lagrange 27-point per cell displacement formulation is used in combination with discontinuous 8-point per cell linear formulations for the pressure and volume ratio fields. Hence, the mesh of the cube is converted to a tri-quadratic mesh for the displacement field. The regions for the pressure and the volume ratio are created on a disconnected mesh for the generation of discontinuous fields.

..  code-block:: python

    mesh1  = fem.Cube(n=5)
    mesh2 = mesh1.convert(
        order=2, 
        calc_points=True, 
        calc_midfaces=True, 
        calc_midvolumes=True
    )

    region  = fem.RegionTriQuadraticHexahedron(mesh2)
    region0 = fem.RegionHexahedron(mesh1.disconnect(), quadrature=region.quadrature)

    displacement = fem.Field(region,  dim=3)
    pressure     = fem.Field(region0, dim=1)
    volumeratio  = fem.Field(region0, dim=1, values=1)

    field = fem.FieldContainer(fields=[displacement, pressure, volumeratio])

Boundary conditions are enforced on the displacement field. For the pre-defined loadcases like the clamped uniaxial compression, the boundaries are automatically applied on the first field.

..  code-block:: python

    import numpy as np

    boundaries, loadcase = fem.dof.uniaxial(field, clamped=True)

The Step and Job definitions are identical to single field formulations.

..  code-block:: python

    step = fem.Step(
        items=[solid], 
        ramp={boundaries["move"]: fem.math.linsteps([0, -0.4], num=12)},
        boundaries=boundaries
    )
    job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"])
    job.evaluate(filename="result.xdmf")

The deformed cube is finally visualized by a XDMF output file with the help of Paraview.

.. image:: images/threefield_cube.png
   :width: 600px