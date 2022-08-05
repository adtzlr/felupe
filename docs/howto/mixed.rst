Mixed-Field Problems
~~~~~~~~~~~~~~~~~~~~

FElupe supports mixed-field formulations in a similar way it can handle (default) single-field variations. The definition of a mixed-field variation is shown for the hydrostatic-volumetric selective three-field-variation with independend fields for displacements :math:`\boldsymbol{u}`, pressure :math:`p` and volume ratio :math:`J`. The total potential energy for nearly-incompressible hyperelasticity is formulated with a determinant-modified deformation gradient. We take the :ref:`tutorial-getting-started` tutorial and modify it accordingly. The built-in Neo-Hookean material model is used as an argument of :class:`felupe.ThreeFieldVariation` for mixed-field problems.

..  code-block:: python

    import felupe as fe

    neohooke = fe.constitution.NeoHooke(mu=1.0, bulk=5000.0)
    umat = fe.constitution.ThreeFieldVariation(neohooke)

Next, let's create a meshed cube. Two regions, one for the displacements and another one for the pressure and the volume ratio are created.

..  code-block:: python

    mesh  = fe.Cube(n=6)

    region  = fe.RegionHexahedron(mesh)
    region0 = fe.RegionConstantHexahedron(mesh)

    dV = region.dV

    displacement = fe.Field(region,  dim=3)
    pressure     = fe.Field(region0, dim=1)
    volumeratio  = fe.Field(region0, dim=1, values=1)

    field = fe.FieldContainer((displacement, pressure, volumeratio))

Boundary conditions are enforced in the same way as in Getting Started.

..  code-block:: python

    import numpy as np

    f1 = lambda x: np.isclose(x, 1)

    boundaries = fe.dof.symmetry(displacement)
    boundaries["right"] = fe.Boundary(displacement, fx=f1, skip=(1, 0, 0))
    boundaries["move" ] = fe.Boundary(displacement, fx=f1, skip=(0, 1, 1), value=-0.4)

    dof0, dof1 = fe.dof.partition(field, boundaries)
    ext0 = fe.dof.apply(field, boundaries, dof0)

The Newton-Rhapson iterations are coded quite similar to the one used in :ref:`tutorial-getting-started`. FElupe provides a Mixed-field version of it's :class:`felupe.IntegralForm`, called :class:`felupe.IntegralFormMixed`. It assumes that the first field operates on the gradient and all the others don't. The resulting system vector with incremental values of the fields has to be splitted at the field-offsets in order to update the fields.

..  code-block:: python

    for iteration in range(8):

        F, p, J = field.extract()
        
        linearform   = fe.IntegralForm(umat.gradient([F, p, J]), field, dV)
        bilinearform = fe.IntegralForm(umat.hessian([F, p, J]), field, dV, field)

        r = linearform.assemble().toarray()[:, 0]
        K = bilinearform.assemble()
        
        system = fe.solve.partition(field, K, dof1, dof0, r)
        dfield = np.split(fe.solve.solve(*system, ext0), field.offsets)
        
        field += dfield

        norm = np.linalg.norm(dfield[0])
        print(iteration, norm)

        if norm < 1e-12:
            break

    fe.tools.save(region, field, filename="result.vtk")

The deformed cube is finally visualized by a VTK output file with the help of Paraview.

.. image:: images/threefield_cube.png
   :width: 600px