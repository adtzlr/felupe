Non-homogenous shear
--------------------

.. admonition:: Plane strain hyperelastic non-homogenous shear loadcase
   :class: note

   * define a non-homogenous shear loadcase
   
   * use a mixed hyperelastic formulation in plane strain
   
   * assign a micro-sphere material formulation
   
   * export and plot stress results


Two rubber blocks of height :math:`H` and length :math:`L`, both glued to a 
rigid plate on their top and bottom faces, are subjected to a displacement 
controlled non-homogenous shear deformation by :math:`u_{ext}` in combination 
with a compressive normal force :math:`F`. What is being looked for is the 
shear stress as a function of ?

.. image:: images/shear.svg
   :width: 400px


Let's create the mesh. An additional center-point is created for a multi-point
constraint.

..  code-block:: python

    import felupe as fe

    H = 1.0
    L = 2.0
    
    a = min(L / 21, H / 21)

    mesh = fe.Rectangle((0, 0), (L, H), n=(round(L / a), round(H / a)))
    mesh.points = np.vstack((mesh.points, [0, 2 * H]))
    mesh.update(mesh.cells)

.. image:: images/shear_mesh.png
   :width: 400px

A numeric quad-region created on the mesh in combination with a vector-valued 
displacement field represents the rubber. A uniaxial loadcase is applied on the 
displacement field. This involves setting up symmetry planes as well as the 
absolute value of the prescribed displacement in direction :math:`x` at the 
mesh-points on the right-end face of the rectangle.

..  code-block:: python

    region  = fe.RegionQuad(mesh)
    region0 = fe.RegionConstantQuad(mesh)
    
    displacement = fe.Field(region, dim=2)
    pressure     = fe.Field(region)
    volumeratio  = fe.Field(region, value=1)
    fields       = fe.FieldMixed((displacement, pressure, fields))

    f0 = lambda y: np.isclose(y, 0)
    f2 = lambda y: np.isclose(y, 2 * L)
    
    boundaries = fe.dof.symmetry(displacement, axes=(True, False))
    boundaries["shear"] = fe.Boundary(displacement, fy=f0, skip=(0, 1), value=0.1)
    boundaries["top"] = fe.Boundary(displacement, fy=f2, skip=(0, 1))
    
    dof0, dof1, offsets = fe.dof.partition(fields, boundaries)
    ext0 = fe.dof.apply(displacement, boundaries)


The material behavior is defined through a built-in isotropic linear-elastic material formulation for plane stress problems. The deformation gradient is extracted from the displacement field. In the undeformed state it is filled with the identity matrix at every integration point of every cell in the mesh.

..  code-block:: python

    umat = fe.LinearElasticPlaneStrain(E=210000, nu=0.3)
    F = displacement.extract()
    

The weak form of linear elasticity is assembled into the stiffness matrix, where the constitutive elasticity matrix is generated with :func:`umat.hessian`. Please note that although the elasticity tensor does not depend on the deformation gradient for linear elasticity, FElupe extracts the shape of the deformation gradient in :func:`umat.hessian`.

.. math::

   \delta W_{int} = - \int_v \delta \boldsymbol{\varepsilon} : \mathbb{C} : \boldsymbol{\varepsilon} \ dv


..  code-block:: python

    K = fe.IntegralForm(
        fun=umat.hessian(F), 
        v=displacement, 
        dV=region.dV, 
        u=displacement, 
        grad_v=True,
        grad_u=True,
    ).assemble()

The linear equation system may now be solved. First, a partition into active and inactive degrees of freedom is performed. This partitioned system is then passed to the solver. The resulting displacements are directly added to the displacement field.

..  code-block:: python

    system = fe.solve.partition(displacement, K, dof1, dof0)
    displacement += fe.solve.solve(*system, u0ext=u0ext)

Once again, let's evaluate the deformation gradient and the stress. This process is also called *stress recovery*.

..  code-block:: python

    F = displacement.extract()
    stress = umat.gradient(F)

However, the stress results are still located at the numeric integration points. Let's project them to mesh points. Beside the stress tensor we are also interested in the equivalent stress von Mises. For the two-dimensional case is calculated as:

.. math::

   \sigma_{vM} = \sqrt{\sigma_{11}^2 + \sigma_{22}^2 + 3 \ \sigma_{12}^2 + \sigma_{11} \ \sigma_{22}}


..  code-block:: python

    import numpy as np
    
    vonmises = np.sqrt(
        stress[0, 0] ** 2 + stress[1, 1] ** 2 + 3 * stress[0, 1] ** 2 +
        stress[0, 0] * stress[0, 1]
    )
    
    stress_projected = fe.project(stress, region)
    vonmises_projected = fe.project(vonmises, region)


Results are saved as VTK-files, where additional point-data is passed within the ``point_data`` argument. Stresses are normalized by the mean value of the stress at the right end-face in order to visualize a normalized stress distribution over the plate.
    
..  code-block:: python

    right = mesh.points[:, 0] == L
    bottom = mesh.points[:, 1] == 0

    fe.save(
        region, 
        displacement, 
        filename="plate_with_hole.vtk",
        point_data={
            "Stress": (stress_projected / 
                stress_projected[right].mean(axis=0)
            ),
            "Stress-von-Mises": (vonmises_projected / 
                vonmises_projected[right].mean(axis=0)
            ),
        },
    )


.. image:: images/platewithhole_stress.png

The normal stress distribution over the hole at :math:`x=0` is plotted with matplotlib.

..  code-block:: python

    import matplotlib.pyplot as plt

    plt.plot(
        mesh.points[:, 1][left] / h, 
        (stress_projected / 
            stress_projected[right].mean(axis=0)
        )[:, 0][left],
        "o-"
    )
    
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    
    plt.grid()
    
    plt.xlabel(r"$y/h\ \longrightarrow$")
    plt.ylabel(r"$\sigma_{11}(x=0, y)\ /\ \sigma_{11}(x=x_{max})$ $\longrightarrow$")

.. image:: images/platewithhole_stressplot.png