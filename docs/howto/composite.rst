Composite Regions with Solid Bodies
-----------------------------------

This section demonstrates how to set up a problem with two regions, each associated to a seperated solid body. First, a (total) region is created.

..  code-block:: python

    import felupe as fe
    import numpy as np

    n = 16
    mesh = fe.Cube(n=n)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)


In a second step, sub-sets for points and cells are created from which two sub-regions and sub-fields are initiated. All field values are linked, that means they share their values array.
    
..  code-block:: python

    points = np.arange(mesh.npoints)[np.logical_or.reduce((
        mesh.points[:,0] == 0,
        mesh.points[:,0] == 0 + 1/(n - 1),
        mesh.points[:,0] == 0.5 - 1/(n - 1) / 2,
        mesh.points[:,0] == 0.5 + 1/(n - 1) / 2,
        mesh.points[:,0] == 1 - 1/(n - 1),
        mesh.points[:,0] == 1,
    ))]
    cells = np.isin(mesh.cells, points).sum(1) == mesh.cells.shape[1]

    mesh_rubber = mesh.copy()
    mesh_rubber.update(mesh_rubber.cells[~cells])

    mesh_steel = mesh.copy()
    mesh_steel.update(mesh_steel.cells[cells])
    
    region_rubber = fe.RegionHexahedron(mesh_rubber)
    displacement_rubber = fe.Field(region_rubber, dim=3)

    region_steel = fe.RegionHexahedron(mesh_steel)
    displacement_steel = fe.Field(region_steel, dim=3)

    # link fields
    displacement_steel.values = displacement_rubber.values = displacement.values


The displacement boundaries are created on the total field.

..  code-block:: python

    boundaries, dof0, dof1, ext0 = fe.dof.uniaxial(
        displacement, move=-0.25
    )


The rubber is associated to a Neo-Hookean material formulation whereas the steel is modeled by a linear elastic material formulation. For each material a solid body is created.

..  code-block:: python

    neohooke = fe.NeoHooke(mu=1.0, bulk=2.0)
    linearelastic = fe.LinearElastic(E=210000.0, nu=0.3)

    rubber = fe.SolidBody(neohooke, displacement_rubber)
    steel = fe.SolidBody(linearelastic, displacement_steel)


Inside the Newton-Rhapson iterations both the internal force vector and the tangent stiffness matrix are assembled and summed up from contributions of both solid bodies.

..  code-block:: python

    r = rubber.assemble.vector()
    r+= steel.assemble.vector()

    for iteration in range(8):

        K = rubber.assemble.matrix()
        K+= steel.assemble.matrix()

        system = fe.solve.partition(displacement, K, dof1, dof0, r)
        du = fe.solve.solve(*system, ext0)

        displacement += du
        
        r = rubber.assemble.vector(displacement_rubber)
        r+= steel.assemble.vector(displacement_steel)

        norm = fe.math.norm(du)
        print(iteration, norm)

        if norm < 1e-12:
            break

..  code-block:: shell

    0 9.636630560459622
    1 0.3116645161396399
    2 0.005354041194053836
    3 2.825485818694591e-05
    4 1.0857485921106448e-09
    5 9.016379080063146e-16

Results and cauchy stresses may be exported either for the total region (take care of result-averaging at region intersections!) or for sub-regions only.

.. image:: images/composite2_total.png
   :width: 600px

..  code-block:: python

    s = rubber.evaluate.cauchy_stress()
    cauchy_stress = fe.project(fe.math.tovoigt(s), region_rubber)
    
    fe.save(region, displacement, filename="result.vtk")

    fe.save(region_rubber, displacement_rubber, filename="result_rubber.vtk",
        point_data={"CauchyStress": cauchy_stress})

.. image:: images/composite2_rubber_cauchy.png
   :width: 600px
