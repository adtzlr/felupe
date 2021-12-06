Composite regions
-----------------

This section demonstrates how to set up a problem with two regions, each associated to a seperated material model but with the same kind of weak form formulation (displacement-based). First, a (total) region is created.

..  code-block:: python

    import felupe as fe
    import numpy as np

    n = 16
    mesh = fe.Cube(n=n)
    region = fe.RegionHexahedron(mesh)
    displacement = fe.Field(region, dim=3)


In a second step, sub-sets for points and cells are created from which two sub-regions and sub-fields are initiated. Both sub-fields are added to a list of composite fields.
    
..  code-block:: python

    rubber = mesh.copy()
    rubber.update(rubber.cells[~cells])

    rregion = fe.RegionHexahedron(rubber)
    rdisplacement = fe.Field(rregion, dim=3)

    steel = mesh.copy()
    steel.update(steel.cells[cells])

    sregion = fe.RegionHexahedron(steel)
    sdisplacement = fe.Field(sregion, dim=3)
    
    fields = [rdisplacement, sdisplacement]

The displacement boundaries are created on the total field whereas the partition of the degrees of freedom is performed on the sub-fields.

..  code-block:: python

    boundaries, dof0, dof1, ext0 = fe.dof.uniaxial(
        displacement, move=-0.25
    )
    
    dofs = [
        fe.dof.partition(rdisplacement, boundaries),
        fe.dof.partition(sdisplacement, boundaries),
    ]

The rubber is associated to a Neo-Hookean material formulation whereas the steel is modeled by a linear elastic material formulation.

..  code-block:: python

    umats = [fe.NeoHooke(mu=1.0, bulk=2.0), fe.LinearElastic(E=210000.0, nu=0.3)]

The integral (weak) linear and bilinear forms are created with functions in order to re-use them for both sub-fields.

..  code-block:: python

    def fun(umat, field):
        r = fe.IntegralForm(
            umat.gradient(field.extract()), field, field.region.dV, None, True
        ).assemble().toarray()[:, 0]
        return r

    def jac(umat, field):
        K = fe.IntegralForm(
            umat.hessian(field.extract()), field, field.region.dV, field, True, True
        ).assemble()
        return K
    
    funs = [fun, fun]
    jacs = [jac, jac]


Inside the Newton-Rhapson iterations both the internal force vector and the tangent stiffness matrix are assembled and summed up from contributions of both sub-regions.

..  code-block:: python

    for iteration in range(8):
    
        r = sum([f(umat, field) for f, umat, field in zip(funs, umats, fields)])
        K = sum([j(umat, field) for j, umat, field in zip(jacs, umats, fields)])

        system = fe.solve.partition(displacement, K, dof1, dof0, r)
        du = fe.solve.solve(*system, ext0)

        displacement += du
        
        for field, (d0, d1) in zip(fields, dofs):
        
            field.values.ravel()[d1] += du[d1]
            field.values.ravel()[d0] += du[d0]
        
        norm = fe.math.norm(du)
        print(iteration, norm)

        if norm < 1e-12:
            break

..  code-block:: shell

    0 9.636630560448182
    1 0.31166451613964075
    2 0.005354041194053835
    3 2.8254858186935622e-05
    4 1.0857486092949548e-09
    5 9.475677365353017e-16

Results and cauchy stresses may be exported either for the total region (take care of result-averaging at region intersections!) or for sub-regions only.

.. image:: images/composite_total.png
   :width: 600px

..  code-block:: python

    from felupe.math import dot, det, transpose, tovoigt

    F = fields[0].extract()
    s = dot(umats[0].gradient(F), transpose(F)) / det(F)

    cauchy = fe.project(tovoigt(s), rregion)
    
    fe.save(region, displacement, filename="result.vtk")

    fe.save(fields[0].region, fields[0], filename="result_rubber.vtk",
            point_data={"Cauchy": cauchy})

.. image:: images/composite_rubber_cauchy.png
   :width: 600px