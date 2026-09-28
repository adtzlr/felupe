# -*- coding: utf-8 -*-
"""
 _______  _______  ___      __   __  _______  _______
|       ||       ||   |    |  | |  ||       ||       |
|    ___||    ___||   |    |  | |  ||    _  ||    ___|
|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___
|    ___||    ___||   |___ |       ||    ___||    ___|
|   |    |   |___ |       ||       ||   |    |   |___
|___|    |_______||_______||_______||___|    |_______|

This file is part of felupe.

Felupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Felupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Felupe.  If not, see <http://www.gnu.org/licenses/>.

"""

import numpy as np
import pytest

import felupe as fem


def pre(values=0):
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)
    u = fem.Field(r, dim=3, values=values)
    v = fem.FieldContainer([u])

    return r, v


def pre_axi():
    m = fem.Rectangle(n=3)
    e = fem.Quad()
    q = fem.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r, dim=2)
    v = fem.FieldContainer([u])

    print(m), print(r), print(v)

    return r, v


def pre_mixed():
    m = fem.Cube(n=3)
    e = fem.Hexahedron()
    q = fem.GaussLegendre(1, 3)
    r = fem.Region(m, e, q)

    u = fem.Field(r, dim=3)
    p = fem.Field(r)
    J = fem.Field(r, values=1.0)

    f = fem.FieldContainer([u, p, J])
    g = fem.FieldsMixed(fem.RegionHexahedron(m), n=3, disconnect=True)
    f2 = u & p & J
    f3 = (u & p) & J
    f4 = u & (p & J)
    f5 = u.as_container() & (p & J)
    assert [np.allclose(fi, f2i) for fi, f2i in zip(f.extract(), f2.extract())]
    assert [np.allclose(fi, f3i) for fi, f3i in zip(f.extract(), f3.extract())]
    assert [np.allclose(fi, f4i) for fi, f4i in zip(f.extract(), f4.extract())]
    assert [np.allclose(fi, f5i) for fi, f5i in zip(f.extract(), f5.extract())]

    f & None, u & None

    print(m), print(r), print(f)

    u.values[0] = np.ones(3)
    assert np.all(f.values()[0][0] == 1)
    assert len(g.fields) == 3

    fem.Field(r, dim=9, values=np.eye(3))

    return r, f, u, p, J


def pre_axi_mixed():
    m = fem.Rectangle(n=3)
    e = fem.Quad()
    q = fem.GaussLegendre(1, 2)
    r = fem.Region(m, e, q)

    u = fem.FieldAxisymmetric(r, dim=2)
    p = fem.Field(r)
    J = fem.Field(r, values=1.0)

    f = fem.FieldContainer((u, p, J))

    region = fem.RegionQuad(m)
    fem.FieldsMixed(region, axisymmetric=True)
    fem.FieldsMixed(region, planestrain=True)
    with pytest.raises(ValueError):
        fem.FieldsMixed(region, axisymmetric=True, planestrain=True)

    with pytest.warns(UserWarning):
        fem.FieldsMixed(region, axisymmetric=True, n=3, dim=2)

    u.values[0] = np.ones(2)
    assert np.all(f.values()[0][0] == 1)

    return r, f, u, p, J


def test_axi():
    r, u = pre_axi()
    u += u.values()

    r, f, u, p, J = pre_axi_mixed()

    u.extract()
    u.extract(grad=False)
    u.extract(grad=True, sym=True)
    u.extract(grad=True, add_identity=False)


def test_mixed_lagrange():
    order = 4

    m = fem.Cube(n=order + 1)
    md = fem.Cube(n=order)

    m.update(
        cells=np.arange(m.npoints).reshape(1, -1), cell_type="VTK_LAGRANGE_HEXAHEDRON"
    )
    md.update(
        cells=np.arange(md.npoints).reshape(1, -1), cell_type="VTK_LAGRANGE_HEXAHEDRON"
    )

    m = fem.mesh.CubeArbitraryOrderHexahedron(order=order)
    md = fem.mesh.CubeArbitraryOrderHexahedron(order=order - 1)

    r = fem.RegionLagrange(m, order=order, dim=3)
    g = fem.FieldsMixed(r, mesh=md)

    assert len(g.fields) == 3


def test_3d():
    r, u = pre()
    u += u.values()

    with pytest.raises(ValueError):
        r, u = pre(values=np.ones(2))


def test_3d_mixed():
    r, f, u, p, J = pre_mixed()

    f.evaluate.deformation_gradient()
    f.evaluate.strain()
    f.evaluate.log_strain()
    f.evaluate.green_lagrange_strain()
    f.evaluate.right_cauchy_green_deformation()

    f.extract()
    f.extract(grad=False)
    f.extract(grad=(False,))
    f.extract(grad=True, sym=True)
    f.extract(grad=True, add_identity=False)

    u.extract()
    u.extract(grad=False)
    u.extract(grad=True, sym=True)
    u.extract(grad=True, add_identity=False)

    J.fill(1.0)

    u + u.values
    u - u.values
    u * u.values
    J / J.values

    u + u
    u - u
    u * u
    J / J

    J /= J.values
    J /= J

    J *= J.values
    J += J.values
    J -= J.values

    J *= J
    J += J
    J -= J

    dof = [0, 1]
    u[dof]
    f[0][dof]

    u.values.fill(1)
    p.values.fill(1)
    J.values.fill(1)

    df = [u.values.copy(), p.values.copy(), J.values.copy()]

    f + df
    f - df
    f * df
    f / df

    f += df
    f -= df
    f *= df
    f /= df

    df_1d = np.concatenate([dfi.ravel() for dfi in df])

    f + df_1d
    f - df_1d
    f * df_1d
    f / df_1d

    f += df_1d
    f -= df_1d
    f *= df_1d
    f /= df_1d

    v = u.copy()
    g = f.copy()

    assert np.allclose(v.values, u.values)
    assert np.allclose(g[0].values, f[0].values)


def test_view():
    mesh = fem.Rectangle(n=6)
    region = fem.RegionQuad(mesh)
    field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
    plotter = field.plot(off_screen=True)
    # img = mesh.screenshot(transparent_background=True)
    # ax = mesh.imshow()


def test_link():
    mesh = fem.Cube(n=2)
    region = fem.RegionHexahedron(mesh)
    field = fem.Field(region, dim=3) & fem.Field(region, dim=1)
    field.link()

    assert field[0].values is field[1].values

    field1 = fem.Field(region, dim=3) & fem.Field(region, dim=1)
    field2 = fem.Field(region, dim=3) & fem.Field(region, dim=1)
    field1.link(field2)

    assert field1[0].values is field2[0].values
    assert field1[1].values is field2[1].values


def test_toplevel():
    meshes = [
        fem.Rectangle(n=3),
        fem.Rectangle(n=3).translate(1, axis=0).triangulate(),
    ]
    container = fem.MeshContainer(meshes, merge=True)
    field = fem.Field.from_mesh_container(container).as_container()
    regions = [
        fem.RegionQuad(container.meshes[0]),
        fem.RegionTriangle(container.meshes[1]),
    ]
    fields = [
        fem.FieldContainer([fem.FieldPlaneStrain(regions[0], dim=2)]),
        fem.FieldContainer([fem.FieldPlaneStrain(regions[1], dim=2)]),
    ]
    boundaries = fem.dof.uniaxial(field, clamped=True, return_loadcase=False)
    umats = [
        fem.LinearElastic(E=2.1e5, nu=0.3),
        fem.LinearElastic(E=1.0, nu=0.3),
    ]
    solids = [
        fem.SolidBody(umat=umats[0], field=fields[0]),
        fem.SolidBody(umat=umats[1], field=fields[1]),
    ]
    step = fem.Step(items=solids, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(x0=field)


def test_toplevel_merge():

    mesh1 = fem.Rectangle(n=3)
    displacement1 = fem.FieldAxisymmetric(fem.RegionQuad(mesh1), dim=2)
    field1 = fem.FieldContainer([displacement1])

    mesh2 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3)
    displacement2 = fem.FieldAxisymmetric(fem.RegionQuad(mesh2), dim=2)
    field2 = fem.FieldContainer([displacement2])

    with pytest.raises(TypeError):
        x0 = (field1 & displacement2).merge()

    x0 = (field1 & field2).merge()

    umat = fem.NeoHookeCompressible(mu=1, lmbda=2)
    solid1 = fem.SolidBody(umat, field1)
    solid2 = fem.SolidBody(umat, field2)

    boundaries = fem.dof.uniaxial(x0, clamped=True, return_loadcase=False)

    step = fem.Step(items=[solid1, solid2], boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(x0=x0)


def test_merge():

    # empty list of field containers can't be merged
    with pytest.raises(ValueError):
        fem.field.merge([])

    # only field containers can be merged
    mesh = fem.Rectangle(n=3)
    with pytest.raises(TypeError):
        fem.field.merge([fem.Field(fem.RegionQuad(mesh))])

    # field containers with a different number of fields can't be merged
    field1 = fem.FieldsMixed(fem.RegionQuad(mesh), n=3)
    field2 = fem.FieldsMixed(fem.RegionQuad(mesh.translate(1, 0)), n=2)
    with pytest.raises(TypeError):
        fem.field.merge([field1, field2])

    # fields with different dimensions can't be merged
    field1 = fem.FieldsMixed(fem.RegionQuad(mesh), n=2)
    field2 = fem.FieldsMixed(fem.RegionQuad(mesh.translate(1, 0)), n=2, dim=2)
    with pytest.raises(ValueError):
        fem.field.merge([field1, field2])

    # field containers with multiple fields on the same region
    region = fem.RegionQuad(fem.Rectangle(n=3))
    field = fem.FieldContainer(
        [fem.Field(region, dim=2), fem.Field(region, dim=3, values=1.0)]
    )
    x0 = fem.field.merge([field])
    assert field.fieldsizes == x0.fieldsizes == [18, 27]
    assert np.allclose(x0[1].values, 1.0)

    # a mixture of fields on shared and on separate regions is not supported
    region1 = fem.RegionQuad(fem.Rectangle(n=3))
    field1 = fem.FieldContainer([fem.Field(region1, dim=2), fem.Field(region1)])
    region2 = fem.RegionQuad(fem.Rectangle(a=(1, 0), b=(2, 1), n=3))
    field2 = fem.FieldContainer(
        [fem.Field(region2, dim=2), fem.Field(fem.RegionQuad(region2.mesh))]
    )
    with pytest.raises(TypeError):
        fem.field.merge([field1, field2])

    # dual meshes without point coordinates can't be merged by coordinates
    mesh1 = fem.Rectangle(n=3)
    mesh2 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3)
    fields = []
    for mesh in [mesh1, mesh2]:
        dual = fem.Mesh(np.zeros_like(mesh.points), mesh.cells[::-1], "quad")
        fields.append(
            fem.FieldContainer(
                [
                    fem.Field(fem.RegionQuad(mesh), dim=2),
                    fem.Field(fem.RegionQuad(dual, grad=False)),
                ]
            )
        )
    with pytest.raises(ValueError):
        fem.field.merge(fields)


def reference_solution(field, items, move=0.2):
    boundaries = fem.dof.uniaxial(field, clamped=True, move=move, return_loadcase=False)
    step = fem.Step(items=items, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate(x0=field)
    return field


def max_difference_by_points(field, reference):
    points = np.round(field.fields[0].region.mesh.points, 8)
    points_reference = np.round(reference.fields[0].region.mesh.points, 8)
    values = dict(zip(map(tuple, points_reference), reference[0].values))

    return max(
        np.abs(values[tuple(p)] - v).max()
        for p, v in zip(points, field[0].values)
        if tuple(p) in values
    )


@pytest.mark.parametrize("taylor_hood", [False, True])
def test_merge_mixed(taylor_hood):

    def create_mesh(a, b, n):
        mesh = fem.Rectangle(a=a, b=b, n=n)
        if taylor_hood:
            mesh = mesh.triangulate().add_midpoints_edges()
        return mesh

    Region = fem.RegionQuadraticTriangle if taylor_hood else fem.RegionQuad
    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=500)

    # two mixed-field containers (u, p, J)
    mesh1 = create_mesh(a=(0, 0), b=(1, 1), n=4)
    mesh2 = create_mesh(a=(1, 0), b=(2, 1), n=4)
    field1 = fem.FieldsMixed(Region(mesh1), n=3, planestrain=True)
    field2 = fem.FieldsMixed(Region(mesh2), n=3, planestrain=True)

    x0 = (field1 & field2).merge()

    assert field1.fieldsizes == field2.fieldsizes == x0.fieldsizes
    assert np.allclose(field1.offsets, x0.offsets)
    assert np.allclose(x0[2].values, 1.0)
    assert field1.x0 is field2.x0 is x0

    if taylor_hood:
        # continuous pressure on the (merged) corner points
        assert x0[1].values.shape == (7 * 4, 1)
    else:
        # cell-wise constant pressure
        assert x0[1].values.shape == (2 * 9, 1)

    solids = [fem.SolidBody(umat, field1), fem.SolidBody(umat, field2)]
    reference_solution(x0, solids)

    # compare with the solution on a single mesh
    mesh = create_mesh(a=(0, 0), b=(2, 1), n=(7, 4))
    field = fem.FieldsMixed(Region(mesh), n=3, planestrain=True)
    reference_solution(field, [fem.SolidBody(umat, field)])

    assert max_difference_by_points(x0, field) < 1e-10


def test_merge_mixed_element_types():

    # hexahedrons with disconnected (constant) and quadratic tetrahedrons with
    # continuous dual fields
    mesh1 = fem.Cube(n=3)
    mesh2 = fem.Cube(a=(1, 0, 0), b=(2, 1, 1), n=3).triangulate().add_midpoints_edges()
    mesh3 = fem.Cube(a=(2, 0, 0), b=(3, 1, 1), n=3)

    fields = [
        fem.FieldsMixed(fem.RegionHexahedron(mesh1), n=3),
        fem.FieldsMixed(fem.RegionQuadraticTetra(mesh2), n=3),
        fem.FieldsMixed(fem.RegionHexahedron(mesh3), n=3),
    ]

    x0 = fem.field.merge(fields, decimals=8)

    # 2 x 8 hexahedrons (constant) + 27 corner points of the tetrahedrons
    assert x0[1].values.shape == (2 * 8 + 27, 1)

    for field in fields:
        assert field.fieldsizes == x0.fieldsizes

    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=500)
    solids = [fem.SolidBody(umat, field) for field in fields]
    reference_solution(x0, solids)

    assert np.isclose(x0[0].values[:, 0].max(), 0.2)


def test_merge_fewer_points():

    mesh = fem.Rectangle(n=2)
    mesh.points[2] = mesh.points[0]

    displacement = fem.FieldAxisymmetric(fem.RegionQuad(mesh), dim=2)
    field = fem.FieldContainer([displacement])

    x0 = fem.field.merge([field])
    assert len(field[0].values) == len(x0[0].values) == 3


def test_field_dual():

    mesh = fem.Cube(n=3).convert(2, 1, 1, 1)
    region = fem.RegionTriQuadraticHexahedron(mesh)
    field_dual = fem.FieldDual(
        region,
        dim=3,
        calc_points=True,
        disconnect=False,
        grad=True,
    )
    field_container = fem.FieldContainer([field_dual])
    assert field_container[0] is field_dual


def test_field_take():

    mesh = fem.Cube(n=3).convert(2, 1, 1, 1)
    region = fem.RegionTriQuadraticHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)], take=[0, 0])

    F, u = field.extract(grad=[True, False])

    assert F.shape[:2] == (3, 3)
    assert u.shape[:1] == (3,)

    assert len(F.shape) == 4
    assert len(u.shape) == 3

    assert len(field.fieldsizes) == 1
    assert len(field.offsets) == 0


if __name__ == "__main__":
    test_axi()
    test_3d()
    test_3d_mixed()
    test_mixed_lagrange()
    test_view()
    test_link()
    test_toplevel()
    test_toplevel_merge()
    test_merge()
    test_merge_fewer_points()
    test_field_dual()
    test_field_take()
