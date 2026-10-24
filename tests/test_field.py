# SPDX-License-Identifier: GPL-3.0-or-later

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
    with pytest.raises(TypeError):
        fem.field.merge([fem.Field(fem.RegionQuad(fem.Rectangle(n=3)))])

    # the first fields have different dimensions
    field_1 = fem.FieldContainer([fem.Field(fem.RegionQuad(fem.Rectangle(n=3)), dim=2)])
    field_2 = fem.FieldContainer(
        [fem.Field(fem.RegionQuad(fem.Rectangle(a=(1, 0), b=(2, 1), n=3)), dim=3)]
    )

    with pytest.raises(ValueError):
        fem.field.merge([field_1, field_2])

    # the second fields are continuous in the first and dual in the second container
    region = fem.RegionQuad(fem.Rectangle(n=3))
    field_1 = fem.FieldContainer([fem.Field(region, dim=2), fem.Field(region)])
    field_2 = fem.FieldsMixed(
        fem.RegionQuad(fem.Rectangle(a=(1, 0), b=(2, 1), n=3)), n=2
    )
    x0 = fem.field.merge([field_1, field_2])

    assert x0.fieldsizes == [30, 9, 4]  # compact: points of the first container
    assert field_1.slots is None
    assert field_2.slots == [0, 2]
    assert field_2[1].values is x0[2].values

    # a field container of an item with wrong sizes of the fields can't be embedded
    with pytest.raises(ValueError):
        fem.tools.embed(
            fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field_2).assemble.vector(),
            shape=(43, 1),
            field=field_2,
            x=fem.FieldContainer([x0[0], x0[1], x0[1]]),
        )


def test_merge_mixed():

    # one mesh, split into two meshes
    mesh = fem.Cube(n=4)
    mask = mesh.points[mesh.cells].mean(axis=1)[:, 0] < 0.5
    meshes = [mesh.copy(), mesh.copy()]
    meshes[0].update(cells=mesh.cells[mask])
    meshes[1].update(cells=mesh.cells[~mask])
    meshes = [m.sweep() for m in meshes]

    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=50)

    def evaluate(fields, x0):
        boundaries = fem.dof.uniaxial(x0, move=0.3, return_loadcase=False)
        solids = [fem.SolidBody(umat, f) for f in fields]
        step = fem.Step(items=solids, boundaries=boundaries)
        fem.Job(steps=[step]).evaluate(x0=x0)

    field = fem.FieldsMixed(fem.RegionHexahedron(mesh), n=3)
    evaluate([field], field)

    fields = [fem.FieldsMixed(fem.RegionHexahedron(m), n=3) for m in meshes]
    x0 = fem.field.merge(fields)

    # the dual fields are stacked and the values are linked to the top-level field
    assert len(x0) == 3
    assert len(x0[1].values) == mesh.ncells
    assert np.allclose(x0[2].values, 1.0)
    assert np.all(fields[1].offsets == x0.offsets)
    assert all(f.values is g.values for f, g in zip(fields[1].fields, x0.fields))
    assert fields[0].x0 is fields[1].x0 is x0

    evaluate(fields, x0)

    def sort(points, values):
        return values[np.lexsort(np.round(points, 8).T[::-1])]

    u = sort(mesh.points, field[0].values)
    u_merged = sort(x0[0].region.mesh.points, x0[0].values)
    assert np.allclose(u, u_merged)
    assert np.allclose(np.sort(field[1].values.ravel()), np.sort(x0[1].values.ravel()))


def test_merge_mixed_formulations():

    # a displacement field, a Q2/P0 and a Taylor-Hood P2/P1 mixed field container
    quad8 = fem.Rectangle(a=(0, 0), b=(1, 1), n=3).add_midpoints_edges()
    tri6 = fem.Rectangle(a=(1, 0), b=(2, 1), n=3).triangulate().add_midpoints_edges()
    tri6b = fem.Rectangle(a=(2, 0), b=(3, 1), n=3).triangulate().add_midpoints_edges()

    fields = [
        fem.FieldContainer(
            [fem.FieldPlaneStrain(fem.RegionQuadraticTriangle(tri6b), dim=2)]
        ),
        fem.FieldsMixed(fem.RegionQuadraticQuad(quad8), n=3, planestrain=True),
        fem.FieldsMixed(fem.RegionQuadraticTriangle(tri6), n=3, planestrain=True),
    ]
    npoints = fields[1][1].region.mesh.npoints + fields[2][1].region.mesh.npoints
    x0 = fem.field.merge(fields)
    assert len(x0[1].values) == npoints

    umat = fem.NearlyIncompressible(fem.NeoHooke(mu=1), bulk=50)
    solids = [
        fem.SolidBody(fem.NeoHooke(mu=1, bulk=50), fields[0]),
        fem.SolidBody(umat, fields[1]),
        fem.SolidBody(umat, fields[2]),
    ]
    boundaries = fem.dof.uniaxial(x0, move=0.3, clamped=False, return_loadcase=False)
    step = fem.Step(items=solids, boundaries=boundaries)
    fem.Job(steps=[step]).evaluate()

    # homogeneous deformation (patch test)
    assert np.allclose(x0[1].values, x0[1].values[0])
    for field in fields:
        assert np.allclose(field.extract()[0][0, 0], 1.1)


def test_merge_third_medium_contact():

    # a C-shaped body and a third medium in between (see examples/ex20)
    t, L, H = 0.1, 1.0, 0.5
    a = fem.Rectangle(a=(0, 0), b=(t, t), n=2)
    b = a.translate(move=H - t, axis=1)
    c = fem.Rectangle(a=(t, 0), b=(L, t), n=(7, 2))
    d = c.translate(move=H - t, axis=1)
    e = fem.Rectangle(a=(0, t), b=(t, H - t), n=(2, 4))
    body = fem.MeshContainer([a, b, c, d, e], merge=True).stack()
    medium = fem.Rectangle(a=(t, t), b=(L, H - t), n=(7, 4))

    region_body = fem.RegionQuad(body)
    region_medium = fem.RegionQuad(medium)
    fields = [
        fem.FieldContainer([fem.FieldPlaneStrain(region_body, dim=2)]),
        fem.FieldContainer(
            [
                fem.FieldPlaneStrain(region_medium, dim=2),
                fem.Field(region_medium, dim=9),
            ],
            take=[0, 1, 1],
        ),
    ]
    x0 = fem.field.merge(fields, decimals=6)

    # the tensor-valued field of the medium has only the points of the medium
    points = np.unique(np.vstack([body.points, medium.points]).round(6), axis=0)
    assert x0.fieldsizes == [2 * len(points), 9 * medium.npoints]
    assert fields[1][0].region is not fields[1][1].region
    assert np.allclose(
        np.sort(fields[1][1].region.mesh.points, axis=0),
        np.sort(medium.points, axis=0),
    )
    assert np.all(fields[1].offsets == x0.offsets)

    G, K = 5 / 14, 5 / 3
    solid = fem.SolidBodyNearlyIncompressible(
        fem.NeoHooke(mu=G), fields[0], bulk=50 * G
    )
    third_medium = fem.SolidBody(
        fem.ThirdMediumContactMixed(
            fem.NeoHookeCompressible(mu=G, lmbda=K),
            gamma=5e-6,
            alpha_r=5e-6 * K * L**2,
            p_r=1e-2,
        ),
        fields[1],
        grad=[True, False, True],
    )

    # no orphan degrees of freedom
    from felupe.tools._newton import jac_items

    matrix = jac_items([solid, third_medium], x0).tocsr()
    assert np.all(np.diff(matrix.indptr) > 0)

    bounds = fem.BoundaryDict(
        fixed=fem.Boundary(x0[0], fx=0),
        move=fem.Boundary(x0[0], fx=L, fy=H, mode="and", skip=(1, 0)),
    )
    step = fem.Step(
        items=[solid, third_medium],
        ramp={bounds["move"]: fem.math.linsteps([0, -0.2], num=2)},
        boundaries=bounds,
    )
    fem.Job(steps=[step], plugins=[fem.LinesearchPlugin()]).evaluate()

    assert fields[1][1].values is x0[1].values
    assert np.all(fem.math.det(third_medium.results.kinematics[0]) > 0)


def test_merge_three_field_variation_third_medium():

    # a C-shaped body and a third medium in between (see examples/ex20)
    t, L, H = 0.1, 1.0, 0.5
    a = fem.Rectangle(a=(0, 0), b=(t, t), n=2)
    b = a.translate(move=H - t, axis=1)
    c = fem.Rectangle(a=(t, 0), b=(L, t), n=(7, 2))
    d = c.translate(move=H - t, axis=1)
    e = fem.Rectangle(a=(0, t), b=(t, H - t), n=(2, 4))
    body = fem.MeshContainer([a, b, c, d, e], merge=True).stack()
    medium = fem.Rectangle(a=(t, t), b=(L, H - t), n=(7, 4))

    G, K = 5 / 14, 5 / 3

    def evaluate(order, three_field_variation=True):
        region_medium = fem.RegionQuad(medium)
        fields = {
            "body": fem.FieldsMixed(fem.RegionQuad(body), n=3, planestrain=True),
            "medium": fem.FieldContainer(
                [
                    fem.FieldPlaneStrain(region_medium, dim=2),
                    fem.Field(region_medium, dim=9),
                ],
                take=[0, 1, 1],
            ),
        }
        if not three_field_variation:
            fields["body"] = fem.FieldContainer(
                [fem.FieldPlaneStrain(fem.RegionQuad(body), dim=2)]
            )

        fields = [fields[name] for name in order]
        x0 = fem.field.merge(fields, decimals=6)

        items = {
            "body": fem.SolidBody(
                fem.ThreeFieldVariation(fem.NeoHooke(mu=G, bulk=50 * G)),
                fields[order.index("body")],
            ),
            "medium": fem.SolidBody(
                fem.ThirdMediumContactMixed(
                    fem.NeoHookeCompressible(mu=G, lmbda=K),
                    gamma=5e-6,
                    alpha_r=5e-6 * K * L**2,
                    p_r=1e-2,
                ),
                fields[order.index("medium")],
                grad=[True, False, True],
            ),
        }
        if not three_field_variation:
            items["body"] = fem.SolidBodyNearlyIncompressible(
                fem.NeoHooke(mu=G), fields[order.index("body")], bulk=50 * G
            )

        items = [items[name] for name in order]

        # no orphan degrees of freedom
        from felupe.tools._newton import jac_items

        matrix = jac_items(items, x0).tocsr()
        assert np.all(np.diff(matrix.indptr) > 0)

        bounds = fem.BoundaryDict(
            fixed=fem.Boundary(x0[0], fx=0),
            move=fem.Boundary(x0[0], fx=L, fy=H, mode="and", skip=(1, 0)),
        )
        step = fem.Step(
            items=items,
            ramp={bounds["move"]: fem.math.linsteps([0, -0.2], num=2)},
            boundaries=bounds,
        )
        fem.Job(steps=[step], plugins=[fem.LinesearchPlugin()]).evaluate()

        points = x0[0].region.mesh.points
        sort = np.lexsort(points.round(6).T[::-1])

        return fields, x0, x0[0].values[sort]

    fields, x0, u_1 = evaluate(["body", "medium"])
    assert x0.fieldsizes[1:] == [body.ncells, body.ncells, 9 * medium.npoints]
    assert fields[0].slots is None
    assert fields[1].slots == [0, 3]

    fields, x0, u_2 = evaluate(["medium", "body"])
    assert x0.fieldsizes[1:] == [9 * medium.npoints, body.ncells, body.ncells]
    assert fields[0].slots is None
    assert fields[1].slots == [0, 2, 3]
    assert all(f.values is x0[s].values for f, s in zip(fields[1], [0, 2, 3]))

    # the order of the bodies doesn't change the solution
    assert np.allclose(u_1, u_2)

    # Q1/P0 three-field variation = nearly-incompressible solid body (condensed)
    fields, x0, u_3 = evaluate(["body", "medium"], three_field_variation=False)
    assert np.allclose(u_1, u_3)


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


def assert_checkpoint_shares_geometry(field, checkpoint):
    "The geometry is shared, the field containers, fields and values are copied."
    copy = checkpoint["field"]

    assert copy is not field
    assert copy.evaluate.field is copy
    assert copy.region is field.region

    for f, c in zip(field.fields, copy.fields):
        assert c is not f
        assert c.region is f.region
        assert c.indices is f.indices
        assert c.values is not f.values
        assert np.all(c.values == f.values)

    # the internal list of fields refers to the copied fields
    for c in copy._list_of_fields_and_field_containers:
        assert any(c is other for other in copy.fields)


def test_container_checkpoint():
    region = fem.RegionHexahedron(fem.Cube(n=3))
    field = fem.FieldContainer([fem.Field(region, dim=3)])
    field[0].values[:] = np.random.default_rng(1).random(field[0].values.shape)
    values = field[0].values.copy()

    checkpoint = field.checkpoint()
    assert_checkpoint_shares_geometry(field, checkpoint)
    assert "field.x0" not in checkpoint

    # the copy is independent of the field container
    F = field.extract()[0].copy()
    field[0].values *= 2
    assert np.allclose(checkpoint["field"][0].values, values)
    assert np.allclose(checkpoint["field"].extract()[0], F)

    # a checkpoint may be restored several times
    for repeat in range(2):
        field.restore(checkpoint)
        assert np.allclose(field[0].values, values)
        field[0].values += 1.0

    assert np.allclose(checkpoint["field"][0].values, values)

    # the copied field container may be pickled
    import pickle

    checkpoint_pickled = pickle.loads(pickle.dumps(checkpoint))
    field.restore(checkpoint_pickled)
    assert np.allclose(field[0].values, values)


def test_container_checkpoint_mixed():
    mesh = fem.Cube(n=3)

    fields = {
        "mixed": fem.FieldsMixed(fem.RegionHexahedron(mesh), n=3),
        "dual": fem.FieldsMixed(
            fem.RegionQuadraticHexahedron(fem.Cube(n=3).add_midpoints_edges()),
            n=2,
        ),
        "axisymmetric": fem.FieldContainer(
            [fem.FieldAxisymmetric(fem.RegionQuad(fem.Rectangle(n=3)), dim=2)]
        ),
        "planestrain": fem.FieldContainer(
            [fem.FieldPlaneStrain(fem.RegionQuad(fem.Rectangle(n=3)), dim=2)]
        ),
    }

    assert any(isinstance(f, fem.FieldDual) for f in fields["dual"].fields)

    for field in fields.values():
        for f in field.fields:
            f.values[:] = np.random.default_rng(2).random(f.values.shape)

        values = [f.values.copy() for f in field.fields]
        checkpoint = field.checkpoint()
        assert_checkpoint_shares_geometry(field, checkpoint)

        for f in field.fields:
            f.values += 1.0

        field.restore(checkpoint)

        for f, v in zip(field.fields, values):
            assert np.allclose(f.values, v)

    # the parent region of a dual field is shared
    dual = [f for f in fields["dual"].fields if isinstance(f, fem.FieldDual)][0]
    copy = fields["dual"].checkpoint()["field"]
    dual_copy = [f for f in copy.fields if isinstance(f, fem.FieldDual)][0]
    assert dual_copy.__args__[0] is dual.__args__[0]


def test_container_checkpoint_x0():
    mesh_1 = fem.Cube(a=(0, 0, 0), b=(1, 1, 1), n=3)
    mesh_2 = fem.Cube(a=(1, 0, 0), b=(2, 1, 1), n=3)
    field_1 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_1), dim=3)])
    field_2 = fem.FieldContainer([fem.Field(fem.RegionHexahedron(mesh_2), dim=3)])
    x0 = fem.field.merge([field_1, field_2])

    x0[0].values[:] = np.random.default_rng(3).random(x0[0].values.shape)
    values = x0[0].values.copy()
    values_1 = field_1[0].values.copy()

    checkpoint = field_1.checkpoint()
    assert_checkpoint_shares_geometry(field_1, checkpoint)

    # one copy of the top-level field container (with the shared mesh container)
    copy_x0 = checkpoint["field.x0"]
    assert checkpoint["field"].x0 is copy_x0
    assert copy_x0 is not x0
    assert copy_x0.region is x0.region
    assert copy_x0.mesh_container is x0.mesh_container
    assert copy_x0[0].values is not x0[0].values

    x0[0].values += 1.0
    field_1[0].values += 1.0
    field_1.restore(checkpoint)

    assert np.allclose(x0[0].values, values)
    assert np.allclose(field_1[0].values, values_1)


def test_container_checkpoint_shared_mesh():
    """Changes of the mesh after a checkpoint is created are visible in the copied
    field container of the checkpoint (the mesh is shared), the restore of the values
    is not affected."""

    mesh = fem.Cube(n=3)
    region = fem.RegionHexahedron(mesh)
    field = fem.FieldContainer([fem.Field(region, dim=3)])

    checkpoint = field.checkpoint()
    mesh.points[:, 0] *= 2

    assert np.all(checkpoint["field"].region.mesh.points[:, 0] == mesh.points[:, 0])

    field[0].values += 1.0
    field.restore(checkpoint)
    assert np.allclose(field[0].values, 0.0)


def test_planestrain_axi_out():
    "The results are written to the given output arrays of the fields in 3d-space."

    mesh = fem.Rectangle(a=(0, 1), b=(2, 3), n=4).add_midpoints_edges()
    mesh = mesh.add_midpoints_faces()
    region = fem.RegionBiQuadraticQuad(mesh, hess=True)
    rng = np.random.default_rng(1)

    for Field, dim in [
        (fem.FieldPlaneStrain, 2),
        (fem.FieldAxisymmetric, 1),
        (fem.FieldAxisymmetric, 2),
    ]:
        values = rng.standard_normal((mesh.npoints, dim))
        field = Field(region, dim=dim, values=values)

        methods = [
            (field.interpolate, {}),
            (field.grad, {}),
            (field.extract, {}),
            (field.extract, {"grad": False}),
        ]

        if dim > 1:
            methods.append((field.grad, {"sym": True}))

        if Field is fem.FieldPlaneStrain:
            methods.append((field.hess, {}))

        for method, kwargs in methods:
            res = method(**kwargs)

            # the out-of-plane components of the output array are overwritten too
            out = np.full_like(res, np.nan)

            assert method(out=out, **kwargs) is out
            assert np.array_equal(out, res)

    # the kinematics of a solid body are re-used
    for Field in [fem.FieldPlaneStrain, fem.FieldAxisymmetric]:
        field = Field(fem.RegionQuad(fem.Rectangle(n=3)), dim=2).as_container()
        solid = fem.SolidBody(fem.NeoHooke(mu=1, bulk=2), field)
        F = solid.results.kinematics[0]
        solid.assemble.vector(field)

        assert solid.results.kinematics[0] is F


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
    test_merge_mixed()
    test_merge_mixed_formulations()
    test_merge_third_medium_contact()
    test_merge_three_field_variation_third_medium()
    test_merge_fewer_points()
    test_field_dual()
    test_field_take()
    test_container_checkpoint()
    test_container_checkpoint_mixed()
    test_container_checkpoint_x0()
    test_container_checkpoint_shared_mesh()
    test_planestrain_axi_out()
