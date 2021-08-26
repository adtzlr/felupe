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

from collections import namedtuple
from copy import deepcopy as copy

import numpy as np
from scipy.sparse import csr_matrix as sparsematrix
from scipy.interpolate import interp1d

import meshio

from .forms import IntegralFormMixed
from .solve import partition, solve
from .math import identity, grad, dot, transpose, eigvals, det, interpolate, norms
from .field import Field

from .mesh import revolve
from .region import Region
from .quadrature import GaussLegendre
from .element import Quad, Hexahedron

from .doftools import apply, partition as dofpartition, extend as dofextend


def dofresiduals(region, r, rref, dof1=None):

    rx = r
    ry = rref

    ry[ry == 0] = np.nan

    rxy = rx / ry

    rxy[region.mesh.cells_per_point == 1] = rx[region.mesh.cells_per_point == 1]

    if dof1 is None:
        return rxy
    else:
        return rxy.ravel()[dof1]


def newtonrhapson(
    region,
    fields,
    u0ext,
    fun_f,
    fun_A,
    dof1,
    dof0,
    unstack,
    maxiter=20,
    tol=1e-6,
    parallel=True,
):

    # extract fields
    # u, p, J = fields
    dV = region.dV

    # deformation gradient at integration points
    F = identity(grad(fields[0])) + grad(fields[0])

    # PK1 stress and elasticity matrix
    f = fun_f(F, *[interpolate(f) for f in fields[1:]])
    A = fun_A(F, *[interpolate(f) for f in fields[1:]])

    # assembly
    r = IntegralFormMixed(f, fields, dV).assemble(parallel=parallel).toarray()[:, 0]
    K = IntegralFormMixed(A, fields, dV, fields).assemble(parallel=parallel)

    converged = False

    for iteration in range(maxiter):

        system = partition(fields, K, dof1, dof0, r)
        dfields = np.split(solve(*system, u0ext), unstack)

        if np.any(np.isnan(dfields[0])):
            break
        else:
            for field, dfield in zip(fields, dfields):
                field += dfield

        # deformation gradient at integration points
        F = identity(grad(fields[0])) + grad(fields[0])

        # PK1 stress and elasticity matrix
        f = fun_f(F, *[interpolate(f) for f in fields[1:]])

        # residuals and stiffness matrix components
        r = IntegralFormMixed(f, fields, dV).assemble(parallel=parallel).toarray()[:, 0]

        rref = np.linalg.norm(r[dof0])
        uref = np.linalg.norm(fields[0].values.ravel()[dof0])
        if rref == 0:
            rref = 1
        if uref == 0:
            uref = 1
        norm_r = np.linalg.norm(r[dof1[dof1 < len(dfields[0])]]) / rref

        norm_u = np.linalg.norm(dfields[0][dof1[dof1 < len(dfields[0])]]) / uref
        norm_dfields = norms(dfields[1:])

        info_r = f"#{iteration+1:2d}: |r|={norm_r:1.3e} |u|={norm_u:1.3e}"
        info_f = [f"(|δ{2+i}|={norm_f:1.3e})" for i, norm_f in enumerate(norm_dfields)]

        print(" ".join([info_r, *info_f]))

        if norm_r < tol:  # np.array(norm_fields).sum() +
            converged = True
            break
        else:
            # elasticity matrix
            A = fun_A(F, *[interpolate(f) for f in fields[1:]])

            # assembly of stiffness matrix components
            K = IntegralFormMixed(A, fields, dV, fields).assemble(parallel=parallel)

    Result = namedtuple(
        "Result", ["fields", "r", "K", "F", "f", "A", "unstack", "converged"]
    )

    return Result(copy(fields), r, K, F, f, A, unstack, converged)


def incsolve(
    fields,
    region,
    f,
    A,
    bounds,
    move,
    boundary="move",
    filename="out",
    maxiter=8,
    tol=1e-6,
    parallel=True,
):

    res = []

    # dofs to dismiss and to keep
    dof0, dof1, unstack = dofpartition(fields, bounds)
    # solve newton iterations and save result
    for increment, move_t in enumerate(move):

        print(f"\nINCREMENT {increment+1:2d}   (move={move_t:1.3g})")
        # set new value on boundary
        bounds[boundary].value = move_t

        # obtain external displacements for prescribed dofs
        u0ext = apply(fields[0], bounds, dof0)

        Result = newtonrhapson(
            region,
            fields,
            u0ext,
            f,
            A,
            dof1,
            dof0,
            unstack,
            maxiter=maxiter,
            tol=tol,
            parallel=parallel,
        )

        fields = copy(Result.fields)

        if not Result.converged:
            # reset counter for last converged increment and break
            increment = increment - 1
            break
        else:
            # save results and go to next increment
            res.append(Result)
            save(region, *Result, filename=filename + ".vtk")
            # save(region, *Result, filename=filename + f"_{increment+1:d}")
            print("SAVED TO FILE")

    savehistory(region, res, filename=filename + ".xdmf")

    return res


def topoints(values, region, sym=True, mode="tensor"):

    rows = region.mesh.cells.T.ravel()
    cols = np.zeros_like(rows)

    if mode == "tensor":
        ndim = values.shape[0]
        if sym:
            if ndim == 3:
                ij = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]
            elif ndim == 2:
                ij = [(0, 0), (1, 1), (0, 1)]
        else:
            if ndim == 3:
                ij = [
                    (0, 0),
                    (0, 1),
                    (0, 2),
                    (1, 0),
                    (1, 1),
                    (1, 2),
                    (2, 0),
                    (2, 1),
                    (2, 2),
                ]
            elif ndim == 2:
                ij = [(0, 0), (0, 1), (1, 0), (1, 1)]

        out = Field(region, dim=len(ij)).values

        for a, (i, j) in enumerate(ij):
            out[:, a] = (
                sparsematrix(
                    (values.reshape(ndim, ndim, -1)[i, j], (rows, cols)),
                    shape=(region.mesh.npoints, 1),
                ).toarray()[:, 0]
                / region.mesh.cells_per_point
            )

    elif mode == "scalar":
        out = sparsematrix(
            (values.ravel(), (rows, cols)), shape=(region.mesh.npoints, 1)
        ).toarray()[:, 0]
        out = out / region.mesh.cells_per_point

    return out


def save(
    region,
    fields,
    r=None,
    K=None,
    F=None,
    f=None,
    A=None,
    unstack=None,
    converged=True,
    filename="result.vtk",
):

    if unstack is not None:
        u = fields[0]
    else:
        u = fields

    mesh = region.mesh

    point_data = {
        "Displacements": u.values,
    }

    if r is not None:
        if unstack is not None:
            reactionforces = np.split(r, unstack)[0]
        else:
            reactionforces = r

        point_data["ReactionForce"] = reactionforces.reshape(*u.values.shape)

    if f is not None:
        # 1st Piola Kirchhoff stress
        if unstack is not None:
            P = f[0]
        else:
            P = f

        # cauchy stress at integration points
        s = dot(P, transpose(F)) / det(F)
        sp = np.sort(eigvals(s), axis=0)

        # shift stresses to points and average nodal values
        cauchy = topoints(s, region=region, sym=True)
        cauchyprinc = [topoints(sp_i, region=region, mode="scalar") for sp_i in sp]

        point_data["CauchyStress"] = cauchy

        point_data["MaxPrincipalCauchyStress"] = cauchyprinc[2]
        point_data["IntPrincipalCauchyStress"] = cauchyprinc[1]
        point_data["MinPrincipalCauchyStress"] = cauchyprinc[0]

        point_data["MaxPrincipalShearCauchyStress"] = cauchyprinc[2] - cauchyprinc[0]

    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[
            (mesh.cell_type, mesh.cells),
        ],  # [:, : mesh.edgepoints]},
        # Optionally provide extra data on points, cells, etc.
        point_data=point_data,
    )

    if filename is not None:
        mesh.write(filename)


def savehistory(region, results, filename="result_history.xdmf"):

    mesh = region.mesh
    points = mesh.points
    cells = [
        (mesh.cell_type, mesh.cells),
    ]  # [:, : mesh.edgepoints]}

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(points, cells)

        for inc, result in enumerate(results):
            fields, r, K, F, f, A, unstack, converged = result

            if unstack is not None:
                reactionforces = np.split(r, unstack)[0]
            else:
                reactionforces = r

            u = fields[0]

            point_data = {
                "Displacements": u.values,
                "ReactionForce": reactionforces.reshape(*u.values.shape),
            }

            if f is not None:
                # cauchy stress at integration points
                s = dot(f[0], transpose(F)) / det(F)
                sp = eigvals(s)

                # shift stresses to points and average nodal values
                cauchy = topoints(s, region=region, sym=True)
                cauchyprinc = [
                    topoints(sp_i, region=region, mode="scalar") for sp_i in sp
                ]

                point_data["CauchyStress"] = cauchy

                point_data["MaxPrincipalCauchyStress"] = cauchyprinc[2]
                point_data["IntPrincipalCauchyStress"] = cauchyprinc[1]
                point_data["MinPrincipalCauchyStress"] = cauchyprinc[0]

            writer.write_data(inc, point_data=point_data)


def force(results, boundary):
    return np.array(
        [
            (((np.split(res.r, res.unstack)[0]).reshape(-1, 3))[boundary.points]).sum(0)
            for res in results
        ]
    )


def moment(results, boundary, point=np.zeros(3)):

    points = results[0].fields[0].region.mesh.points
    points = point.reshape(-1, 3)

    indices = np.array([(1, 2), (2, 0), (0, 1)])

    if points.shape[0] == 1:
        points = np.tile(points, (len(results), 1))

    moments = []

    for pt, res in zip(points, results):
        displacements = res.fields[0].values
        d = ((points + displacements) - pt)[boundary.points]

        force = (np.split(res.r, res.unstack)[0]).reshape(-1, 3)
        f = force[boundary.points]

        moments.append([(f[:, i] * d[:, i[::-1]]).sum() for i in indices])

    return np.array(moments)


def curve(x, y):
    if len(y) > 1:
        kind = "linear"
    if len(y) > 2:
        kind = "quadratic"
    if len(y) > 3:
        kind = "cubic"

    f = interp1d(x[: len(y)], y, kind=kind)
    xx = np.linspace(x[0], x[: len(y)][-1])
    return np.array([x[: len(y)], y]), np.array([xx, f(xx)])


def axito3d(
    mesh_axi, element_axi, quadrature_axi, region_axi, field_axi, n=11, phi=180
):
    """Axisymmetric -> 3D expansion
    ===============================

    Restrictions
    ------------
    * only Quad1-Elements are supported
    * no mixed-field support
    * quadrature must be GaussLegendre based
    """

    mesh_3d = revolve(mesh_axi, n=n, phi=phi)
    values_3d = revolve((field_axi.values, mesh_axi.cells), n=n, phi=phi)[0]

    edict = {Quad: Hexahedron}

    element_3d = edict[type(element_axi)]()

    order = int((quadrature_axi.npoints / 2 - 1) ** 3)
    quadrature_3d = GaussLegendre(order=order, dim=3)
    region_3d = Region(mesh_3d, element_3d, quadrature_3d)
    field_3d = Field(region_3d, dim=3, values=values_3d)

    return mesh_3d, element_3d, quadrature_3d, region_3d, field_3d
