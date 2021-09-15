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

from copy import deepcopy

from numpy.linalg import norm
import numpy as np

import meshio

from .math import norms, dot, transpose, det, eigvals, extract
from . import solve as solvetools
from .doftools import partition as dofpartition, apply
from .form import IntegralFormMixed
from . import utils


def update(y, dx, inplace=True):
    "Update field values."

    if inplace:
        x = y
    else:
        x = deepcopy(y)

    if isinstance(x, tuple) or isinstance(x, list):
        for field, dfield in zip(x, dx):
            field += dfield
    elif "mixed" in str(type(x)):
        for field, dfield in zip(x.fields, dx):
            field += dfield
    else:
        x += dx

    return x


def solve(K, f, field, dof0, dof1, ext, unstack=None):
    "Solve linear equation system K dx = b"

    if isinstance(field, tuple) or isinstance(field, list):
        return _solve_mixed(K, f, field, dof0, dof1, ext, unstack)

    elif "mixed" in str(type(field)):
        return _solve_mixed(K, f, field, dof0, dof1, ext, unstack)

    else:
        return _solve_single(K, f, field, dof0, dof1, ext)


def _solve_single(K, f, field, dof0, dof1, ext):
    "Solve linear equation system K dx = b"
    system = solvetools.partition(field, K, dof1, dof0, -f)
    dx = solvetools.solve(*system, ext)
    return dx


def _solve_mixed(K, f, field, dof0, dof1, ext, unstack):
    "Solve linear equation system K dx = b"
    system = solvetools.partition(field, K, dof1, dof0, -f)
    dfields = np.split(solvetools.solve(*system, ext), unstack)
    return dfields


def check(dfields, fields, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dfields is valid."

    if (
        isinstance(fields, tuple)
        or isinstance(fields, list)
        or "mixed" in str(type(fields))
    ):
        return _check_mixed(dfields, fields, f, dof1, dof0, tol_f, tol_x, verbose)

    else:
        x = fields
        dx = dfields
        return _check_single(dx, x, f, dof1, dof0, tol_f, tol_x, verbose)


def _check_single(dx, x, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dx is valid."

    # get reference values of "f" and "x"
    ref_f = 1 if np.linalg.norm(f[dof0]) == 0 else np.linalg.norm(f[dof0])
    ref_x = 1 if np.linalg.norm(x[dof0]) == 0 else np.linalg.norm(x[dof0])

    norm_f = np.linalg.norm(f[dof1]) / ref_f
    norm_x = np.linalg.norm(dx.ravel()[dof1]) / ref_x

    if verbose:
        info_r = f"|r|={norm_f:1.3e} |u|={norm_x:1.3e}"
        print(info_r)

    if norm_f < tol_f and norm_x < tol_x:
        success = True
    else:
        success = False

    return success


def _check_mixed(dfields, fields, f, dof1, dof0, tol_f=1e-3, tol_x=1e-3, verbose=1):
    "Check if solution dfields is valid."

    if "mixed" in str(type(fields)):
        fields = fields.fields

    x = fields[0]
    dx = dfields[0]

    # get reference values of "f" and "x"
    ref_f = 1 if np.linalg.norm(f[dof0]) == 0 else np.linalg.norm(f[dof0])
    ref_x = 1 if np.linalg.norm(x[dof0]) == 0 else np.linalg.norm(x[dof0])

    norm_f = np.linalg.norm(f[dof1[dof1 < len(dx)]]) / ref_f
    norm_x = np.linalg.norm(dx.ravel()[dof1[dof1 < len(dx)]]) / ref_x

    norm_dfields = norms(dfields[1:])

    if verbose:
        info_r = f"|r|={norm_f:1.3e} |u|={norm_x:1.3e}"
        info_f = [f"(|δ{2+i}|={norm_f:1.3e})" for i, norm_f in enumerate(norm_dfields)]

        print(" ".join([info_r, *info_f]))

    if norm_f < tol_f and norm_x < tol_x:
        success = True
    else:
        success = False

    return success


class Result:
    def __init__(self, x, y=None, fun=None, jac=None, success=None, iterations=None):
        "Result class."
        self.x = x
        self.y = y
        self.fun = fun
        self.jac = jac
        self.success = success
        self.iterations = iterations


def newtonrhapson(
    fun,
    x0,
    jac,
    solve=np.linalg.solve,
    maxiter=8,
    pre=lambda x: x,
    update=lambda x, dx: x + dx,
    check=lambda dx, x, f: norm(dx) < np.finfo(float).eps,
    args=(),
    kwargs={},
    kwargs_solve={},
    kwargs_check={},
    export_jac=False,
):
    """
    General-purpose Newton-Rhapson algorithm
    ========================================

    (Nonlinear) equilibrium equations `f`, as a function `f(x)` of the
    unknowns `x`, are solved by linearization of `f` at given unknowns `x0`.

        f(x0) = 0                                          (1)

        f(x0 + dx)     =  f(x0) + (df/dx)(x0) dx (= 0)     (2)
        (df/dx)(x0) dx = -f(x0)

        dx = solve(df/dx(x0), -f(x0))                      (3)

         x = x0 + dx                                       (4)

    Repeated evaluations of Eq.(3) and Eq.(4) lead to an incrementally updated
    solution of `x` which is shown in equation (4). Herein `xn` refer to the
    inital unknowns whereas `x` to the updated unknowns (the subscript `n+1`
    is dropped for readability).

        dx = solve(df/dx(xn), -f(xn))                      (5)

         x = xn + dx                                       (6)

    Eq.(5) and Eq.(6) are repeated until `check(dx, x, f)` returns `True`.

    """

    # copy x0
    x = x0
    # x = deepcopy(x0)

    # pre-evaluate function at given unknowns "x"
    f = fun(pre(x), *args, **kwargs)

    # iteration loop
    for iteration in range(maxiter):

        # evaluate jacobian at unknowns "x"
        K = jac(pre(x), *args, **kwargs)

        # solve linear system and update solution
        result = solve(K, -f, **kwargs_solve)
        x = update(x, result)

        # evaluate function at unknowns "x"
        f = fun(pre(x), *args, **kwargs)

        # check success of solution
        success = check(result, x, f, **kwargs_check)

        if success:
            break

    Res = Result(x=x, y=pre(x), fun=f, success=success, iterations=1 + iteration)

    if export_jac:
        Res.jac = K

    return Res


def incsolve(
    field,
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
    verbose=1,
):

    res = []

    # dofs to dismiss and to keep
    dof0, dof1, unstack = dofpartition(field, bounds)
    # solve newton iterations and save result
    for increment, move_t in enumerate(move):

        if verbose > 0:
            print(f"\nINCREMENT {increment+1:2d}   (move={move_t:1.3g})")

        # set new value on boundary
        bounds[boundary].value = move_t

        # obtain external displacements for prescribed dofs
        u0ext = apply(field.fields[0], bounds, dof0)

        def fun(x):
            linearform = IntegralFormMixed(f(*x), field, region.dV)
            return linearform.assemble(parallel=parallel).toarray()[:, 0]

        def jac(x):
            bilinearform = IntegralFormMixed(A(*x), field, region.dV, field)
            return bilinearform.assemble(parallel=parallel)

        Result = newtonrhapson(
            fun,
            field,
            jac,
            solve=solve,
            pre=extract,
            update=update,
            check=check,
            kwargs_solve={
                "field": field,
                "ext": u0ext,
                "dof0": dof0,
                "dof1": dof1,
                "unstack": unstack,
            },
            kwargs_check={
                "tol_f": tol,
                "tol_x": tol,
                "dof0": dof0,
                "dof1": dof1,
                "verbose": verbose,
            },
        )

        Result.F = Result.y[0]
        Result.f = f(*Result.y)
        Result.unstack = unstack

        fields = deepcopy(Result.x)

        if not Result.success:
            # reset counter for last converged increment and break
            increment = increment - 1
            break
        else:
            # save results and go to next increment
            res.append(Result)
            utils.save(
                region,
                fields.fields,
                Result.fun,
                Result.jac,
                Result.F,
                Result.f,
                None,
                unstack,
                Result.success,
                filename=filename + ".vtk",
            )
            # save(region, *Result, filename=filename + f"_{increment+1:d}")
            if verbose > 0:
                print("SAVED TO FILE")

    savehistory(region, res, filename=filename + ".xdmf")

    return res


def savehistory(region, results, filename="result_history.xdmf"):

    mesh = region.mesh
    points = mesh.points
    # cells = {mesh.cell_type: mesh.cells}  # [:, : mesh.edgepoints]}
    cells = [
        (mesh.cell_type, mesh.cells),
    ]

    with meshio.xdmf.TimeSeriesWriter(filename) as writer:
        writer.write_points_cells(points, cells)

        for inc, result in enumerate(results):
            fields = result.x
            r = result.fun
            F = result.F
            f = result.f
            unstack = result.unstack

            if unstack is not None:
                reactionforces = np.split(r, unstack)[0]
            else:
                reactionforces = r

            if isinstance(fields, tuple) or isinstance(fields, list):
                u = fields[0]
            elif "mixed" in str(type(fields)):
                u = fields.fields[0]
            else:
                u = fields

            point_data = {
                "Displacements": u.values,
                "ReactionForce": reactionforces.reshape(*u.values.shape),
            }

            if f is not None:
                # cauchy stress at integration points
                s = dot(f[0], transpose(F)) / det(F)
                sp = eigvals(s)

                # shift stresses to points and average nodal values
                cauchy = utils.topoints(s, region=region, sym=True)
                cauchyprinc = [
                    utils.topoints(sp_i, region=region, mode="scalar") for sp_i in sp
                ]

                point_data["CauchyStress"] = cauchy

                point_data["MaxPrincipalCauchyStress"] = cauchyprinc[2]
                point_data["IntPrincipalCauchyStress"] = cauchyprinc[1]
                point_data["MinPrincipalCauchyStress"] = cauchyprinc[0]

            writer.write_data(inc, point_data=point_data)
