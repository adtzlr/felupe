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

from ._boundary import Boundary
from ._tools import partition, apply


def _get_first_field(field):
    "Get first field of FieldContainer."

    return field.fields[0]


def symmetry(field, axes=(True, True, True), x=0, y=0, z=0, bounds=None):
    "Create symmetry boundary conditions."

    # convert axes to array and slice by mesh dimension
    enforce = np.array(axes).astype(bool)[: field.dim]

    # create search functions for x,y,z - axes
    fx = lambda v: np.isclose(v, x)
    fy = lambda v: np.isclose(v, y)
    fz = lambda v: np.isclose(v, z)

    # invert boolean identity matrix and use its rows
    # for the skip argument (a symmetry condition on
    # axis "z" fixes all displacements u_z=0 but keeps
    # in-plane displacements active)
    skipax = ~np.eye(3).astype(bool)
    kwarglist = [
        {"fx": fx, "skip": skipax[0][: field.dim]},
        {"fy": fy, "skip": skipax[1][: field.dim]},
        {"fz": fz, "skip": skipax[2][: field.dim]},
    ]

    if bounds is None:
        bounds = {}
    labels = ["symx", "symy", "symz"]

    # loop over symmetry conditions and add them to a new dict
    for a, (symaxis, kwargs) in enumerate(zip(enforce, kwarglist[: field.dim])):
        if symaxis:
            bounds[labels[a]] = Boundary(field, **kwargs)

    return bounds


def uniaxial(field, right=1, move=0.2, clamped=False):
    """Define boundaries for uniaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0."""

    f = _get_first_field(field)

    f1 = lambda x: np.isclose(x, right)

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=f1, skip=(1, 0, 0))

    bounds["move"] = Boundary(f, fx=f1, skip=(0, 1, 1), value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dof0, dof1, ext0


def biaxial(field, right=1, move=0.2, clamped=False):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0.

    Note that `clamped=True` is not a valid loadcase for a cube. Use a cross-
    like shape instead where the clamped faces at fx=1 and fy=1 do not share
    mesh-points."""

    f = _get_first_field(field)

    f1 = lambda x: np.isclose(x, right)

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=f1, skip=(1, 0, 0))
        bounds["top"] = Boundary(f, fy=f1, skip=(0, 1, 0))

    bounds["move-x"] = Boundary(f, fx=f1, skip=(0, 1, 1), value=move)
    bounds["move-y"] = Boundary(f, fy=f1, skip=(1, 0, 1), value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dof0, dof1, ext0


def planar(field, right=1, move=0.2, clamped=False):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0."""

    f = _get_first_field(field)

    f1 = lambda x: np.isclose(x, right)

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=f1, skip=(1, 0, 0))

    bounds["move"] = Boundary(f, fx=f1, skip=(0, 1, 1), value=move)
    bounds["fix-y"] = Boundary(f, fy=f1, skip=(1, 0, 1))

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dof0, dof1, ext0


def shear(field, bottom=0, top=1, move=0.2, sym=True):
    """Define boundaries for shear loading between two clamped plates. The
    bottom plate remains fixed while the shear is applied at the top plate."""

    f = _get_first_field(field)

    f0 = lambda x: np.isclose(x, bottom)
    f1 = lambda x: np.isclose(x, top)

    if sym:
        bounds = symmetry(f, axes=(False, False, True))
    else:
        bounds = {}

    bounds["bottom"] = Boundary(f, fy=f0)
    bounds["top"] = Boundary(f, fy=f1, skip=(1, 0, 0))
    bounds["move"] = Boundary(f, fy=f1, skip=(0, 1, 1), value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dof0, dof1, ext0
