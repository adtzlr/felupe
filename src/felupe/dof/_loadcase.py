# -*- coding: utf-8 -*-
"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from ._boundary import Boundary
from ._tools import apply, partition


def _get_first_field(field):
    "Get first field of FieldContainer."

    return field.fields[0]


def symmetry(field, axes=(True, True, True), x=0, y=0, z=0, bounds=None):
    "Create symmetry boundary conditions."

    # convert axes to array and slice by mesh dimension
    enforce = np.array(axes).astype(bool)[: field.dim]

    # invert boolean identity matrix and use its rows
    # for the skip argument (a symmetry condition on
    # axis "z" fixes all displacements u_z=0 but keeps
    # in-plane displacements active)
    skipax = ~np.eye(3).astype(bool)
    kwarglist = [
        {"fx": x, "skip": skipax[0][: field.dim]},
        {"fy": y, "skip": skipax[1][: field.dim]},
        {"fz": z, "skip": skipax[2][: field.dim]},
    ]

    if bounds is None:
        bounds = {}
    labels = ["symx", "symy", "symz"]

    # loop over symmetry conditions and add them to a new dict
    for a, (symaxis, kwargs) in enumerate(zip(enforce, kwarglist[: field.dim])):
        if symaxis:
            bounds[labels[a]] = Boundary(field, **kwargs)

    return bounds


def uniaxial(field, right=None, move=0.2, axis=0, clamped=False, left=None, sym=True):
    """Define boundaries for uniaxial loading along a given axis on (a quarter of) a
    model (x > 0, y > 0, z > 0) with optional symmetries at x=0, y=0 and z=0."""

    f = _get_first_field(field)

    fx = ["fx", "fy", "fz"][axis]

    mask = np.ones(3, dtype=bool)
    mask[axis] = False

    active = tuple(mask.astype(int))
    inactive = tuple((~mask).astype(int))

    if not hasattr(sym, "__len__"):
        sym = (sym, sym, sym)

    if right is None:
        right = f.region.mesh.points[:, axis].max()

    bounds = symmetry(f, axes=sym)

    if not sym[axis]:
        if left is None:
            left = f.region.mesh.points[:, axis].min()

        bounds["leftx"] = Boundary(f, skip=active, **{fx: left})

    if clamped:
        bounds["right"] = Boundary(f, skip=inactive, **{fx: right})

        if not sym[axis]:
            bounds["leftyz"] = Boundary(f, skip=inactive, **{fx: left})

    bounds["move"] = Boundary(f, skip=active, value=move, **{fx: right})

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def biaxial(field, right=None, move=0.2, clamped=False):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0.

    Note that `clamped=True` is not a valid loadcase for a cube. Use a cross-
    like shape instead where the clamped faces at fx=1 and fy=1 do not share
    mesh-points."""

    f = _get_first_field(field)

    if right is None:
        right = f.region.mesh.points[:, 0].max()

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=right, skip=(1, 0, 0))
        bounds["top"] = Boundary(f, fy=right, skip=(0, 1, 0))

    bounds["move-x"] = Boundary(f, fx=right, skip=(0, 1, 1), value=move)
    bounds["move-y"] = Boundary(f, fy=right, skip=(1, 0, 1), value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def planar(field, right=None, move=0.2, clamped=False):
    """Define boundaries for biaxial loading on a quarter model (x > 0, y > 0,
    z > 0) with symmetries at x=0, y=0 and z=0."""

    f = _get_first_field(field)

    if right is None:
        right = f.region.mesh.points[:, 0].max()

    bounds = symmetry(f)

    if clamped:
        bounds["right"] = Boundary(f, fx=right, skip=(1, 0, 0))

    bounds["move"] = Boundary(f, fx=right, skip=(0, 1, 1), value=move)
    bounds["fix-y"] = Boundary(f, fy=right, skip=(1, 0, 1))

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)


def shear(
    field,
    bottom=None,
    top=None,
    move=0.2,
    axis_shear=0,
    axis_compression=1,
    compression=(0, 0),
    sym=True,
):
    """Define boundaries for shear loading between two clamped plates. The
    bottom plate remains fixed while the shear is applied at the top plate."""

    f = _get_first_field(field)

    if bottom is None:
        bottom = f.region.mesh.points[:, axis_compression].min()

    if top is None:
        top = f.region.mesh.points[:, axis_compression].max()

    if sym:
        axes = [True, True, True]
        axes[axis_shear] = False
        axes[axis_compression] = False

        bounds = symmetry(f, axes=axes)
    else:
        bounds = {}

    fy = ["fx", "fy", "fz"][axis_compression]

    skip_compression = [0, 0, 0]
    skip_compression[axis_compression] = 1

    not_skip_compression = [1, 1, 1]
    not_skip_compression[axis_compression] = 0

    not_skip_thickness = [0, 0, 0]
    not_skip_thickness[axis_compression] = 1
    not_skip_thickness[axis_shear] = 1

    not_skip_shear = [1, 1, 1]
    not_skip_shear[axis_shear] = 0

    bounds["bottom"] = Boundary(f, **{fy: bottom}, skip=skip_compression)
    bounds["top"] = Boundary(f, **{fy: top}, skip=not_skip_thickness)
    bounds["compression_bottom"] = Boundary(
        f, **{fy: bottom}, skip=not_skip_compression, value=compression[0]
    )
    bounds["compression_top"] = Boundary(
        f, **{fy: top}, skip=not_skip_compression, value=-compression[1]
    )
    bounds["move"] = Boundary(f, **{fy: top}, skip=not_skip_shear, value=move)

    dof0, dof1 = partition(field, bounds)
    ext0 = apply(field, bounds, dof0)

    return bounds, dict(dof0=dof0, dof1=dof1, ext0=ext0)
