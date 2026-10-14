# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from ..math import det, dot, eigvalsh, transpose
from . import topoints


def save(
    region,
    field,
    forces=None,
    gradient=None,
    filename="result.vtu",
    cell_data=None,
    point_data=None,
):
    """Write field-data to a VTU file.

    Parameters
    ----------
    region : Region
        The region to be saved.
    field : FieldContainer
        The field container to be saved.
    forces : ndarray, optional
        Array with reaction forces to be saved (default is None).
    gradient : list of ndarray, optional
        The result of ``umat.gradient()`` with the first Piola-Kirchhoff stress tensor
        as the first list item (default is None).
    filename : str, optional
        The filename for the results (default is "result.vtu").
    cell_data : dict or None, optional
        Additional dict with cell data (default is None).
    point_data : dict or None, optional
        Additional dict with point data (default is None).

    Examples
    --------
    >>> import felupe as fem
    >>>
    >>> mesh = fem.Cube(n=6)
    >>> region = fem.RegionHexahedron(mesh)
    >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
    >>>
    >>> boundaries = fem.dof.uniaxial(
    ...     field, clamped=True, move=0.3, return_loadcase=False
    ... )
    >>>
    >>> umat = fem.NeoHooke(mu=1)
    >>> solid = fem.SolidBodyNearlyIncompressible(umat, field, bulk=5000)
    >>> step = fem.Step(items=[solid], boundaries=boundaries)
    >>> job = fem.CharacteristicCurve(steps=[step], boundary=boundaries["move"]).evaluate()

    >>> fem.save(region, field, forces=job.res.fun, gradient=solid.results.stress)

    """

    u = field.fields[0]
    mesh = region.mesh

    offsets = field.offsets

    if point_data is None:
        point_data = {}

    point_data["Displacements"] = u.values

    if forces is not None:
        reactionforces = np.split(forces, offsets)[0]
        point_data["Reaction Force"] = reactionforces.reshape(*u.values.shape)

    if gradient is not None:
        # 1st Piola Kirchhoff stress
        F = field.extract()[0]
        P = gradient[0]

        # cauchy stress at integration points
        s = dot(P, transpose(F)) / det(F)
        sp = eigvalsh(s)

        # shift stresses to points and average nodal values
        cauchy = topoints(s, region=region)
        cauchyprinc = topoints(sp, region=region)

        point_data["Cauchy Stress"] = cauchy

        point_data["Cauchy Stress (Max. Principal)"] = cauchyprinc[:, 2]
        point_data["Cauchy Stress (Int. Principal)"] = cauchyprinc[:, 1]
        point_data["Cauchy Stress (Min. Principal)"] = cauchyprinc[:, 0]

        point_data["Cauchy Stress (Max. Principal Shear)"] = (
            cauchyprinc[:, 2] - cauchyprinc[:, 0]
        )

    import meshio

    mesh = meshio.Mesh(
        points=mesh.points,
        cells=[
            (mesh.cell_type, mesh.cells),
        ],
        point_data=point_data,
        cell_data=cell_data,
    )

    mesh.write(filename)
