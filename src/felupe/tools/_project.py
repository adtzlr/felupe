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
from scipy.sparse import csr_matrix as sparsematrix
from scipy.sparse.linalg import spsolve

from ..assembly import IntegralFormCartesian
from ..element import (
    QuadraticTetra,
    QuadraticTriangle,
    Tetra,
    TetraMINI,
    Triangle,
    TriangleMINI,
)
from ..field import Field
from ..quadrature import Tetrahedron as TetrahedronQuadrature
from ..quadrature import Triangle as TriangleQuadrature
from ..region import Region


def topoints(values, region, average=True, mean=False):
    """Shift array of values located at quadrature points of cells to mesh-points.

    Parameters
    ----------
    values : ndarray of shape (..., q, c)
        Array with values located at the quadrature points ``q`` of cells ``c``.
    region : Region
        A region used to shift the values to the mesh-points.
    average : bool, optional
        A flag to return values averaged at mesh-points if True and the mesh of the
        region is not already disconnected or to return values on a disconnected mesh
        with discontinuous values at cell transitions if False (default is True).
    mean : bool, optional
        A flag to shift the cell-means of the values to the mesh-points (default is
        False).

    Returns
    -------
    Field.values : ndarray of shape (p, ...)
        Array of values shifted to the mesh-points ``p``.

    Notes
    -----
    If the number of quadrature-points per cell is greater than the number of points-
    per-cell, then the values are trimmed.
    """

    if mean:
        weights = region.quadrature.weights

        # np.average(keepdims=True) requires numpy >= 1.23.0
        values = np.average(values, axis=-2, weights=weights)
        values = np.expand_dims(values, axis=-2)

    shape = values.shape[:-2]  # tensor-components
    size = np.prod(shape).astype(int)  # tensor-size (enforce int for empty tuple)
    points_per_cell = region.mesh.cells.shape[1]

    if values.shape[-2] == 1:
        # broadcast single quadrature-point values to number of points-per-cell
        values = np.broadcast_to(values, (*shape, points_per_cell, values.shape[-1]))

    elif values.shape[-2] > points_per_cell:
        # trim the values to the number of points-per-cell
        values = values[..., :points_per_cell, :]

    if average:
        # trailing axes (of broadcasted and trimmed values)
        axes = values.shape[-2:]

        # create a field to output the values at mesh-points
        field = Field(region, dim=size)

        # field-values as column vector for result of sparse-matrix
        field.values = field.values.reshape(-1, 1)
        field.values = sparsematrix(
            (values.reshape(-1, *axes).T.ravel(), field.indices.ai),
            shape=field.indices.shape,
        ).toarray(out=field.values)

        # reshape field-values to original shape
        field.values = field.values.reshape(-1, *shape)

        # divide the result by the number of cells per mesh-point
        out = np.einsum(
            "a...,a->a...",
            field.values,
            1 / region.mesh.cells_per_point,
            out=field.values,
        )
    else:
        out = np.einsum("...qc->cq...", values)
        out = out.reshape(-1, *shape)

    return out


def extrapolate(values, region, average=True, mean=False):
    """Extrapolate (and optionally average) an array with values at quadrature points to
    mesh-points.

    Parameters
    ----------
    values : ndarray of shape (..., q, c)
        Array with values located at the quadrature points ``q`` of cells ``c``.
    region : Region
        A region used to extrapolate the values to the mesh-points.
    average : bool, optional
        A flag to return values averaged at mesh-points if True and the mesh of the
        region is not already disconnected or to return values on a disconnected mesh
        with discontinuous values at cell transitions if False (default is True).
    mean : bool, optional
        A flag to take the cell-means, averaged by the quadrature weights, of the
        values (default is False).

    Returns
    -------
    Field.values : ndarray of shape (p, ...)
        Array of values projected to the mesh-points ``p``.
    """

    dim = values.shape[:-2]
    size = int(np.prod(dim))
    weights = region.quadrature.weights

    if mean:
        weights = region.quadrature.weights

        # np.average(keepdims=True) requires numpy >= 1.23.0
        values = np.average(values, axis=-2, weights=weights)
        values = np.expand_dims(values, axis=-2)

        # broadcast mean values to number of points-per-cell
        points_per_cell = region.mesh.cells.shape[1]
        values = np.broadcast_to(values, (*dim, points_per_cell, values.shape[-1]))

    # transpose values
    idx = np.arange(len(values.shape))
    idx[: len(dim)] = idx[: len(dim)][::-1]
    values = values.transpose(idx)

    # reshape from (shape, nint.points, nelements) to 1d-values
    u = values.T.reshape(-1, size)

    # disconnect the mesh
    m = region.mesh.disconnect()

    if mean:
        # region on disconnected mesh with original quadrature scheme
        # a single-point quadrature would be sufficient
        # but would require additional (not available) informations
        r = Region(m, region.element, region.quadrature, grad=False)
    else:
        # region on disconnected mesh with inverse quadrature scheme
        r = Region(m, region.element, region.quadrature.inv(), grad=False)

    # field for values on disconnected mesh; project values to mesh-points
    f = Field(r, dim=size, values=u)
    v = f.interpolate()

    if mean:
        # due to the usage of the original quadrature scheme the averaging must be
        # applied again
        # np.average(keepdims=True) requires numpy >= 1.23.0
        v = np.average(v, axis=-2, weights=weights).reshape(size, 1, -1)

        # broadcast averaged values to match the number of element-points
        shape = np.array([*v.shape[:-2], len(region.element.points), v.shape[-1]])
        v = np.broadcast_to(v, shape=shape)

    if average:
        # create dummy field for values on original mesh
        # (used for calculation of sparse-matrix indices)
        g = Field(region, dim=size)

        # average values
        w = sparsematrix(
            (v.T.ravel(), g.indices.ai), shape=(size * region.mesh.npoints, 1)
        ).toarray().reshape(-1, size) / region.mesh.cells_per_point.reshape(-1, 1)

    else:
        w = v.T

    return w.reshape(-1, *dim)


def project(values, region, average=True, mean=False, dV=None, solver=spsolve):
    r"""Project given values at quadrature-points to mesh-points.

    Parameters
    ----------
    values : ndarray of shape (..., q, c)
        Array with values located at the quadrature points ``q`` of cells ``c``.
    region : Region
        A region used to project the values to the mesh-points.
    average : bool, optional
        A flag to return values averaged at mesh-points if True and the mesh of the
        region is not already disconnected or to return values on a disconnected mesh
        with discontinuous values at cell transitions if False (default is True).
    mean : bool, optional
        A flag to extrapolate the cell-means of the values to the mesh-points, see
        :func:`felupe.tools.extrapolate`. If True, ``dV`` and ``solver`` are ignored.
        Default is False.
    dV : ndarray of shape (q, c) or None, optional
        Differential volumes located at the quadrature points of the cells. If None,
        the differential volumes are taken from the region (default is None).
    solver : callable, optional
        A function for a sparse solver with signature ``x=solver(A, b)``. Default is
        :func:`scipy.sparse.linalg.spsolve`.

    Returns
    -------
    Field.values : ndarray of shape (p, ...)
        Array of values projected to the mesh-points ``p``.

    Notes
    -----
    The projection finds :math:`\hat{\boldsymbol{u}}`, located at mesh-points ``p``, for
    values :math:`v`, evaluated at quadrature-points ``q``, such that the variation of
    the minimization problem in Eq. :eq:`project` is solved.

    ..  math::
        :label: project

        \Pi &= \int_V \frac{1}{2} (u - v) \cdot (u - v)\ dV \quad \rightarrow \quad \min

        \delta \Pi &= \int_V (u - v) \cdot \delta u\ dV = 0

    This leads to assembled system-matrix and -vector(s) as shown in
    Eq. :eq:`project-forms`

    ..  math::
        :label: project-forms

        \int_V v\ \cdot \delta u\ dV &\qquad \rightarrow \qquad \hat{\boldsymbol{A}}

        \int_V u\ \cdot \delta u\ dV &\qquad \rightarrow \qquad \hat{\boldsymbol{b}}

    of an equation system to be solved, see Eq. :eq:`project-solve`. The right-arrows
    in Eq. :eq:`project-forms` represent the assembly of the integral forms into the
    system matrix or vectors.

    ..  math::
        :label: project-solve

        \hat{\boldsymbol{A}}\ \hat{\boldsymbol{u}} = \hat{\boldsymbol{b}}

    The region, on which the values are projected to, may contain a mesh with a
    different (e.g. a disconnected) points- but same cells-array. The only requirement
    on the region is that it must use a compatible quadrature scheme. With the values at
    mesh-points at hand, new fields may be created to project between fields on
    different regions. If the array of differential volumes is not given, then the
    region must be created with ``Region(grad=True)`` to include the differential
    volumes.

    ..  warning::

        Triangular element formulations require regions with quadrature schemes with
        higher integration orders, the default choices are not sufficient for
        projection. For :class:`~felupe.RegionTriangle` and :class:`~felupe.RegionTetra`
        a second-order scheme is used.

        ..  code-block::

            import felupe as fem

            mesh = fem.Rectangle(n=2).triangulate().add_midpoints_edges()
            quadrature = fem.TriangleQuadrature(order=5)
            fem.RegionQuadraticTriangle(mesh, quadrature=quadrature)

        For :class:`~felupe.RegionQuadraticTriangle` and
        :class:`~felupe.RegionQuadraticTetra` use a fifth-order scheme.

        ..  code-block::

            mesh = fem.Cube(n=2).triangulate().add_midpoints_edges()
            quadrature = fem.TetrahedronQuadrature(order=5)
            fem.RegionQuadraticTetra(mesh, quadrature=quadrature)

    Examples
    --------

    ..  code-block::

        import felupe as fem

        mesh = fem.Rectangle(n=2).triangulate()
        region = fem.RegionTriangle(mesh)
        field = fem.FieldAxisymmetric(region, dim=2)
        values = field.extract()

        region2 = region.copy()
        region2.reload(quadrature=fem.TriangleQuadrature(order=2))
        values_projected = project(values, region2)
    """

    if mean:
        return extrapolate(values, region, average=average, mean=mean)

    mesh = None
    if not average:
        mesh = region.mesh.disconnect()  # for non-continuous results

    # quadrature schemes for projection
    # triangles and tetrahedrons require quadratic quadratures for projection
    element = None
    quadrature = None
    schemes = {
        Triangle: TriangleQuadrature(order=2),
        Tetra: TetrahedronQuadrature(order=2),
        TriangleMINI: TriangleQuadrature(order=5),
        TetraMINI: TetrahedronQuadrature(order=5),
        QuadraticTriangle: TriangleQuadrature(order=5),
        QuadraticTetra: TetrahedronQuadrature(order=5),
    }
    scheme = schemes.get(type(region.element))
    if scheme is not None and region.quadrature.npoints < scheme.npoints:
        if region.quadrature.npoints == 1:
            quadrature = scheme
        else:
            raise ValueError(
                " ".join(
                    [
                        "Quadrature not supported (order is too low).",
                        "Take the cell-means with `project(mean=True)` or use a",
                        "higher-order quadrature scheme.",
                    ]
                )
            )

    # copy and reload the region if necessary
    if mesh is not None or element is not None or quadrature is not None:
        region = region.copy(mesh, element, quadrature)

    shape = values.shape[:-2]  # tensor-components
    size = np.prod(shape).astype(int)  # tensor-size (enforce int for empty tuple)
    axes = values.shape[-2:]  # trailing axes

    if dV is None:
        dV = region.dV  # take the differential volumes from the region to project

    # lhs (volume matrix) is constant for all items
    v = u = Field(region, dim=1)  # 1d-field for lhs
    A = IntegralFormCartesian(np.ones((1, 1)), v=v, dV=dV, u=u).assemble()

    # fix diagonal items of the matrix for points not connected to cells
    zeros_on_diagonal = A.diagonal() == 0
    if np.any(zeros_on_diagonal):
        A = A.tolil()
        A[zeros_on_diagonal, zeros_on_diagonal] = 1
        A = A.tocsr()

    # field of unknowns with projected values
    x = Field(region, dim=size)

    # solve with individual right-hand-sides (per tensor-component of values)
    b = IntegralFormCartesian(values.reshape(size, *axes), v=x, dV=dV).assemble()
    x.values[:] = spsolve(A, b.toarray().reshape(-1, size)).reshape(x.values.shape)

    return x.values.reshape(-1, *shape)
