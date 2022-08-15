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

from ._region import Region
from ._boundary import RegionBoundary
from ..element import (
    ConstantQuad,
    Quad,
    ConstantHexahedron,
    Hexahedron,
    QuadraticHexahedron,
    TriQuadraticHexahedron,
    Triangle,
    QuadraticTriangle,
    Tetra,
    QuadraticTetra,
    TriangleMINI,
    TetraMINI,
    ArbitraryOrderLagrange,
)
from ..quadrature import (
    GaussLegendre,
    GaussLegendreBoundary,
    Triangle as TriangleQuadrature,
    Tetrahedron as TetraQuadrature,
)
from ..mesh import Mesh


class RegionConstantQuad(Region):
    "A region with a constant quad element."

    def __init__(self, mesh, offset=0, npoints=None):

        element = ConstantQuad()
        quadrature = GaussLegendre(order=1, dim=2)

        if npoints is not None:
            npts = npoints
        else:
            npts = offset + mesh.ncells

        points = np.zeros((npts, mesh.dim), dtype=int)
        cells = offset + np.arange(mesh.ncells).reshape(-1, 1)
        m = Mesh(points, cells, mesh.cell_type)

        super().__init__(m, element, quadrature, grad=False)


class RegionQuad(Region):
    "A region with a quad element."

    def __init__(self, mesh):

        element = Quad()
        quadrature = GaussLegendre(order=1, dim=2)

        super().__init__(mesh, element, quadrature)


class RegionQuadBoundary(RegionBoundary):
    "A region with a quad element."

    def __init__(self, mesh, only_surface=True, mask=None, ensure_3d=False):

        element = Quad()
        quadrature = GaussLegendreBoundary(order=1, dim=2)

        super().__init__(
            mesh,
            element,
            quadrature,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
        )


class RegionConstantHexahedron(Region):
    "A region with a constant hexahedron element."

    def __init__(self, mesh, offset=0, npoints=None):

        element = ConstantHexahedron()
        quadrature = GaussLegendre(order=1, dim=3)

        if npoints is not None:
            npts = npoints
            ncells = min(npoints, mesh.ncells)
        else:
            npts = offset + mesh.ncells
            ncells = mesh.ncells

        points = np.zeros((npts, mesh.dim), dtype=int)
        cells = offset + np.arange(ncells).reshape(-1, 1)
        m = Mesh(points, cells, mesh.cell_type)

        super().__init__(m, element, quadrature, grad=False)


class RegionHexahedron(Region):
    "A region with a hexahedron element."

    def __init__(self, mesh):

        element = Hexahedron()
        quadrature = GaussLegendre(order=1, dim=3)

        super().__init__(mesh, element, quadrature)


class RegionHexahedronBoundary(RegionBoundary):
    "A region with a hexahedron element."

    def __init__(self, mesh, only_surface=True, mask=None):

        element = Hexahedron()
        quadrature = GaussLegendreBoundary(order=1, dim=3)

        super().__init__(
            mesh, element, quadrature, only_surface=only_surface, mask=mask
        )


class RegionQuadraticHexahedron(Region):
    "A region with a (serendipity) quadratic hexahedron element."

    def __init__(self, mesh):

        element = QuadraticHexahedron()
        quadrature = GaussLegendre(order=2, dim=3)

        super().__init__(mesh, element, quadrature)


class RegionTriQuadraticHexahedron(Region):
    "A region with a tri-quadratic (lagrange) hexahedron element."

    def __init__(self, mesh):

        element = TriQuadraticHexahedron()
        quadrature = GaussLegendre(order=2, dim=3)

        super().__init__(mesh, element, quadrature)


class RegionLagrange(Region):
    "A region with an arbitrary order lagrange element."

    def __init__(self, mesh, order, dim):

        element = ArbitraryOrderLagrange(order, dim)
        quadrature = GaussLegendre(order, dim, permute=False)

        super().__init__(mesh, element, quadrature)


class RegionTriangle(Region):
    "A region with a triangle element."

    def __init__(self, mesh):

        element = Triangle()
        quadrature = TriangleQuadrature(order=1)

        super().__init__(mesh, element, quadrature)


class RegionTetra(Region):
    "A region with a tetra element."

    def __init__(self, mesh):

        element = Tetra()
        quadrature = TetraQuadrature(order=1)

        super().__init__(mesh, element, quadrature)


class RegionTriangleMINI(Region):
    "A region with a triangle-MINI element."

    def __init__(self, mesh):

        element = TriangleMINI()
        quadrature = TriangleQuadrature(order=2)

        super().__init__(mesh, element, quadrature)


class RegionTetraMINI(Region):
    "A region with a tetra-MINI element."

    def __init__(self, mesh):

        element = TetraMINI()
        quadrature = TetraQuadrature(order=2)

        super().__init__(mesh, element, quadrature)


class RegionQuadraticTriangle(Region):
    "A region with a quadratic triangle element."

    def __init__(self, mesh):

        element = QuadraticTriangle()
        quadrature = TriangleQuadrature(order=1)

        super().__init__(mesh, element, quadrature)


class RegionQuadraticTetra(Region):
    "A region with a quadratic tetra element."

    def __init__(self, mesh):

        element = QuadraticTetra()
        quadrature = TetraQuadrature(order=2)

        super().__init__(mesh, element, quadrature)
