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

from ..element import (
    ArbitraryOrderLagrange,
    BiQuadraticQuad,
    ConstantHexahedron,
    ConstantQuad,
    Hexahedron,
    Quad,
    QuadraticHexahedron,
    QuadraticQuad,
    QuadraticTetra,
    QuadraticTriangle,
    Tetra,
    TetraMINI,
    Triangle,
    TriangleMINI,
    TriQuadraticHexahedron,
)
from ..mesh import Mesh
from ..quadrature import GaussLegendre, GaussLegendreBoundary
from ..quadrature import Tetrahedron as TetraQuadrature
from ..quadrature import Triangle as TriangleQuadrature
from ._boundary import RegionBoundary
from ._region import Region


class RegionConstantQuad(Region):
    "A region with a constant quad element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendre(order=1, dim=2),
        grad=False,
    ):

        element = ConstantQuad()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuad(Region):
    "A region with a quad element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=1, dim=2), grad=True):

        element = Quad()

        if len(mesh.cells.T) > 4:
            mesh = Mesh(mesh.points, mesh.cells[:, :4], "quad")

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuadraticQuad(Region):
    "A region with a (serendipity) quadratic quad element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=2, dim=2), grad=True):

        element = QuadraticQuad()

        if len(mesh.cells.T) > 8:
            mesh = Mesh(mesh.points, mesh.cells[:, :8], "quad8")

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionBiQuadraticQuad(Region):
    "A region with a bi-quadratic (lagrange) quad element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=2, dim=2), grad=True):

        element = BiQuadraticQuad()

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuadBoundary(RegionBoundary):
    "A region with a quad element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=1, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
    ):

        element = Quad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
        )


class RegionQuadraticQuadBoundary(RegionBoundary):
    "A boundary region with a quadratic quad element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
    ):

        element = QuadraticQuad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
        )


class RegionBiQuadraticQuadBoundary(RegionBoundary):
    "A boundary region with a bi-quadratic quad element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=2),
        grad=True,
        only_surface=True,
        mask=None,
        ensure_3d=False,
    ):

        element = BiQuadraticQuad()

        super().__init__(
            mesh,
            element,
            quadrature,
            grad=grad,
            only_surface=only_surface,
            mask=mask,
            ensure_3d=ensure_3d,
        )


class RegionConstantHexahedron(Region):
    "A region with a constant hexahedron element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendre(order=1, dim=3),
        grad=False,
    ):

        element = ConstantHexahedron()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionHexahedron(Region):
    "A region with a hexahedron element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=1, dim=3), grad=True):

        element = Hexahedron()

        if len(mesh.cells.T) > 8:
            mesh = Mesh(mesh.points, mesh.cells[:, :8], "hexahedron")

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionHexahedronBoundary(RegionBoundary):
    "A region with a hexahedron element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=1, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
    ):

        element = Hexahedron()
        super().__init__(
            mesh, element, quadrature, grad=grad, only_surface=only_surface, mask=mask
        )


class RegionQuadraticHexahedron(Region):
    "A region with a (serendipity) quadratic hexahedron element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=2, dim=3), grad=True):

        element = QuadraticHexahedron()

        if len(mesh.cells.T) > 20:
            mesh = Mesh(mesh.points, mesh.cells[:, :20], "hexahedron20")

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuadraticHexahedronBoundary(RegionBoundary):
    "A boundary region with a (serendipity) quadratic hexahedron element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
    ):

        element = QuadraticHexahedron()
        super().__init__(
            mesh, element, quadrature, grad=grad, only_surface=only_surface, mask=mask
        )


class RegionTriQuadraticHexahedron(Region):
    "A region with a tri-quadratic (lagrange) hexahedron element."

    def __init__(self, mesh, quadrature=GaussLegendre(order=2, dim=3), grad=True):

        element = TriQuadraticHexahedron()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionTriQuadraticHexahedronBoundary(RegionBoundary):
    "A boundary region with a tri-quadratic (lagrange) hexahedron element."

    def __init__(
        self,
        mesh,
        quadrature=GaussLegendreBoundary(order=2, dim=3),
        grad=True,
        only_surface=True,
        mask=None,
    ):

        element = TriQuadraticHexahedron()
        super().__init__(
            mesh, element, quadrature, grad=grad, only_surface=only_surface, mask=mask
        )


class RegionLagrange(Region):
    "A region with an arbitrary order lagrange element."

    def __init__(self, mesh, order, dim, quadrature=None, grad=True):

        if quadrature is None:
            quadrature = GaussLegendre(order=order, dim=dim, permute=False)

        element = ArbitraryOrderLagrange(order, dim)
        self.order = order

        super().__init__(mesh, element, quadrature, grad=grad)


class RegionTriangle(Region):
    "A region with a triangle element."

    def __init__(self, mesh, quadrature=TriangleQuadrature(order=1), grad=True):

        element = Triangle()

        if len(mesh.cells.T) > 3:
            m = Mesh(mesh.points, mesh.cells[:, :3], "triangle")
        else:
            m = mesh

        super().__init__(m, element, quadrature, grad=grad)


class RegionTetra(Region):
    "A region with a tetra element."

    def __init__(self, mesh, quadrature=TetraQuadrature(order=1), grad=True):

        element = Tetra()

        if len(mesh.cells.T) > 4:
            m = Mesh(mesh.points, mesh.cells[:, :4], "tetra")
        else:
            m = mesh

        super().__init__(m, element, quadrature, grad=grad)


class RegionTriangleMINI(Region):
    "A region with a triangle-MINI element."

    def __init__(self, mesh, quadrature=TriangleQuadrature(order=2), grad=True):

        element = TriangleMINI()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionTetraMINI(Region):
    "A region with a tetra-MINI element."

    def __init__(self, mesh, quadrature=TetraQuadrature(order=2), grad=True):

        element = TetraMINI()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuadraticTriangle(Region):
    "A region with a quadratic triangle element."

    def __init__(self, mesh, quadrature=TriangleQuadrature(order=2), grad=True):

        element = QuadraticTriangle()
        super().__init__(mesh, element, quadrature, grad=grad)


class RegionQuadraticTetra(Region):
    "A region with a quadratic tetra element."

    def __init__(self, mesh, quadrature=TetraQuadrature(order=2), grad=True):

        element = QuadraticTetra()
        super().__init__(mesh, element, quadrature, grad=grad)
