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

from ..math import sym as symmetric
from ._base import Field


class FieldAxisymmetric(Field):
    r"""An axisymmetric :class:`~felupe.Field` on points of a two-dimensional
    :class:`~felupe.Region` with dimension ``dim`` (default is 2) and initial point
    ``values`` (default is 0).
    
    Parameters
    ----------
    region : Region
        The region on which the field will be created.
    dim : int, optional
        The dimension of the field (default is 2).
    values : float or array, optional
        A single value for all components of the field or an array of
        shape (region.mesh.npoints, dim)`. Default is 0.0.
    
    Notes
    -----
    * component 1 =  axial component
    * component 2 = radial component

    ..  code-block::

         x_2 (radial direction)

          ^
          |        _
          |       / \
        --|-----------------> x_1 (axial rotation axis)
                  \_^

    This is a modified :class:`Field` in which the radial coordinates are evaluated at
    the numeric integration points ``q`` for each cell ``c``. The :meth:`grad`-method is
    modified in such a way that it does not only contain the in-plane 2d-gradient but
    also the circumferential stretch, see Eq. :eq:`gradient_axi`.
    
    ..  math::
        :label: gradient_axi
        
        \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} = 
            \begin{bmatrix}
                \left( 
                    \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} 
                \right)_{2d} & \boldsymbol{0} \\
                \boldsymbol{0}^T & \frac{u_r}{R}
            \end{bmatrix}
    
    See Also
    --------
    felupe.Field : Field on points of a :class:`~felupe.Region` with dimension ``dim``
       and initial point ``values``.

    """

    def __init__(self, region, dim=2, values=0.0):
        # init base Field
        super().__init__(region, dim=dim, values=values)

        # create scalar-valued field of radial point values
        self.scalar = Field(region, dim=1, values=region.mesh.points[:, 1])

        # interpolate radial point values to integration points of each cell
        # in the region
        self.radius = self.scalar.interpolate()

    def _interpolate_2d(self, out=None):
        """Interpolate 2D field values at points and evaluate them at the
        integration points of all cells in the region."""

        # interpolated field values "aI"
        # evaluated at quadrature point "q"
        # for cell "c"
        return np.einsum(
            "ca...,aqc->...qc",
            self.values[self.region.mesh.cells],
            self.region.h,
            out=out,
        )

    def interpolate(self, out=None):
        # out-argument is not supported
        # if out is not None:
        #     out = out[:2]

        # extend dimension of in-plane 2d-gradient
        return np.pad(self._interpolate_2d(out=None), ((0, 1), (0, 0), (0, 0)))

    def _grad_2d(self, sym=False, out=None):
        """In-plane 2D gradient as partial derivative of field values at points
        w.r.t. the undeformed coordinates, evaluated at the integration points
        of all cells in the region. Optionally, the symmetric part of the
        gradient is returned.

        Arguments
        ---------
        sym : bool, optional (default is False)
            Calculate the symmetric part of the gradient.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).

        Returns
        -------
        ndarray
            In-plane 2D-gradient as partial derivative of field values at points
            w.r.t. undeformed coordinates, evaluated at the integration points
            of all cells in the region.
        """

        # gradient as partial derivative of field component "I" at point "a"
        # w.r.t. undeformed coordinate "J" evaluated at quadrature point "q"
        # for each cell "c"
        g = np.einsum(
            "ca...,aJqc->...Jqc",
            self.values[self.region.mesh.cells],
            self.region.dhdX,
            out=out,
        )

        if sym:
            return symmetric(g, out=g)
        else:
            return g

    def grad(self, sym=False, out=None):
        """3D-gradient as partial derivative of field values at points w.r.t.
        the undeformed coordinates, evaluated at the integration points of all
        cells in the region. Optionally, the symmetric part of the gradient is
        returned.

        ..  code-block::

                        |  dudX(2d) :   0   |
            dudX(axi) = | ..................|
                        |     0     : u_r/R |

        Arguments
        ---------
        sym : bool, optional
            Calculate the symmetric part of the gradient (default is False).
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).

        Returns
        -------
        ndarray
            Full 3D-gradient as partial derivative of field values at points
            w.r.t. undeformed coordinates, evaluated at the integration points
            of all cells in the region.
        """

        # out-argument is not supported
        # if out is not None:
        #     out = out[:2, :2]

        # extend dimension of in-plane 2d-gradient
        g = np.pad(self._grad_2d(sym=sym, out=None), ((0, 1), (0, 1), (0, 0), (0, 0)))

        # set dudX_33 = u_r / R
        g[-1, -1] = self.interpolate()[1] / self.radius

        return g
