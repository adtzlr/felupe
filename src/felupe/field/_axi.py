# SPDX-License-Identifier: GPL-3.0-or-later

from ._base import Field
from ._embed import embed, in_plane


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
    dtype : data-type or None, optional
        Data-type of the array containing the field values.

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

    def __init__(self, region, dim=1, values=0.0, dtype=None):

        # init base Field
        super().__init__(region, dim=dim, values=values, dtype=dtype)

        # create scalar-valued field of radial point values
        self.scalar = Field(region, dim=1, values=region.mesh.points[:, 1], dtype=dtype)

        # interpolate radial point values to integration points of each cell
        # in the region
        self.radius = self.scalar.interpolate()

    def interpolate(self, dtype=None, out=None, order="C"):
        # a scalar field has no out-of-plane component
        axes = int(self.dim > 1)

        values = super().interpolate(dtype=dtype, out=in_plane(out, axes), order=order)
        return embed(values, out, axes)

    def grad(self, sym=False, dtype=None, out=None, order="C"):
        r"""3D-gradient as partial derivative of field values at points w.r.t.
        the undeformed coordinates, evaluated at the integration points of all
        cells in the region. Optionally, the symmetric part of the gradient is
        returned.

        ..  math::

            \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} =
                \begin{bmatrix}
                    \left(
                        \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}
                    \right)_{2d} & \boldsymbol{0} \\
                    \boldsymbol{0}^T & \frac{u_r}{R}
                \end{bmatrix}

        Arguments
        ---------
        sym : bool, optional
            Calculate the symmetric part of the gradient (default is False).
        dtype : data-type or None, optional
            If provided, forces the calculation to use the data type specified. Default
            is None.
        out : None or ndarray, optional
            A location into which the result is stored. If provided, it must have a
            shape that the inputs broadcast to. If not provided or None, a freshly-
            allocated array is returned (default is None).
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the output. 'C' means it should be C
            contiguous. 'F' means it should be Fortran contiguous, 'A' means it should
            be 'F' if the inputs are all 'F', 'C' otherwise. 'K' means it should be as
            close to the layout as the inputs as is possible, including arbitrarily
            permuted axes. Default is 'C'.

        Returns
        -------
        ndarray
            Full 3D-gradient as partial derivative of field values at points
            w.r.t. undeformed coordinates, evaluated at the integration points
            of all cells in the region.
        """

        # a scalar field has no out-of-plane component
        axes = 2 * int(self.dim > 1)

        g = super().grad(sym=sym, dtype=dtype, out=in_plane(out, axes), order=order)
        g = embed(g, out, axes)

        if self.dim > 1:
            # set dudX_33 = u_r / R
            g[-1, -1] = self.interpolate(dtype=dtype, order=order)[1] / self.radius

        return g
