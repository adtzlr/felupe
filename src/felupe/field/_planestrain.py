# SPDX-License-Identifier: GPL-3.0-or-later

from ._base import Field
from ._embed import embed, in_plane


class FieldPlaneStrain(Field):
    r"""A plane strain :class:`~felupe.Field` on points of a two-dimensional
    :class:`~felupe.Region` with dimension ``dim`` (default is 2) and initial point
    ``values`` (default is 0).

    Parameters
    ----------
    region : Region
        The region on which the field will be created.
    dim : int, optional
        The dimension of the field (default is 2).
    values : float or array
        A single value for all components of the field or an array of
        shape (region.mesh.npoints, dim)`. Default is 0.0.
    dtype : data-type or None, optional
        Data-type of the array containing the field values.

    Notes
    -----
    This is a modified :class:`~felupe.Field` for plane strain. The :meth:`grad`-method
    is modified in such a way that the in-plane 2d-gradient is embedded in 3d-space, see
    Eq. :eq:`gradient_planestrain`.

    ..  math::
        :label: gradient_planestrain

        \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}} =
            \begin{bmatrix}
                \left(
                    \frac{\partial \boldsymbol{u}}{\partial \boldsymbol{X}}
                \right)_{2d} & \boldsymbol{0} \\
                \boldsymbol{0}^T & 0
            \end{bmatrix}

    See Also
    --------
    felupe.Field : Field on points of a :class:`~felupe.Region` with dimension ``dim``
       and initial point ``values``.

    """

    def __init__(self, region, dim=2, values=0.0, dtype=None):
        # init base Field
        super().__init__(region, dim=dim, values=values, dtype=dtype)

    def interpolate(self, dtype=None, out=None, order="C"):
        values = super().interpolate(dtype=dtype, out=in_plane(out, 1), order=order)
        return embed(values, out, 1)

    def grad(self, sym=False, dtype=None, out=None, order="C"):
        """3D-gradient as partial derivative of field values at points w.r.t.
        the undeformed coordinates, evaluated at the integration points of all
        cells in the region. Optionally, the symmetric part of the gradient is
        returned.

        ..  code-block::

                                |  dudX(2d) :   0   |
            dudX(planestrain) = | ..................|
                                |     0     :   0   |

        Parameters
        ----------
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

        g = super().grad(sym=sym, dtype=dtype, out=in_plane(out, 2), order=order)
        return embed(g, out, 2)

    def hess(self, dtype=None, out=None, order="C"):
        """3D-Hessian as second partial derivative of field values at points w.r.t.
        the undeformed coordinates, evaluated at the integration points of all
        cells in the region.

        Parameters
        ----------
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
            Full 3D-hessian as second partial derivative of field values at points
            w.r.t. undeformed coordinates, evaluated at the integration points
            of all cells in the region.
        """

        h = super().hess(dtype=dtype, out=in_plane(out, 3), order=order)
        return embed(h, out, 3)
