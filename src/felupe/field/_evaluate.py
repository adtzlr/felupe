# SPDX-License-Identifier: GPL-3.0-or-later

from ..math import right_cauchy_green_deformation as right_cauchy_green
from ..math import strain, strain_stretch_1d


class EvaluateFieldContainer:
    r"""Methods to evaluate the deformation gradient and strain measures of a field
    container.

    Parameters
    ----------
    field : FieldContainer
        A container for fields.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Rectangle(n=4)
        >>> region = fem.RegionQuad(mesh)
        >>> field = fem.FieldContainer([fem.FieldPlaneStrain(region, dim=2)])
        >>>
        >>> evaluate = fem.field.EvaluateFieldContainer(field)
        >>> F = evaluate.deformation_gradient()
        >>>
        >>> F.shape  # (3, 3, nquadraturepoints, ncells)
        (3, 3, 4, 9)

        >>> F[..., 0, 0]  # deformation gradient of first cell, first quadrature point
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])

    See Also
    --------
    felupe.FieldContainer : A container which holds one or multiple (mixed) fields.
    """

    def __init__(self, field):
        self.field = field

    def deformation_gradient(self):
        r"""Return the Deformation gradient tensor.

        .. math::
           :label: deformation-gradient-tensor

           \boldsymbol{F} &= \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}}

           \boldsymbol{F} &= \sum_\alpha \lambda_\alpha
               \ \boldsymbol{n}_\alpha \otimes \boldsymbol{N}_\alpha

        """
        return self.field[0].extract()

    def right_cauchy_green_deformation(self):
        r"""Return the right Cauchy-Green deformation tensor.

        .. math::
           :label:right-cauchy-green-deformation-tensor

           \boldsymbol{F} &= \frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}}

           \boldsymbol{C} &= \boldsymbol{F}^T \boldsymbol{F}

           \boldsymbol{C} &= \sum_\alpha \lambda^2_\alpha
               \ \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

        """
        return right_cauchy_green(self.field)

    def strain(self, fun=strain_stretch_1d, tensor=True, asvoigt=False, n=0, **kwargs):
        r"""Return the Lagrangian strain tensor or its principal values.

        .. math::
           :label: lagrangian-strain-tensor

           \boldsymbol{E} = \sum_\alpha f_\alpha \left( \lambda_\alpha \right)
               \ \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

        By default, the Seth-Hill strain-stretch relation with a strain exponent of
        zero is used, see Eq. :eq:`seth-hill-strain-tensor`.

        .. math::
           :label: seth-hill-strain-tensor

           \boldsymbol{E} = \sum_\alpha \frac{1}{k} \left( \lambda_\alpha^k - 1 \right)
               \ \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

        Parameters
        ----------
        fun : callable, optional
            A callable for the one-dimensional strain-stretch relation. Its Signature
            must be ``lambda stretch, **kwargs: strain`` (default is the log. strain,
            :func:`~felupe.math.strain_stretch_1d` with ``k=0``).
        tensor : bool, optional
            Assemble and return the strain tensor if True or return its principal values
            only if False. Default is True.
        asvoigt : bool, optional
            Return the symmetric strain tensor in reduced vector storage (default is
            False).
        n : int, optional
            The index of the displacement field (default is 0).
        **kwargs : dict, optional
            Optional keyword-arguments are passed to the 1d strain-stretch relation.

        Returns
        -------
        ndarray of shape (N, N, ...) tensor, (N!, ...) asvoigt or (N, ...) princ. values
            The strain tensor or its principal values.

        See Also
        -------
        math.strain : Compute a Lagrangian strain tensor.
        math.strain_stretch_1d : Compute the Seth-Hill strains.
        """
        return strain(
            self.field, fun=fun, tensor=tensor, asvoigt=asvoigt, n=0, **kwargs
        )

    def log_strain(self, tensor=True, asvoigt=False, n=0):
        r"""Return the logarithmic Lagrangian strain tensor or its principal values.

        .. math::
           :label: log-strain-tensor

           \boldsymbol{E} = \sum_\alpha \ln(\lambda_\alpha) \
               \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

        Parameters
        ----------
        tensor : bool, optional
            Assemble and return the strain tensor if True or return its principal values
            only if False. Default is True.
        asvoigt : bool, optional
            Return the symmetric strain tensor in reduced vector storage (default is
            False).
        n : int, optional
            The index of the displacement field (default is 0).

        Returns
        -------
        ndarray of shape (N, N, ...) tensor, (N!, ...) asvoigt or (N, ...) princ. values
            The strain tensor or its principal values.

        See Also
        -------
        math.strain_stretch_1d : Compute the Seth-Hill strains.
        math.strain : Compute a Lagrangian strain tensor.
        """
        return strain(self.field, tensor=tensor, asvoigt=asvoigt, k=0)

    def green_lagrange_strain(self, tensor=True, asvoigt=False, n=0):
        r"""Return the Green-Lagrange Lagrangian strain tensor or its principal values.

        .. math::
           :label: seth-hill-strain-tensor

           \boldsymbol{E} = \sum_\alpha \frac{1}{2} \left( \lambda_\alpha^2 - 1 \right)
               \ \boldsymbol{N}_\alpha \otimes \boldsymbol{N}_\alpha

        Parameters
        ----------
        tensor : bool, optional
            Assemble and return the strain tensor if True or return its principal values
            only if False. Default is True.
        asvoigt : bool, optional
            Return the symmetric strain tensor in reduced vector storage (default is
            False).
        n : int, optional
            The index of the displacement field (default is 0).

        Returns
        -------
        ndarray of shape (N, N, ...) tensor, (N * (N + 1) / 2, ...) asvoigt or (N, ...) princ. values
            The strain tensor or its principal values.

        See Also
        -------
        math.strain : Compute a Lagrangian strain tensor.
        math.strain_stretch_1d : Compute the Seth-Hill strains.
        """
        return strain(self.field, tensor=tensor, asvoigt=asvoigt, k=2)
