# SPDX-License-Identifier: GPL-3.0-or-later

import inspect

from ._helpers import Evaluate


class EvaluateSolidBody(Evaluate):
    r"""Methods to evaluate the stresses and elasticity tensors of a solid body.

    Parameters
    ----------
    solid : SolidBody or SolidBodyNearlyIncompressible
        A solid body.

    Examples
    --------
    ..  pyvista-plot::

        >>> import felupe as fem
        >>>
        >>> mesh = fem.Cube(n=3)
        >>> region = fem.RegionHexahedron(mesh)
        >>> field = fem.FieldContainer([fem.Field(region, dim=3)])
        >>> umat = fem.NeoHooke(mu=1.0, bulk=2.0)
        >>> solid = fem.SolidBody(umat, field)
        >>>
        >>> evaluate = fem.mechanics.EvaluateSolidBody(solid)
        >>> s = evaluate.cauchy_stress()
        >>> s.shape  # (3, 3, nquadraturepoints, ncells)
        (3, 3, 8, 8)

    See Also
    --------
    felupe.SolidBody : A solid body with methods for the assembly of sparse vectors/matrices.
    felupe.SolidBodyNearlyIncompressible : A (nearly) incompressible solid body with
        methods for the assembly of sparse vectors/matrices.
    """

    def __init__(self, solid):
        self.solid = solid

    def gradient(self, field=None, *args, **kwargs):
        r"""Evaluate the first partial derivative of the strain energy density w.r.t.
        the kinematics (the first Piola-Kirchhoff stress tensor) and update internal
        state variables in the results of the solid body.

        .. math::
           :label: solid-body-gradient

           \boldsymbol{P} = \frac{\partial \psi}{\partial \boldsymbol{F}}

        Parameters
        ----------
        field : FieldContainer or None, optional
            A field container with one or more fields (default is None, which uses the
            current field of the solid body).
        *args : tuple, optional
            Optional positional arguments passed to the material gradient function.
        **kwargs : dict, optional
            Optional keyword arguments passed to the material gradient function.

        Returns
        -------
        list or tuple of ndarray
            The evaluated gradients (the first entry is the first Piola-Kirchhoff
            stress tensor :math:`\boldsymbol{P}`).
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        if "args" in kwargs:
            args = kwargs.pop("args")
        if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
            extra_kwargs = kwargs.pop("kwargs")
            kwargs.update(extra_kwargs)
        if "parallel" in inspect.signature(self.solid._gradient).parameters:
            parallel = kwargs.pop("parallel", False)
            return self.solid._gradient(
                field=field, parallel=parallel, args=args, kwargs=kwargs
            )
        return self.solid._gradient(field=field, args=args, kwargs=kwargs)

    def hessian(self, field=None, *args, **kwargs):
        r"""Evaluate the second partial derivative of the strain energy density w.r.t.
        the kinematics (the fourth-order elasticity tensor) and store the elasticity
        in the results of the solid body.

        .. math::
           :label: solid-body-hessian

           \boldsymbol{\mathcal{A}} = \frac{\partial^2 \psi}{
               \partial \boldsymbol{F}\ \partial \boldsymbol{F}
           }

        Parameters
        ----------
        field : FieldContainer or None, optional
            A field container with one or more fields (default is None, which uses the
            current field of the solid body).
        *args : tuple, optional
            Optional positional arguments passed to the material hessian function.
        **kwargs : dict, optional
            Optional keyword arguments passed to the material hessian function.

        Returns
        -------
        list or tuple of ndarray
            The evaluated elasticity tensors (the first entry is the fourth-order
            elasticity tensor :math:`\boldsymbol{\mathcal{A}}`).
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        if "args" in kwargs:
            args = kwargs.pop("args")
        if "kwargs" in kwargs and isinstance(kwargs["kwargs"], dict):
            extra_kwargs = kwargs.pop("kwargs")
            kwargs.update(extra_kwargs)
        if "parallel" in inspect.signature(self.solid._hessian).parameters:
            parallel = kwargs.pop("parallel", False)
            return self.solid._hessian(
                field=field, parallel=parallel, args=args, kwargs=kwargs
            )
        return self.solid._hessian(field=field, args=args, kwargs=kwargs)

    def stress(self, field=None, **kwargs):
        r"""Evaluate and return the first Piola-Kirchhoff stress tensor (or engineering
        stress in linear elasticity).

        .. math::
           :label: solid-body-stress

           \boldsymbol{P} = \frac{\partial \psi}{\partial \boldsymbol{F}}

        Parameters
        ----------
        field : FieldContainer or None, optional
            A field container with one or more fields (default is None, which uses the
            current field of the solid body).
        **kwargs : dict, optional
            Optional keyword arguments passed to the material gradient function.

        Returns
        -------
        ndarray
            The first Piola-Kirchhoff stress tensor :math:`\boldsymbol{P}`.

        See Also
        --------
        felupe.mechanics.EvaluateSolidBody.cauchy_stress : Return the Cauchy stress tensor.
        felupe.mechanics.EvaluateSolidBody.kirchhoff_stress : Return the Kirchhoff stress tensor.
        """
        return self.gradient(field=field, **kwargs)[0]

    def cauchy_stress(self, field=None):
        r"""Return the Cauchy stress tensor.

        .. math::
           :label: solid-body-cauchy-stress

           \boldsymbol{\sigma} = \frac{1}{J} \boldsymbol{P} \boldsymbol{F}^T

        Parameters
        ----------
        field : FieldContainer or None, optional
            A field container with one or more fields (default is None, which uses the
            current field of the solid body).

        Returns
        -------
        ndarray
            The Cauchy stress tensor :math:`\boldsymbol{\sigma}`.

        Notes
        -----
        For two-dimensional tensors, the volume ratio :math:`J` is not available and
        the Kirchhoff stress tensor is returned (with a warning).

        See Also
        --------
        felupe.mechanics.cauchy_stress : Return the Cauchy stress tensor from given stress
            and deformation gradient tensors.
        felupe.mechanics.EvaluateSolidBody.kirchhoff_stress : Return the Kirchhoff stress tensor.
        """
        return self.solid._cauchy_stress(field=field)

    def kirchhoff_stress(self, field=None):
        r"""Return the Kirchhoff stress tensor.

        .. math::
           :label: solid-body-kirchhoff-stress

           \boldsymbol{\tau} = \boldsymbol{P} \boldsymbol{F}^T

        Parameters
        ----------
        field : FieldContainer or None, optional
            A field container with one or more fields (default is None, which uses the
            current field of the solid body).

        Returns
        -------
        ndarray
            The Kirchhoff stress tensor :math:`\boldsymbol{\tau}`.

        See Also
        --------
        felupe.mechanics.kirchhoff_stress : Return the Kirchhoff stress tensor from given
            stress and deformation gradient tensors.
        felupe.mechanics.EvaluateSolidBody.cauchy_stress : Return the Cauchy stress tensor.
        """
        return self.solid._kirchhoff_stress(field=field)
