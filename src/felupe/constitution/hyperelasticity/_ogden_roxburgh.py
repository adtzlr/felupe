# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.special import erf

from ...math import dya
from .._base import ConstitutiveMaterial, _set_material_parameters


class OgdenRoxburgh(ConstitutiveMaterial):
    r"""`Ogden-Roxburgh <https://doi.org/10.1098%2Frspa.1999.0431>`_ Pseudo-Elastic
    material formulation for an isotropic treatment of the load-history dependent
    Mullins-softening of rubber-like materials.

    Parameters
    ----------
    material : NeoHooke, Hyperelastic, Material or MaterialAD
        An isotropic hyperelastic (user) material definition.
    r : float
        Reciprocal value of the maximum relative amount of softening. i.e. ``r=3`` means
        the shear modulus of the base material scales down from :math:`1` (no softening)
        to :math:`1 - 1/3 = 2/3` (maximum softening).
    m : float
        The initial Mullins softening modulus.
    beta : float
        Maximum deformation-dependent part of the Mullins softening modulus.
    parallel : bool or None, optional
        A flag to evaluate the gradient and the hessian in chunks of cells by a pool of
        threads. If None, the flag is taken from the material (default is None).

    Notes
    -----
    ..  math::

        \eta(\psi, \psi_\text{max}) &= 1 - \frac{1}{r} \text{erf} \left(
            \frac{\psi_\text{max} - \psi}{m + \beta~\psi_\text{max}}
        \right)

        \boldsymbol{P} &= \eta \frac{\partial \psi}{\partial \boldsymbol{F}}

        \mathbb{A} &= \frac{\partial^2 \psi}{\partial \boldsymbol{F} \partial
        \boldsymbol{F}} + \frac{\partial \eta}{\partial \psi} \frac{\partial \psi}
        {\partial \boldsymbol{F}} \otimes \frac{\partial \psi}{\partial \boldsymbol{F}}

    Examples
    --------
    ..  plot::

        >>> import felupe as fem
        >>>
        >>> neo_hooke = fem.NeoHooke(mu=1.0)
        >>> umat = fem.OgdenRoxburgh(material=neo_hooke, r=3.0, m=1.0, beta=0.0)
        >>>
        >>> ax = umat.plot(
        ...     ux=fem.math.linsteps([1, 1.5, 1, 2, 1, 2.5, 1], num=15),
        ...     ps=None,
        ...     bx=None,
        ...     incompressible=True,
        ... )

    """

    def __init__(self, material, r, m, beta, parallel=None):
        # isotropic hyperelastic material formulation
        self.material = self.fun = material

        if parallel is None:
            parallel = getattr(material, "parallel", False)

        self.parallel = parallel

        # ogden-roxburgh material parameters
        self.r = r
        self.m = m
        self.beta = beta

        self.kwargs = {
            "r": self.r,
            "m": self.m,
            "beta": self.beta,
            **self.material.kwargs,
        }

        # initial variables for calling
        # ``self.gradient(self.x)`` and ``self.hessian(self.x)``
        self.x = [np.eye(3), np.zeros(1)]

    def _set_parameters(self, parameters):
        """Set the values of material parameters of the Ogden-Roxburgh material
        formulation (``r``, ``m`` and ``beta``) and of the hyperelastic material."""

        super()._set_parameters(parameters)
        _set_material_parameters(self.material, parameters)

    def _gradient(self, x):
        # unpack variables into deformation gradient and state variables
        F, statevars = x[0], x[-1]

        # material parameters alias
        r, m, beta = self.r, self.m, self.beta

        # isotropic material formulation: evaluate
        # * the strain energy function and
        # * the first Piola-Kirchhoff stress tensor
        W = self.material.function([F, statevars])[0]
        P = self.material.gradient([F, statevars])[0]

        # get the maximum load-history strain energy function
        Wmax = np.maximum(W, statevars[0])
        z = (Wmax - W) / (m + beta * Wmax)

        # softening function
        eta = 1 - erf(z) / r

        # update the state variables
        statevars_new = statevars.copy()
        statevars_new[0] = Wmax

        return [eta * P, statevars_new]

    def _hessian(self, x):
        # unpack variables into deformation gradient and state variables
        F, statevars = x[0], x[-1]

        # material parameters alias
        r, m, beta = self.r, self.m, self.beta

        # isotropic material formulation: evaluate
        # * the strain energy function and
        # * the first Piola-Kirchhoff stress tensor as well as
        # * the according fourth-order elasticity tensor
        W = self.material.function([F, statevars])[0]
        P = self.material.gradient([F, statevars])[0]
        A = self.material.hessian([F, statevars])[0]

        # get the maximum load-history strain energy function
        Wmax = np.maximum(W, statevars[0])
        z = (Wmax - W) / (m + beta * Wmax)

        # softening function
        eta = 1 - erf(z) / r

        # derivative of softening function
        detadz = -2 / (np.sqrt(np.pi) * r) * np.exp(-(z**2))
        dzdW = -1 / (m + beta * Wmax)
        detadW = detadz * dzdW

        # set non-softened derivative to zero
        detadW[np.isclose(eta, 1)] = 0

        return [eta * A + detadW * dya(P, P)]
