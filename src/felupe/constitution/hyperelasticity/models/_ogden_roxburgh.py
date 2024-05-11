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
from tensortrax import Tensor, Δ, Δδ, f, δ
from tensortrax.math import array, maximum, tanh


def ogden_roxburgh(C, Wmax_n, material, r, m, beta, **kwargs):
    r"""`Ogden-Roxburgh <https://doi.org/10.1098%2Frspa.1999.0431>`_ Pseudo-Elastic
    material formulation for an isotropic treatment of the load-history dependent
    Mullins-softening of rubber-like materials.

    Parameters
    ----------
    C : tensortrax.Tensor
        Right Cauchy-Green deformation tensor.
    Wmax_n : ndarray
        State variable: value of the maximum strain energy density in load-history from
        the previous solution.
    material : callable
        Isotropic strain-energy density function. Optional keyword arguments are passed
        to :func:`material <material(C, **kwargs)>`.
    r : float
        Reciprocal value of the maximum relative amount of softening. i.e. ``r=3`` means
        the shear modulus of the base material scales down from :math:`1` (no softening)
        to :math:`1 - 1/3 = 2/3` (maximum softening).
    m : float
        The initial Mullins softening modulus.
    beta : float
        Maximum deformation-dependent part of the Mullins softening modulus.
    **kwargs : dict
        Optional keyword arguments are passed to the isotropic strain energy density
        function :func:`material <material(C, **kwargs)>`.

    Notes
    -----
    ..  note::
        This implementation uses the hyperbolic tangent instead of the Gauss error
        function.

    The strain energy density function and the Mullins-effect related evolution are
    given in Eq. :eq:`psi-ogden-roxburgh`.

    ..  math::
        :label: psi-ogden-roxburgh

        \psi &= \eta \hat{\psi} + \phi

        \eta(\psi, \psi_\text{max}) &= 1 - \frac{1}{r} \tanh \left(
            \frac{\psi_\text{max} - \psi}{m + \beta~\psi_\text{max}}
        \right)

        \delta \phi &= -\delta \eta \ \hat{\psi}

        \delta \psi &= \eta \ \delta \hat{\psi}

    Examples
    --------

    ..  pyvista-plot::
        :context:

        >>> import felupe as fem
        >>>
        >>> umat = fem.Hyperelastic(
        ...     fem.ogden_roxburgh,
        ...     material=fem.neo_hooke,
        ...     r=3,
        ...     m=1,
        ...     beta=0,
        ...     mu=1,
        ...     nstatevars=1
        ... )
        >>> ux = fem.math.linsteps(
        ...     [1, 1.5, 1, 2, 1, 2.5, 1], num=[10, 10, 20, 20, 30, 30]
        ... )
        >>> ax = umat.plot(ux=ux, bx=None, ps=None, incompressible=True)

    ..  pyvista-plot::
        :include-source: False
        :context:
        :force_static:

        >>> import pyvista as pv
        >>>
        >>> fig = ax.get_figure()
        >>> chart = pv.ChartMPL(fig)
        >>> chart.show()

    """

    W = material(C, **kwargs)
    Wmax = maximum(W, array(Wmax_n, like=W))

    def pseudo_elastic_strain_energy(W, Wmax):
        "Strain energy density function with custom variations."

        # evolution equation
        η = 1 - 1 / r * tanh((Wmax - W) / (m + beta * Wmax))

        # custom first- and second-partial derivatives
        return Tensor(
            x=f(η) * f(W),
            δx=f(η) * δ(W),
            Δx=f(η) * Δ(W),
            Δδx=δ(η) * Δ(W) + f(η) * Δδ(W),
            ntrax=W.ntrax,
        )

    return pseudo_elastic_strain_energy(W, Wmax), Wmax
