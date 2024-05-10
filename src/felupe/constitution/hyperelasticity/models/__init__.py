"""
Models
======
This module contains strain energy density functions for material model formulations to
be used as the ``fun``-argument in :func:`~felupe.Hyperelastic`. A strain energy density
function must be formulated in terms of the right Cauchy-Green deformation tensor. The
gradient as well as the hessian of the strain energy density function is carried out
by automatic differentiation using :mod:`tensortrax`. Hence, all math-functions must be
taken from :mod:`tensortrax.math`.
"""

from ._alexander import alexander
from ._arruda_boyce import arruda_boyce
from ._extended_tube import extended_tube
from ._finite_strain_viscoelastic import finite_strain_viscoelastic
from ._helpers import isochoric_volumetric_split
from ._mooney_rivlin import mooney_rivlin
from ._neo_hooke import neo_hooke
from ._ogden import ogden
from ._ogden_roxburgh import ogden_roxburgh
from ._saint_venant_kirchhoff import saint_venant_kirchhoff
from ._third_order_deformation import third_order_deformation
from ._van_der_waals import van_der_waals
from ._yeoh import yeoh

__all__ = [
    "alexander",
    "arruda_boyce",
    "extended_tube",
    "finite_strain_viscoelastic",
    "isochoric_volumetric_split",
    "mooney_rivlin",
    "neo_hooke",
    "ogden",
    "ogden_roxburgh",
    "saint_venant_kirchhoff",
    "third_order_deformation",
    "van_der_waals",
    "yeoh",
]

# default (stable) material parameters
alexander.kwargs = dict(C1=0, C2=0, C3=0, gamma=100, k=0)
arruda_boyce.kwargs = dict(C1=0, limit=100)
extended_tube.kwargs = dict(Gc=0, Ge=0, beta=1, delta=0)
mooney_rivlin.kwargs = dict(C10=0, C01=0)
neo_hooke.kwargs = dict(mu=0)
ogden.kwargs = dict(mu=[0, 0], alpha=[2, -2])
ogden_roxburgh.kwargs = dict(r=100, m=1, beta=0, material=neo_hooke, mu=0)
saint_venant_kirchhoff.kwargs = dict(mu=0.0, lmbda=0.0)
third_order_deformation.kwargs = dict(C10=0, C01=0, C11=0, C20=0, C30=0)
van_der_waals.kwargs = dict(mu=0, beta=0, a=0, limit=100)
yeoh.kwargs = dict(C10=0, C20=0, C30=0)
