from .__about__ import __version__
from . import math
from . import mesh
from . import quadrature
from . import dof
from . import element
from . import tools
from . import constitution
from . import solve

from ._region import Region
from ._form import IntegralForm, IntegralFormMixed, IntegralFormAxisymmetric
from ._field import Field, FieldAxisymmetric, FieldMixed
from .dof import Boundary


__all__ = [
    "__version__",
]
