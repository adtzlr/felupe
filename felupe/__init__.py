from .__about__ import __version__
from . import math
from . import mesh
from . import quadrature
from . import dof
from . import element

# from . import field
from . import tools

# from . import region
from . import constitution
from . import solve
from . import utils

from ._region import Region
from .form import IntegralForm, IntegralFormMixed, IntegralFormAxisymmetric
from .field import Field, FieldAxisymmetric, FieldMixed
from .dof import Boundary


__all__ = [
    "__version__",
]
