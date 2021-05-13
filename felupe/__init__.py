from .__about__ import __version__
from . import math
from . import mesh
from . import quadrature
from . import doftools
from . import element
from . import field

# from . import region
from . import constitution
from . import solve
from . import utils

from .region import Region
from .forms import IntegralForm, IntegralFormMixed
from .field import Field
from .doftools import Boundary


__all__ = [
    "__version__",
]
