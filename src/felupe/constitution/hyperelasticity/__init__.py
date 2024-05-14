"""
Hyperelastic material model formulations.

This module contains automatic-differentiation (ad) and non-ad based hyperelastic
material model formulations:

a) :class:`~felupe.Hyperelastic` to be used with strain energy density functions for
   material model formulations from :mod:`constitution.hyperelasticity.models` which are
   implemented with automatic differentiation using
   `tensortrax <https://github.com/adtzlr/tensortrax>`_ and

b) manually-defined hyperelastic material model formulations (no automatic
   differentation) :mod:`constitution.hyperelasticity.core`.
"""
from ._hyperelastic import Hyperelastic

__all__ = [
    "Hyperelastic",
]
