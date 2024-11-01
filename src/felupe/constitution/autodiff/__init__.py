"""
felupe.constitution.autodiff
============================
This module contains constitutive material classes and models for several backends. The
primary used backend in FElupe is based on :mod:`tensortrax`.
"""

from . import jax, tensortrax

__all__ = ["jax", "tensortrax"]
