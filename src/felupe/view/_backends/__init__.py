# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting backends of the scenes."""

from ._base import PlottingBackend
from ._pyvista import PyVistaBackend
from ._registry import (
    backend_of,
    get_plotting_backend,
    register_plotting_backend,
    set_plotting_backend,
    take_screenshot,
)

__all__ = [
    "PlottingBackend",
    "PyVistaBackend",
    "backend_of",
    "get_plotting_backend",
    "register_plotting_backend",
    "take_screenshot",
    "set_plotting_backend",
]
