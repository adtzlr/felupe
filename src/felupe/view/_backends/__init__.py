# SPDX-License-Identifier: GPL-3.0-or-later

"""Plotting backends of the scenes."""

from ._base import PlottingBackend
from ._matplotlib import MatplotlibBackend, MatplotlibPlotter
from ._pyvista import PyVistaBackend
from ._registry import (
    backend_of,
    get_plotting_backend,
    register_plotting_backend,
    require_pyvista,
    set_plotting_backend,
    take_screenshot,
)
from ._webgl import WebGLBackend, WebGLPlotter

__all__ = [
    "MatplotlibBackend",
    "MatplotlibPlotter",
    "PlottingBackend",
    "PyVistaBackend",
    "WebGLBackend",
    "WebGLPlotter",
    "backend_of",
    "get_plotting_backend",
    "register_plotting_backend",
    "require_pyvista",
    "take_screenshot",
    "set_plotting_backend",
]
