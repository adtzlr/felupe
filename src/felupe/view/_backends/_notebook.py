# SPDX-License-Identifier: GPL-3.0-or-later

"""Helpers of the plotting backends for notebooks."""


def in_kernel():
    "Return True if the code runs in an IPython kernel, e.g. of Jupyter or VS Code."

    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False

    return getattr(get_ipython(), "kernel", None) is not None
