# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util

import pytest

# the module of the widget of the dashboard requires the optional dependency anywidget
collect_ignore = []
if importlib.util.find_spec("anywidget") is None:  # pragma: no cover
    collect_ignore.append("src/felupe/plugins/_dashboard/_widget.py")


def pytest_configure(config):
    "Render the plots of the tests and of the docstring examples off-screen."

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:  # pragma: no cover
        pass

    try:
        import pyvista

        pyvista.OFF_SCREEN = True
    except ImportError:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def _doctest_in_tmp_path(request):
    """Run the docstring examples in a temporary directory, i.e. the files written by
    the examples (e.g. XDMF-files or animations) don't end up in the working directory,
    and close the matplotlib-figures of the examples afterwards."""

    if type(request.node).__name__ != "DoctestItem":
        yield
        return

    monkeypatch = request.getfixturevalue("monkeypatch")
    monkeypatch.chdir(request.getfixturevalue("tmp_path"))

    yield

    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:  # pragma: no cover
        pass
