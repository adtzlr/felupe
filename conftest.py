"""
This file is part of FElupe.

FElupe is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FElupe is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FElupe.  If not, see <http://www.gnu.org/licenses/>.
"""

import pytest


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
