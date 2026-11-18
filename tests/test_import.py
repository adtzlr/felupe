# SPDX-License-Identifier: GPL-3.0-or-later

import subprocess
import sys


def test_import():
    import felupe as fem

    assert hasattr(fem, "constitution")


def test_import_scipy_lazy():
    """The import of FElupe doesn't import the SciPy modules which are slow to import
    and which are only used by a few functions (#1297)."""

    code = "\n".join(
        [
            "import sys",
            "import scipy.sparse.linalg",
            "before = set(sys.modules)",
            "import felupe",
            "print(' '.join(sorted(set(sys.modules) - before)))",
        ]
    )
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    modules = output.stdout.split()

    assert "felupe" in modules

    for name in ["interpolate", "special", "constants", "optimize", "spatial", "fft"]:
        package = f"scipy.{name}"
        imported = [m for m in modules if m == package or m.startswith(package + ".")]
        assert not imported, f"import felupe imports {package}"


if __name__ == "__main__":
    test_import()
    test_import_scipy_lazy()
