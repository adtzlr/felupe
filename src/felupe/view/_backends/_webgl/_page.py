# SPDX-License-Identifier: GPL-3.0-or-later

"""The scene of the WebGL viewer and its self-contained HTML page with the inline
module, styles and scene. The arrays of a scene are encoded by base64 for the page
or transferred as binary buffers to the Jupyter widget."""

import base64
import html
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

DTYPES = {"f4": np.float32, "u4": np.uint32, "u1": np.uint8}


def array(values, dtype="f4"):
    """Return a dict of a flat array of the scene with a data type of the viewer
    (``"f4"``, ``"u4"`` or ``"u1"``, little-endian). The array is encoded by
    :func:`encode`."""

    values = np.ascontiguousarray(
        values, dtype=np.dtype(DTYPES[dtype]).newbyteorder("<")
    ).ravel()

    return {"dtype": dtype, "size": int(values.size), "data": values}


def encode(value, binary=False):
    """Return a copy of a scene (or of a part of it), where the data of the arrays is
    encoded by base64 (for JSON) or as bytes (for the binary buffers of a widget)."""

    if isinstance(value, dict):
        data = value.get("data")
        if isinstance(data, np.ndarray) and set(value) == {"dtype", "size", "data"}:
            data = data.tobytes()
            if not binary:
                data = base64.b64encode(data).decode("ascii")
            return {**value, "data": data}

        return {key: encode(item, binary) for key, item in value.items()}

    if isinstance(value, list):
        return [encode(item, binary) for item in value]

    return value


def decode(item):
    """Return the flat array of an array of the scene, given by an array, by bytes or
    encoded by base64 (e.g. for the tests)."""

    dtype = np.dtype(DTYPES[item["dtype"]]).newbyteorder("<")
    data = item["data"]

    if isinstance(data, str):
        data = base64.b64decode(data)

    return np.frombuffer(memoryview(data).cast("B"), dtype=dtype)


def viewer():
    "Return the source code of the module of the viewer."

    return (HERE / "viewer.js").read_text(encoding="utf-8")


def page(scene, height=None):
    """Return the HTML page of the viewer for a scene (a JSON-serializable dict).

    Parameters
    ----------
    scene : dict
        The scene with the objects, the legend and the options of the camera.
    height : str or None, optional
        The height of the viewer as CSS length, e.g. ``"480px"`` (default is None).
        If None, the viewer fills the height of the window.
    """

    module = viewer()
    text = (HERE / "viewer.html").read_text(encoding="utf-8")

    # the scene is inline JSON in a script: "<" is escaped (e.g. "</script>")
    config = json.dumps(encode(scene), separators=(",", ":"), allow_nan=False)
    config = config.replace("<", "\\u003c")

    text = text.replace("/*HEIGHT*/", height or "100vh")
    text = text.replace("/*SCENE*/", config)
    return text.replace("/*MODULE*/", module.replace("</script", "<\\/script"))


def iframe(text, height="480px", title="FElupe"):
    "Return an ``iframe`` with a page in its ``srcdoc`` attribute."

    # the page is embedded in a single line (e.g. in an indented block of reST)
    srcdoc = html.escape(text, quote=True).replace("\n", "&#10;")
    return (
        f'<iframe class="felupe-webgl" title="{html.escape(title)}" '
        f'style="display: block; width: 100%; height: {height}; border: 0;" '
        f'srcdoc="{srcdoc}"></iframe>'
    )


def in_kernel():
    "Return True if the code runs in an IPython kernel, e.g. of Jupyter or VS Code."

    try:
        from IPython import get_ipython
    except ImportError:  # pragma: no cover
        return False

    return getattr(get_ipython(), "kernel", None) is not None
