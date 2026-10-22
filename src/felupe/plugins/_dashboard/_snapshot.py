# SPDX-License-Identifier: GPL-3.0-or-later

"""A static snapshot of the dashboard, a self-contained HTML page (e.g. for the
documentation or for a result which is shared)."""

import html
from pathlib import Path

from ._file import GEOMETRY, STATE, page


class DashboardSnapshot:
    """A static snapshot of the dashboard of a job, see
    :meth:`DashboardWidgetPlugin.snapshot() <felupe.DashboardWidgetPlugin.snapshot>`.

    Parameters
    ----------
    source : DashboardWidget or DashboardState
        The widget or the state of the dashboard of a job.

    Attributes
    ----------
    html : str
        The self-contained HTML page of the snapshot (the data is copied, i.e. the
        snapshot is not changed by the next jobs).

    Notes
    -----
    The rich representation of the snapshot (``_repr_html_``) embeds the page in an
    ``iframe``, e.g. in the output of a cell of a notebook or in an example of the
    documentation which is built by sphinx-gallery (with ``"capture_repr":
    ("_repr_html_",)``). The height of the ``iframe`` follows the dashboard and the
    theme follows the colours of the parent page (light or dark).
    """

    def __init__(self, source):
        snapshot = dict(
            geometry={key: getattr(source, key) for key in GEOMETRY},
            state={key: getattr(source, key) for key in STATE},
        )
        self.html = page({"snapshot": snapshot})

    def _repr_html_(self):
        # the page is embedded in a single line (e.g. in an indented block of reST)
        srcdoc = html.escape(self.html, quote=True).replace("\n", "&#10;")
        return (
            '<iframe class="felupe-dashboard" title="FElupe dashboard" '
            'style="display: block; width: 100%; height: 560px; border: 0;" '
            f'srcdoc="{srcdoc}"></iframe>'
        )

    def save(self, filename):
        """Save the snapshot as a self-contained HTML file, e.g. ``"job.html"``.

        Parameters
        ----------
        filename : str or pathlib.Path
            The filename of the HTML file.

        Returns
        -------
        pathlib.Path
            The path of the HTML file.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html, encoding="utf-8")
        return path
