# SPDX-License-Identifier: GPL-3.0-or-later

from platform import architecture, machine, platform

from ..__about__ import __version__ as version


def logo():
    """Return the FElupe logo.

    Returns
    -------
    str
        The FElupe logo as a multi-line string.

    Examples
    --------
    >>> import felupe as fem
    >>> print(fem.tools.logo())
     _______  _______  ___      __   __  _______  _______
    |       ||       ||   |    |  | |  ||       ||       |
    |    ___||    ___||   |    |  | |  ||    _  ||    ___|
    |   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___
    |    ___||    ___||   |___ |       ||    ___||    ___|
    |   |    |   |___ |       ||       ||   |    |   |___
    |___|    |_______||_______||_______||___|    |_______|
    """
    return "\n".join(
        [
            " _______  _______  ___      __   __  _______  _______",
            "|       ||       ||   |    |  | |  ||       ||       |",
            "|    ___||    ___||   |    |  | |  ||    _  ||    ___|",
            "|   |___ |   |___ |   |    |  |_|  ||   |_| ||   |___",
            "|    ___||    ___||   |___ |       ||    ___||    ___|",
            "|   |    |   |___ |       ||       ||   |    |   |___",
            "|___|    |_______||_______||_______||___|    |_______|",
        ]
    )


def runs_on():
    """Return the FElupe version and the platform information on which FElupe runs.

    Returns
    -------
    str
        A multi-line string with the FElupe version and the platform, machine and
        architecture information.
    """
    return "\n".join(
        [
            f"FElupe Version {version}",
            f"{platform(terse=True)} {machine()} {architecture()[0]}",
        ]
    )
