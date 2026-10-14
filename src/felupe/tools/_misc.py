# SPDX-License-Identifier: GPL-3.0-or-later

from platform import architecture, machine, platform

from ..__about__ import __version__ as version


def logo():
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
    return "\n".join(
        [
            f"FElupe Version {version}",
            f"{platform(terse=True)} {machine()} {architecture()[0]}",
        ]
    )
