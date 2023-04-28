# -*- coding: utf-8 -*-
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
