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


class UpdateItem:
    """A helper class to update an item in the parent class when the value changes.

    Parameters
    ----------
    parent : object
        The parent object that contains the item to be updated.
    key : str
        The key of the item's parameter to be updated in the parent object.

    """

    def __init__(self, parent, key):
        self.parent = parent
        self.key = key

    def update(self, value):
        update_method = getattr(self.parent, f"_update_{self.key}")
        update_method(value)
