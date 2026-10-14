# SPDX-License-Identifier: GPL-3.0-or-later


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
