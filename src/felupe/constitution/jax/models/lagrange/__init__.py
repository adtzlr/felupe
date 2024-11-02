from ._morph import morph

__all__ = [
    "morph",
]

# default (stable) material parameters
morph.kwargs = dict(p=[0, 0, 0, 0, 0, 1, 0, 0])
