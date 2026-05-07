import warnings

from ._event_dispatcher import Context, EventDispatcher
from ._hello_world import hello_world
from ._misc import logo, runs_on
from ._newton import NewtonResult, IterationState
from ._newton import fun_items as fun
from ._newton import jac_items as jac
from ._newton import newtonraphson


def newtonrhapson(*args, **kwargs):
    """Deprecated alias for :func:`felupe.newtonraphson`."""
    warnings.warn(
        "`newtonrhapson` is deprecated since v10.2.0 and will be removed in a future "
        "release. Use `newtonraphson` in new code.",
        DeprecationWarning,
        stacklevel=2,
    )
    return newtonraphson(*args, **kwargs)


from ._post import curve, force, moment
from ._project import extrapolate, project, topoints
from ._save import save
from ._solve import solve

__all__ = [
    "fun",
    "jac",
    "newtonraphson",
    "newtonrhapson",
    "curve",
    "force",
    "hello_world",
    "moment",
    "extrapolate",
    "project",
    "topoints",
    "logo",
    "runs_on",
    "save",
    "solve",
    "NewtonResult",
    "EventDispatcher",
    "Context",
    "IterationState",
]
