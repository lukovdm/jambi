"""janitrace: write probabilistic models as plain Python, compile them to JANI.

    from janitrace import Model, Int, action, flip, choose, guard, to_jani

Write each action as an ordinary Python function over the state `s`; janitrace
runs it on symbolic values, enumerates every path through it and emits the
corresponding guarded probabilistic edges. The same function also runs
concretely (see `simulate`), so models can be unit-tested and simulated.
"""
from .core import (
    Bool, Expr, Int, Model, ModelError, Reward, V,
    action, choose, flip, guard, ite, maximum, minimum, reward, uniform,
    Emax, Emin, F, Pmax, Pmin,
    simulate, to_jani, write_jani,
)

__version__ = "0.1.0"

__all__ = [
    "Bool", "Expr", "Int", "Model", "ModelError", "Reward", "V",
    "action", "choose", "flip", "guard", "ite", "maximum", "minimum", "reward", "uniform",
    "Emax", "Emin", "F", "Pmax", "Pmin",
    "simulate", "to_jani", "write_jani",
]
