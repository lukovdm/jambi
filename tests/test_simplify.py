import pytest

pytest.importorskip("z3")

from janitrace import Model, Int, ModelError, action, flip, to_jani
from janitrace.simplify import quick


class Queue(Model):
    n = Int(0, 10, init=0)

    @action
    def step(s):
        if s.n < 10 and flip(0.7):
            s.n += 1
        if flip(0.6) and s.n > 0:        # n+1 > 0 is always true on the arrival path
            s.n -= 1


def test_quick_folds_constants_and_units():
    assert quick({"op": "+", "left": 2, "right": 3}) == 5
    assert quick({"op": "-", "left": {"op": "+", "left": "n", "right": 1}, "right": 1}) == "n"
    assert quick({"op": "∧", "left": True, "right": "b"}) == "b"
    assert quick({"op": "ite", "if": False, "then": 1, "else": "x"}) == "x"


def test_simplification_prunes_paths_and_keeps_structure_valid():
    plain, simp = {}, {}
    a = to_jani(Queue, stats=plain)
    b = to_jani(Queue, simplify=True, stats=simp)
    assert simp["runs"] < plain["runs"]
    count = lambda j: sum(len(e["destinations"]) for e in j["automata"][0]["edges"])
    assert count(b) <= count(a)


def test_bound_violation_is_reported_with_a_counterexample():
    class Buggy(Model):
        n = Int(0, 10, init=0)

        @action
        def step(s):
            if flip(0.7):
                s.n += 1                 # no capacity check

    with pytest.raises(ModelError, match=r"leaves \[0, 10\].*'n': 10"):
        to_jani(Buggy, simplify=True)
