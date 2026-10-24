"""End-to-end checks against Storm (skipped if stormpy is not installed)."""
import os
import sys

import pytest

pytest.importorskip("stormpy")
pytest.importorskip("z3")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from janitrace.storm import check  # noqa: E402


@pytest.mark.parametrize("simplify", [False, True])
def test_server_values(simplify):
    from server import Server
    v = check(Server, simplify=simplify)
    assert v["overflow_max"] == pytest.approx(0.156506, abs=1e-6)
    assert v["overflow_min"] == pytest.approx(6.97946e-07, rel=1e-4)


def test_list_model_plain_equals_simplified():
    from queues import make_round_robin
    M = make_round_robin(3, 5)
    a, b = check(M), check(M, simplify=True)
    for k in a:
        assert a[k] == pytest.approx(b[k], abs=1e-9)


def test_rewards_checked_by_storm():
    from janitrace import Model, Int, Reward, action, flip, guard, reward, Emin, V

    class Walk(Model):
        x = Int(0, 1, init=0)
        steps = Reward()

        @action
        def step(s):
            guard(s.x == 0)
            reward("steps")
            if flip(0.25):
                s.x = 1

        @action
        def stay(s):
            guard(s.x == 1)

    v = check(Walk, {"e": Emin("steps", V.x == 1)})
    assert v["e"] == pytest.approx(4.0)          # geometric with p = 1/4
