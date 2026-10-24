import json
import random
from fractions import Fraction

import pytest

from janitrace import (Model, Int, Bool, Reward, ModelError, V, action, properties, choose, flip,
                       guard, reward, uniform, simulate, to_jani, Pmax, F)
from janitrace.core import Expr


def edges(j):
    return j["automata"][0]["edges"]


def const(p):
    """Probability expression -> Fraction (constant probabilities only)."""
    if isinstance(p, (int, float)):
        return Fraction(p)
    assert p["op"] == "/"
    return Fraction(p["left"], p["right"])


class Coin(Model):
    x = Int(0, 3, init=0)

    @action
    def step(s):
        guard(s.x < 3)
        if flip(0.25):
            s.x += 1


def test_expressions_emit_jani():
    n = Expr("n")
    assert (n + 1 < 10).j == {"op": "<", "left": {"op": "+", "left": "n", "right": 1}, "right": 10}
    assert (3 - n).j == {"op": "-", "left": 3, "right": "n"}
    assert (~(n == 2)).j == {"op": "¬", "exp": {"op": "=", "left": "n", "right": 2}}


def test_simple_model_structure():
    j = to_jani(Coin)
    assert j["type"] == "mdp"
    assert j["variables"][0] == {"name": "x", "initial-value": 0,
                                 "type": {"kind": "bounded", "base": "int",
                                          "lower-bound": 0, "upper-bound": 3}}
    (e,) = edges(j)
    assert e["guard"]["exp"] == {"op": "<", "left": "x", "right": 3}
    assert sorted(const(d["probability"]["exp"]) for d in e["destinations"]) == \
        [Fraction(1, 4), Fraction(3, 4)]


def test_probabilities_sum_to_one_per_edge():
    class Dice(Model):
        a = Int(0, 6, init=0)
        b = Int(0, 1, init=0)

        @action
        def roll(s):
            guard(s.a == 0)
            s.a = uniform(6) + 1
            if flip(0.5):
                s.b = 1

    for e in edges(to_jani(Dice)):
        assert sum(const(d["probability"]["exp"]) for d in e["destinations"]) == 1


def test_python_if_on_symbolic_value_splits_guards():
    class Branchy(Model):
        x = Int(0, 5, init=0)

        @action
        def step(s):
            if s.x < 2:
                s.x += 2
            else:
                s.x = 0

    es = edges(to_jani(Branchy))
    assert len(es) == 2
    guards = {json.dumps(e["guard"]["exp"], ensure_ascii=False) for e in es}
    assert '{"op": "<", "left": "x", "right": 2}' in guards


def test_choose_gives_nondeterministic_edges():
    class Pick(Model):
        x = Int(0, 2, init=0)

        @action
        def step(s):
            s.x = choose(3)

    assert len(edges(to_jani(Pick))) == 3


def test_choose_after_flip_is_rejected():
    class Bad(Model):
        x = Int(0, 1, init=0)

        @action
        def step(s):
            if flip(0.5) and choose():
                s.x = 1

    with pytest.raises(ModelError, match="choose"):
        to_jani(Bad)


def test_guard_after_flip_is_rejected():
    class Bad(Model):
        x = Int(0, 1, init=0)

        @action
        def step(s):
            if flip(0.5):
                guard(s.x == 0)

    with pytest.raises(ModelError, match="guard"):
        to_jani(Bad)


def test_lists_flatten_and_support_symbolic_index():
    class Queues(Model):
        q = [Int(0, 2, init=i) for i in range(3)]
        ptr = Int(0, 2, init=0)

        @action
        def step(s):
            if s.q[s.ptr] > 0:
                s.q[s.ptr] -= 1
            s.ptr = (s.ptr + 1) % len(s.q)

    j = to_jani(Queues)
    names = [v["name"] for v in j["variables"]]
    assert names == ["q_0", "q_1", "q_2", "ptr"]
    assert [v["initial-value"] for v in j["variables"]][:3] == [0, 1, 2]
    assert len(edges(j)) >= 3            # at least one branch per possible index


def test_list_index_out_of_range():
    class Bad(Model):
        q = [Int(0, 1, init=0) for _ in range(2)]

        @action
        def step(s):
            s.q[2] = 1

    with pytest.raises(IndexError):
        to_jani(Bad)


def test_rewards_are_transient_variables():
    class Paid(Model):
        x = Int(0, 1, init=0)
        cost = Reward()

        @action
        def step(s):
            reward("cost", 2)
            s.x = 1

    j = to_jani(Paid)
    assert {"name": "cost", "type": "real", "initial-value": 0, "transient": True} in j["variables"]
    (d,) = edges(j)[0]["destinations"]
    assert {"ref": "cost", "value": 2} in d["assignments"]


def test_properties():
    j = to_jani(Coin, {"reach": Pmax(F(V.x == 3, steps=5))})
    (p,) = j["properties"]
    assert p["name"] == "reach"
    assert p["expression"]["values"]["op"] == "Pmax"
    assert p["expression"]["values"]["exp"]["step-bounds"] == {"upper": 5}


def test_class_properties_are_the_default():
    class WithProps(Coin):
        @properties
        def props():
            return {"reach": Pmax(F(V.x == 3))}

    assert [p["name"] for p in to_jani(WithProps)["properties"]] == ["reach"]
    assert to_jani(WithProps, {})["properties"] == []      # explicit argument wins
    assert to_jani(Coin)["properties"] == []


def test_simulate_runs_same_code():
    run = simulate(Coin, 50, random.Random(0))
    xs = [st["x"] for st in run]
    assert xs[0] == 0 and all(b - a in (0, 1) for a, b in zip(xs, xs[1:]))
    assert all(0 <= x <= 3 for x in xs)
