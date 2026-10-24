"""Core of janitrace: tracing Python step functions into JANI.

The same action function runs in two modes:
  * symbolic: every path through the function is enumerated (a Python `if`
    on a symbolic value becomes a choice point, explored both ways by
    re-running the function), and the paths are compiled into JANI edges and
    destinations;
  * concrete: plain ints/bools and random sampling, for simulation and tests.

Rules for an action body (one MDP step = nondeterminism, then probability):
  * guard(c) and choose() must come before the first flip()/uniform();
  * the function must be pure apart from writes to the state `s`;
  * loops must have a concrete bound (they are unrolled while tracing);
  * list lengths are fixed when the model class is created.
"""
from __future__ import annotations

import json
import random
from fractions import Fraction

# --------------------------------------------------------------------------
# Symbolic expressions: operator overloading emits JANI expression JSON
# --------------------------------------------------------------------------

def lift(v):
    if isinstance(v, Expr):
        return v.j
    if isinstance(v, bool):
        return v
    if isinstance(v, Fraction):
        return v.numerator if v.denominator == 1 else \
            {"op": "/", "left": v.numerator, "right": v.denominator}
    if isinstance(v, (int, float)):
        return v
    raise TypeError(f"cannot use {v!r} in a JANI expression")


class Expr:
    __hash__ = object.__hash__

    def __init__(self, j):
        self.j = j

    def _b(op):
        return lambda a, b: Expr({"op": op, "left": lift(a), "right": lift(b)})

    def _r(op):
        return lambda a, b: Expr({"op": op, "left": lift(b), "right": lift(a)})

    __add__, __radd__ = _b("+"), _r("+")
    __sub__, __rsub__ = _b("-"), _r("-")
    __mul__, __rmul__ = _b("*"), _r("*")
    __truediv__, __rtruediv__ = _b("/"), _r("/")
    __mod__, __rmod__ = _b("%"), _r("%")
    __lt__, __le__ = _b("<"), _b("≤")
    __gt__, __ge__ = _b(">"), _b("≥")
    __eq__, __ne__ = _b("="), _b("≠")
    __and__, __rand__ = _b("∧"), _r("∧")
    __or__, __ror__ = _b("∨"), _r("∨")

    def __invert__(self):
        return Expr({"op": "¬", "exp": self.j})

    def __bool__(self):
        # The key trick: `if expr:` becomes a branch point of the tracer.
        return _ctx.branch(self)

    def __repr__(self):
        return f"Expr({json.dumps(self.j, ensure_ascii=False)})"


def ite(c, a, b):
    if not isinstance(c, Expr):
        return a if c else b
    return Expr({"op": "ite", "if": c.j, "then": lift(a), "else": lift(b)})


def minimum(a, b):
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        return min(a, b)
    return Expr({"op": "min", "left": lift(a), "right": lift(b)})


def maximum(a, b):
    if not isinstance(a, Expr) and not isinstance(b, Expr):
        return max(a, b)
    return Expr({"op": "max", "left": lift(a), "right": lift(b)})


# --------------------------------------------------------------------------
# Model declarations
# --------------------------------------------------------------------------

class Int:
    def __init__(self, lo, hi, init):
        self.lo, self.hi, self.init = lo, hi, init

    def jani_type(self):
        return {"kind": "bounded", "base": "int",
                "lower-bound": self.lo, "upper-bound": self.hi}


class Bool:
    def __init__(self, init):
        self.init = init

    def jani_type(self):
        return "bool"


def _is_decl_list(v):
    """A plain list/tuple of Int/Bool declarations is a list of state variables."""
    return isinstance(v, (list, tuple)) and len(v) > 0 and \
        all(isinstance(x, (Int, Bool)) for x in v)


class Reward:
    """A reward structure: a JANI transient real variable, accumulated per step."""

    def jani_decl(self, name):
        return {"name": name, "type": "real", "initial-value": 0, "transient": True}


def action(fn):
    fn._is_action = True
    return fn


def properties(fn):
    """Mark a model function that returns {name: property expression}.

    It takes no arguments; to_jani/write_jani/check use its properties when
    none are passed explicitly."""
    fn._is_properties = True
    return fn


class Model:
    @classmethod
    def variables(cls):
        out = {}
        for klass in reversed(cls.__mro__):
            for k, v in vars(klass).items():
                if isinstance(v, (Int, Bool)):
                    out[k] = v
                elif _is_decl_list(v):
                    out.update({f"{k}_{i}": x for i, x in enumerate(v)})
        return out

    @classmethod
    def lists(cls):
        out = {}
        for klass in reversed(cls.__mro__):
            for k, v in vars(klass).items():
                if _is_decl_list(v):
                    out[k] = len(v)
        return out

    @classmethod
    def rewards(cls):
        out = {}
        for klass in reversed(cls.__mro__):
            for k, v in vars(klass).items():
                if isinstance(v, Reward):
                    out[k] = v
        return out

    @classmethod
    def all_properties(cls):
        """The properties of every @properties function, merged in MRO order."""
        out = {}
        for klass in reversed(cls.__mro__):
            for v in vars(klass).values():
                if getattr(v, "_is_properties", False):
                    out.update(v())
        return out

    @classmethod
    def actions(cls):
        return {k: v for k, v in vars(cls).items()
                if callable(v) and getattr(v, "_is_action", False)}


class State:
    """Attribute bag; in symbolic mode values are Expr, in concrete mode ints."""

    def __init__(self, values, lists=None):
        object.__setattr__(self, "_v", dict(values))
        object.__setattr__(self, "_lists", dict(lists or {}))

    def __getattr__(self, k):
        if k in self._lists:
            return ListView(self, k, self._lists[k])
        try:
            return self._v[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, val):
        if k in self._lists:
            raise AttributeError(f"assign elements of list {k!r}, e.g. s.{k}[i] = ...")
        if k not in self._v:
            raise AttributeError(f"unknown state variable {k!r}")
        self._v[k] = val


class ListView:
    """s.q as seen inside an action: a fixed-length list of state variables."""

    def __init__(self, state, name, n):
        self._s, self._name, self._n = state, name, n

    def __len__(self):
        return self._n

    def _index(self, i):
        if isinstance(i, Expr):
            for j in range(self._n):       # symbolic index: one branch per value
                if i == j:
                    return j
            raise Disabled                 # index out of range on this path
        if not -self._n <= i < self._n:
            raise IndexError(f"{self._name}[{i}] out of range (length {self._n})")
        return i % self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(self._n)[i]]
        return self._s._v[f"{self._name}_{self._index(i)}"]

    def __setitem__(self, i, val):
        self._s._v[f"{self._name}_{self._index(i)}"] = val

    def __iter__(self):
        return (self[j] for j in range(self._n))


class SymVar(Expr):
    """V.x is the variable x; V.q[i] is element i of list q."""

    def __getitem__(self, i):
        return Expr(f"{self.j}_{i}")


class Symbols:
    """`V.n` gives the symbolic variable n, for writing properties."""

    def __getattr__(self, k):
        return SymVar(k)


V = Symbols()

# --------------------------------------------------------------------------
# Execution contexts
# --------------------------------------------------------------------------

class Disabled(Exception):
    pass


class ModelError(Exception):
    pass


class ConcreteCtx:
    def __init__(self, rng, scheduler):
        self.rng, self.scheduler = rng, scheduler
        self.earned = {}

    def branch(self, e):
        raise ModelError(f"symbolic value {e!r} reached concrete execution")

    def flip(self, p):
        return self.rng.random() < float(p)

    def reward(self, name, value):
        self.earned[name] = self.earned.get(name, 0) + value

    def choose(self, n=None):
        return self.scheduler(self.rng) if n is None else self.rng.randrange(n)

    def uniform(self, n):
        return self.rng.randrange(n)

    def guard(self, c):
        if not c:
            raise Disabled


class TraceCtx:
    """Enumerates all paths of an action by re-running it (DFS over choices)."""

    def __init__(self, prefix, oracle=None):
        self.prefix = prefix
        self.oracle = oracle
        self.trail = []          # number of options at each choice point
        self.pos = 0
        self.prob = Fraction(1)
        self.pre = []            # conditions before the first flip -> guard
        self.post = []           # conditions after a flip -> probability
        self.nondet = []         # choose() outcomes
        self.rewards = {}        # reward name -> accumulated expression
        self.flipped = False

    def _point(self, n):
        d = self.prefix[self.pos] if self.pos < len(self.prefix) else 0
        self.trail.append(n)
        self.pos += 1
        return d

    def branch(self, e):
        if self.oracle is not None:
            known = self.oracle.decide(self.pre + self.post, e.j)
            if known is not None:
                return known        # implied or contradicted: no branch point
        taken = self._point(2) == 0
        cond = e.j if taken else {"op": "¬", "exp": e.j}
        (self.post if self.flipped else self.pre).append(cond)
        return taken

    def flip(self, p):
        if isinstance(p, Expr):
            raise ModelError("flip() needs a concrete probability "
                             "(use ite() before it if it depends on state)")
        p = Fraction(p).limit_denominator(10**9)
        self.flipped = True
        if self._point(2) == 0:
            self.prob *= p
            return True
        self.prob *= 1 - p
        return False

    def reward(self, name, value):
        old = self.rewards.get(name, 0)
        self.rewards[name] = value if (isinstance(old, int) and old == 0) else old + value

    def uniform(self, n):
        self.flipped = True
        d = self._point(n)
        self.prob *= Fraction(1, n)
        return d

    def choose(self, n=None):
        if self.flipped:
            raise ModelError("choose() after flip(): an MDP step resolves "
                             "nondeterminism before probability")
        d = self._point(2 if n is None else n)
        c = (d == 0) if n is None else d
        self.nondet.append(c)
        return c

    def guard(self, c):
        if self.flipped:
            raise ModelError("guard() after flip() would lose probability mass")
        if not c:
            raise Disabled


_ctx = None


def flip(p):
    return _ctx.flip(p)


def reward(name, value=1):
    """Earn `value` on reward structure `name` in this step (may depend on state)."""
    return _ctx.reward(name, value)


def uniform(n):
    """Probabilistic choice of an index in range(n), each with probability 1/n."""
    return _ctx.uniform(n)


def choose(n=None):
    """choose() -> bool, choose(n) -> index in range(n); resolved by the scheduler."""
    return _ctx.choose(n)


def guard(c):
    return _ctx.guard(c)


def _conj(conds):
    if not conds:
        return True
    e = conds[0]
    for c in conds[1:]:
        e = {"op": "∧", "left": e, "right": c}
    return e


def trace_action(fn, var_names, oracle=None, stats=None, lists=None):
    """Return a list of paths: (pre, post, nondet, prob, assignments)."""
    global _ctx
    paths, prefix = [], []
    while True:
        ctx = TraceCtx(prefix, oracle)
        _ctx = ctx
        originals = {k: Expr(k) for k in var_names}
        s = State(originals, lists)
        try:
            fn(s)
            assigns = {k: lift(v) for k, v in s._v.items()
                       if v is not originals[k]}
            assigns.update({k: lift(v) for k, v in ctx.rewards.items()})
            paths.append((ctx.pre, ctx.post, tuple(ctx.nondet), ctx.prob, assigns))
        except Disabled:
            pass
        finally:
            _ctx = None
        if stats is not None:
            stats["runs"] = stats.get("runs", 0) + 1
        # backtrack to the deepest choice point with an unexplored option
        decisions = [prefix[i] if i < len(prefix) else 0
                     for i in range(len(ctx.trail))]
        while decisions and decisions[-1] == ctx.trail[len(decisions) - 1] - 1:
            decisions.pop()
        if not decisions:
            return paths
        decisions[-1] += 1
        prefix = decisions


# --------------------------------------------------------------------------
# Compilation to JANI
# --------------------------------------------------------------------------

def _edges_for(name, fn, var_names, oracle=None, stats=None, lists=None):
    groups = {}
    for pre, post, nondet, prob, assigns in trace_action(fn, var_names, oracle, stats, lists):
        key = (json.dumps(pre, ensure_ascii=False), nondet)
        groups.setdefault(key, (pre, []))[1].append((post, prob, assigns))

    edges = []
    for (_, _), (pre, dests) in groups.items():
        merged = {}  # identical assignments -> list of probability terms
        for post, prob, assigns in dests:
            if prob == 0:
                continue
            k = json.dumps(sorted(assigns.items()), ensure_ascii=False)
            term = prob if not post else ite(Expr(_conj(post)), prob, 0)
            merged.setdefault(k, (assigns, []))[1].append(term)
        jdests = []
        for assigns, terms in merged.values():
            if all(isinstance(t, Fraction) for t in terms):
                p = lift(sum(terms))
            else:
                acc = terms[0]
                for t in terms[1:]:
                    acc = acc + t if isinstance(acc, Expr) else t + acc
                p = lift(acc)
            jdests.append({
                "location": "l",
                "probability": {"exp": p},
                "assignments": [{"ref": k, "value": v}
                                for k, v in sorted(assigns.items())],
            })
        edges.append({"location": "l", "action": name,
                      "guard": {"exp": _conj(pre)}, "destinations": jdests})
    return edges


def to_jani(model_cls, properties=None, simplify=False, stats=None):
    """Compile a Model subclass to a JANI model (a dict ready for json.dump).

    properties: {name: property expression}, built with Pmin/Pmax/Emin/Emax;
    defaults to the model's @properties functions.
    simplify=True uses Z3 (see simplify.py) to prune infeasible branches
    while tracing, normalise expressions, merge equal destinations and
    check bounds and probability sums.
    stats: optional dict that receives tracing statistics (paths run, Z3 queries)."""
    if properties is None:
        properties = model_cls.all_properties()
    vs = model_cls.variables()
    oracle = None
    if simplify:
        from .simplify import Oracle
        oracle = Oracle(vs)
    edges, actions = [], []
    for name, fn in model_cls.actions().items():
        actions.append({"name": name})
        edges += _edges_for(name, fn, list(vs), oracle, stats, model_cls.lists())
    if oracle is not None:
        edges = oracle.postprocess(edges)
        if stats is not None:
            stats["z3_queries"] = oracle.queries
    return {
        "jani-version": 1,
        "name": model_cls.__name__,
        "type": "mdp",
        "features": [],
        "actions": actions,
        "variables": [{"name": k, "type": v.jani_type(), "initial-value": v.init}
                      for k, v in vs.items()]
                     + [r.jani_decl(k) for k, r in model_cls.rewards().items()],
        "properties": [{"name": n, "expression": e}
                       for n, e in properties.items()],
        "automata": [{"name": "main", "locations": [{"name": "l"}],
                      "initial-locations": ["l"], "edges": edges}],
        "system": {"elements": [{"automaton": "main"}]},
    }


# Property helpers -----------------------------------------------------------

def F(target, steps=None):
    e = {"op": "U", "left": True, "right": lift(target)}
    if steps is not None:
        e["step-bounds"] = {"upper": steps}
    return e


def _query(op, path):
    return {"op": "filter", "fun": "values",
            "values": {"op": op, "exp": path}, "states": {"op": "initial"}}


def Emin(reward, reach):
    """Minimal expected accumulated reward until `reach` holds."""
    return _query("Emin", None) | {"values": {"op": "Emin", "exp": reward,
            "accumulate": ["steps"], "reach": lift(reach)}}


def Emax(reward, reach):
    return _query("Emax", None) | {"values": {"op": "Emax", "exp": reward,
            "accumulate": ["steps"], "reach": lift(reach)}}


def Pmax(path):
    return _query("Pmax", path)


def Pmin(path):
    return _query("Pmin", path)


# --------------------------------------------------------------------------
# Concrete simulation with the very same action functions
# --------------------------------------------------------------------------

def write_jani(path, model_cls, properties=None, **kwargs):
    """Compile `model_cls` and write the JANI file; returns the JANI dict."""
    j = to_jani(model_cls, properties, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(j, f, indent=1, ensure_ascii=False)
    return j


def simulate(model_cls, steps, rng=None, scheduler=lambda r: r.random() < 0.5):
    """Run the actions concretely for up to `steps` steps; returns the list of states.

    Enabled actions are tried in random order; choose() is resolved by
    `scheduler` (choose(n) uniformly at random)."""
    global _ctx
    rng = rng or random.Random()
    s = State({k: v.init for k, v in model_cls.variables().items()}, model_cls.lists())
    acts = list(model_cls.actions().values())
    trace = [dict(s._v)]
    for _ in range(steps):
        rng.shuffle(acts)
        for fn in acts:
            nxt = State(s._v, s._lists)
            _ctx = ConcreteCtx(rng, scheduler)
            try:
                fn(nxt)
            except Disabled:
                continue
            finally:
                _ctx = None
            s = nxt
            break
        else:
            break  # deadlock
        trace.append(dict(s._v))
    return trace
