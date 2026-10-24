"""Z3 backend for janitrace (optional; needs the z3-solver package).

Used in two places:
  1. while tracing: `decide` answers whether a branch condition is already
     implied (or contradicted) by the variable bounds and the conditions on
     the current path, so the tracer does not split there;
  2. after tracing: `postprocess` normalises expressions, drops identity
     assignments and dead edges/destinations, merges destinations that are
     semantically equal, and statically checks every edge (probabilities sum
     to one, no assignment leaves its variable's bounds).
"""
import json
from fractions import Fraction

import z3

from .core import Int, ModelError, lift

K = z3.Z3_OP_UNINTERPRETED

# --------------------------------------------------------------------------
# JANI <-> Z3
# --------------------------------------------------------------------------

_BIN = {
    "+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
    "%": lambda a, b: a % b,
    "<": lambda a, b: a < b, "≤": lambda a, b: a <= b,
    ">": lambda a, b: a > b, "≥": lambda a, b: a >= b,
    "=": lambda a, b: a == b, "≠": lambda a, b: a != b,
    "∧": lambda a, b: z3.And(a, b), "∨": lambda a, b: z3.Or(a, b),
    "min": lambda a, b: z3.If(a <= b, a, b),
    "max": lambda a, b: z3.If(a >= b, a, b),
}


def _real(x):
    return z3.ToReal(x) if z3.is_int(x) else x


def to_z3(j, env):
    if isinstance(j, bool):
        return z3.BoolVal(j)
    if isinstance(j, int):
        return z3.IntVal(j)
    if isinstance(j, float):
        f = Fraction(j).limit_denominator(10**12)
        return z3.RealVal(f"{f.numerator}/{f.denominator}")
    if isinstance(j, str):
        return env[j]
    op = j["op"]
    if op == "ite":
        return z3.If(to_z3(j["if"], env), to_z3(j["then"], env), to_z3(j["else"], env))
    if op == "¬":
        return z3.Not(to_z3(j["exp"], env))
    a, b = to_z3(j["left"], env), to_z3(j["right"], env)
    if op == "/":
        return _real(a) / _real(b)
    return _BIN[op](a, b)


def _neg_coeff(e):
    """For a*x with a negative numeral a, return (|a|, x); else None."""
    if z3.is_mul(e) and e.num_args() == 2:
        c, x = e.arg(0), e.arg(1)
        if (z3.is_int_value(c) or z3.is_rational_value(c)):
            v = Fraction(c.as_fraction().numerator, c.as_fraction().denominator) \
                if z3.is_rational_value(c) else Fraction(c.as_long())
            if v < 0:
                return -v, x
    return None


_CMP = {z3.Z3_OP_LE: "≤", z3.Z3_OP_LT: "<", z3.Z3_OP_GE: "≥", z3.Z3_OP_GT: ">"}
_NEG_CMP = {z3.Z3_OP_LE: ">", z3.Z3_OP_LT: "≥", z3.Z3_OP_GE: "<", z3.Z3_OP_GT: "≤"}


def _fold(op, xs):
    acc = xs[0]
    for x in xs[1:]:
        acc = {"op": op, "left": acc, "right": x}
    return acc


def from_z3(e):
    if z3.is_true(e):
        return True
    if z3.is_false(e):
        return False
    if z3.is_int_value(e):
        return e.as_long()
    if z3.is_rational_value(e):
        f = e.as_fraction()
        return lift(Fraction(f.numerator, f.denominator))
    k = e.decl().kind()
    if z3.is_const(e) and k == K:
        return e.decl().name()
    ch = e.children()
    if k == z3.Z3_OP_ADD:
        acc = None
        for c in ch:
            neg = _neg_coeff(c)
            if neg is None and (z3.is_int_value(c) or z3.is_rational_value(c)):
                v = c.as_fraction() if z3.is_rational_value(c) else Fraction(c.as_long())
                neg = (-Fraction(v.numerator, v.denominator), None) if v < 0 else None
            if neg is not None and acc is not None:
                coef, x = neg
                term = lift(coef) if x is None else (
                    from_z3(x) if coef == 1 else {"op": "*", "left": lift(coef), "right": from_z3(x)})
                acc = {"op": "-", "left": acc, "right": term}
            else:
                t = from_z3(c)
                acc = t if acc is None else {"op": "+", "left": acc, "right": t}
        return acc
    if k == z3.Z3_OP_MUL:
        return _fold("*", [from_z3(c) for c in ch])
    if k == z3.Z3_OP_SUB:
        return _fold("-", [from_z3(c) for c in ch])
    if k == z3.Z3_OP_UMINUS:
        return {"op": "-", "left": 0, "right": from_z3(ch[0])}
    if k in _CMP:
        return {"op": _CMP[k], "left": from_z3(ch[0]), "right": from_z3(ch[1])}
    if k == z3.Z3_OP_EQ:
        return {"op": "=", "left": from_z3(ch[0]), "right": from_z3(ch[1])}
    if k == z3.Z3_OP_DISTINCT and len(ch) == 2:
        return {"op": "≠", "left": from_z3(ch[0]), "right": from_z3(ch[1])}
    if k == z3.Z3_OP_NOT:
        inner = ch[0]
        ik = inner.decl().kind()
        if ik in _NEG_CMP:   # Not(a <= b)  ->  a > b
            return {"op": _NEG_CMP[ik], "left": from_z3(inner.arg(0)),
                    "right": from_z3(inner.arg(1))}
        return {"op": "¬", "exp": from_z3(inner)}
    if k == z3.Z3_OP_AND:
        return _fold("∧", [from_z3(c) for c in ch])
    if k == z3.Z3_OP_OR:
        return _fold("∨", [from_z3(c) for c in ch])
    if k == z3.Z3_OP_ITE:
        return {"op": "ite", "if": from_z3(ch[0]), "then": from_z3(ch[1]),
                "else": from_z3(ch[2])}
    if k == z3.Z3_OP_TO_REAL:
        return from_z3(ch[0])
    raise NotImplementedError(str(e))


_FOLD = {"+": lambda a, b: a + b, "-": lambda a, b: a - b, "*": lambda a, b: a * b,
         "<": lambda a, b: a < b, "≤": lambda a, b: a <= b, ">": lambda a, b: a > b,
         "≥": lambda a, b: a >= b, "=": lambda a, b: a == b, "≠": lambda a, b: a != b,
         "min": min, "max": max}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def quick(j):
    """Cheap bottom-up simplification: constant folding and unit laws."""
    if not isinstance(j, dict):
        return j
    op = j["op"]
    if op == "ite":
        c = quick(j["if"])
        if c is True:
            return quick(j["then"])
        if c is False:
            return quick(j["else"])
        t, e = quick(j["then"]), quick(j["else"])
        return t if t == e else {"op": "ite", "if": c, "then": t, "else": e}
    if op == "¬":
        x = quick(j["exp"])
        if isinstance(x, bool):
            return not x
        if isinstance(x, dict) and x["op"] == "¬":
            return x["exp"]
        return {"op": "¬", "exp": x}
    if "left" not in j:
        return j
    a, b = quick(j["left"]), quick(j["right"])
    if op in _FOLD and _num(a) and _num(b):
        return _FOLD[op](a, b)
    if op == "∧":
        if a is False or b is False:
            return False
        if a is True:
            return b
        if b is True:
            return a
    if op == "∨":
        if a is True or b is True:
            return True
        if a is False:
            return b
        if b is False:
            return a
    if op in ("+", "-") and b == 0:
        return a
    if op == "+" and a == 0:
        return b
    if op == "*" and (a == 1 or b == 1):
        return b if a == 1 else a
    if op == "-" and isinstance(a, dict) and a["op"] == "+" and _num(a["right"]) and _num(b):
        c = a["right"] - b                 # (x + c1) - c2  ->  x + (c1 - c2)
        return a["left"] if c == 0 else {"op": "+", "left": a["left"], "right": c}
    return {"op": op, "left": a, "right": b}


def size(j):
    return len(json.dumps(j, ensure_ascii=False))


def const_value(j):
    """Fraction if j is a numeric constant, else None."""
    if isinstance(j, bool):
        return None
    if isinstance(j, (int, float)):
        return Fraction(j)
    if isinstance(j, dict) and j.get("op") == "/" and \
            isinstance(j["left"], int) and isinstance(j["right"], int):
        return Fraction(j["left"], j["right"])
    return None


# --------------------------------------------------------------------------
# The oracle
# --------------------------------------------------------------------------

class Oracle:
    def __init__(self, variables):
        self.vars = variables
        self.env, bounds = {}, []
        for k, v in variables.items():
            if isinstance(v, Int):
                x = z3.Int(k)
                bounds += [x >= v.lo, x <= v.hi]
            else:
                x = z3.Bool(k)
            self.env[k] = x
        self.solver = z3.Solver()
        self.solver.add(*bounds)
        self.cache = {}
        self._zmemo, self._smemo = {}, {}
        self.queries = 0
        self.validate = True
        self.report = {"pruned_branches": 0, "dead_edges": 0, "dead_dests": 0,
                       "merged_dests": 0, "identity_assignments": 0}

    def z(self, j):
        """JANI -> Z3, memoised on the expression's text (paths repeat a lot)."""
        if not isinstance(j, dict):
            return to_z3(j, self.env)
        key = json.dumps(j, ensure_ascii=False)
        r = self._zmemo.get(key)
        if r is None:
            r = self._zmemo[key] = to_z3(j, self.env)
        return r

    def sat(self, conds):
        self.queries += 1
        self.solver.push()
        self.solver.add(*conds)
        r = self.solver.check()
        m = self.solver.model() if r == z3.sat else None
        self.solver.pop()
        return m if r == z3.sat else None

    def valid(self, ctx, claim):
        return self.sat(ctx + [z3.Not(claim)]) is None

    # -- used during tracing ---------------------------------------------
    def decide(self, ctx, cond):
        key = (json.dumps(ctx, ensure_ascii=False), json.dumps(cond, ensure_ascii=False))
        if key not in self.cache:
            c, e = [self.z(x) for x in ctx], self.z(cond)
            if self.valid(c, e):
                res = True
            elif self.valid(c, z3.Not(e)):
                res = False
            else:
                res = None
            self.cache[key] = res
            if res is not None:
                self.report["pruned_branches"] += 1
        return self.cache[key]

    # -- used after tracing ----------------------------------------------
    def simp(self, j):
        j = quick(j)
        if not isinstance(j, dict):
            return j                       # constant or variable: nothing to do
        key = json.dumps(j, ensure_ascii=False)
        if key in self._smemo:
            return self._smemo[key]
        try:
            out = from_z3(z3.simplify(self.z(j), arith_lhs=True))
        except NotImplementedError:
            out = j
        if size(out) > size(j):
            out = j
        self._smemo[key] = out
        return out

    def _add(self, a, b):
        ca, cb = const_value(a), const_value(b)
        if ca is not None and cb is not None:
            return lift(ca + cb)
        return self.simp({"op": "+", "left": a, "right": b})

    def postprocess(self, edges):
        errors, out = [], []
        for e in edges:
            g = self.simp(e["guard"]["exp"])
            gz = [self.z(g)]
            if self.sat(gz) is None:
                self.report["dead_edges"] += 1
                continue
            groups = {}   # normalised assignment map -> probability
            for d in e["destinations"]:
                p = self.simp(d["probability"]["exp"])
                if self.valid(gz, self.z(p) == 0):
                    self.report["dead_dests"] += 1
                    continue
                amap = {}
                for a in d["assignments"]:
                    v = self.simp(a["value"])
                    if v == a["ref"]:
                        self.report["identity_assignments"] += 1
                        continue
                    amap[a["ref"]] = v
                key = json.dumps(sorted(amap.items()), ensure_ascii=False)
                if key in groups:
                    self.report["merged_dests"] += 1
                    groups[key] = (amap, self._add(groups[key][1], p))
                else:
                    groups[key] = (amap, p)
            # semantic merge: different syntax, equal effect under the guard
            dests = list(groups.values())
            merged = []
            for amap, p in dests:
                for i, (bmap, q) in enumerate(merged):
                    if set(amap) != set(bmap):
                        continue           # cheap filter before asking Z3
                    refs = set(amap)
                    same = z3.And(*[self.z(amap.get(r, r)) == self.z(bmap.get(r, r))
                                    for r in refs]) if refs else z3.BoolVal(True)
                    if self.valid(gz, same):
                        self.report["merged_dests"] += 1
                        merged[i] = (bmap, self._add(q, p))
                        break
                else:
                    merged.append((amap, p))
            # static checks: translation validation of this edge
            total = z3.Sum([_real(self.z(p)) for _, p in merged])
            m = self.sat(gz + [total != 1]) if self.validate else None
            if m is not None:
                errors.append(f"edge {e['action']}: probabilities do not sum to 1 in {m}")
            for amap, p in (merged if self.validate else []):
                for ref, v in amap.items():
                    var = self.vars.get(ref)
                    if not isinstance(var, Int):
                        continue
                    vz = self.z(v)
                    m = self.sat(gz + [self.z(p) > 0,
                                       z3.Or(vz < var.lo, vz > var.hi)])
                    if m is not None:
                        state = {str(d): m[d] for d in m.decls()}
                        errors.append(f"edge {e['action']}: {ref} := "
                                      f"{json.dumps(v, ensure_ascii=False)} leaves "
                                      f"[{var.lo}, {var.hi}] in state {state}")
            out.append({**e, "guard": {"exp": g}, "destinations": [
                {"location": "l", "probability": {"exp": p},
                 "assignments": [{"ref": r, "value": v} for r, v in sorted(amap.items())]}
                for amap, p in merged]})
        if errors:
            raise ModelError("static check failed:\n  " + "\n  ".join(errors))
        return out
