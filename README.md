# Jambi

Write probabilistic models as plain Python and compile them to
[JANI](https://jani-spec.org) for probabilistic model checkers such as
[Storm](https://www.stormchecker.org).

Each action is an ordinary Python function over the state. janitrace runs it on
symbolic values the way JAX or MLX trace array code: every path through the
function is enumerated, and each path becomes a guarded, probabilistic JANI
edge. Ordinary Python (loops with fixed bounds, helper functions,
comprehensions, classes) acts as a macro language and is unrolled while
tracing. The same function also runs on concrete values, so a model can be
simulated and unit-tested with exactly the code that gets verified.

```python
from janitrace import Model, Int, Bool, action, properties, flip, choose, guard, Pmax, F, V, write_jani

class Server(Model):
    n = Int(0, 10, init=0)
    broken = Bool(init=False)

    @action
    def step(s):
        guard(not s.broken)
        fast = choose()                                 # the scheduler decides
        if s.n < 10 and flip(0.7):                      # a job arrives
            s.n += 1
        if flip(0.9 if fast else 0.6) and s.n > 0:      # a job is served
            s.n -= 1
        if fast and flip(0.05):
            s.broken = True

    @action
    def repair(s):
        guard(s.broken)
        if flip(0.25):
            s.broken = False

    @properties
    def props():
        return {"overflow": Pmax(F(V.n == 10, steps=40))}

write_jani("server.jani", Server)
```

## Install

```bash
pip install .            # core: no dependencies
pip install ".[z3]"      # + Z3-based simplification and static checks
pip install ".[all]"     # + stormpy, for janitrace.storm and the examples
```

Requires Python 3.10+.

## The modelling language

| Construct | Meaning |
|---|---|
| `x = Int(lo, hi, init=..)`, `b = Bool(init=..)` | state variables |
| `q = [Int(0, 5, init=0) for _ in range(K)]` | a list of state variables; the length is fixed when the class is created, and the list becomes `q_0 … q_{K-1}` |
| `@action def name(s): ...` | an action; `s.x` reads and writes the state |
| `guard(cond)` | the action is enabled only if `cond` holds |
| `choose()` / `choose(n)` | nondeterministic choice (bool / index), resolved by the scheduler |
| `flip(p)` / `uniform(n)` | probabilistic choice (bool with probability `p` / uniform index) |
| `reward("name", v)` with `name = Reward()` | earn `v` on reward structure `name` in this step |
| `ite`, `minimum`, `maximum` | branch-free expressions |
| `Pmin`, `Pmax`, `F(target, steps=k)`, `Emin`, `Emax` | properties; refer to variables as `V.x` or `V.q[i]` |
| `@properties def props(): return {"name": Pmax(...)}` | the model's properties, used by `to_jani`, `write_jani` and `check` unless others are passed |
| a function without `@action` | a helper, called as `ModelName.helper(s, ...)` from the actions |

A plain Python `if` on a symbolic value is allowed: janitrace explores both
branches by re-running the function. Indexing a list with a state variable
(`s.q[s.ptr]`) branches over the possible indices.

Rules for an action body, since one MDP step resolves nondeterminism first and
probability second:

- `guard()` and `choose()` must come before the first `flip()` or `uniform()`;
- the function must not have side effects other than writing `s`, because it
  is run once per path;
- loops need a bound known at tracing time.

Violations raise `ModelError` while compiling.

## Z3 simplification (optional)

`to_jani(Model, simplify=True)` uses Z3 to:

- skip branches whose condition is already implied or contradicted by the
  variable bounds and the current path;
- normalise expressions and merge destinations with equal effect;
- prove that every edge's probabilities sum to 1;
- prove that no assignment leaves its variable's bounds, reporting a
  counterexample state otherwise. For example:
  `n := 1 + n leaves [0, 10] in state {'n': 10}`.

Pruning matters most when Python control flow creates many infeasible guard
combinations. In the Ludo example, it cut 1037 edges to 141 and Storm's sparse
build from about 11 s to 0.6 s. For arithmetic-heavy models it changes little.

## Checking with Storm

```python
from janitrace.storm import check
check(Server, simplify=True)
# {'overflow': 0.1565...}
check(Server, {"long": Pmax(F(V.n == 10, steps=80))})   # other properties
```

Or write the file with `write_jani` and use the Storm command line:
`storm --jani server.jani --janiproperty`.

## Examples

- `examples/server.py`: the quick-start model, simulated and checked.
- `examples/queues.py`: lists of state variables, including indexing with a
  state variable.
- `examples/acas.py`: an ACAS Xu-flavoured collision avoidance MDP with alert
  rewards and an optional alert budget.
- `examples/ludo.py`: a two-player, two-pawn "Mens erger je niet" game against
  a fixed opponent policy, with the game logic as helpers inside the model.
- `examples/benchmark.py`: plain against Z3-simplified compilation, with Storm
  build times and bisimulation.

## Tests

```bash
pip install ".[test]"
pytest            # Storm tests are skipped if stormpy is not installed
```

## Limitations

- Output is a single automaton with one location.
- There is no synchronisation between automata yet, and no continuous time or
  partial observability.
- Lists have a fixed length.
- Probabilities passed to `flip` must be concrete numbers. Use a Python `if` or
  `ite` first if they depend on the state.

## Status

This is an early research prototype.
