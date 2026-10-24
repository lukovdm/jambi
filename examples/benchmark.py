"""Plain vs Z3-simplified compilation, Storm build times, and bisimulation.

    python benchmark.py ludo 12 | acas 20 16 | queues 4 5
"""
import json, sys, time

import stormpy

from janitrace import to_jani


def timed(f):
    t = time.perf_counter()
    r = f()
    return r, time.perf_counter() - t


def bisimulation(model, props):
    try:                                   # stormpy >= 2.0
        import stormpy.transformations as T
        return T.perform_bisimulation(model, props, bisimulation_type=T.BisimulationType.STRONG)
    except ImportError:                    # stormpy 1.x
        return stormpy.perform_bisimulation(model, props, stormpy.BisimulationType.STRONG)


def value(model, prop):
    r = stormpy.model_checking(model, prop)
    s = model.initial_states[0]
    return r.get_value(s) if hasattr(r, "get_value") else r.at(s)


def bench(M, props=None):
    props = M.all_properties() if props is None else props
    for simplify in (False, True):
        stats = {}
        j, tc = timed(lambda: to_jani(M, props, simplify=simplify, stats=stats))
        edges = j["automata"][0]["edges"]
        jani, ps = stormpy.parse_jani_model_from_string(json.dumps(j, ensure_ascii=False))
        model, tb = timed(lambda: stormpy.build_model(jani, ps))
        vals = ", ".join(f"{n}={value(model, p):.6f}" for n, p in zip(props, ps))
        print(f"{'z3   ' if simplify else 'plain'} compile {tc:6.2f}s  paths={stats['runs']:5}  "
              f"edges={len(edges):5}  dests={sum(len(e['destinations']) for e in edges):5}  "
              f"build {tb:6.2f}s  {model.nr_states} states  {vals}")
    q, tq = timed(lambda: bisimulation(model, ps))
    print(f"bisimulation: {model.nr_states} -> {q.nr_states} states in {tq:.2f}s")


if __name__ == "__main__":
    which, args = sys.argv[1], list(map(int, sys.argv[2:]))
    if which == "ludo":
        from ludo import make_ludo
        bench(make_ludo(*args))
    elif which == "acas":
        from acas import make_acas
        M = make_acas(*args)
        bench(M, {k: M.all_properties()[k] for k in ("nmac_min", "nmac_max")})
    else:
        from queues import make_farm
        bench(make_farm(*args))
