"""Quick start: a server queue whose scheduler picks fast or slow service."""
import random

from janitrace import (Model, Int, Bool, action, properties, flip, choose, guard, simulate,
                       write_jani, Pmax, Pmin, F, V)

CAP = 10


class Server(Model):
    n = Int(0, CAP, init=0)          # jobs waiting
    broken = Bool(init=False)

    @action
    def step(s):
        guard(not s.broken)
        fast = choose()                                  # nondeterministic: the scheduler
        if s.n < CAP and flip(0.7):                      # a job arrives
            s.n += 1
        if flip(0.9 if fast else 0.6) and s.n > 0:       # a job is served
            s.n -= 1
        if fast and flip(0.05):                          # fast service may break the server
            s.broken = True

    @action
    def repair(s):
        guard(s.broken)
        if flip(0.25):
            s.broken = False

    @properties
    def props():
        return {"overflow_max": Pmax(F(V.n == CAP, steps=40)),
                "overflow_min": Pmin(F(V.n == CAP, steps=40))}


if __name__ == "__main__":
    j = write_jani("server.jani", Server, simplify=True)
    print("wrote server.jani with", len(j["automata"][0]["edges"]), "edges")
    print("a concrete run:", [st["n"] for st in simulate(Server, 15, random.Random(0))])
    try:
        from janitrace.storm import check
        print(check(Server))      # overflow_max ~ 0.156506, overflow_min ~ 6.98e-07
    except ImportError:
        print("install stormpy to model-check")
