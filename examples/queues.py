"""K queues as a list of state variables: a nondeterministic dispatcher,
and a round-robin server that indexes the list with a state variable."""
from janitrace import Model, Int, action, flip, choose, Pmax, Pmin, F, V


def any_full(K, C):
    e = V.q[0] == C
    for i in range(1, K):
        e = e | (V.q[i] == C)
    return e


def make_farm(K, C):
    """The dispatcher (the scheduler) sends each arrival to a queue."""
    class Farm(Model):
        q = [Int(0, C, init=0) for _ in range(K)]      # length fixed at model init

        @action
        def step(s):
            t = choose(K)
            if s.q[t] < C and flip(0.8):
                s.q[t] += 1
            for i in range(K):
                if flip(0.25) and s.q[i] > 0:
                    s.q[i] -= 1

    goal = any_full(K, C)
    return Farm, {"pmax": Pmax(F(goal, steps=30)), "pmin": Pmin(F(goal, steps=30))}


def make_round_robin(K, C):
    """Indexing with a state variable: a server visits the queues in turn."""
    class RoundRobin(Model):
        q = [Int(0, C, init=0) for _ in range(K)]
        ptr = Int(0, K - 1, init=0)

        @action
        def step(s):
            t = choose(K)                                # where the arrival goes
            if s.q[t] < C and flip(0.8):
                s.q[t] += 1
            if flip(0.6) and s.q[s.ptr] > 0:             # symbolic index
                s.q[s.ptr] -= 1
            s.ptr = (s.ptr + 1) % K

    goal = any_full(K, C)
    return RoundRobin, {"pmax": Pmax(F(goal, steps=30)), "pmin": Pmin(F(goal, steps=30))}


if __name__ == "__main__":
    import random
    from janitrace import simulate
    from janitrace.storm import check
    for make in (make_farm, make_round_robin):
        M, props = make(3, 5)
        print(make.__name__, check(M, props, simplify=True))
    M, _ = make_round_robin(3, 5)
    print("a concrete run:", [st for st in simulate(M, 5, random.Random(1))])
