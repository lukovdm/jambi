"""ACAS Xu-flavoured collision avoidance MDP.

Relative lateral geometry on a grid: the intruder closes in over `tau` steps,
`dy` is the lateral separation. Each step the system issues an advisory
(clear of conflict, weak/strong left, weak/strong right); the pilot complies
probabilistically (faster for a repeated advisory), the intruder randomly
changes its lateral rate. Rewards count alerts and sense reversals;
optionally at most B steps may carry an advisory. NMAC = |dy| <= 1 when tau reaches 0.
"""
from janitrace import (Model, Int, Bool, Reward, action, flip, choose, guard, reward,
                       minimum, maximum, Pmax, Pmin, Emin, F, V)

COC, WL, SL, WR, SR = range(5)
TARGET = {WL: -1, SL: -2, WR: 1, SR: 2}


def make_acas(T=30, D=16, B=None, advisories=True):
    class Acas(Model):
        tau = Int(0, T, init=T)          # steps to closest approach
        dy = Int(-D, D, init=0)          # lateral separation (head-on start)
        vy = Int(-2, 2, init=0)          # ownship lateral rate
        vi = Int(-2, 2, init=0)          # intruder lateral rate
        adv = Int(0, 4, init=COC)        # previous advisory
        nmac = Bool(init=False)
        alert = Reward()                 # +1 per step with an advisory
        reversal = Reward()              # +1 when the advisory flips left <-> right

        @action
        def fly(s):
            guard(s.tau > 0)
            a = choose(5) if advisories else COC
            if a != COC:
                if B is not None:
                    guard(s.alerts < B)
                    s.alerts += 1
                reward("alert")
                if (a in (WL, SL) and (s.adv == WR or s.adv == SR)) or \
                   (a in (WR, SR) and (s.adv == WL or s.adv == SL)):
                    reward("reversal")
                p = 0.9 if a == s.adv else 0.5      # pilots respond faster to a repeated advisory
                if flip(p):
                    t = TARGET[a]
                    if s.vy < t:
                        s.vy += 1
                    elif s.vy > t:
                        s.vy -= 1
            if flip(0.3):                            # intruder manoeuvres
                if flip(0.5):
                    s.vi = minimum(s.vi + 1, 2)
                else:
                    s.vi = maximum(s.vi - 1, -2)
            s.dy = maximum(minimum(s.dy + s.vy - s.vi, D), -D)
            s.tau -= 1
            if s.tau == 0 and -1 <= s.dy <= 1:
                s.nmac = True
            s.adv = a

        @action
        def done(s):                                 # absorbing after closest approach
            guard(s.tau == 0)

    if B is not None:
        Acas.alerts = Int(0, B, init=0)  # alerting budget: steps with an advisory
    props = {"nmac_min": Pmin(F(V.nmac)), "nmac_max": Pmax(F(V.nmac)),
             "alerts_min": Emin("alert", V.tau == 0)}
    return Acas, props


if __name__ == "__main__":
    from janitrace.storm import check
    for B in (0, 1, 2, 4):
        M, props = make_acas(T=20, D=16, B=B)
        v = check(M, {"nmac_min": props["nmac_min"]}, simplify=True)
        print(f"alert budget {B}: minimal NMAC probability {v['nmac_min']:.6f}")
