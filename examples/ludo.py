"""Two-player, two-pawn 'Mens erger je niet' (Ludo) on a shared ring of L squares.

Player A (us) chooses which pawn to move; player B follows a fixed policy
(move its most advanced legal pawn). A 6 is needed to enter and gives
another turn; landing on an opponent's pawn sends it back to base; own pawns
block. Positions are relative to each player's start: 0 = base, 1..L = ring,
L+1 = finished. B's start square is opposite A's.
"""
from janitrace import (Model, Int, Bool, action, choose, guard, uniform,
                       minimum, ite, Pmax, Pmin, F, V)


def make_ludo(L=12):
    FIN, H = L + 1, L // 2
    A, B = ["a0", "a1"], ["b0", "b1"]

    def other_view(r):
        """Square r (1..L) in one player's coordinates, seen by the other."""
        return ite(r > H, r - H, r + H)

    def target(s, mine, k):
        """New position of pawn k, or None if it cannot move."""
        p, d = getattr(s, mine[k]), s.die
        if p == 0:
            if d == 6 and getattr(s, mine[1 - k]) != 1:
                return 1
            return None
        if p == FIN:
            return None
        np = minimum(p + d, FIN)
        if np != FIN and getattr(s, mine[1 - k]) == np:
            return None                            # blocked by own pawn
        return np

    def move(s, mine, opp, k, np):
        setattr(s, mine[k], np)
        if np != FIN:
            sq = other_view(np)
            for j in range(2):
                if getattr(s, opp[j]) == sq:
                    setattr(s, opp[j], 0)          # captured
        if s.die != 6:
            s.turn = 1 - s.turn
        s.die = 0

    def over(s):
        return (s.a0 == FIN and s.a1 == FIN) or (s.b0 == FIN and s.b1 == FIN)

    class Ludo(Model):
        a0 = Int(0, FIN, init=0)
        a1 = Int(0, FIN, init=0)
        b0 = Int(0, FIN, init=0)
        b1 = Int(0, FIN, init=0)
        turn = Int(0, 1, init=0)
        die = Int(0, 6, init=0)                    # 0 = not rolled yet

        @action
        def roll(s):
            guard(s.die == 0 and not over(s))
            s.die = uniform(6) + 1

        @action
        def move_a(s):                             # nondeterministic: our choice
            guard(s.turn == 0 and s.die > 0)
            k = choose(2)
            np = target(s, A, k)
            guard(np is not None)
            move(s, A, B, k, np)

        @action
        def pass_a(s):
            guard(s.turn == 0 and s.die > 0)
            guard(target(s, A, 0) is None and target(s, A, 1) is None)
            if s.die != 6:
                s.turn = 1
            s.die = 0

        @action
        def move_b(s):                             # fixed opponent policy
            guard(s.turn == 1 and s.die > 0)
            t0, t1 = target(s, B, 0), target(s, B, 1)
            guard(t0 is not None or t1 is not None)
            if t0 is not None and (t1 is None or s.b0 >= s.b1):
                move(s, B, A, 0, t0)
            else:
                move(s, B, A, 1, t1)

        @action
        def pass_b(s):
            guard(s.turn == 1 and s.die > 0)
            guard(target(s, B, 0) is None and target(s, B, 1) is None)
            if s.die != 6:
                s.turn = 0
            s.die = 0

        @action
        def end(s):
            guard(over(s))

    a_wins = (V.a0 == FIN) & (V.a1 == FIN)
    return Ludo, {"win_max": Pmax(F(a_wins)), "win_min": Pmin(F(a_wins))}


if __name__ == "__main__":
    from janitrace.storm import check
    M, props = make_ludo(L=12)
    v = check(M, props, simplify=True)
    print(f"P(we win): best play {v['win_max']:.4f}, worst play {v['win_min']:.4f}")
