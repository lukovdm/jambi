"""Two-player, two-pawn 'Mens erger je niet' (Ludo) on a shared ring of L squares.

Player A (us) chooses which pawn to move; player B follows a fixed policy
(move its most advanced legal pawn). A 6 is needed to enter and gives
another turn; landing on an opponent's pawn sends it back to base; own pawns
block. Positions are relative to each player's start: 0 = base, 1..L = ring,
L+1 = finished. B's start square is opposite A's.
"""
from janitrace import (Model, Int, action, properties, choose, guard, uniform,
                       minimum, ite, Pmax, Pmin, F, V)


def make_ludo(L=12):
    FIN = L + 1                                    # position of a finished pawn
    H = L // 2                                     # offset between A's and B's start

    class Ludo(Model):
        a = [Int(0, FIN, init=0), Int(0, FIN, init=0)]   # our pawns
        b = [Int(0, FIN, init=0), Int(0, FIN, init=0)]   # opponent's pawns
        turn = Int(0, 1, init=0)                   # 0 = A, 1 = B
        die = Int(0, 6, init=0)                    # 0 = not rolled yet

        # Helpers: plain functions without @action, so they are not edges.
        # Call them as Ludo.<name>(...) from the actions.

        def other_view(r):
            """Square r (1..L) in one player's coordinates, seen by the other."""
            return ite(r > H, r - H, r + H)

        def target(s, mine, k):
            """New position of pawn mine[k], or None if it cannot move."""
            p = mine[k]
            if p == 0:
                if s.die == 6 and mine[1 - k] != 1:
                    return 1
                return None
            if p == FIN:
                return None
            np = minimum(p + s.die, FIN)
            if np != FIN and mine[1 - k] == np:
                return None                        # blocked by own pawn
            return np

        def move(s, mine, opp, k, np):
            mine[k] = np
            if np != FIN:
                sq = Ludo.other_view(np)
                for j in range(2):
                    if opp[j] == sq:
                        opp[j] = 0                 # captured
            Ludo.end_turn(s)

        def end_turn(s):
            if s.die != 6:                         # a 6 gives another turn
                s.turn = 1 - s.turn
            s.die = 0

        def stuck(s, mine):
            return Ludo.target(s, mine, 0) is None and Ludo.target(s, mine, 1) is None

        def over(s):
            return (s.a[0] == FIN and s.a[1] == FIN) or (s.b[0] == FIN and s.b[1] == FIN)

        # Actions

        @action
        def roll(s):
            guard(s.die == 0 and not Ludo.over(s))
            s.die = uniform(6) + 1

        @action
        def move_a(s):                             # nondeterministic: our choice
            guard(s.turn == 0 and s.die > 0)
            k = choose(2)
            np = Ludo.target(s, s.a, k)
            guard(np is not None)
            Ludo.move(s, s.a, s.b, k, np)

        @action
        def pass_a(s):
            guard(s.turn == 0 and s.die > 0 and Ludo.stuck(s, s.a))
            Ludo.end_turn(s)

        @action
        def move_b(s):                             # fixed opponent policy
            guard(s.turn == 1 and s.die > 0)
            t0, t1 = Ludo.target(s, s.b, 0), Ludo.target(s, s.b, 1)
            guard(t0 is not None or t1 is not None)
            if t0 is not None and (t1 is None or s.b[0] >= s.b[1]):
                Ludo.move(s, s.b, s.a, 0, t0)
            else:
                Ludo.move(s, s.b, s.a, 1, t1)

        @action
        def pass_b(s):
            guard(s.turn == 1 and s.die > 0 and Ludo.stuck(s, s.b))
            Ludo.end_turn(s)

        @action
        def end(s):
            guard(Ludo.over(s))

        @properties
        def props():
            a_wins = (V.a[0] == FIN) & (V.a[1] == FIN)
            return {"win_max": Pmax(F(a_wins)), "win_min": Pmin(F(a_wins))}

    return Ludo


if __name__ == "__main__":
    from janitrace.storm import check
    v = check(make_ludo(L=12), simplify=True)
    print(f"P(we win): best play {v['win_max']:.4f}, worst play {v['win_min']:.4f}")
