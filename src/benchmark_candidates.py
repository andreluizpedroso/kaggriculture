"""Head-to-head benchmark of candidate submission files (main.py) against
built-in bots and our current agent, both seats, several seeds.

Entry points are resolved with Kaggle's own loader rule -- execute the file
and take the LAST callable in its namespace -- not `module.agent`, because
files built from layered wrappers can leave a stale inner `agent` bound
(see the 2945 Farm notebook, "Five lessons about evaluation" #1).

Usage: python -m src.benchmark_candidates [seeds]
"""

import sys
import time
from pathlib import Path

from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = {
    "farm2945": ROOT / "reference_agents" / "thomastschinkel_2945" / "main.py",
    "current": ROOT / "main.py",
    "tetsutani": ROOT / "reference_agents" / "tetsutani" / "main.py",
    "romanrozen": ROOT / "reference_agents" / "romanrozen" / "main.py",
    "boatlee": ROOT / "reference_agents" / "boatlee" / "main.py",
}
OPPONENTS = ["random", "starter", "current", "tetsutani", "romanrozen", "boatlee"]
TURN_BUDGET_S = 1.0


def load_last_callable(path):
    ns = {"__name__": f"cand_{path.parent.name}"}
    exec(compile(path.read_bytes(), str(path), "exec"), ns)
    return [v for v in ns.values() if callable(v)][-1]


def _timed(fn, sink):
    def wrapped(obs):
        t0 = time.perf_counter()
        out = fn(obs)
        sink.append(time.perf_counter() - t0)
        return out
    return wrapped


def run_game(cand_fn, opp, seed, seat):
    times = []
    fns = [_timed(cand_fn, times), opp]
    if seat == 1:
        fns.reverse()
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run(fns)
    final = env.steps[-1]
    mine, theirs = (final[seat].reward, final[1 - seat].reward)
    return mine, theirs, max(times) * 1000 if times else 0.0


def main():
    seeds = list(range(1, int(sys.argv[1]) + 1)) if len(sys.argv) > 1 else list(range(1, 6))
    name = "farm2945"
    cand = load_last_callable(CANDIDATES[name])
    print(f"candidate={name} entry={cand.__name__} seeds={seeds} (both seats)")
    for opp_name in OPPONENTS:
        opp = load_last_callable(CANDIDATES[opp_name]) if opp_name in CANDIDATES else opp_name
        rows = []
        for seed in seeds:
            for seat in (0, 1):
                rows.append(run_game(cand, opp, seed, seat))
        wins = sum(1 for m, t, _ in rows if m > t)
        avg_me = sum(m for m, _, _ in rows) / len(rows)
        avg_opp = sum(t for _, t, _ in rows) / len(rows)
        worst = max(r[2] for r in rows)
        print(f"vs {opp_name:<8} wins={wins}/{len(rows)} avg_bank {name}={avg_me:,.0f} opp={avg_opp:,.0f} worst_turn={worst:.1f}ms", flush=True)


if __name__ == "__main__":
    main()
