"""Search graphs of all five algorithms on ONE handwritten puzzle.

    python run_search_graph.py [--seed 7] [--misreads 2] [--givens 30] [--goal unique|solvable] [--show]

For each algorithm (UCS, Greedy, A*, BFS, DFS) the same first reading is searched and the nodes are drawn in the
order the algorithm popped them:  x = pop order, y = correction cost g, colour = number of corrections,
gold star = the state the CSP accepted.  A last panel compares nodes expanded / CSP tests.
Writes results/search_graph.png (and opens a window with --show).
"""
import argparse
import numpy as np
import matplotlib
from sudoku_reader.nn import MLP
from sudoku_reader.generator import ImagePools, make_puzzle, make_mosaic
from sudoku_reader.integration import read_cells, solve_from_probs

ALGOS = ("ucs", "greedy", "astar", "bfs", "dfs")
LABEL = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*", "bfs": "BFS", "dfs": "DFS"}
COLORS = {"ucs": "tab:blue", "greedy": "tab:orange", "astar": "tab:green", "bfs": "tab:red", "dfs": "tab:purple"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--goal", default="unique", choices=["solvable", "unique"])
    ap.add_argument("--budget", type=int, default=5000, help="max expansions per algorithm")
    ap.add_argument("--show", action="store_true", help="open a window instead of only saving the PNG")
    a = ap.parse_args()
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load("data/mnist_1to9.npz")
    net = MLP.load("models/mlp.npz")
    pools = ImagePools(d["x_test"], d["y_test"], net.predict_proba(d["x_test"]))
    rng = np.random.default_rng(a.seed)
    puzzle, solution = make_puzzle(rng, a.givens)
    imgs, mis = make_mosaic(puzzle, pools, rng, a.misreads)
    probs, blank = read_cells(imgs, net)

    reps = {}
    print(f"puzzle seed={a.seed}, {a.givens} givens, {a.misreads} injected misreads at "
          f"{[f'r{c // 9 + 1}c{c % 9 + 1}' for c in mis]}, goal={a.goal}\n")
    print(f"{'algorithm':9s} {'status':10s} {'expanded':>8s} {'generated':>9s} {'CSP tests':>9s} {'#corr':>5s} "
          f"{'cost':>6s} {'exact':>5s}")
    for algo in ALGOS:
        r = solve_from_probs(probs, blank, algo=algo, goal=a.goal, max_expansions=a.budget)
        reps[algo] = r
        s = r.search
        print(f"{LABEL[algo]:9s} {r.status:10s} {s.expanded:8d} {s.generated:9d} {s.goal_tests:9d} "
              f"{len(r.corrections):5d} {(f'{s.cost:.2f}' if r.status == 'solved' else '-'):>6s} "
              f"{str(r.solution == solution):>5s}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    ymax = max((t[1] for r in reps.values() for t in r.search.trace), default=1) * 1.05 + 0.1
    for ax, algo in zip(axes.flat, ALGOS):
        r = reps[algo]
        tr = r.search.trace
        sc = ax.scatter([t[0] for t in tr], [t[1] for t in tr], c=[t[2] for t in tr], cmap="viridis",
                        vmin=0, vmax=3, s=14)
        ok = [t for t in tr if t[3]]
        ax.scatter([t[0] for t in ok], [t[1] for t in ok], marker="*", s=260, color="gold", ec="k", zorder=5,
                   label="goal (CSP accepts)")
        ax.set_ylim(-0.1, ymax)
        ax.set_xlabel("order popped from the frontier"); ax.set_ylabel("correction cost g")
        ax.set_title(f"{LABEL[algo]}: {r.search.expanded} expanded, {r.search.goal_tests} CSP tests"
                     f" ({r.status})", fontsize=10, color=COLORS[algo])
        if ok:
            ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=list(axes.flat), label="#corrections in the state", shrink=0.6, pad=0.01, ticks=[0, 1, 2, 3])

    ax = axes.flat[5]
    x = np.arange(len(ALGOS))
    ex = [reps[al].search.expanded for al in ALGOS]
    ax.bar(x, np.maximum(ex, 0.3), color=[COLORS[al] for al in ALGOS])
    for i, v in enumerate(ex):
        ax.text(i, max(v, 0.3), str(v), ha="center", va="bottom", fontsize=9)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([LABEL[al] for al in ALGOS])
    ax.set_ylabel("nodes expanded (log)")
    ax.set_title("Effort on this puzzle")
    fig.suptitle(f"Search graphs, same puzzle (seed {a.seed}, {a.misreads} misreads, goal = {a.goal})", fontsize=12)
    out = "results/search_graph.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print("\nwrote", out)
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
