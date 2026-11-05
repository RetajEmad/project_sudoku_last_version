"""Integrated demo.   python run_demo.py [--seed 7] [--misreads 2] [--givens 30] [--algo ucs|greedy|astar|bfs|dfs] [--goal solvable|unique]

Shows: the handwritten input, the network's first reading, the AC-3 conflict, the search trace, the corrected
and solved grid, the CSP with/without FC and AC-3, and the system with the NN-guided correction ON versus OFF.
Writes results/demo_overview.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sudoku_reader.nn import MLP
from sudoku_reader.generator import ImagePools, make_puzzle, make_mosaic, mosaic_picture
from sudoku_reader.integration import read_cells, solve_from_probs, format_grid
from sudoku_reader.csp import SudokuSolver, SOLVER_VARIANTS, find_conflicts


def draw_grid(ax, grid, colors=None, title=""):
    colors = colors or {}
    for c, col in colors.items():
        ax.add_patch(Rectangle((c % 9, 8 - c // 9), 1, 1, color=col, alpha=0.55, lw=0))
    for i in range(10):
        ax.plot([0, 9], [i, i], "k", lw=2 if i % 3 == 0 else 0.5)
        ax.plot([i, i], [0, 9], "k", lw=2 if i % 3 == 0 else 0.5)
    for c, v in enumerate(grid):
        if v:
            ax.text(c % 9 + .5, 8 - c // 9 + .5, str(v), ha="center", va="center", fontsize=11)
    ax.set_xlim(0, 9); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--misreads", type=int, default=2)
    ap.add_argument("--givens", type=int, default=30)
    ap.add_argument("--algo", default="ucs", choices=["ucs", "greedy", "astar", "bfs", "dfs"])
    ap.add_argument("--goal", default="unique", choices=["solvable", "unique"])
    a = ap.parse_args()

    d = np.load("data/mnist_1to9.npz"); net = MLP.load("models/mlp.npz")
    pools = ImagePools(d["x_test"], d["y_test"], net.predict_proba(d["x_test"]))
    rng = np.random.default_rng(a.seed)
    puzzle, solution = make_puzzle(rng, a.givens)
    imgs, mis = make_mosaic(puzzle, pools, rng, a.misreads)

    print("=" * 70, "\nINPUT: 81 handwritten cell images (%d given, %d blank); %d hard digits injected at %s"
          % (sum(1 for v in puzzle if v), sum(1 for v in puzzle if not v), a.misreads,
             [f"r{c // 9 + 1}c{c % 9 + 1}" for c in mis]))
    probs, blank = read_cells(imgs, net)
    rep = solve_from_probs(probs, blank, algo=a.algo, goal=a.goal)
    first = rep.first_reading
    wrong = [c for c in range(81) if first[c] != puzzle[c]]
    print("\n[1] NN first reading ([x] = misread):\n" + format_grid(first, mark=wrong))
    print("\n[2] AC-3 on the first reading:", "CONFLICT - a domain was emptied" if rep.ac3_conflict else "no conflict found",
          "| duplicate pairs:", [(f"r{i // 9 + 1}c{i % 9 + 1}", f"r{j // 9 + 1}c{j % 9 + 1}") for i, j in rep.first_reading_conflicts])
    s = rep.search
    print(f"\n[3] Search ({a.algo.upper()}, goal={a.goal}): status={rep.status}; nodes expanded={s.expanded}, generated={s.generated}, "
          f"CSP goal tests={s.goal_tests}")
    for cell, old, new, cost in rep.corrections:
        print(f"    correction r{cell // 9 + 1}c{cell % 9 + 1}: {old} -> {new}  cost={cost:.3f}  "
              f"({'correct' if new == puzzle[cell] else 'WRONG'})")
    if rep.status != "solved":
        print("   ", rep.message)
        return
    print("\n[4] Solved grid:\n" + format_grid(rep.solution, mark=[c for c, *_ in rep.corrections]))
    print("    exact solution recovered:", rep.solution == solution)

    fixed = list(first)
    for c, _, new, _ in rep.corrections:
        fixed[c] = new
    print("\n[5] CSP on the corrected puzzle (static ordering so the pruning effect is visible):")
    cs = {}
    for name, kw in SOLVER_VARIANTS.items():
        st = SudokuSolver(ordering="static", **kw).solve(fixed, 1).stats
        cs[name] = st
        print(f"    {name:10s} assignments={st.assignments:6d} backtracks={st.backtracks:6d} "
              f"fc_prunes={st.fc_prunes:6d} ac3_removed={st.ac3_removed:4d}")
    off = SudokuSolver().solve(list(first), 1)
    print("\n[6] NN-guided correction ON : exact =", rep.solution == solution)
    print("    NN-guided correction OFF: ", "no solution (contradiction)" if not off.solved else
          ("solved but exact = %s" % (off.solutions[0] == solution)))

    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(mosaic_picture(imgs), cmap="gray_r")
    for c in mis:
        ax.add_patch(Rectangle(((c % 9) * 28, (c // 9) * 28), 28, 28, fill=False, ec="red", lw=2))
    for i in range(10):
        ax.plot([0, 252], [i * 28] * 2, "k", lw=1.5 if i % 3 == 0 else .3)
        ax.plot([i * 28] * 2, [0, 252], "k", lw=1.5 if i % 3 == 0 else .3)
    ax.axis("off"); ax.set_title("(a) Handwritten input (red = injected hard digit)", fontsize=10)

    conf_cells = {c: "red" for pair in rep.first_reading_conflicts for c in pair}
    conf_cells.update({c: "orange" for c in wrong if c not in conf_cells})
    draw_grid(fig.add_subplot(2, 3, 2), first, conf_cells,
              "(b) NN reading + AC-3\nred = duplicate-digit conflict, orange = other misread")

    ax = fig.add_subplot(2, 3, 3)
    tr = s.trace
    xs = [t[0] for t in tr]; gs = [t[1] for t in tr]
    ax.scatter(xs, gs, c=[t[2] for t in tr], cmap="viridis", s=18)
    ok = [t for t in tr if t[3]]
    ax.scatter([t[0] for t in ok], [t[1] for t in ok], marker="*", s=250, color="gold", ec="k", zorder=5, label="goal (CSP accepts)")
    ax.set_xlabel("order popped from the queue"); ax.set_ylabel("correction cost g")
    ax.set_title(f"(c) {a.algo.upper()} trace: {s.expanded} expanded, {s.goal_tests} CSP tests\ncolour = #corrections", fontsize=10)
    ax.legend()

    draw_grid(fig.add_subplot(2, 3, 4), rep.solution, {c: "limegreen" for c, *_ in rep.corrections},
              "(d) Corrected + solved (green = corrected cell)")

    ax = fig.add_subplot(2, 3, 5)
    names = list(cs)
    ax.bar(names, [cs[n].assignments for n in names], color=["tab:red", "tab:orange", "tab:green"])
    for i, n in enumerate(names):
        ax.text(i, cs[n].assignments, f"{cs[n].assignments}\nbt={cs[n].backtracks}", ha="center", va="bottom", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("assignments (log)"); ax.set_title("(e) CSP: effect of FC and AC-3", fontsize=10)

    ax = fig.add_subplot(2, 3, 6); ax.axis("off")
    ax.text(0, .95, "(f) NN-guided correction ON vs OFF", fontsize=11, weight="bold", va="top")
    ax.text(0, .78, f"OFF (trust first reading):\n  " +
            ("no solution - puzzle looks contradictory" if not off.solved else f"solved, exact = {off.solutions[0] == solution}"),
            fontsize=10, va="top")
    ax.text(0, .45, f"ON ({a.algo.upper()}, goal={a.goal}):\n  {len(rep.corrections)} correction(s), total cost "
            f"{rep.search.cost:.2f}\n  exact solution recovered = {rep.solution == solution}\n  "
            f"nodes expanded = {s.expanded}", fontsize=10, va="top")
    plt.tight_layout(); plt.savefig("results/demo_overview.png", dpi=140)
    print("\nwrote results/demo_overview.png")


if __name__ == "__main__":
    main()
