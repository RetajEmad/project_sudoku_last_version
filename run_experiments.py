"""The three experiments of Project 7.   python run_experiments.py [--exp 1 2 3] [--quick]

Exp 1  UCS vs Greedy vs A* vs BFS vs DFS : nodes expanded + exact-recovery rate, 0-4 injected misreads (30 givens)
Exp 2  BT vs BT+FC vs BT+FC+AC-3 : assignments / backtracks / time on puzzles with 25, 30, 35 givens
Exp 3  NN-guided correction (UCS) vs no correction vs random correction : exact-recovery rate

Every number is mean +- std over SEEDS independent seeds (each seed = its own generated puzzles).
Raw rows go to results/raw_exp*.csv, tables to results/table_exp*.md, plots to results/exp*.png.
"""
import argparse, csv, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sudoku_reader.nn import MLP
from sudoku_reader.generator import ImagePools, make_puzzle, make_mosaic
from sudoku_reader.integration import read_cells, solve_from_probs
from sudoku_reader.csp import SudokuSolver, SOLVER_VARIANTS

TEAM_SEED = 2026          # "give each team its own generator seed": change this number
SEEDS = 5
OUT = Path("results")
_ctx = {}
ALGOS = ("ucs", "greedy", "astar", "bfs", "dfs")           # the five search algorithms of experiment 1
LABEL = {"ucs": "UCS", "greedy": "Greedy", "astar": "A*", "bfs": "BFS", "dfs": "DFS"}
COLORS = {"ucs": "tab:blue", "greedy": "tab:orange", "astar": "tab:green", "bfs": "tab:red", "dfs": "tab:purple"}


def ctx():
    if not _ctx:
        d = np.load("data/mnist_1to9.npz")
        net = MLP.load("models/mlp.npz")
        _ctx["net"] = net
        _ctx["pools"] = ImagePools(d["x_test"], d["y_test"], net.predict_proba(d["x_test"]))
    return _ctx


def rng_for(*key):
    return np.random.default_rng([TEAM_SEED, *key])


# ------------------------------------------------------------------ experiment 1
def exp1_seed(args):
    seed, n_per, goals = args
    c = ctx(); rows = []
    for m in range(5):
        rng = rng_for(1, seed, m)
        for i in range(n_per):
            puz, sol = make_puzzle(rng, 30)
            imgs, mis = make_mosaic(puz, c["pools"], rng, m)
            probs, blank = read_cells(imgs, c["net"])
            for goal in goals:
                for algo in ALGOS:
                    t = time.perf_counter()
                    r = solve_from_probs(probs, blank, algo=algo, goal=goal, max_expansions=5000)
                    rows.append(dict(seed=seed, misreads=m, goal=goal, algo=algo, puzzle=i, status=r.status,
                                     recovered=int(r.solution == sol), expanded=r.search.expanded,
                                     generated=r.search.generated, goal_tests=r.search.goal_tests,
                                     cost=r.search.cost if r.status == "solved" else np.nan,
                                     n_corr=len(r.corrections), seconds=time.perf_counter() - t))
    return rows


# ------------------------------------------------------------------ experiment 2
def exp2_seed(args):
    seed, n_per, cap = args
    rows = []
    for ng in (25, 30, 35):
        rng = rng_for(2, seed, ng)
        puzzles = [make_puzzle(rng, ng)[0] for _ in range(n_per)]
        for order in ("static", "mrv"):
            for name, kw in SOLVER_VARIANTS.items():
                for i, p in enumerate(puzzles):
                    t = time.perf_counter()
                    r = SudokuSolver(ordering=order, max_assignments=cap, **kw).solve(p, 1)
                    s = r.stats
                    rows.append(dict(seed=seed, givens=ng, ordering=order, variant=name, puzzle=i, status=r.status,
                                     assignments=s.assignments, backtracks=s.backtracks,
                                     fc_prunes=s.fc_prunes, ac3_removed=s.ac3_removed,
                                     seconds=time.perf_counter() - t))
    return rows


# ------------------------------------------------------------------ experiment 3
def exp3_seed(args):
    seed, n_per, budget = args
    c = ctx(); rows = []; log = []
    from sudoku_reader.integration import solve_no_correction
    for m in (1, 2, 3):
        rng = rng_for(3, seed, m)
        for i in range(n_per):
            puz, sol = make_puzzle(rng, 30)
            imgs, mis = make_mosaic(puz, c["pools"], rng, m)
            probs, blank = read_cells(imgs, c["net"])
            s0 = solve_no_correction(imgs, c["net"])
            base = dict(seed=seed, misreads=m, puzzle=i)
            rows.append(dict(base, goal="-", method="no correction", recovered=int(s0 == sol), expanded=0))
            for goal in ("solvable", "unique"):
                for method, algo in (("NN-guided (UCS)", "ucs"), ("random", "random")):
                    r = solve_from_probs(probs, blank, algo=algo, goal=goal, max_expansions=budget,
                                         rng=rng_for(33, seed, m, i))
                    rows.append(dict(base, goal=goal, method=method, recovered=int(r.solution == sol),
                                     expanded=r.search.expanded))
                    if algo == "ucs" and goal == "unique":
                        for cell, old, new, cost in r.corrections:
                            log.append(dict(seed=seed, misreads=m, puzzle=i, row=cell // 9 + 1, col=cell % 9 + 1,
                                            read=old, new=new, cost=round(cost, 4), true=puz[cell] or sol[cell],
                                            fixed_a_real_misread=int(cell in mis and new == sol[cell])))
    return rows, log


# ------------------------------------------------------------------ helpers
def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def agg(rows, keys, val):
    """mean and std over seeds of the per-seed mean of `val` for each key combination."""
    out = {}
    for k in sorted({tuple(r[x] for x in keys) for r in rows}, key=str):
        per_seed = []
        for s in sorted({r["seed"] for r in rows}):
            v = [r[val] for r in rows if tuple(r[x] for x in keys) == k and r["seed"] == s and r[val] == r[val]]
            if v:
                per_seed.append(np.mean(v))
        out[k] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0, len(per_seed))
    return out


def md_table(head, body):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in body]) + "\n"


def fmt(ms, nd=1):
    return f"{ms[0]:.{nd}f} ± {ms[1]:.{nd}f}"


def run_pool(fn, argl):
    with Pool(min(len(argl), 5)) as p:
        return p.map(fn, argl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (2 seeds, 2 puzzles)")
    a = ap.parse_args()
    seeds = 2 if a.quick else SEEDS
    OUT.mkdir(exist_ok=True)
    colors = COLORS

    if 1 in a.exp:
        n = 2 if a.quick else 20
        rows = sum(run_pool(exp1_seed, [(s, n, ("solvable", "unique")) for s in range(seeds)]), [])
        write_csv(OUT / "raw_exp1.csv", rows)
        body = []
        ex = agg(rows, ("goal", "algo", "misreads"), "expanded")
        rc = agg(rows, ("goal", "algo", "misreads"), "recovered")
        gt = agg(rows, ("goal", "algo", "misreads"), "goal_tests")
        ct = agg(rows, ("goal", "algo", "misreads"), "cost")
        for goal in ("solvable", "unique"):
            for algo in ALGOS:
                for m in range(5):
                    k = (goal, algo, m)
                    body.append([goal, LABEL[algo], m, fmt(ex[k]), fmt(gt[k]),
                                 fmt(ct[k], 2) if ct[k][2] else "-", f"{100 * rc[k][0]:.0f} ± {100 * rc[k][1]:.0f}"])
        # --- the search graph: nodes expanded (log) for both goals + recovery rate, all five algorithms
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
        w = 0.8 / len(ALGOS)
        for j, (goal, title) in enumerate((("solvable", "Nodes expanded (goal: solvable)"),
                                           ("unique", "Nodes expanded (goal: unique)"))):
            for k, algo in enumerate(ALGOS):
                mu = [ex[(goal, algo, m)][0] for m in range(5)]
                sd = [ex[(goal, algo, m)][1] for m in range(5)]
                ax[j].bar(np.arange(5) + (k - (len(ALGOS) - 1) / 2) * w, np.maximum(mu, 0.3), w, yerr=sd,
                          label=LABEL[algo], color=colors[algo], capsize=2)
            ax[j].set_yscale("log"); ax[j].set_xlabel("injected misreads"); ax[j].set_title(title)
            ax[j].set_xticks(range(5)); ax[j].legend(fontsize=8)
        for goal, ls in (("solvable", "--"), ("unique", "-")):
            for k, algo in enumerate(ALGOS):
                mu = np.array([rc[(goal, algo, m)][0] for m in range(5)]) * 100
                sd = np.array([rc[(goal, algo, m)][1] for m in range(5)]) * 100
                ax[2].errorbar(np.arange(5) + (k - 2) * .04, mu, sd, ls=ls, marker="o", ms=4,
                               color=colors[algo], label=f"{LABEL[algo]} / {goal}", capsize=2)
        ax[2].set_ylabel("exact solution recovered (%)"); ax[2].set_xlabel("injected misreads")
        ax[2].set_xticks(range(5)); ax[2].legend(fontsize=6, ncol=2); ax[2].set_title("Recovery rate")
        plt.tight_layout(); plt.savefig(OUT / "exp1_search.png", dpi=140)
        (OUT / "table_exp1.md").write_text(md_table(
            ["goal", "algorithm", "misreads", "nodes expanded", "goal tests (CSP calls)", "correction cost", "exact recovery % "], body))
        print("exp1 done")

    if 2 in a.exp:
        n = 2 if a.quick else 20
        rows = sum(run_pool(exp2_seed, [(s, n, 100000) for s in range(seeds)]), [])
        for r in rows:
            r["capped"] = int(r["status"] == "budget")
        write_csv(OUT / "raw_exp2.csv", rows)
        body = []
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        for oi, order in enumerate(("static", "mrv")):
            sub = [r for r in rows if r["ordering"] == order]
            asg = agg(sub, ("variant", "givens"), "assignments")
            bkt = agg(sub, ("variant", "givens"), "backtracks")
            sec = agg(sub, ("variant", "givens"), "seconds")
            cap = agg(sub, ("variant", "givens"), "capped")
            for v in SOLVER_VARIANTS:
                for ng in (25, 30, 35):
                    k = (v, ng)
                    body.append([order, v, ng, fmt(asg[k], 0), fmt(bkt[k], 0), fmt(sec[k], 3),
                                 f"{100 * cap[k][0]:.0f}%"])
            for qi, (stat, lab) in enumerate(((asg, "assignments"), (sec, "seconds"))):
                for k, v in enumerate(SOLVER_VARIANTS):
                    mu = [stat[(v, ng)][0] for ng in (25, 30, 35)]
                    sd = [stat[(v, ng)][1] for ng in (25, 30, 35)]
                    ax[oi, qi].bar(np.arange(3) + (k - 1) * 0.27, mu, 0.27, yerr=sd, label=v, capsize=2)
                ax[oi, qi].set_yscale("log"); ax[oi, qi].set_xticks(range(3)); ax[oi, qi].set_xticklabels([25, 30, 35])
                ax[oi, qi].set_xlabel("givens"); ax[oi, qi].set_title(f"{lab} per puzzle ({order} ordering)")
                ax[oi, qi].legend(fontsize=7)
        plt.tight_layout(); plt.savefig(OUT / "exp2_csp.png", dpi=140)
        (OUT / "table_exp2.md").write_text(md_table(
            ["ordering", "solver", "givens", "assignments", "backtracks", "seconds", "hit 100k cap"], body))
        print("exp2 done")

    if 3 in a.exp:
        n = 2 if a.quick else 10
        res = run_pool(exp3_seed, [(s, n, 1000) for s in range(seeds)])
        rows = sum([r[0] for r in res], []); log = sum([r[1] for r in res], [])
        write_csv(OUT / "raw_exp3.csv", rows)
        if log:
            write_csv(OUT / "correction_log.csv", log)
        rc = agg(rows, ("goal", "method", "misreads"), "recovered")
        body = []
        methods = ["no correction", "random", "NN-guided (UCS)"]
        for goal in ("-", "solvable", "unique"):
            for meth in methods:
                if (goal == "-") != (meth == "no correction"):
                    continue
                for m in (1, 2, 3):
                    body.append([goal, meth, m, f"{100 * rc[(goal, meth, m)][0]:.0f} ± {100 * rc[(goal, meth, m)][1]:.0f}"])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for j, goal in enumerate(("solvable", "unique")):
            for k, meth in enumerate(methods):
                g = "-" if meth == "no correction" else goal
                mu = [100 * rc[(g, meth, m)][0] for m in (1, 2, 3)]
                sd = [100 * rc[(g, meth, m)][1] for m in (1, 2, 3)]
                ax[j].bar(np.arange(3) + (k - 1) * 0.27, mu, 0.27, yerr=sd, label=meth, capsize=2)
            ax[j].set_xticks(range(3)); ax[j].set_xticklabels([1, 2, 3]); ax[j].set_xlabel("injected misreads")
            ax[j].set_title(f"Exact recovery (goal: {goal})"); ax[j].legend(fontsize=8)
        ax[0].set_ylabel("%")
        plt.tight_layout(); plt.savefig(OUT / "exp3_guided.png", dpi=140)
        (OUT / "table_exp3.md").write_text(md_table(["goal", "method", "misreads", "exact recovery %"], body))
        print("exp3 done")


if __name__ == "__main__":
    main()
