"""Supplement to experiment 3: how much does the random-correction baseline depend on its expansion budget?
(goal = unique, 30 givens, 5 seeds x 10 puzzles).  python extra_random_budget.py -> results/table_extra_random_budget.md"""
import numpy as np
from run_experiments import ctx, rng_for, agg, fmt, md_table, OUT, SEEDS
from sudoku_reader.generator import make_puzzle, make_mosaic
from sudoku_reader.integration import read_cells, solve_from_probs

rows = []
c = ctx()
for seed in range(SEEDS):
    for m in (1, 2, 3):
        rng = rng_for(4, seed, m)
        for i in range(10):
            puz, sol = make_puzzle(rng, 30)
            imgs, _ = make_mosaic(puz, c["pools"], rng, m)
            probs, blank = read_cells(imgs, c["net"])
            for method, algo, budget in (("NN-guided UCS (budget 1000)", "ucs", 1000), ("random, budget 50", "random", 50),
                                         ("random, budget 200", "random", 200), ("random, budget 1000", "random", 1000)):
                r = solve_from_probs(probs, blank, algo=algo, goal="unique", max_expansions=budget,
                                     rng=rng_for(44, seed, m, i))
                rows.append(dict(seed=seed, method=method, misreads=m, recovered=int(r.solution == sol), expanded=r.search.expanded))
rc = agg(rows, ("method", "misreads"), "recovered"); ex = agg(rows, ("method", "misreads"), "expanded")
methods = ["NN-guided UCS (budget 1000)", "random, budget 50", "random, budget 200", "random, budget 1000"]
body = [[mt, m, f"{100 * rc[(mt, m)][0]:.0f} ± {100 * rc[(mt, m)][1]:.0f}", fmt(ex[(mt, m)])] for mt in methods for m in (1, 2, 3)]
(OUT / "table_extra_random_budget.md").write_text(md_table(["method", "misreads", "exact recovery %", "nodes expanded"], body))
print(open(OUT / "table_extra_random_budget.md").read())
