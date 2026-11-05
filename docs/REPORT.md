# Handwritten Sudoku Reader and Solver: Neural Network + Search + CSP
*Draft report generated with the code base; the team must review it, add their own reflections, and keep it <= 12 pages.*

## 1. Problem
A handwritten Sudoku is read cell by cell by a neural network. One misread digit usually makes the puzzle contradictory
(two equal digits in a unit) or, worse, silently changes the answer. We build a system that (1) reads the grid, (2) detects
that the reading is inconsistent, (3) searches for the *most probable* set of corrections, and (4) solves the corrected puzzle.
If nothing works it reports "unreadable" together with the least confident cells. Users: puzzle apps and education.

## 2. Data (see `docs/data_dictionary.md`)
Real: MNIST digit images and labels (digit 0 removed; 45,068 / 9,009 / 9,020 images for train / validation / test).
Generated: Sudoku puzzles (unique solution, 25/30/35 givens), the 9x9 mosaics, and the injected hard digits.
This is **not** photographed real-world Sudoku: every cell is an isolated MNIST image placed on a perfect grid.

## 3. The neural network
A 784-256-128-9 ReLU network trained with Adam for 12 epochs (NumPy), then **temperature scaling** (T = 1.745, fitted on the validation set only)
so that probabilities can serve as costs. Baseline: multinomial logistic regression.

| model (test, 9,020 digits) | accuracy | ECE raw | ECE after temp. scaling | NLL raw -> scaled |
|---|---|---|---|---|
| MLP (ours) | **98.0 %** | 0.0103 | **0.0048** | 0.0755 -> 0.0633 |
| Logistic regression | 92.3 % | 0.0058 | 0.0151 | 0.2806 -> 0.2785 |

Scaling halves the MLP's calibration error. For logistic regression the NLL improved marginally but ECE got *worse*
(temperature scaling optimises NLL, not ECE), which we report as it is. Confusion matrix and reliability diagram: `results/classifier_calibration.png`.

## 4. Search over corrections
* **State**: a set of corrections to the first reading (at most 3, each cell changed at most once).
* **Actions**: change a *candidate* cell to one of its alternatives. Candidates = the K = 10 least confident cells; alternatives = the other digits in the network's top-3.
  The space therefore has at most 1 + 20 + 180 + 960 = 1,161 states.
* **Cost** of one correction = `max(0, log p(read) - log p(new))`; costs add up, so UCS returns the correction set with the largest joint probability *under the network's independence assumption*.
* **Goal test = the CSP**: `solvable` (>= 1 solution; minimum version) or `unique` (exactly 1; extension). The zero-correction root is tested first.
* **Algorithms** (same space, same goal test, only the frontier order differs): UCS (priority g); Greedy (priority = number of duplicate pairs, ties by g);
  A* (g + h); **BFS** (FIFO: fewest corrections first, costs ignored); **DFS** (LIFO: deepest state first, depth limit = 3 corrections, first goal found is returned);
  random-priority baseline. BFS and DFS are uninformed: they do not use the network's probabilities and are not cost-optimal.
  Each state is generated at most once, so DFS cannot loop.
* **A\* heuristic**: every duplicate-digit pair needs at least one of its two cells changed, so
  `h = max over pairs of (cheapest possible change of either cell)`, or infinity when a pair can no longer be repaired. This never overestimates
  (each pair needs at least that much, and max <= the sum). Evidence: unit test against brute force, and in experiment 1 A* and UCS returned **identical cost in all 870 paired instances** (448 `solvable` + 422 `unique`).
* **Counters**: nodes expanded, nodes generated, CSP goal tests, plus the CSP counters of every test.

## 5. CSP
81 variables, domains 1-9, all-different on 27 units, givens = singleton domains. Three independent switches:
plain **Backtracking** (consistency check against assigned peers), **Forward Checking** (prune peers after each assignment), **AC-3** (once, before search).
AC-3 detects contradictory givens (two equal givens empty a domain, no search needed). Counters: assignments, backtracks, consistency checks, FC prunes, AC-3 revisions/removals.
Two variable orderings are available (static row-major with givens first, and MRV); both are reported in experiment 2.

## 6. Integration
`NN -> blank rule -> first reading -> AC-3 conflict check -> search (UCS/Greedy/A*/BFS/DFS) -> CSP goal test -> solved grid`, or
`Unreadable: ... Least confident: (r,c) read d @ p` when no corrected grid within 3 corrections passes. Demo: `run_demo.py` (`results/demo_overview.png`).
In the demo (seed 7, two hard digits, goal `unique`) the first reading has a duplicate-digit conflict; UCS expands 14 nodes, corrects `3->1` (cost 0.68) and `6->4` (cost 1.73), and the exact solution is recovered, whereas trusting the first reading gives "no solution". `run_search_graph.py` runs all five algorithms on this puzzle and draws their search graphs (`results/search_graph.png`): Greedy 3, A* 7, UCS 14, BFS 73 and DFS 457 expanded nodes, all returning the same correct pair.

## 7. Experiments (5 seeds; mean ± std over seeds; 20 puzzles per seed unless stated)
### Exp. 1: UCS vs Greedy vs A\* vs BFS vs DFS (30 givens; full table incl. both goals in `results/table_exp1.md`, graph in `results/exp1_search.png`)
Nodes expanded, goal = `solvable` (mean ± std over 5 seeds x 20 puzzles):

| misreads | UCS | Greedy | A* | BFS | DFS |
|---|---|---|---|---|---|
| 1 | 14.1 ± 25.7 | 12.5 ± 26.0 | **1.0 ± 0.1** | 14.0 ± 25.9 | 109.3 ± 59.1 |
| 2 | 13.0 ± 5.3 | **2.1 ± 0.3** | 3.2 ± 0.6 | 25.9 ± 4.5 | 100.4 ± 40.9 |
| 3 | 120.4 ± 67.1 | 73.8 ± 75.4 | **10.1 ± 2.9** | 252.8 ± 77.5 | 186.3 ± 117.5 |
| 4 | 567.1 ± 111.2 | 525.7 ± 136.3 | **37.1 ± 4.2** | 636.4 ± 123.7 | 596.6 ± 117.6 |

Exact recovery % (`solvable` / `unique`):

| misreads | UCS | Greedy | A* | BFS | DFS |
|---|---|---|---|---|---|
| 1 | 83 ± 7 / 97 ± 4 | 83 ± 7 / 97 ± 4 | 83 ± 7 / 97 ± 4 | 83 ± 7 / 96 ± 5 | 62 ± 12 / 85 ± 11 |
| 2 | 67 ± 4 / 88 ± 8 | 67 ± 4 / 88 ± 8 | 67 ± 4 / 88 ± 8 | 65 ± 6 / 87 ± 9 | 54 ± 11 / 81 ± 7 |
| 3 | 50 ± 13 / 76 ± 13 | 49 ± 11 / 75 ± 13 | 50 ± 13 / 76 ± 13 | 46 ± 10 / 76 ± 12 | 70 ± 15 / 85 ± 11 |
| 4 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

*Reading*:
* **Informed vs uninformed.** UCS, A* and Greedy use the network's costs and recover the same fraction of puzzles; A* does so with the fewest nodes at 1, 3 and 4 misreads (about 15x fewer than UCS at 4 misreads, 12x at 3) and the same cost as UCS in all 870 paired instances (448 `solvable` + 422 `unique`). Greedy is cheapest at 2 misreads but unreliable (std 75 at 3 misreads) and returned a costlier set than UCS in 8 of those 870 instances.
* **BFS** ignores the costs and expands the states level by level, so it needs about 2x the nodes of UCS at 3 misreads (253 vs 120; 349 vs 156 with `unique`) and 7-26x those of A* (median ratio, 2-3 misreads). Its recovery is almost identical to UCS because the first accepted state with the fewest corrections is usually also the cheapest one; it returned a costlier set than UCS in 21 of 870 instances.
* **DFS** dives to three-correction states before it tests simple fixes, so it is the most expensive at 1-2 misreads (109 and 100 nodes vs 1.0 and 3.2 for A*), makes more corrections than needed (1.17 vs 0.85 on average at 1 misread) and returns a costlier set than UCS in 150 of 870 instances. Because it accepts the first state the CSP likes, its recovery is lower at 1-2 misreads (62 vs 83 % `solvable`, 85 vs 97 % `unique`).
  At 3 misreads DFS appears *better* (70 vs 50 %, 85 vs 76 %). We believe this is the same effect as the random baseline in experiment 3 (a search that reaches three-correction states early matches exactly when three misreads exist), but we did not test that explanation, and the difference is only about 1.5 standard deviations with 5 seeds.
* **The large std at 1 misread (±25)** comes from one puzzle per goal that was reported unreadable after the search had exhausted the whole 1,161-state space; the median at 1 misread is 1 node for every algorithm.
* Recovery is 0 at 4 misreads because the system is limited to 3 corrections: this is a design limit, not noise. (A wrong 3-correction grid is still accepted as "solved" in 55 % / 32 % of these puzzles, which is the *unique is not true* limit below.)
* *Where failures come from*: with `solvable`, a misread that does not break the puzzle is accepted as it is; `unique` removes most of these, see Limits.

*Note on reproducibility*: after BFS and DFS were added, the UCS and Greedy rows reproduce the earlier run exactly; A* differs by 1-2 expanded nodes in about 25 of 3,000 rows (float32 rounding at the 1e-7 level changes ties between equal priorities).

### Exp. 2: BT vs BT+FC vs BT+FC+AC-3 (solving generated puzzles, first solution, cap 100,000 assignments)
| ordering | givens | BT assignments | BT+FC | BT+FC+AC-3 | seconds BT / FC / FC+AC-3 |
|---|---|---|---|---|---|
| static | 25 | 31,218 ± 6,284 (13 % hit the cap) | 9,457 ± 3,356 | 6,249 ± 2,063 | 1.30 / 0.13 / 0.10 |
| static | 30 | 2,893 ± 968 | 825 ± 257 | 488 ± 146 | 0.10 / 0.009 / 0.014 |
| static | 35 | 578 ± 125 | 203 ± 19 | 119 ± 25 | 0.019 / 0.002 / 0.007 |
| MRV | 25 | 379 ± 140 | 318 ± 107 | 318 ± 107 | 0.046 / 0.008 / 0.017 |
| MRV | 30 | 125 ± 21 | 115 ± 16 | 115 ± 16 | 0.011 / 0.003 / 0.011 |
| MRV | 35 | 84 ± 2 | 83 ± 1 | 83 ± 1 | 0.005 / 0.001 / 0.007 |

*Reading*: with a fixed ordering FC cuts assignments about 3x and AC-3 a further ~1.5-1.7x (plain BT at 25 givens is capped in 13 % of puzzles, so its mean is an *under*-estimate).
With the MRV heuristic the gap almost vanishes, and **AC-3 adds no pruning benefit at all on these puzzles, only 2-7x more time**: AC-3 on a not-equal constraint only propagates fixed values, which MRV+FC already exploits.
The honest conclusion is that ordering matters more than AC-3 for Sudoku; AC-3's real value here is the cheap early contradiction detection (zero assignments) used by the integration.

### Exp. 3: NN-guided correction vs no correction vs random correction (50 puzzles per level; expansion budget 1000 for the searching methods)
| misreads | no correction | random / solvable | NN-guided UCS / solvable | random / unique | NN-guided UCS / unique |
|---|---|---|---|---|---|
| 1 | 0 ± 0 | 50 ± 7 | **88 ± 8** | 60 ± 7 | **94 ± 9** |
| 2 | 0 ± 0 | 24 ± 15 | **74 ± 17** | 56 ± 25 | **92 ± 13** |
| 3 | 0 ± 0 | 58 ± 15 | 60 ± 17 | 76 ± 5 | **82 ± 19** |

*Reading and a caveat*: trusting the first reading never recovers the solution once a hard digit is present. NN-guided correction is clearly better than random at 1-2 misreads and only slightly better at 3 (82 vs 76 % with `unique`, within the spread).
**The random baseline looks deceptively good (and non-monotone, 58 % at 3 vs 24 % at 2 misreads)** because a budget of 1000 expansions almost *enumerates the whole 1,161-state space*, and at exactly 3 misreads the 3-correction cap leaves no room for wrong supersets.
`results/table_extra_random_budget.md` (5 seeds x 10 puzzles, goal unique) shows the budget dependence: random reaches 22 / 6 / 2 % at budget 50 and 74 / 52 / 64 % at budget 1000, while NN-guided UCS needs only 3 / 38 / 63 expansions on average for 96 / 82 / 94 %.
The network's probabilities therefore buy *efficiency* (and better accuracy at 1-2 misreads), not an impossible-to-reach solution. Standard deviations are wide because each seed has only 10 puzzles per level here.

## 8. Limits (please keep in the final version)
1. **Controlled errors.** Misreads are injected by picking test images the network gets wrong (about 190 such images exist, so they are re-used across puzzles). Recovery rates are for *those* levels, not for natural handwriting or scanned pages.
2. **Unique is not true.** The `unique` goal accepts any correction that yields exactly one solution; it can accept a wrong-but-unique grid (a unit test with 5 removed givens shows this). In experiment 1 (`unique`, UCS, 100 puzzles per level) the failures are 3 / 12 / 24 at 1 / 2 / 3 misreads: at 1 misread 2 were first readings accepted unchanged and 1 was reported unreadable; at 2 misreads 10 returned a wrong-but-unique corrected grid; at 3 misreads 16 did and 8 were reported unreadable.
3. **Candidate set.** Only the 10 least-confident cells and top-3 digits are considered; a confidently wrong digit is unreachable. In a 40-puzzle pilot the true digit was reachable for 100 / 100 / 97.5 % of puzzles at 1 / 2 / 3 misreads (not part of the final experiment pipeline).
4. **Max 3 corrections**, so >= 4 misreads cannot be recovered (0 %).
5. **Independence assumption.** Adding log-costs assumes cells are read independently; neighbouring digits in real grids are not independent of image conditions.
6. **Blank rule.** The brightness rule has a thin margin on real digits (min mean 0.028 vs threshold 0.02); a scanned page with noise or a grid-line would break it.
7. **Statistics.** 5 seeds x 20 puzzles (10 in exp. 3) give wide standard deviations; the ranking of A* vs UCS vs BFS nodes is clear; small recovery differences between UCS, Greedy, A* and BFS are not significant, and the DFS differences at 3 misreads are within about 1.5 standard deviations.
8. Calibration reported with ECE on 15 equal-width bins; ECE is itself noisy on 9,020 samples.

## 9. Conclusions
Search over network-probability costs plus a CSP goal test turns an unusable misread grid into the exact solution in 73-94 % of 1-2-misread cases (82 % with 3 misreads under the `unique` goal), at a few to some dozens of expanded nodes.
A* gives the same optimal answers as UCS with several times fewer nodes, while the uninformed BFS and DFS spend more nodes and (DFS) return costlier, less often exact corrections. Forward checking is the decisive CSP technique under a naive ordering, but with MRV even FC and AC-3 are almost redundant on Sudoku.
The `unique` goal is the better acceptance test, and neither goal can guarantee truth.
