# 10-minute presentation + 5-minute live demo (plan)

## Slides (10 min)
1. (1 min) Problem: handwritten Sudoku; one misread digit makes the puzzle "unsolvable" or silently wrong.
2. (1 min) Architecture: NN -> AC-3 -> Search -> CSP loop (README diagram).
3. (1.5 min) The NN: 98.0 % test accuracy vs 92.3 % logistic regression; temperature scaling halves the ECE;
   probabilities = correction costs. (`results/classifier_calibration.png`)
4. (1.5 min) Search design: state/action/cost/goal, K=10 candidates, UCS vs Greedy vs A* vs BFS vs DFS (heuristic + admissibility argument; BFS/DFS as the uninformed baselines).
5. (1.5 min) CSP design: BT, FC, AC-3 switches and counters.
6. (2 min) Experiments 1-3 (one slide each plot: `exp1_search.png`, `exp2_csp.png`, `exp3_guided.png`).
7. (1 min) Limits: controlled misreads, uniqueness is not truth, 4 misreads exceed the 3-correction limit.

## Live demo (5 min) - commands
1. `python run_demo.py --seed 7 --misreads 2` (30 s): show input mosaic, first reading, AC-3 conflict, corrections, solved grid.
2. **Search graphs** (1 min): point at panel (c) and the printed counters, then run `python run_search_graph.py` and show `results/search_graph.png`: the same puzzle searched by UCS, Greedy, A*, BFS and DFS (nodes popped, cost g, goal star).
3. **CSP pruning with/without FC and AC-3** (1 min): section [5] of the output / panel (e): BT vs BT+FC vs BT+FC+AC3 assignment counts.
4. **NN on vs off** (1 min): section [6]: without correction -> "no solution (contradiction)"; with -> exact solution.
5. **Impossible case** (1 min): `python -m pytest tests -k "unreadable or impossible" -v` shows the clear "no solution"/"Unreadable ... least confident cells" messages.
