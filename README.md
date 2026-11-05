# Project 7 - Handwritten Sudoku Reader and Solver

A neural network reads a Sudoku written in handwritten digits (MNIST images). Because it sometimes misreads a
digit and breaks the puzzle, **search** looks for the cheapest set of corrections (guided by the network's
probabilities) and a **CSP solver** (Backtracking / Forward Checking / AC-3) tests every corrected puzzle.

```
81 cell images --NN--> digit probabilities --first reading--> AC-3 finds conflict
      --> Search (UCS / Greedy / A* / BFS / DFS) proposes corrections --> CSP (BT+FC+AC-3) accepts or rejects --> solved grid
                                                            \--> "unreadable" + least-confident cells if nothing works
```


## Setup (Python 3.12, tested on Windows; ~1 CPU core is enough)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run (exact commands, in this order)
```bash
python make_data.py            # downloads MNIST, drops digit 0, writes data/mnist_1to9.npz   (~20 MB download)
python train_nn.py             # trains MLP + logistic-regression baseline, calibrates, writes models/ and results/
python -m pytest               # 52 tests (needs make_data.py for one test; it is skipped otherwise)
python run_demo.py             # the integrated demo  -> results/demo_overview.png
python run_demo.py --seed 3 --misreads 3 --algo astar --goal solvable      # other demo settings (ucs|greedy|astar|bfs|dfs)
python run_search_graph.py     # search graphs of all 5 algorithms on one puzzle -> results/search_graph.png
python run_search_graph.py --seed 3 --misreads 3 --goal solvable --show     # other puzzle, open a window
python run_experiments.py      # the three experiments, 5 seeds (~10 min on 1 core)
python run_experiments.py --quick                                          # 15-second smoke run
python extra_random_budget.py  # supplement to experiment 3 (random-baseline budget)
```
Everything is seeded. Change `TEAM_SEED` at the top of `run_experiments.py` to give your team its own generator seed.

## Layout
| path | content |
|---|---|
| `make_data.py` | reproducible data download + preprocessing (digit 0 dropped, 0-1 scaling, 50k/10k/10k split) |
| `sudoku_reader/nn.py` | NumPy MLP (784-256-128-9), temperature scaling, ECE, confusion matrix |
| `sudoku_reader/csp.py` | Sudoku CSP: Backtracking, Forward Checking, AC-3 as separate switches + counters |
| `sudoku_reader/search.py` | correction search: UCS, Greedy, A*, BFS, DFS, random baseline, brute-force checker, counters |
| `sudoku_reader/generator.py` | seeded unique-solution puzzle generator, handwritten mosaic builder, hard-digit injection |
| `sudoku_reader/integration.py` | NN -> AC-3 -> search -> CSP pipeline, "unreadable" report |
| `train_nn.py`, `run_demo.py`, `run_search_graph.py`, `run_experiments.py` | scripts (`run_search_graph.py` draws the 5-algorithm search graph) |
| `tests/` | 52 tests: >=5 normal, >=3 edge, >=2 impossible (clear "no solution"/"unreadable" messages); `test_bfs_dfs.py` covers BFS/DFS |
| `results/` | metrics, tables (`table_exp*.md`), plots, raw CSVs, `correction_log.csv` |
| `docs/` | data dictionary, report draft, presentation/demo plan, contribution-statement template |

## Important modelling choices (details in `docs/REPORT.md`)
* **Search space.** State = set of corrections to the first reading (max 3). Candidate cells = the K=10 least
  confident cells; alternatives = the network's top-3 digits. Cost = `max(0, log p(read) - log p(new))`.
* **Five algorithms, one search space.** All use the same states, actions, costs and CSP goal test; only the frontier
  order differs: UCS = lowest g, Greedy = fewest duplicate pairs, A* = g + h, **BFS = fewest corrections first (FIFO,
  ignores the costs), DFS = deepest first (LIFO, depth limit = max corrections, takes the first goal it meets)**.
  BFS and DFS are uninformed, so they are neither cost-optimal nor guided by the network's probabilities.
* **Goal test = the CSP.** `solvable` (>=1 solution; the minimum version) or `unique` (exactly one; the extension).
* **A\* heuristic.** For each duplicate-digit pair at least one of its two cells must change, so
  `h = max over pairs of the cheapest change of either cell` (admissible; tested against brute force).
* **Hard digits.** To have something to correct, `m` of the given cells are filled with test images the network
  actually misreads (correct digit still in its top-3). All other cells use correctly classified test images.
  So reported recovery rates are for *controlled* misread levels, not for natural handwriting.
* **Not done (optional items):** maintaining arc consistency during search (MAC), a second goal heuristic for Greedy.
