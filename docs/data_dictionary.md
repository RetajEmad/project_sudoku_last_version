# Data dictionary - Project 7

## Real data
| item | type | real / generated | description |
|---|---|---|---|
| MNIST image | 28x28 float32, 0-1 | **real** | handwritten digit, centred; 0 = background, 1 = ink |
| MNIST label | int 1..9 | **real** | digit shown (digit 0 removed: Sudoku has no 0) |
| split | train 45,068 / val 9,009 / test 9,020 | **real** | 50,000/10,000/10,000 original split minus zeros; train and test come from different writers; test set is untouched by training and calibration |

Source: public GitHub copy of MNIST (`mnist.pkl.gz`, sha256 `f11bb9e41d6c1b6c124aa38fd605497bdcfe2ee08cf7c2bb5a41ab5d759e1416`)
because yann.lecun.com was not reachable from our build machine. `make_data.py` asserts the official MNIST test-label counts.
Licence/terms of the original page were **not** re-checked by us: confirm before redistributing the files (we do not ship them).

## Generated data (all seeded)
| item | how it is made |
|---|---|
| Sudoku solution | randomised backtracking with MRV (`generator.random_full_grid`) |
| Puzzle | cells removed in random order while the solution stays **unique** (checked with the CSP, 2-solution cut-off) until the requested number of givens (25/30/35) |
| Blank cell image | zeros + `|N(0, 0.004)|` noise; recognised by the **brightness rule** `mean pixel < 0.02` (not by the NN). Every real digit in MNIST has mean >= 0.028 (checked by a test) |
| Mosaic | each given cell gets a random MNIST **test** image of that digit |
| Clean pool | test images the calibrated network classifies correctly |
| Hard pool (injection rule) | test images the network misreads but whose true digit is in its top-3 classes (about 190 images over 9 digits) |
| Misread level m | m distinct given cells (random, digit must have a hard image) use hard images; every other cell uses a clean image |
| Seeds | `rng = default_rng([TEAM_SEED, experiment, seed, level])`; 5 seeds per experiment |

## Model outputs
| item | description |
|---|---|
| `probs` (81x9) | temperature-scaled softmax per cell; blank cells are ignored |
| first reading | argmax digit per non-blank cell |
| correction cost | `max(0, log p(read) - log p(new))`, additive over corrected cells |

## Result files (`results/`)
| file | columns |
|---|---|
| `raw_exp1.csv` | (algo = ucs, greedy, astar, bfs, dfs) seed, misreads, goal, algo, puzzle, status, recovered (0/1), expanded, generated, goal_tests, cost, n_corr, seconds |
| `raw_exp2.csv` | seed, givens, ordering, variant, puzzle, status, assignments, backtracks, fc_prunes, ac3_removed, seconds, capped |
| `raw_exp3.csv` | seed, misreads, puzzle, goal, method, recovered, expanded |
| `correction_log.csv` | seed, misreads, puzzle, row, col, read, new, cost, true, fixed_a_real_misread (NN-guided UCS, goal=unique) |
| `classifier_metrics.json` | accuracy, ECE and NLL before/after temperature scaling, NN and logistic regression |
