import math, numpy as np, pytest
from pathlib import Path
from sudoku_reader.search import make_problem, search, brute_force_best, heuristic_astar, goal_checker
from sudoku_reader.integration import solve_from_probs, format_grid
from sudoku_reader.generator import make_puzzle, make_mosaic, ImagePools, BLANK_MEAN_THRESHOLD
from sudoku_reader.csp import find_conflicts
from helpers import PUZZLE, SOLUTION, probs_from_grid


def misread_instance():
    """Cell 4 (row 1, col 5) holds a true 7; reading it as 1 duplicates the 1s at cells 12 and 67."""
    g = list(PUZZLE)
    cell = 4
    g[cell] = 1
    return g, cell

# ---------------- normal cases ----------------
def test_ucs_known_answer_single_misread():                                    # normal 6
    g, cell = misread_instance()
    assert find_conflicts(g)                                                  # the misread breaks the puzzle
    p, blank = probs_from_grid(g, overrides={cell: {1: 0.52, 7: 0.45}})
    rep = solve_from_probs(p, blank, algo="ucs")
    assert rep.status == "solved" and rep.solution == SOLUTION
    (c, old, new, cost), = rep.corrections
    assert (c, old, new) == (cell, 1, 7) and cost == pytest.approx(math.log(0.52 / 0.45))
    assert rep.ac3_conflict                                                   # AC-3 flagged the first reading

def test_clean_reading_needs_zero_corrections():                               # normal 7
    p, blank = probs_from_grid(PUZZLE)
    rep = solve_from_probs(p, blank, goal="unique")
    assert rep.status == "solved" and rep.corrections == [] and not rep.ac3_conflict
    assert rep.search.expanded == 0

def test_ucs_equals_brute_force_cost_and_astar_is_optimal():                  # normal 8
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(6):
        sol_puz, sol = make_puzzle(rng, 32)
        g = list(sol_puz); cells = [i for i in range(81) if g[i]]
        wrong = rng.choice(cells, 2, replace=False)
        ov = {}
        for c in wrong:
            alt = int(rng.integers(1, 10))
            while alt == g[c]:
                alt = int(rng.integers(1, 10))
            ov[int(c)] = {alt: 0.5, g[c]: 0.3 + 0.1 * rng.random()}
            g[c] = alt
        p, blank = probs_from_grid(g, conf=0.97, overrides=ov)
        prob = make_problem(p, blank, K=8)
        best_cost, _ = brute_force_best(prob)
        for algo in ("ucs", "astar"):
            r = search(prob, algo)
            assert r.status == "found" and r.cost == pytest.approx(best_cost), (algo, r.cost, best_cost)
        checked += 1
    assert checked == 6

def test_astar_heuristic_never_overestimates_and_expands_no_more_than_ucs():   # normal 9
    g, cell = misread_instance()
    p, blank = probs_from_grid(g, overrides={cell: {1: 0.52, 7: 0.45}})
    prob = make_problem(p, blank)
    opt = search(prob, "ucs").cost
    assert heuristic_astar(prob, frozenset(), list(prob.base)) <= opt + 1e-9
    assert search(prob, "astar").expanded <= search(prob, "ucs").expanded

def test_unique_goal_rejects_solvable_but_ambiguous_grid():                    # normal 10
    g = list(PUZZLE)
    for c in (0, 1, 3, 4, 6):                    # blank extra cells so the first reading has many solutions
        g[c] = 0
    p, blank = probs_from_grid(g)
    assert solve_from_probs(p, blank, goal="solvable").corrections == []
    rep = solve_from_probs(p, blank, goal="unique")
    # the ambiguous first reading is NOT accepted as-is (see REPORT: uniqueness cannot tell a wrong-but-unique fix
    # from the truth, so the answer may be wrong - this test only checks the goal is really enforced)
    assert rep.corrections != [] or rep.status == "unreadable"

def test_generator_is_seeded_and_unique():                                     # normal 11
    a = make_puzzle(np.random.default_rng(11), 28)
    b = make_puzzle(np.random.default_rng(11), 28)
    assert a == b and sum(1 for v in a[0] if v) == 28
    assert goal_checker("unique")(list(a[0]))[0]

def test_correction_costs_are_never_negative():                                # normal 12
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(9) * 0.3, size=81)
    prob = make_problem(p, np.zeros(81, bool))
    assert all(c >= 0 for alts in prob.candidates.values() for _, c in alts)

DATA = Path(__file__).resolve().parents[1] / "data" / "mnist_1to9.npz"

@pytest.mark.skipif(not DATA.exists(), reason="run make_data.py first")
def test_blank_rule_separates_blank_from_every_real_digit():                   # normal 13
    d = np.load(DATA)
    x = np.concatenate([d["x_train"], d["x_val"], d["x_test"]])
    assert x.reshape(len(x), -1).mean(1).min() > BLANK_MEAN_THRESHOLD
    blank = np.abs(np.random.default_rng(0).normal(0, 0.004, (1000, 784)))
    assert blank.mean(1).max() < BLANK_MEAN_THRESHOLD

# ---------------- edge cases ----------------
def test_all_blank_page_is_trivially_solvable():                               # edge 5
    rep = solve_from_probs(np.full((81, 9), 1 / 9), np.ones(81, bool))
    assert rep.status == "solved" and rep.corrections == []

def test_exactly_max_corrections_is_allowed():                                 # edge 6
    g = list(PUZZLE); ov = {}
    givens = [i for i in range(81) if PUZZLE[i]][:3]
    for c in givens:
        alt = PUZZLE[c] % 9 + 1
        ov[c] = {alt: 0.5, PUZZLE[c]: 0.45}
        g[c] = alt
    p, blank = probs_from_grid(g, overrides=ov)
    rep = solve_from_probs(p, blank, max_corrections=3, goal="unique")
    assert rep.status == "solved" and len(rep.corrections) <= 3

def test_too_few_allowed_corrections_gives_unreadable():                       # edge 7 (also impossible-type)
    g = list(PUZZLE); ov = {}
    for c in [i for i in range(81) if PUZZLE[i]][:2]:
        alt = PUZZLE[c] % 9 + 1
        ov[c] = {alt: 0.5, PUZZLE[c]: 0.45}; g[c] = alt
    p, blank = probs_from_grid(g, overrides=ov)
    assert solve_from_probs(p, blank, max_corrections=1, goal="unique").status == "unreadable"

# ---------------- impossible cases ----------------
def test_impossible_confident_wrong_digit_reports_unreadable_with_cells():     # impossible 3
    g, cell = misread_instance()
    p, blank = probs_from_grid(g, conf=0.9999)            # network is (wrongly) certain: no plausible alternative
    prob_k = 1                                            # only one candidate cell allowed -> cannot reach the culprit
    rep = solve_from_probs(p, blank, K=prob_k)
    assert rep.status == "unreadable" and rep.message.startswith("Unreadable")
    assert rep.solution is None and "least confident" in rep.message.lower()

def test_impossible_budget_message():                                          # impossible 4
    g, cell = misread_instance()
    p, blank = probs_from_grid(g, overrides={cell: {1: 0.52, 7: 0.45}})
    rep = solve_from_probs(p, blank, algo="random", max_expansions=0, rng=np.random.default_rng(0))
    assert rep.status == "budget" and "Stopped" in rep.message   # root (uncorrected) fails, 0 expansions allowed
