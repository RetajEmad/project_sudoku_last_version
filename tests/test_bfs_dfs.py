"""Tests for the two uninformed search algorithms added to the correction search (BFS, DFS)."""
import itertools
import numpy as np
import pytest
from sudoku_reader.search import make_problem, search, brute_force_best, goal_checker
from sudoku_reader.integration import solve_from_probs
from sudoku_reader.generator import make_puzzle
from helpers import PUZZLE, SOLUTION, probs_from_grid

ALGOS = ["ucs", "greedy", "astar", "bfs", "dfs"]


def misread_instance():
    """Cell 4 holds a true 7 read as 1 (duplicates the 1s at cells 12 and 67)."""
    g = list(PUZZLE)
    g[4] = 1
    return probs_from_grid(g, overrides={4: {1: 0.52, 7: 0.45}})


def two_misread_instance(seed=3):
    rng = np.random.default_rng(seed)
    puz, sol = make_puzzle(rng, 32)
    g = list(puz)
    ov = {}
    for c in rng.choice([i for i in range(81) if g[i]], 2, replace=False):
        alt = int(rng.integers(1, 10))
        while alt == g[c]:
            alt = int(rng.integers(1, 10))
        ov[int(c)] = {alt: 0.5, g[c]: 0.3 + 0.1 * rng.random()}
        g[c] = alt
    return probs_from_grid(g, conf=0.97, overrides=ov)


# ---------------- normal cases ----------------
@pytest.mark.parametrize("algo", ["bfs", "dfs"])
def test_bfs_dfs_recover_known_single_misread(algo):                                  # normal 1
    p, blank = misread_instance()
    rep = solve_from_probs(p, blank, algo=algo, goal="unique")
    assert rep.status == "solved" and rep.solution == SOLUTION
    assert [(c, old, new) for c, old, new, _ in rep.corrections] == [(4, 1, 7)]


@pytest.mark.parametrize("algo", ALGOS)
def test_every_algorithm_returns_a_grid_the_csp_accepts(algo):                        # normal 2
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8)
    r = search(prob, algo, "solvable")
    assert r.status == "found" and len(r.corrections) <= prob.max_corrections
    assert goal_checker("solvable")(list(r.solution))[0]
    # the reported cost is the sum of the individual correction costs
    assert r.cost == pytest.approx(sum(c for *_, c in r.corrections))


def test_bfs_finds_the_fewest_corrections():                                          # normal 3
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8)
    check = goal_checker("solvable")
    cells = list(prob.candidates)
    fewest = None
    for k in range(prob.max_corrections + 1):          # brute force by number of corrections
        for cs in itertools.combinations(cells, k):
            for ds in itertools.product(*[prob.candidates[c] for c in cs]):
                if check(prob.apply(frozenset((c, d) for c, (d, _) in zip(cs, ds))))[0]:
                    fewest = k
                    break
            if fewest is not None:
                break
        if fewest is not None:
            break
    assert len(search(prob, "bfs", "solvable").corrections) == fewest


def test_bfs_is_not_cost_optimal_but_ucs_is():                                        # normal 4
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8)
    best, _ = brute_force_best(prob)
    assert search(prob, "ucs").cost == pytest.approx(best)
    for algo in ("bfs", "dfs"):                         # uninformed: never cheaper than the optimum
        assert search(prob, algo).cost >= best - 1e-9


def test_dfs_expands_deeper_states_first():                                           # normal 5
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8)
    depths = [t[2] for t in search(prob, "dfs", "solvable").trace]
    assert depths[0] == 0 and 1 in depths               # root, then it dives
    assert depths.index(1) == 1                          # first pop after the root is a 1-correction state
    b = [t[2] for t in search(prob, "bfs", "solvable").trace]
    assert b == sorted(b)                                # BFS pops states level by level


def test_bfs_and_dfs_clean_reading_needs_no_expansion():                              # normal 6
    p, blank = probs_from_grid(PUZZLE)
    for algo in ("bfs", "dfs"):
        rep = solve_from_probs(p, blank, algo=algo, goal="unique")
        assert rep.status == "solved" and rep.corrections == [] and rep.search.expanded == 0


# ---------------- edge cases ----------------
@pytest.mark.parametrize("algo", ["bfs", "dfs"])
def test_depth_limit_is_respected(algo):                                              # edge 1
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8, max_corrections=2)
    r = search(prob, algo, "solvable")
    assert all(t[2] <= 2 for t in r.trace)


@pytest.mark.parametrize("algo", ["bfs", "dfs"])
def test_each_state_generated_at_most_once(algo):                                     # edge 2
    p, blank = two_misread_instance()
    prob = make_problem(p, blank, K=8)
    r = search(prob, algo, "unique", max_expansions=5000)
    n_states = sum(len(list(itertools.combinations(range(len(prob.candidates)), k))) * 2 ** k
                   for k in range(prob.max_corrections + 1))      # upper bound: every candidate has 2 alternatives
    assert r.generated <= n_states and r.goal_tests <= n_states
    assert r.goal_tests in (r.expanded, r.expanded + 1)


def test_ucs_and_astar_results_unchanged_by_new_algorithms():                         # edge 3
    p, blank = misread_instance()
    prob = make_problem(p, blank)
    assert search(prob, "ucs").cost == pytest.approx(search(prob, "astar").cost)


def test_unknown_algorithm_is_rejected():                                             # edge 4
    p, blank = misread_instance()
    with pytest.raises(ValueError):
        search(make_problem(p, blank), "nonsense")


# ---------------- impossible cases ----------------
@pytest.mark.parametrize("algo", ["bfs", "dfs"])
def test_unreachable_misread_reports_unreadable(algo):                                # impossible 1
    g = list(PUZZLE)
    g[4] = 1
    p, blank = probs_from_grid(g, conf=0.9999)           # confidently wrong, culprit not among the candidates
    rep = solve_from_probs(p, blank, algo=algo, K=1)
    assert rep.status == "unreadable" and rep.message.startswith("Unreadable")
    assert rep.solution is None and "least confident" in rep.message.lower()


@pytest.mark.parametrize("algo", ["bfs", "dfs"])
def test_budget_message(algo):                                                        # impossible 2
    p, blank = misread_instance()
    rep = solve_from_probs(p, blank, algo=algo, max_expansions=0)
    assert rep.status == "budget" and "Stopped" in rep.message
