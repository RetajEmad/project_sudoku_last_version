import pytest
from sudoku_reader.csp import SudokuSolver, SOLVER_VARIANTS, ac3, CSPStats, ALL, find_conflicts
from helpers import PUZZLE, SOLUTION

# ---------------- normal cases ----------------
@pytest.mark.parametrize("name", list(SOLVER_VARIANTS))
@pytest.mark.parametrize("ordering", ["mrv", "static"])
def test_all_variants_solve_known_puzzle(name, ordering):                       # normal 1 (6 sub-cases)
    r = SudokuSolver(ordering=ordering, **SOLVER_VARIANTS[name]).solve(PUZZLE, 2)
    assert r.status == "solved" and len(r.solutions) == 1 and r.solutions[0] == SOLUTION

def test_forward_checking_needs_fewer_assignments_than_plain_bt():              # normal 2
    bt = SudokuSolver(fc=False, ac3=False, ordering="static").solve(PUZZLE).stats
    fc = SudokuSolver(fc=True, ac3=False, ordering="static").solve(PUZZLE).stats
    assert fc.assignments < bt.assignments and fc.fc_prunes > 0 and bt.fc_prunes == 0

def test_ac3_prunes_domains_and_counts_work():                                  # normal 3
    doms = [(1 << (v - 1)) if v else ALL for v in PUZZLE]
    st = CSPStats()
    assert ac3(doms, st)
    assert st.ac3_removed > 0 and st.ac3_revisions >= st.ac3_removed
    assert all(doms[i] != 0 for i in range(81))
    assert all(doms[i] & (1 << (SOLUTION[i] - 1)) for i in range(81))           # never prunes the true digit

def test_switches_are_independent():                                            # normal 4
    s = SudokuSolver(fc=False, ac3=True).solve(PUZZLE).stats
    assert s.ac3_revisions > 0 and s.fc_prunes == 0 and s.consistency_checks >= 0
    s = SudokuSolver(fc=False, ac3=False).solve(PUZZLE).stats
    assert s.ac3_revisions == 0

def test_find_conflicts_reports_duplicates_once():                              # normal 5
    g = [0] * 81
    g[0] = g[8] = 4          # same row
    g[3 * 9] = 4             # same column as cell 0
    assert find_conflicts(g) == [(0, 8), (0, 27)]

# ---------------- edge cases ----------------
def test_empty_grid_is_solvable_and_has_many_solutions():                       # edge 1
    r = SudokuSolver().solve([0] * 81, 2)
    assert len(r.solutions) == 2

def test_complete_valid_grid_returns_itself():                                  # edge 2
    r = SudokuSolver().solve(list(SOLUTION), 2)
    assert r.solutions == [SOLUTION]

def test_single_blank_cell():                                                   # edge 3
    g = list(SOLUTION); g[40] = 0
    assert SudokuSolver().solve(g, 2).solutions == [SOLUTION]

def test_budget_stops_search_cleanly():                                         # edge 4
    r = SudokuSolver(fc=False, ac3=False, ordering="static", max_assignments=50).solve(PUZZLE)
    assert r.status == "budget" and not r.solved

# ---------------- impossible cases ----------------
@pytest.mark.parametrize("name", list(SOLVER_VARIANTS))
def test_duplicate_givens_give_clear_no_solution(name):                         # impossible 1
    g = list(PUZZLE); g[2] = 5                      # second 5 in row 0
    r = SudokuSolver(**SOLVER_VARIANTS[name]).solve(g)
    assert r.status == "no_solution" and r.message.startswith("No solution")
    if name == "BT+FC+AC3":
        assert "AC-3" in r.message and r.stats.assignments == 0   # caught before any search

def test_conflict_free_but_unsolvable_grid():                                   # impossible 2
    g = [0] * 81
    g[1:9] = [1, 2, 3, 4, 5, 6, 7, 8]               # row 0 uses 1..8 -> cell 0 must be 9 ...
    g[27] = 9                                       # ... but column 0 already has a 9
    assert find_conflicts(g) == []                  # no duplicate pair exists
    for name in SOLVER_VARIANTS:
        r = SudokuSolver(**SOLVER_VARIANTS[name]).solve(g)
        assert r.status == "no_solution", name
