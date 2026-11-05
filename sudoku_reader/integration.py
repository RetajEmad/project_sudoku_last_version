"""NN -> AC-3 -> Search -> CSP pipeline."""
from dataclasses import dataclass, field
import numpy as np
from .csp import SudokuSolver, CSPStats, find_conflicts, ac3, ALL
from .generator import BLANK_MEAN_THRESHOLD
from .search import make_problem, search, goal_checker, SearchResult


def read_cells(images, model):
    """images (81,28,28) -> (probs (81,9), blank bool (81,)).  Blank detection = brightness rule, not the NN."""
    blank = images.reshape(81, -1).mean(1) < BLANK_MEAN_THRESHOLD
    probs = np.full((81, 9), 1 / 9)
    if (~blank).any():
        probs[~blank] = model.predict_proba(images[~blank])
    return probs, blank


@dataclass
class Report:
    status: str                       # 'solved' | 'unreadable' | 'no_solution' | 'budget'
    message: str
    first_reading: tuple
    ac3_conflict: bool
    first_reading_conflicts: list
    corrections: list = field(default_factory=list)   # (cell, read, new, cost)
    solution: tuple | None = None
    search: SearchResult | None = None
    csp_stats: CSPStats = field(default_factory=CSPStats)


def solve_from_probs(probs, blank, algo="ucs", goal="solvable", K=10, max_corrections=3,
                     max_expansions=5000, rng=None):
    """Pipeline on class probabilities (so it can be tested without images)."""
    prob = make_problem(probs, blank, K=K, max_corrections=max_corrections)
    grid = list(prob.base)
    # 1. AC-3 on the first reading: does it already contradict itself?
    doms = [(1 << (v - 1)) if v else ALL for v in grid]
    conflict = not ac3(doms, CSPStats())
    pairs = find_conflicts(grid)
    # 2. Search over corrections; the goal test is the CSP (BT + FC + AC-3).  Zero corrections is the root state,
    #    so a first reading that already satisfies the goal wins immediately.
    res = search(prob, algo, goal, max_expansions, rng=rng, check=goal_checker(goal))
    if res.status == "found":
        msg = "Solved with %d correction(s)." % len(res.corrections)
        return Report("solved", msg, prob.base, conflict, pairs, res.corrections, res.solution, res, res.csp)
    status = "unreadable" if res.status == "unreadable" else "budget"
    return Report(status, res.message, prob.base, conflict, pairs, [], None, res, res.csp)


def solve_handwritten(images, model, **kw):
    probs, blank = read_cells(images, model)
    return solve_from_probs(probs, blank, **kw)


def solve_no_correction(images, model):
    """Baseline: trust the first reading blindly."""
    probs, blank = read_cells(images, model)
    grid = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    r = SudokuSolver().solve(grid, 1)
    return r.solutions[0] if r.solved else None


def format_grid(g, mark=()):
    rows = []
    for r in range(9):
        cells = []
        for c in range(9):
            d = g[r * 9 + c]
            s = str(d) if d else "."
            cells.append(f"[{s}]" if r * 9 + c in mark else f" {s} ")
        rows.append("".join(cells[:3]) + "|" + "".join(cells[3:6]) + "|" + "".join(cells[6:]))
        if r in (2, 5):
            rows.append("-" * 11 + "+" + "-" * 11 + "+" + "-" * 11)
    return "\n".join(rows)
