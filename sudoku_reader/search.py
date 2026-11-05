"""Correction search.

State  : a set of corrections {(cell, new_digit)} applied to the first reading (<= max_corrections).
Action : pick a cell that has not been corrected yet and change it to one of its top alternatives.
Cost   : max(0, log p(read digit) - log p(new digit))   (never negative, additive over corrections)
Goal   : the corrected grid passes the CSP test ('solvable' = >=1 solution, 'unique' = exactly 1).

Candidate cells are the K least-confident cells of the first reading (K=10 default); alternatives are
the other digits among the network's top-3.  This keeps the space finite: at most sum_k C(K,k)*2^k states.

Algorithms (same space, same goal test, different priority):
  ucs     priority g                      -> cheapest correction set (optimal)
  greedy  priority (#duplicate pairs, g)  -> "most repaired first", not optimal
  astar   priority g + h                  -> h = max over duplicate pairs of the cheapest way to change one of
                                             its two cells (admissible: each pair needs >= 1 change)
  random  random priority                 -> baseline that ignores the network's probabilities
  bfs     FIFO by number of corrections   -> uninformed; fewest corrections first, ignores the costs
  dfs     LIFO (depth limit = max_corrections) -> uninformed; dives to the deepest state first, takes the first
                                             goal it meets (neither optimal nor cheapest)
"""
import heapq, itertools, math
from dataclasses import dataclass, field
import numpy as np
from .csp import SudokuSolver, CSPStats, find_conflicts

INF = float("inf")


@dataclass
class CorrectionProblem:
    base: tuple              # first reading, 0 = blank
    candidates: dict         # cell -> [(digit, cost), ...] sorted by cost
    max_corrections: int = 3
    conf: dict = field(default_factory=dict)   # cell -> confidence of the first reading (all given cells)

    def apply(self, state):
        g = list(self.base)
        for c, d in state:
            g[c] = d
        return g


def make_problem(probs, blank, K=10, topk=3, max_corrections=3):
    """probs: (81, 9) class probabilities (rows of blank cells are ignored); blank: bool (81,)."""
    probs = np.asarray(probs, float)
    base = [0 if blank[i] else int(probs[i].argmax()) + 1 for i in range(81)]
    conf = {i: float(probs[i].max()) for i in range(81) if not blank[i]}
    weakest = sorted(conf, key=lambda i: (conf[i], i))[:K]
    cands = {}
    for i in weakest:
        lp = np.log(np.clip(probs[i], 1e-12, None))
        order = [d for d in np.argsort(-probs[i])[:topk] if d + 1 != base[i]]
        cands[i] = sorted(((int(d) + 1, max(0.0, float(lp[base[i] - 1] - lp[d]))) for d in order),
                          key=lambda t: t[1])
    return CorrectionProblem(tuple(base), cands, max_corrections, conf)


@dataclass
class SearchResult:
    status: str                      # 'found' | 'unreadable' | 'budget'
    corrections: list                # [(cell, read, new, cost)]
    cost: float
    solution: tuple | None
    expanded: int
    generated: int
    goal_tests: int
    csp: CSPStats
    message: str = ""
    trace: list = field(default_factory=list)   # (expanded_index, cost_g, n_corrections, goal?) per pop


def goal_checker(goal="solvable", **solver_kw):
    solver = SudokuSolver(**{"fc": True, "ac3": True, **solver_kw})

    def check(grid):
        r = solver.solve(grid, 2 if goal == "unique" else 1)
        ok = len(r.solutions) == 1 if goal == "unique" else r.solved
        return ok, (r.solutions[0] if ok else None), r.stats
    return check


def _min_cost(problem, cell, used):
    return INF if (cell in used or cell not in problem.candidates) else problem.candidates[cell][0][1]


def heuristic_astar(problem, state, grid):
    used = {c for c, _ in state}
    pairs = find_conflicts(grid)
    if not pairs:
        return 0.0
    if len(state) >= problem.max_corrections:
        return INF
    return max(min(_min_cost(problem, i, used), _min_cost(problem, j, used)) for i, j in pairs)


def search(problem, algo="ucs", goal="solvable", max_expansions=5000, rng=None, check=None):
    check = check or goal_checker(goal)
    rng = rng or np.random.default_rng(0)
    tick = itertools.count()
    csp_total = CSPStats()
    root = frozenset()
    best_g = {root: 0.0}
    cost_of = {c: dict(v) for c, v in problem.candidates.items()}

    def prio(g, state, grid):
        if algo == "ucs":
            return (g,)
        if algo == "astar":
            h = heuristic_astar(problem, state, grid)
            return (g + h,) if h < INF else None
        if algo == "greedy":
            return (len(find_conflicts(grid)), g)
        if algo == "random":
            return (float(rng.random()),)
        if algo in ("bfs", "dfs"):
            return (0,)         # order comes only from the tie-break counter below (FIFO / LIFO)
        raise ValueError(algo)

    sign = -1 if algo == "dfs" else 1      # heap pops the smallest key: +tick = FIFO (BFS), -tick = LIFO (DFS)
    pq = [(prio(0.0, root, list(problem.base)), sign * next(tick), 0.0, root)]
    expanded = generated = tests = 0
    trace = []
    while pq:
        _, _, g, state = heapq.heappop(pq)
        if g > best_g.get(state, INF) + 1e-12:
            continue
        grid = problem.apply(state)
        ok, sol, st = check(grid)
        tests += 1
        csp_total.add(st)
        trace.append((expanded, g, len(state), ok))
        if ok:
            corr = [(c, problem.base[c], d, cost_of[c][d]) for c, d in sorted(state)]
            return SearchResult("found", corr, g, sol, expanded, generated, tests, csp_total, "", trace)
        if expanded >= max_expansions:
            return SearchResult("budget", [], INF, None, expanded, generated, tests, csp_total,
                                f"Stopped after {expanded} expansions.", trace)
        expanded += 1
        if len(state) >= problem.max_corrections:
            continue
        used = {c for c, _ in state}
        moves = [(cell, d, cst) for cell, alts in problem.candidates.items() if cell not in used
                 for d, cst in alts]
        if algo == "dfs":
            moves.reverse()        # LIFO: push in reverse so the least-confident cell / cheapest digit is popped first
        for cell, d, cst in moves:
            ns = state | {(cell, d)}
            ng = g + cst
            if ng < best_g.get(ns, INF) - 1e-12:
                best_g[ns] = ng
                p = prio(ng, ns, problem.apply(ns))
                if p is None:
                    continue
                generated += 1
                heapq.heappush(pq, (p, sign * next(tick), ng, ns))
    weak = sorted(problem.conf.items(), key=lambda t: t[1])[:5]
    msg = ("Unreadable: no solvable grid within %d corrections of the %d least-confident cells. Least confident: "
           % (problem.max_corrections, len(problem.candidates)) +
           ", ".join(f"(r{c // 9 + 1},c{c % 9 + 1}) read {problem.base[c]} @ {p:.2f}" for c, p in weak))
    return SearchResult("unreadable", [], INF, None, expanded, generated, tests, csp_total, msg, trace)


def brute_force_best(problem, goal="solvable", check=None):
    """Exhaustive reference: cheapest correction set (<= max_corrections) passing the goal test."""
    check = check or goal_checker(goal)
    best = (INF, None)
    cells = list(problem.candidates)
    for k in range(problem.max_corrections + 1):
        for cs in itertools.combinations(cells, k):
            for ds in itertools.product(*[problem.candidates[c] for c in cs]):
                cost = sum(cst for _, cst in ds)
                if cost >= best[0]:
                    continue
                state = frozenset((c, d) for c, (d, _) in zip(cs, ds))
                if check(problem.apply(state))[0]:
                    best = (cost, state)
    return best
