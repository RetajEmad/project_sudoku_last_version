"""Sudoku as a CSP: 81 variables (cells), domain {1..9}, all-different on rows/cols/boxes.

Three independent switches
  * Backtracking (always on)       - assigns one variable at a time, checks consistency
  * Forward Checking  (fc=True)    - after an assignment removes that digit from peers' domains
  * Arc consistency AC-3 (ac3=True)- run once before search; shrinks domains and detects conflicts

Domains are 9-bit masks (bit d-1 set <=> digit d still possible).
Givens are ordinary variables with a singleton domain, so two equal givens in a unit are
detected by AC-3 (an arc empties a domain) or by FC/BT when the 2nd one is assigned.
"""
from collections import deque
from dataclasses import dataclass

ALL = 0x1FF


def _build():
    units = []
    for r in range(9):
        units.append([r * 9 + c for c in range(9)])
    for c in range(9):
        units.append([r * 9 + c for r in range(9)])
    for br in range(3):
        for bc in range(3):
            units.append([(3 * br + r) * 9 + 3 * bc + c for r in range(3) for c in range(3)])
    peers = [sorted({x for u in units if i in u for x in u} - {i}) for i in range(81)]
    return units, peers


UNITS, PEERS = _build()


@dataclass
class CSPStats:
    assignments: int = 0          # value assigned to a variable
    backtracks: int = 0           # an assignment that was undone without success
    consistency_checks: int = 0   # peer comparisons made by plain backtracking
    fc_prunes: int = 0            # domain values removed by forward checking
    ac3_revisions: int = 0        # arcs processed by AC-3
    ac3_removed: int = 0          # domain values removed by AC-3

    def add(self, o):
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))
        return self


@dataclass
class SolveResult:
    solutions: list
    stats: CSPStats
    status: str          # 'solved' | 'no_solution' | 'budget'
    message: str = ""

    @property
    def solved(self):
        return len(self.solutions) > 0


def ac3(doms, stats):
    """Classic AC-3 over all arcs (xi, xj), xj a peer of xi.  Returns False on a domain wipe-out."""
    q = deque((i, j) for i in range(81) for j in PEERS[i])
    while q:
        xi, xj = q.popleft()
        stats.ac3_revisions += 1
        dj = doms[xj]
        if dj & (dj - 1) == 0:                 # xj fixed to one digit -> xi may not use it
            if doms[xi] & dj:
                doms[xi] &= ~dj
                stats.ac3_removed += 1
                if doms[xi] == 0:
                    return False
                for xk in PEERS[xi]:
                    if xk != xj:
                        q.append((xk, xi))
    return True


def find_conflicts(grid):
    """Pairs of cells holding the same digit inside one unit: [(i, j), ...] (i < j, no repeats)."""
    out = set()
    for u in UNITS:
        seen = {}
        for c in u:
            d = grid[c]
            if d:
                for o in seen.get(d, []):
                    out.add((min(o, c), max(o, c)))
                seen.setdefault(d, []).append(c)
    return sorted(out)


class SudokuSolver:
    def __init__(self, fc=True, ac3=True, ordering="mrv", max_assignments=None):
        assert ordering in ("mrv", "static")
        self.fc, self.use_ac3, self.ordering, self.max_assignments = fc, ac3, ordering, max_assignments

    # ------------------------------------------------------------------ public
    def solve(self, grid, max_solutions=1):
        stats = CSPStats()
        grid = list(grid)
        doms = [(1 << (v - 1)) if v else ALL for v in grid]
        if self.use_ac3 and not ac3(doms, stats):
            return SolveResult([], stats, "no_solution",
                               "No solution: AC-3 emptied a domain (contradictory givens).")
        self._order = [i for i in range(81) if grid[i]] + [i for i in range(81) if not grid[i]]
        self._sols, self._stats, self._k, self._budget = [], stats, max_solutions, False
        self._bt([0] * 81, doms)
        if self._sols:
            return SolveResult(self._sols, stats, "solved")
        if self._budget:
            return SolveResult([], stats, "budget", "Stopped: assignment budget exhausted.")
        return SolveResult([], stats, "no_solution", "No solution: search space exhausted.")

    # ----------------------------------------------------------------- internal
    def _select(self, val, doms):
        if self.ordering == "static":
            for i in self._order:
                if not val[i]:
                    return i
            return None
        best, best_n = None, 99
        for i in range(81):
            if val[i]:
                continue
            d = doms[i]
            if not self.fc:  # plain BT: legal digits = domain minus digits of assigned peers
                for p in PEERS[i]:
                    if val[p]:
                        d &= ~(1 << (val[p] - 1))
            n = d.bit_count()
            if n < best_n:
                best, best_n = i, n
                if n <= 1:
                    break
        return best

    def _bt(self, val, doms):
        """Returns True when the search must stop (enough solutions or budget hit)."""
        st = self._stats
        var = self._select(val, doms)
        if var is None:
            self._sols.append(tuple(val))
            return len(self._sols) >= self._k
        cand = doms[var]
        for d in range(9):
            if not (cand >> d) & 1:
                continue
            if not self.fc:                      # plain BT: explicit consistency check
                ok = True
                for p in PEERS[var]:
                    st.consistency_checks += 1
                    if val[p] == d + 1:
                        ok = False
                        break
                if not ok:
                    continue
            st.assignments += 1
            if self.max_assignments and st.assignments > self.max_assignments:
                self._budget = True
                return True
            new, ok = doms, True
            if self.fc:
                new = doms[:]
                bit = 1 << d
                new[var] = bit
                for p in PEERS[var]:
                    if new[p] & bit:
                        new[p] &= ~bit
                        st.fc_prunes += 1
                        if new[p] == 0:
                            ok = False
                            break
            val[var] = d + 1
            stop = self._bt(val, new) if ok else False
            val[var] = 0
            if stop:
                return True
            st.backtracks += 1
        return False


SOLVER_VARIANTS = {
    "BT": dict(fc=False, ac3=False),
    "BT+FC": dict(fc=True, ac3=False),
    "BT+FC+AC3": dict(fc=True, ac3=True),
}
