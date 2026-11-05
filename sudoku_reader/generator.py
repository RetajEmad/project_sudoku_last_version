"""Seeded Sudoku + handwritten-mosaic generator.

Puzzle: full grid by randomised backtracking, then cells are removed (random order) while the solution
stays unique, until the requested number of givens is reached.
Mosaic: each given cell gets a real MNIST *test* image of that digit; blank cells get a near-black image.
Hard-digit injection rule (documented in docs/data_dictionary.md):
  * 'clean' pool  = test images the calibrated network classifies correctly
  * 'hard'  pool  = test images it misreads whose true digit is still within its top-3 classes
  * a puzzle with m misreads uses m hard images (distinct cells) and clean images everywhere else.
"""
import numpy as np
from .csp import SudokuSolver, PEERS

BLANK_MEAN_THRESHOLD = 0.02     # blank cell <=> mean pixel brightness below this (rule, not the NN)


def random_full_grid(rng):
    val = [0] * 81
    doms = [0x1FF] * 81

    def rec():
        best, bn = None, 99
        for i in range(81):
            if not val[i]:
                n = doms[i].bit_count()
                if n < bn:
                    best, bn = i, n
        if best is None:
            return True
        digs = [d for d in range(9) if doms[best] >> d & 1]
        rng.shuffle(digs)
        for d in digs:
            saved = [(p, doms[p]) for p in PEERS[best]]
            ok = True
            for p in PEERS[best]:
                if not val[p]:
                    doms[p] &= ~(1 << d)
                    if doms[p] == 0:
                        ok = False
            val[best] = d + 1
            if ok and rec():
                return True
            val[best] = 0
            for p, m in saved:
                doms[p] = m
        return False

    rec()
    return tuple(val)


_UNIQ = SudokuSolver(fc=True, ac3=True)


def make_puzzle(rng, n_givens, attempts=40):
    """Returns (puzzle, solution) with a unique solution and exactly n_givens givens."""
    for _ in range(attempts):
        sol = random_full_grid(rng)
        puz = list(sol)
        cells = list(rng.permutation(81))
        n = 81
        for c in cells:
            if n <= n_givens:
                break
            keep = puz[c]
            puz[c] = 0
            if len(_UNIQ.solve(puz, 2).solutions) == 1:
                n -= 1
            else:
                puz[c] = keep
        if n == n_givens:
            return tuple(puz), sol
    raise RuntimeError(f"could not reach {n_givens} givens with a unique solution")


class ImagePools:
    def __init__(self, x_test, y_test, probs_test):
        pred = probs_test.argmax(1) + 1
        top3 = np.argsort(-probs_test, axis=1)[:, :3] + 1
        self.x = x_test
        self.clean = {d: np.where((y_test == d) & (pred == d))[0] for d in range(1, 10)}
        self.hard = {d: np.where((y_test == d) & (pred != d) & (top3 == d).any(1))[0] for d in range(1, 10)}


def make_mosaic(puzzle, pools, rng, n_misreads=0):
    """Returns (images (81,28,28), misread_cells list). Blank cells = zeros + tiny noise."""
    imgs = np.zeros((81, 28, 28), np.float32)
    givens = [i for i in range(81) if puzzle[i]]
    hard_cells = []
    if n_misreads:
        ok = [c for c in givens if len(pools.hard[puzzle[c]])]
        hard_cells = list(rng.choice(ok, size=n_misreads, replace=False))
    for i in range(81):
        if not puzzle[i]:
            imgs[i] = np.abs(rng.normal(0, 0.004, (28, 28))).astype(np.float32)
        elif i in hard_cells:
            imgs[i] = pools.x[rng.choice(pools.hard[puzzle[i]])]
        else:
            imgs[i] = pools.x[rng.choice(pools.clean[puzzle[i]])]
    return imgs, sorted(int(c) for c in hard_cells)


def mosaic_picture(imgs):
    """(81,28,28) -> one 252x252 picture."""
    return imgs.reshape(9, 9, 28, 28).transpose(0, 2, 1, 3).reshape(252, 252)
