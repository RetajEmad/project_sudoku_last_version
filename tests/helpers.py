import numpy as np

PUZZLE = [int(c) for c in "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOLUTION = tuple(int(c) for c in "534678912672195348198342567859761423426853791713924856961537284287419635345286179")


def probs_from_grid(grid, conf=0.95, overrides=None, rng=None):
    """(81,9) probabilities: read digit gets `conf`, the rest is spread evenly.  overrides: {cell: {digit: p}}."""
    p = np.full((81, 9), 1 / 9)
    blank = np.array([g == 0 for g in grid])
    for i, g in enumerate(grid):
        if g:
            p[i] = (1 - conf) / 8
            p[i, g - 1] = conf
    for c, dist in (overrides or {}).items():
        rest = 1 - sum(dist.values())
        p[c] = rest / (9 - len(dist))
        for d, v in dist.items():
            p[c, d - 1] = v
    return p, blank
