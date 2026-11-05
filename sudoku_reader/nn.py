"""Small MLP digit classifier in plain NumPy + temperature scaling + calibration metrics.

Labels used inside this module are 0..8 (digit-1).  Use `digits_from_probs` to go back to 1..9.
"""
from pathlib import Path
import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes=(784, 256, 128, 9), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.T = 1.0                      # temperature (set by calibration)

    # -- forward / backward ---------------------------------------------------
    def _forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(z if i == len(self.W) - 1 else np.maximum(z, 0))
        return acts

    def logits(self, x):
        x = np.asarray(x, np.float32).reshape(len(x), -1)
        return self._forward(x)[-1]

    def predict_proba(self, x, calibrated=True):
        return softmax(self.logits(x) / (self.T if calibrated else 1.0))

    def fit(self, x, y, xv=None, yv=None, epochs=12, lr=1e-3, batch=128, seed=0, l2=1e-5, log=print):
        rng = np.random.default_rng(seed)
        x = x.reshape(len(x), -1).astype(np.float32)
        m = [np.zeros_like(p) for p in self.W + self.b]
        v = [np.zeros_like(p) for p in self.W + self.b]
        t = 0
        for ep in range(epochs):
            idx = rng.permutation(len(x))
            for s in range(0, len(x), batch):
                bi = idx[s:s + batch]
                acts = self._forward(x[bi])
                p = softmax(acts[-1])
                g = p
                g[np.arange(len(bi)), y[bi]] -= 1
                g /= len(bi)
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    gW[i] = acts[i].T @ g + l2 * self.W[i]
                    gb[i] = g.sum(0)
                    if i:
                        g = (g @ self.W[i].T) * (acts[i] > 0)
                t += 1
                params, grads = self.W + self.b, gW + gb
                for k, (P, G) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * G
                    v[k] = 0.999 * v[k] + 0.001 * G * G
                    P -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            if xv is not None:
                log(f"epoch {ep + 1:2d}  val acc {(self.logits(xv).argmax(1) == yv).mean():.4f}")

    def save(self, path):
        np.savez(path, T=self.T, n=len(self.W), **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        o = cls.__new__(cls)
        n = int(d["n"])
        o.W, o.b, o.T = [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)], float(d["T"])
        return o


def fit_temperature(logits, y):
    """Temperature T>0 minimising validation NLL (grid + local refinement; no extra dependency)."""
    grid = np.linspace(0.3, 5.0, 236)
    nll = [nll_from_logits(logits / T, y) for T in grid]
    k = int(np.argmin(nll))
    fine = np.linspace(grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)], 101)
    return float(fine[int(np.argmin([nll_from_logits(logits / T, y) for T in fine]))])


def nll_from_logits(z, y):
    p = softmax(z)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def ece(probs, y, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            err += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def confusion(pred, y, k=9):
    c = np.zeros((k, k), int)
    np.add.at(c, (y, pred), 1)
    return c
