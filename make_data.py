"""Reproducible data script for Project 7 (MNIST digits 1-9).

Downloads MNIST (50,000 train / 10,000 validation / 10,000 test), drops digit 0
(Sudoku has no zero), scales pixels to 0-1 and writes data/mnist_1to9.npz.

SOURCE NOTE: the original page http://yann.lecun.com/exdb/mnist/ was not
reachable from our build sandbox, so we use a public GitHub copy of the same
data (Michael Nielsen's `mnist.pkl.gz`).  The script verifies the well-known
MNIST test-set label counts, so a wrong/corrupted file is rejected.
If you have the four original LeCun files, convert them to the same pickle
format or just put your own `mnist.pkl.gz` into data/raw/ - the download is skipped.
"""
import gzip, hashlib, pickle, sys, urllib.request
from pathlib import Path
import numpy as np

SEED = 0  # only used for the (deterministic) sanity subsample; the split itself is fixed by the file
URL = ("https://raw.githubusercontent.com/mnielsen/neural-networks-and-deep-learning/"
       "master/data/mnist.pkl.gz")
RAW = Path("data/raw/mnist.pkl.gz")
OUT = Path("data/mnist_1to9.npz")
TEST_COUNTS = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]  # official MNIST test set


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print("downloading", URL)
        urllib.request.urlretrieve(URL, RAW)
    sha = hashlib.sha256(RAW.read_bytes()).hexdigest()
    print("sha256", sha)
    with gzip.open(RAW, "rb") as f:
        train, val, test = pickle.load(f, encoding="latin1")
    (xtr, ytr), (xva, yva), (xte, yte) = train, val, test
    assert xtr.shape == (50000, 784) and xva.shape == (10000, 784) and xte.shape == (10000, 784)
    assert np.bincount(yte, minlength=10).tolist() == TEST_COUNTS, "test labels differ from official MNIST"
    assert 0.0 <= xtr.min() and xtr.max() <= 1.0

    def keep(x, y):
        m = y >= 1                      # drop digit 0
        return x[m].astype(np.float32).reshape(-1, 28, 28), y[m].astype(np.int64)

    out = {}
    for name, (x, y) in dict(train=(xtr, ytr), val=(xva, yva), test=(xte, yte)).items():
        out["x_" + name], out["y_" + name] = keep(x, y)   # labels stay 1..9
        print(name, out["x_" + name].shape)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
