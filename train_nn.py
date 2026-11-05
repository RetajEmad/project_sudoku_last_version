"""Train the digit classifier (NN) and the logistic-regression baseline; calibrate; write metrics + plots."""
import json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sudoku_reader.nn import MLP, fit_temperature, ece, nll_from_logits, softmax, confusion

SEED = 0


def main():
    d = np.load("data/mnist_1to9.npz")
    xtr, ytr, xva, yva, xte, yte = (d["x_train"], d["y_train"] - 1, d["x_val"], d["y_val"] - 1,
                                    d["x_test"], d["y_test"] - 1)
    t0 = time.time()
    net = MLP(seed=SEED)
    net.fit(xtr, ytr, xva.reshape(len(xva), -1), yva, epochs=12, seed=SEED)
    print(f"NN trained in {time.time() - t0:.0f}s")
    zv, zt = net.logits(xva), net.logits(xte)
    net.T = fit_temperature(zv, yva)
    net.save("models/mlp.npz")

    lr = LogisticRegression(max_iter=500)
    t0 = time.time()
    lr.fit(xtr.reshape(len(xtr), -1), ytr)
    print(f"LogReg trained in {time.time() - t0:.0f}s")
    lt = lr.decision_function(xte.reshape(len(xte), -1))
    lv = lr.decision_function(xva.reshape(len(xva), -1))
    T_lr = fit_temperature(lv, yva)

    res = {
        "temperature_nn": net.T, "temperature_lr": T_lr,
        "nn": {"test_acc": float((zt.argmax(1) == yte).mean()),
               "ece_raw": ece(softmax(zt), yte), "ece_calibrated": ece(softmax(zt / net.T), yte),
               "nll_raw": nll_from_logits(zt, yte), "nll_calibrated": nll_from_logits(zt / net.T, yte)},
        "logreg": {"test_acc": float((lt.argmax(1) == yte).mean()),
                   "ece_raw": ece(softmax(lt), yte), "ece_calibrated": ece(softmax(lt / T_lr), yte),
                   "nll_raw": nll_from_logits(lt, yte), "nll_calibrated": nll_from_logits(lt / T_lr, yte)},
        "n_test": int(len(yte)),
    }
    json.dump(res, open("results/classifier_metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    cm = confusion(zt.argmax(1), yte)
    np.savetxt("results/confusion_matrix_nn.csv", cm, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].imshow(np.log1p(cm), cmap="Blues")
    for i in range(9):
        for j in range(9):
            ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax[0].set_xticks(range(9)); ax[0].set_xticklabels(range(1, 10))
    ax[0].set_yticks(range(9)); ax[0].set_yticklabels(range(1, 10))
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title("NN confusion matrix (test)")
    for name, z, T in (("NN raw", zt, 1.0), ("NN calibrated", zt, net.T), ("LogReg raw", lt, 1.0)):
        p = softmax(z / T)
        conf, ok = p.max(1), p.argmax(1) == yte
        edges = np.linspace(0.1, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(ok[m].mean())
        ax[1].plot(xs, ys, "o-", label=name)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability diagram (test)")
    ax[1].legend()
    plt.tight_layout(); plt.savefig("results/classifier_calibration.png", dpi=140)


if __name__ == "__main__":
    main()
