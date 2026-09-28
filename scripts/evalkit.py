"""Splits, probes and metrics shared by the cell- and donor-level tasks.

Design rules enforced here:
  - Donors, never cells, are assigned to folds. A donor's cells are all train
    or all test.
  - Every embedding is scored on the same folds (written once to folds.tsv).
  - Scalers, PCA and classifiers are fit on training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, roc_auc_score)


# ---------------------------------------------------------------- splits
def donor_folds(donor_df, n_splits=5, seed=0, strat_col="adnc"):
    """donor -> fold, stratified by ADNC so every fold spans the stages."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=donor_df.index, name="fold")
    for k, (_, te) in enumerate(skf.split(donor_df.index, donor_df[strat_col])):
        fold.iloc[te] = k
    assert (fold >= 0).all()
    return fold


# ---------------------------------------------------------------- calibration
# Binned ECE, copied verbatim from esm2-rosmap-probe/run_experiment.py
# (expected_calibration_error) so both projects report the identical metric.
# Same binning: equal width, np.digitize default (a value on an edge goes up).
def expected_calibration_error(y, p, n_bins=15):
    """Equal-width-bin ECE, the standard formulation."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return float(ece)


def ece_binary(p, y, n_bins=15):
    return expected_calibration_error(np.asarray(y, float), np.asarray(p, float), n_bins)


def ece_toplabel(P, y_idx, n_bins=15):
    """Top-label ECE: confidence = max prob, correctness = argmax == truth."""
    conf = P.max(1)
    corr = (P.argmax(1) == y_idx).astype(float)
    return ece_binary(conf, corr, n_bins)


def reliability_bins(conf, corr, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)   # same binning as the ECE
    rows = []
    for k in range(n_bins):
        m = b == k
        if m.any():
            rows.append((edges[k], edges[k + 1], int(m.sum()),
                         float(conf[m].mean()), float(corr[m].mean())))
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "mean_conf", "accuracy"])


# ---------------------------------------------------------------- metrics
def multiclass_metrics(y_idx, P, n_classes):
    P = np.clip(P, 1e-12, 1)
    P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1)
    onehot = np.eye(n_classes)[y_idx]
    return {
        "accuracy": accuracy_score(y_idx, pred),
        "balanced_accuracy": balanced_accuracy_score(y_idx, pred),
        "macro_f1": f1_score(y_idx, pred, average="macro",
                             labels=np.arange(n_classes), zero_division=0),
        "log_loss": log_loss(y_idx, P, labels=np.arange(n_classes)),
        "brier": float(((P - onehot) ** 2).sum(1).mean()),
        "ece": ece_toplabel(P, y_idx),
    }


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {
        "auroc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(((p - y) ** 2).mean()),
        "ece": ece_binary(p, y, n_bins=5),    # 84 donors: 15 bins would be ~6 per bin
        "ece_15bin": ece_binary(p, y, n_bins=15),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }


# ---------------------------------------------------------------- probes
def full_proba(clf, X, n_classes):
    """predict_proba padded to all classes (a fold may lack a rare class)."""
    P = np.zeros((X.shape[0], n_classes))
    P[:, clf.classes_] = clf.predict_proba(X)
    return P


def fit_logreg(X, y, C):
    return LogisticRegression(C=C, max_iter=500, tol=1e-3, n_jobs=None).fit(X, y)


def tune_C(X, y, groups, Cs=(0.01, 0.1, 1.0), max_cells=15000, seed=0):
    """Pick C by donor-grouped 3-fold log loss on a training-set subsample."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(len(y), max_cells), replace=False)
    X, y, g = X[idx], y[idx], groups[idx]
    n_classes = int(y.max()) + 1
    scores = {}
    for C in Cs:
        ll = []
        for tr, va in GroupKFold(3).split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            clf = fit_logreg(sc.transform(X[tr]), y[tr], C)
            P = np.clip(full_proba(clf, sc.transform(X[va]), n_classes), 1e-12, 1)
            ll.append(log_loss(y[va], P / P.sum(1, keepdims=True),
                               labels=np.arange(n_classes)))
        scores[C] = float(np.mean(ll))
    return min(scores, key=scores.get), scores


def fit_predict(probe, Xtr, ytr, Xte, n_classes, C=1.0, k=15):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if probe == "logreg":
        clf = fit_logreg(Xtr, ytr, C)
    elif probe == "knn":
        clf = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    else:
        raise ValueError(probe)
    return full_proba(clf, Xte, n_classes)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    o = np.argsort(pv)
    r = pv[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(r, 1)
    q[ok] = out
    return q
