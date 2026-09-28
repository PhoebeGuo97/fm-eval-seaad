"""Task A: cell type classification with donors held out.

For each embedding and each label (Subclass; Supertype restricted to types with
enough cells and donors), fit a linear probe and a kNN probe on 4/5 of donors
and score the held-out fifth. Metrics: accuracy, balanced accuracy, macro-F1,
log loss, Brier, top-label ECE. The unit of inference is the donor: per-donor
macro-F1 is compared between embeddings with a paired Wilcoxon test.

Null: the linear probe refit on training labels permuted across cells, scored
on true test labels. It should sit at chance; if it does not, something leaks.

  python scripts/04_eval_celltype.py [--embeddings pca pca_harmony geneformer_v2_104m]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
sys.path.insert(0, os.path.dirname(__file__))
from common import (read_obs, donor_table, load_embedding, list_embeddings,
                    DONOR, SUBCLASS, SUPERTYPE, CELL_ID, RESULTS, FOLDS)
import evalkit as ek

p = argparse.ArgumentParser()
p.add_argument("--embeddings", nargs="*", default=None)
p.add_argument("--labels", nargs="*", default=[SUBCLASS, SUPERTYPE])
p.add_argument("--n-perm", type=int, default=2)
p.add_argument("--min-cells", type=int, default=100)
p.add_argument("--min-donors", type=int, default=20)
p.add_argument("--baselines", nargs="+", default=["pca_harmony", "pca"],
               help="paired tests are run against each")
a = p.parse_args()

out = os.path.join(RESULTS, "celltype")
os.makedirs(out, exist_ok=True)
obs = read_obs()
ids = obs[CELL_ID].astype(str).values
donors = obs[DONOR].astype(str).values
dt = donor_table(obs)

# folds are made once and reused by every script and embedding
if os.path.exists(FOLDS):
    fold_of = pd.read_csv(FOLDS, sep="\t", index_col=0)["fold"]
    assert set(fold_of.index) == set(dt.index.astype(str)), "folds.tsv donors differ"
else:
    fold_of = ek.donor_folds(dt)
    fold_of.index = fold_of.index.astype(str)
    fold_of.to_csv(FOLDS, sep="\t")
cell_fold = fold_of.loc[donors].to_numpy()
n_folds = int(cell_fold.max()) + 1

embs = a.embeddings or list_embeddings()
print(f"{len(ids):,} cells, {len(dt)} donors, {n_folds} donor folds; embeddings: {embs}")

summary, perdonor, perclass, calib = [], [], [], []
for label in a.labels:
    lab = obs[label].astype(str).values
    keep = np.ones(len(lab), bool)
    if label == SUPERTYPE:
        g = pd.DataFrame({"t": lab, "d": donors})
        n_cells = g.t.value_counts()
        n_don = g.groupby("t").d.nunique()
        ok = n_cells.index[(n_cells >= a.min_cells) & (n_don.reindex(n_cells.index) >= a.min_donors)]
        keep = np.isin(lab, ok)
        print(f"\n{label}: {len(ok)} of {len(n_cells)} types kept "
              f"(>= {a.min_cells} cells and >= {a.min_donors} donors), "
              f"{keep.sum():,} cells")
    classes = np.array(sorted(set(lab[keep])))
    y = np.searchsorted(classes, lab)
    K = len(classes)

    for emb in embs:
        X = load_embedding(emb, ids)
        t0 = time.time()
        tr0 = keep & (cell_fold != 0)
        C, cscores = ek.tune_C(X[tr0], y[tr0], donors[tr0])
        oof = {pr: np.zeros((len(y), K)) for pr in ("logreg", "knn")}
        null_rows = []
        for f in range(n_folds):
            tr = keep & (cell_fold != f)
            te = keep & (cell_fold == f)
            for pr in ("logreg", "knn"):
                oof[pr][te] = ek.fit_predict(pr, X[tr], y[tr], X[te], K, C=C)
            rng = np.random.default_rng(f)
            for r in range(a.n_perm):
                yp = rng.permutation(y[tr])
                Pn = ek.fit_predict("logreg", X[tr], yp, X[te], K, C=C)
                m = ek.multiclass_metrics(y[te], Pn, K)
                null_rows.append(m)
        for pr, P in oof.items():
            m = ek.multiclass_metrics(y[keep], P[keep], K)
            # fold-level spread
            fm = pd.DataFrame([ek.multiclass_metrics(y[keep & (cell_fold == f)],
                                                     P[keep & (cell_fold == f)], K)
                               for f in range(n_folds)])
            for k_, v in m.items():
                summary.append(dict(label=label, embedding=emb, probe=pr, metric=k_,
                                    value=v, fold_sd=fm[k_].std(), dim=X.shape[1], C=C))
            pred = P.argmax(1)
            for d in np.unique(donors[keep]):
                mm = keep & (donors == d)
                perdonor.append(dict(label=label, embedding=emb, probe=pr, donor=d,
                                     n=int(mm.sum()),
                                     accuracy=float((pred[mm] == y[mm]).mean()),
                                     macro_f1=f1_score(y[mm], pred[mm], average="macro",
                                                       zero_division=0)))
            f1s = f1_score(y[keep], pred[keep], average=None, labels=np.arange(K),
                           zero_division=0)
            for c, v in zip(classes, f1s):
                perclass.append(dict(label=label, embedding=emb, probe=pr, cls=c, f1=v))
            rb = ek.reliability_bins(P[keep].max(1), (pred[keep] == y[keep]).astype(float))
            rb.insert(0, "probe", pr); rb.insert(0, "embedding", emb); rb.insert(0, "label", label)
            calib.append(rb)
        if null_rows:
            nm = pd.DataFrame(null_rows).mean()
            for k_, v in nm.items():
                summary.append(dict(label=label, embedding=emb, probe="logreg_permuted",
                                    metric=k_, value=v, fold_sd=np.nan,
                                    dim=X.shape[1], C=C))
        s = pd.DataFrame(summary)
        s = s[(s.label == label) & (s.embedding == emb) & (s.metric == "macro_f1")]
        print(f"  {emb:24s} d={X.shape[1]:4d} C={C:<5} " +
              "  ".join(f"{r.probe}={r.value:.3f}" for r in s.itertuples()) +
              f"  ({time.time()-t0:.0f}s)", flush=True)

S = pd.DataFrame(summary); S.to_csv(f"{out}/summary.tsv", sep="\t", index=False)
D = pd.DataFrame(perdonor); D.to_csv(f"{out}/per_donor.tsv", sep="\t", index=False)
pd.DataFrame(perclass).to_csv(f"{out}/per_class_f1.tsv", sep="\t", index=False)
pd.concat(calib).to_csv(f"{out}/reliability.tsv", sep="\t", index=False)

# paired, donor-level comparison against each baseline
rows = []
for (label, pr), g in D.groupby(["label", "probe"]):
    w = g.pivot(index="donor", columns="embedding", values="macro_f1")
    for base in a.baselines:
        if base not in w:
            continue
        for emb in w.columns:
            if emb == base:
                continue
            diff = (w[emb] - w[base]).dropna()
            p_ = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
            rows.append(dict(label=label, probe=pr, embedding=emb, baseline=base,
                             n_donors=len(diff), median_diff=diff.median(),
                             donors_better=int((diff > 0).sum()), wilcoxon_p=p_))
P = pd.DataFrame(rows)
if len(P):
    P["q"] = ek.bh(P.wilcoxon_p)
P.to_csv(f"{out}/paired_vs_baseline.tsv", sep="\t", index=False)
print("\n" + P.to_string(index=False) if len(P) else "")
print(f"\nwrote {out}/")
