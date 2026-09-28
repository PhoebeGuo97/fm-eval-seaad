"""Task B: donor-level AD neuropathologic change from cell embeddings.

Target: ADNC Intermediate/High (NIA-AA "sufficient explanation for dementia")
versus Not AD/Low. Spearman against the 4-level ordinal is reported too.

Donor features are mean embeddings per subclass. Because cells were capped at
30 per donor per subclass, composition is removed by design: this asks whether
the *state* of cells encodes pathology, not whether their proportions shift.

Models (all L2 logistic regression, prespecified, no tuning at n = 84):
  per-subclass   mean embedding of one subclass -> PCA(10) -> logistic
  combined       per-subclass PCA(10) blocks concatenated over subclasses
                 present in every donor -> logistic (C = 0.1)
  covariates     age, sex, APOE e4 dosage, PMI, fraction of
                 nuclei from 10x Multiome libraries
  combined+cov   both
PCA, scaling and the classifier are fit inside each training fold.

Scoring: repeated stratified 5-fold over donors. The reported statistic is the
AUROC of out-of-fold probabilities averaged over R repeats. The null permutes
the donor labels and recomputes exactly that statistic (same splits, same R),
so observed and null are on the same footing.

  python scripts/05_eval_donor.py [--n-perm 1000] [--repeats 5]
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from common import (read_obs, donor_table, load_embedding, list_embeddings,
                    DONOR, SUBCLASS, CELL_ID, CHEM, RESULTS)
import evalkit as ek

p = argparse.ArgumentParser()
p.add_argument("--embeddings", nargs="*", default=None)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--n-perm", type=int, default=1000, help="combined model")
p.add_argument("--n-perm-subclass", type=int, default=200)
p.add_argument("--n-pc", type=int, default=10)
p.add_argument("--min-cells", type=int, default=5,
               help="donor needs this many cells of a subclass to contribute")
a = p.parse_args()

out = os.path.join(RESULTS, "donor")
os.makedirs(out, exist_ok=True)
obs = read_obs()
ids = obs[CELL_ID].astype(str).values
dt = donor_table(obs)
dt.index = dt.index.astype(str)
y_all = dt["ad_bin"]

print(f"{len(dt)} donors: " + ", ".join(f"{k} {v}" for k, v in
      dt[dt.columns[0]].value_counts().items()))
if "frac_multiome" in dt:
    print("fraction 10xMulti nuclei by ADNC stage (0 Not AD .. 3 High):\n" +
          dt.groupby("adnc")["frac_multiome"].describe()[["count", "mean", "min", "max"]]
          .round(3).to_string())
    print(f"Spearman(frac_multiome, ADNC) = "
          f"{spearmanr(dt.frac_multiome, dt.adnc).correlation:.3f}")

# covariates, imputed with the median (fit on all donors: labels unused)
cov = pd.DataFrame(index=dt.index)
cov["age"] = dt["age"]; cov["male"] = dt["male"]
cov["apoe4"] = dt["apoe4"]; cov["pmi"] = dt["pmi"]
if "frac_multiome" in dt:
    cov["frac_multiome"] = dt["frac_multiome"]
cov = cov.astype(float)
cov = cov.fillna(cov.median())


def split_list(y, repeats):
    return [list(StratifiedKFold(5, shuffle=True, random_state=r).split(y, y))
            for r in range(repeats)]


def featurize(blocks, cov_X, splits):
    """Fold-specific features. PCA and scaling never see labels, so this is
    computed once per split and reused by every permutation."""
    feats = []
    for folds in splits:
        fr = []
        for tr, te in folds:
            Ftr, Fte = [], []
            for B in blocks:
                sc = StandardScaler().fit(B[tr])
                k = min(a.n_pc, len(tr) - 1, B.shape[1])
                pca = PCA(k, random_state=0).fit(sc.transform(B[tr]))
                Ftr.append(pca.transform(sc.transform(B[tr])))
                Fte.append(pca.transform(sc.transform(B[te])))
            if cov_X is not None:
                Ftr.append(cov_X[tr]); Fte.append(cov_X[te])
            Ftr, Fte = np.hstack(Ftr), np.hstack(Fte)
            sc = StandardScaler().fit(Ftr)
            fr.append((tr, te, sc.transform(Ftr), sc.transform(Fte)))
        feats.append(fr)
    return feats


def cv_prob(feats, y, C):
    """OOF probabilities averaged over repeats, and mean/sd AUROC over repeats."""
    probs, aucs = [], []
    for fr in feats:
        pr = np.zeros(len(y))
        for tr, te, Ftr, Fte in fr:
            clf = LogisticRegression(C=C, max_iter=2000).fit(Ftr, y[tr])
            pr[te] = clf.predict_proba(Fte)[:, 1]
        probs.append(pr)
        aucs.append(roc_auc_score(y, pr))
    return np.mean(probs, 0), float(np.mean(aucs)), float(np.std(aucs))


def perm_p(feats, y, C, observed, n_perm, seed):
    rng = np.random.default_rng(seed)
    null = np.array([cv_prob(feats, rng.permutation(y), C)[1] for _ in range(n_perm)])
    return (1 + (null >= observed).sum()) / (1 + n_perm), null


def record(rows, preds, **kw):
    y, pr = kw.pop("y"), kw.pop("prob")
    donors_ = kw.pop("donors")
    m = ek.binary_metrics(y, pr)
    m["spearman_adnc"] = spearmanr(pr, dt.loc[donors_, "adnc"]).correlation
    rows.append({**kw, **m})
    preds.append(pd.DataFrame({"donor": donors_, "y": y, "prob": pr,
                               **{k: kw[k] for k in ("embedding", "model", "subclass")}}))


rows, preds, nulls = [], [], {}
donors = dt.index.to_numpy()
y = y_all.to_numpy()
splits = split_list(y, a.repeats)

# covariates alone
F = featurize([], cov.to_numpy(), splits)
pr, auc, sd = cv_prob(F, y, C=1.0)
pv, null = perm_p(F, y, 1.0, auc, a.n_perm, 1)
nulls["covariates"] = null
record(rows, preds, embedding="none", model="covariates", subclass="all",
       n_donors=len(y), auc_mean_repeats=auc, auc_sd_repeats=sd, perm_p=pv,
       y=y, prob=pr, donors=donors)
print(f"covariates only: AUROC {auc:.3f} (perm p {pv:.3g})")

subclasses = sorted(obs[SUBCLASS].astype(str).unique())
embs = a.embeddings or list_embeddings()
for emb in embs:
    t0 = time.time()
    X = load_embedding(emb, ids)
    df = pd.DataFrame(X, index=ids)
    grp = [obs[DONOR].astype(str).values, obs[SUBCLASS].astype(str).values]
    means = df.groupby(grp).mean()
    counts = df.groupby(grp).size()
    means = means[counts.reindex(means.index) >= a.min_cells]

    full_blocks = []
    for s in subclasses:
        M = means.xs(s, level=1) if s in means.index.get_level_values(1) else None
        if M is None:
            continue
        M = M.reindex([d for d in donors if d in M.index])
        if len(M) == len(donors):
            full_blocks.append(M.loc[donors].to_numpy())
        # per-subclass model on the donors that have this subclass
        ys = y_all.loc[M.index].to_numpy()
        if len(M) < 30 or min(np.bincount(ys, minlength=2)) < 5:
            continue
        sp_ = split_list(ys, a.repeats)
        F = featurize([M.to_numpy()], None, sp_)
        pr, auc, sd = cv_prob(F, ys, C=1.0)
        pv, _ = perm_p(F, ys, 1.0, auc, a.n_perm_subclass, 2)
        record(rows, preds, embedding=emb, model="per_subclass", subclass=s,
               n_donors=len(M), auc_mean_repeats=auc, auc_sd_repeats=sd, perm_p=pv,
               y=ys, prob=pr, donors=M.index.to_numpy())

    for model, cX in (("combined", None), ("combined+cov", cov.to_numpy())):
        F = featurize(full_blocks, cX, splits)
        pr, auc, sd = cv_prob(F, y, C=0.1)
        pv, null = perm_p(F, y, 0.1, auc, a.n_perm, 3)
        nulls[f"{emb}|{model}"] = null
        record(rows, preds, embedding=emb, model=model, subclass=f"{len(full_blocks)} subclasses",
               n_donors=len(y), auc_mean_repeats=auc, auc_sd_repeats=sd, perm_p=pv,
               y=y, prob=pr, donors=donors)
        print(f"  {emb:24s} {model:13s} AUROC {auc:.3f} +/- {sd:.3f} "
              f"perm p {pv:.3g}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)")

R = pd.DataFrame(rows)
R["perm_q_within_embedding"] = np.nan
for emb, g in R[R.model == "per_subclass"].groupby("embedding"):
    R.loc[g.index, "perm_q_within_embedding"] = ek.bh(g.perm_p)
R.to_csv(f"{out}/summary.tsv", sep="\t", index=False)
pd.concat(preds).to_csv(f"{out}/oof_predictions.tsv", sep="\t", index=False)
pd.DataFrame(nulls).to_csv(f"{out}/null_auroc.tsv", sep="\t", index=False)
print(R[R.model != "per_subclass"][["embedding", "model", "auc_mean_repeats",
      "perm_p", "brier", "ece", "spearman_adnc"]].to_string(index=False))
top = R[R.model == "per_subclass"].sort_values("perm_p").head(15)
print("\nbest per-subclass models:\n" + top[["embedding", "subclass", "n_donors",
      "auc_mean_repeats", "perm_p", "perm_q_within_embedding"]].to_string(index=False))
print(f"\nwrote {out}/")
