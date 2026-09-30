"""Could a donor-level win come from sequencing depth rather than biology?

Geneformer's input is a rank list truncated at 4,096 genes, so how many genes a
nucleus detects directly shapes what the model sees. If depth differs by ADNC
stage, an embedding that encodes depth can "predict" pathology without any
biology. Three checks:

  1. Per-donor technical summaries (median genes detected, median log UMIs,
     median mitochondrial fraction, median doublet score, fraction at the
     4,096-token cap proxy) against ADNC: Spearman and a technical-only
     AUROC under the same CV and permutation scheme as 05.
  2. How well each embedding predicts per-donor median genes detected
     (ridge on per-subclass mean embeddings, CV R^2). High = encodes depth.
  3. 05's combined model re-scored with technical covariates added, with the
     conditional null (embedding rows shuffled, covariates fixed).

  python scripts/08_technical_confounds.py
"""
import argparse, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, r2_score
sys.path.insert(0, os.path.dirname(__file__))
from common import (read_obs, donor_table, load_embedding, list_embeddings,
                    DONOR, SUBCLASS, CELL_ID, RESULTS)

p = argparse.ArgumentParser()
p.add_argument("--embeddings", nargs="*", default=None)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--n-perm", type=int, default=500)
p.add_argument("--n-pc", type=int, default=10)
a = p.parse_args()
out = os.path.join(RESULTS, "technical"); os.makedirs(out, exist_ok=True)

obs = read_obs()
ids = obs[CELL_ID].astype(str).values
dt = donor_table(obs); dt.index = dt.index.astype(str)
donors = dt.index.to_numpy(); y = dt["ad_bin"].to_numpy()

g = obs.groupby(obs[DONOR].astype(str), observed=True)
tech = pd.DataFrame({
    "med_genes": g["Genes detected"].median(),
    "med_log_umis": np.log10(g["Number of UMIs"].median()),
    "med_mito": g["Fraction mitochondrial UMIs"].median(),
    "med_doublet": g["Doublet score"].median(),
    "frac_genes_gt_4096": g["Genes detected"].apply(lambda s: (s > 4096).mean()),
}).loc[donors].astype(float)
rows = [dict(metric=c, spearman_adnc=spearmanr(tech[c], dt["adnc"]).correlation,
             spearman_p=spearmanr(tech[c], dt["adnc"]).pvalue,
             mean_ad=tech[c][y == 1].mean(), mean_not_ad=tech[c][y == 0].mean())
        for c in tech]
T = pd.DataFrame(rows); T.to_csv(f"{out}/tech_vs_adnc.tsv", sep="\t", index=False)
print("1. per-donor technical metrics vs ADNC\n" + T.round(3).to_string(index=False))

splits = [list(StratifiedKFold(5, shuffle=True, random_state=r).split(y, y))
          for r in range(a.repeats)]


def feats(blocks, cov, sp):
    res = []
    for folds in sp:
        fr = []
        for tr, te in folds:
            Ftr, Fte = [], []
            for B in blocks:
                s = StandardScaler().fit(B[tr])
                k = min(a.n_pc, len(tr) - 1, B.shape[1])
                pc = PCA(k, random_state=0).fit(s.transform(B[tr]))
                Ftr.append(pc.transform(s.transform(B[tr]))); Fte.append(pc.transform(s.transform(B[te])))
            if cov is not None:
                Ftr.append(cov[tr]); Fte.append(cov[te])
            Ftr, Fte = np.hstack(Ftr), np.hstack(Fte)
            s = StandardScaler().fit(Ftr)
            fr.append((tr, te, s.transform(Ftr), s.transform(Fte)))
        res.append(fr)
    return res


def auc_of(F, yy, C):
    aucs = []
    for fr in F:
        pr = np.zeros(len(yy))
        for tr, te, Ftr, Fte in fr:
            pr[te] = LogisticRegression(C=C, max_iter=2000).fit(Ftr, yy[tr]).predict_proba(Fte)[:, 1]
        aucs.append(roc_auc_score(yy, pr))
    return float(np.mean(aucs))


base_cov = pd.DataFrame({"age": dt.age, "male": dt.male, "apoe4": dt.apoe4, "pmi": dt.pmi,
                         "frac_multiome": dt.get("frac_multiome", 0.0)}).astype(float)
base_cov = base_cov.fillna(base_cov.median())
full_cov = pd.concat([base_cov, tech], axis=1).to_numpy()

rng = np.random.default_rng(5)
Ft = feats([], tech.to_numpy(), splits)
auc_t = auc_of(Ft, y, 1.0)
null_t = [auc_of(Ft, rng.permutation(y), 1.0) for _ in range(a.n_perm)]
print(f"\n   technical metrics alone: AUROC {auc_t:.3f}, perm p "
      f"{(1 + sum(n >= auc_t for n in null_t)) / (1 + a.n_perm):.3g}")

res = []
subclasses = sorted(obs[SUBCLASS].astype(str).unique())
for emb in a.embeddings or list_embeddings():
    X = load_embedding(emb, ids)
    means = pd.DataFrame(X, index=ids).groupby(
        [obs[DONOR].astype(str).values, obs[SUBCLASS].astype(str).values]).mean()
    blocks = []
    for s in subclasses:
        if s in means.index.get_level_values(1):
            M = means.xs(s, level=1)
            if set(donors) <= set(M.index):
                blocks.append(M.loc[donors].to_numpy())
    # 2. does the embedding encode depth? ridge on concatenated per-block PCs, CV R^2
    Z = np.hstack([PCA(min(a.n_pc, B.shape[1]), random_state=0)
                   .fit_transform(StandardScaler().fit_transform(B)) for B in blocks])
    pred = cross_val_predict(make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 20))),
                             Z, tech["med_genes"].to_numpy(), cv=KFold(5, shuffle=True, random_state=0))
    r2 = r2_score(tech["med_genes"], pred)
    # 3. combined + base + technical covariates, conditional null
    F = feats(blocks, full_cov, splits)
    auc = auc_of(F, y, 0.1)
    inull = []
    for _ in range(a.n_perm):
        pm = rng.permutation(len(y))
        inull.append(auc_of(feats([B[pm] for B in blocks], full_cov, splits), y, 0.1))
    pv = (1 + sum(n >= auc for n in inull)) / (1 + a.n_perm)
    res.append(dict(embedding=emb, depth_r2=r2, auc_with_tech_cov=auc,
                    null_mean=float(np.mean(inull)), incr_perm_p=pv))
    print(f"  {emb:30s} depth R2 {r2:6.3f}   AUROC with tech+cov {auc:.3f} "
          f"(embedding shuffled {np.mean(inull):.3f}, p {pv:.3g})", flush=True)
R = pd.DataFrame(res); R.to_csv(f"{out}/embedding_depth_and_adnc.tsv", sep="\t", index=False)
print(f"\nwrote {out}/")
