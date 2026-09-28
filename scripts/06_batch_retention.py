"""How much donor and chemistry signal does each embedding keep?

Within each subclass (so cell type cannot explain it), a kNN classifier
predicts the donor of each cell, 5-fold over cells. Reported as balanced
accuracy and as a multiple of chance (1 / n donors). Same for 10x chemistry
(Multiome vs single-nucleus libraries), with donors held out, since
chemistry varies within donor.

This is not "higher is worse" in general: donor differences include biology,
pathology among it. It shows what the donor-level task has to work with and
whether an embedding is dominated by sample identity, which Kedzierska et al.
2025 reported for zero-shot Geneformer.

  python scripts/06_batch_retention.py
"""
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, os.path.dirname(__file__))
from common import read_obs, load_embedding, list_embeddings, DONOR, SUBCLASS, CHEM, CELL_ID, RESULTS

p = argparse.ArgumentParser()
p.add_argument("--embeddings", nargs="*", default=None)
p.add_argument("--min-cells", type=int, default=300)
a = p.parse_args()
out = os.path.join(RESULTS, "batch"); os.makedirs(out, exist_ok=True)

obs = read_obs()
ids = obs[CELL_ID].astype(str).values
rows = []
for emb in a.embeddings or list_embeddings():
    X = load_embedding(emb, ids)
    for s in sorted(obs[SUBCLASS].astype(str).unique()):
        m = (obs[SUBCLASS].astype(str) == s).to_numpy()
        if m.sum() < a.min_cells:
            continue
        for target in (DONOR, CHEM):
            if target not in obs:
                continue
            yv = obs.loc[m, target].astype(str).to_numpy()
            vc = pd.Series(yv).value_counts()
            ok = np.isin(yv, vc.index[vc >= 5])
            if len(np.unique(yv[ok])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), KNeighborsClassifier(15))
            if target == DONOR:
                cv, g = StratifiedKFold(5, shuffle=True, random_state=0), None
            else:
                # chemistry varies within donor: hold donors out so the probe
                # must find a chemistry signature, not recognize donors
                cv = GroupKFold(5)
                g = obs.loc[m, DONOR].astype(str).to_numpy()[ok]
            pred = cross_val_predict(clf, X[m][ok], yv[ok], cv=cv, groups=g)
            ba = balanced_accuracy_score(yv[ok], pred)
            k = len(np.unique(yv[ok]))
            rows.append(dict(embedding=emb, subclass=s, target=target, n_cells=int(ok.sum()),
                             n_levels=k, balanced_accuracy=ba, x_chance=ba * k))
R = pd.DataFrame(rows)
R.to_csv(f"{out}/batch_retention.tsv", sep="\t", index=False)
agg = R.groupby(["target", "embedding"])[["balanced_accuracy", "x_chance"]].median()
agg.to_csv(f"{out}/batch_retention_summary.tsv", sep="\t")
print("median over subclasses:\n" + agg.round(3).to_string())
