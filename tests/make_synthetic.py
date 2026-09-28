"""Synthetic SEA-AD-shaped object for the smoke test. Known answers:
cell types are separable; ADNC shifts expression in one subclass only;
a 'fm_signal' embedding carries cell type, 'fm_noise' carries nothing."""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

out = sys.argv[1] if len(sys.argv) > 1 else "tests/tmp/data"
os.makedirs(f"{out}/embeddings", exist_ok=True)
rng = np.random.default_rng(1)
n_donor, n_type, per, n_gene = 40, 6, 20, 600
adnc = np.array(["Not AD", "Low", "Intermediate", "High"])[rng.integers(0, 4, n_donor)]
types = [f"T{i}" for i in range(n_type)]
base = rng.gamma(0.5, 1.0, (n_type, n_gene))
rows, obs = [], []
for d in range(n_donor):
    donor_eff = rng.normal(0, 0.15, n_gene)
    for t in range(n_type):
        mu = base[t] * np.exp(donor_eff)
        if t == 2 and adnc[d] in ("Intermediate", "High"):
            mu = mu * np.exp(np.r_[np.full(40, 0.8), np.zeros(n_gene - 40)])
        c = rng.poisson(mu * 3, (per, n_gene))
        rows.append(c)
        for j in range(per):
            obs.append({"Donor ID": f"D{d:02d}", "Subclass": types[t],
                        "Supertype": f"{types[t]}_{j % 2}",
                        "Overall AD neuropathological Change": adnc[d],
                        "Age at Death": str(rng.integers(70, 95)) if j == 0 else None,
                        "Sex": "Male" if d % 2 else "Female",
                        "APOE Genotype": "3/4" if d % 3 == 0 else "3/3",
                        "PMI": 5.0 + d % 7, "method": "10xMulti" if (d + j) % 3 == 0 else "10Xv3.1"})
X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
obs = pd.DataFrame(obs)
# donor-level fields must be constant within donor
for c in ["Age at Death"]:
    obs[c] = obs.groupby("Donor ID")[c].transform("first")
obs["cell_id"] = [f"c{i}" for i in range(len(obs))]
obs.index = obs["cell_id"].values
for c in obs.columns:
    if obs[c].dtype == object and c != "cell_id":
        obs[c] = obs[c].astype("category")
var = pd.DataFrame({"gene_ids": [f"ENSG{i:011d}" for i in range(n_gene)]},
                   index=[f"G{i}" for i in range(n_gene)])
a = ad.AnnData(X, obs=obs, var=var)
a.obsm["X_scVI"] = rng.normal(size=(a.n_obs, 10)).astype(np.float32) + \
    np.eye(n_type, 10)[obs.Subclass.cat.codes.values] * 3
a.write_h5ad(f"{out}/seaad_mtg_84.h5ad")

codes = obs.Subclass.cat.codes.values
for name, sig in (("fm_signal", 4.0), ("fm_noise", 0.0)):
    E = rng.normal(size=(a.n_obs, 64)) + sig * np.eye(n_type, 64)[codes]
    pd.DataFrame(E.astype(np.float32), index=pd.Index(obs.cell_id, name="cell_id"),
                 columns=[f"d{i}" for i in range(64)]).to_parquet(f"{out}/embeddings/{name}.parquet")
print(f"synthetic: {a.n_obs} cells, {n_donor} donors -> {out}")
