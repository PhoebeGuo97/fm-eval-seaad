"""Classical embeddings on the same cells the foundation model sees.

  pca          log-normalized HVGs, scaled, 50 PCs
  pca_harmony  the same 50 PCs corrected for donor with Harmony
  randproj     the same scaled HVG matrix through a 50-d Gaussian random
               projection: a floor for "any 50-d linear summary"
  scvi         the atlas's own obsm['X_scVI'], REFERENCE ONLY. SEA-AD assigned
               Subclass/Supertype in this latent space, so cell type accuracy
               on it is circular. Kept to show the ceiling, never the comparator.

HVG selection, PCA and Harmony are unsupervised and fit on all donors
(transductive). No label touches this step. Stated in the README.

  python scripts/02_baseline_embeddings.py
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
sys.path.insert(0, os.path.dirname(__file__))
from common import H5AD, EMB_DIR, DONOR, CELL_ID

N_HVG, N_PC, SEED = 2000, 50, 0


def save(name, X, ids):
    os.makedirs(EMB_DIR, exist_ok=True)
    df = pd.DataFrame(np.asarray(X, dtype=np.float32), index=pd.Index(ids, name=CELL_ID),
                      columns=[f"d{i}" for i in range(X.shape[1])])
    df.to_parquet(os.path.join(EMB_DIR, f"{name}.parquet"))
    print(f"  {name}: {X.shape}")


a = ad.read_h5ad(H5AD)
ids = a.obs[CELL_ID].astype(str).values
print(f"{a.n_obs:,} cells, {a.obs[DONOR].nunique()} donors")

# seurat_v3 works on counts and needs scikit-misc. FMEVAL_HVG_FLAVOR=seurat is
# only for machines where scikit-misc will not build (the smoke test VM).
flavor = os.environ.get("FMEVAL_HVG_FLAVOR", "seurat_v3")
n_hvg = min(N_HVG, a.n_vars // 2)
if flavor == "seurat_v3":
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor=flavor, batch_key=DONOR)
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
if flavor != "seurat_v3":
    sc.pp.highly_variable_genes(a, n_top_genes=n_hvg, flavor=flavor, batch_key=DONOR)
print(f"  HVG: {int(a.var.highly_variable.sum())} ({flavor})")
a = a[:, a.var.highly_variable].copy()
sc.pp.scale(a, max_value=10)
sc.pp.pca(a, n_comps=N_PC, random_state=SEED)
save("pca", a.obsm["X_pca"], ids)

sc.external.pp.harmony_integrate(a, key=DONOR, basis="X_pca",
                                 adjusted_basis="X_harmony", random_state=SEED)
save("pca_harmony", a.obsm["X_harmony"], ids)

rng = np.random.default_rng(SEED)
R = rng.standard_normal((a.n_vars, N_PC)) / np.sqrt(N_PC)
save("randproj", np.asarray(a.X) @ R, ids)

full = ad.read_h5ad(H5AD, backed="r")
if "X_scVI" in full.obsm:
    save("ref_scvi", np.asarray(full.obsm["X_scVI"]), full.obs[CELL_ID].astype(str).values)
else:
    print("  no X_scVI in object, skipping reference")
