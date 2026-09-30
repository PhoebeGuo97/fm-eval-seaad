"""Dimension-matched sensitivity embeddings.

Geneformer is 768-d; the classical baselines are 50-d. A regularized linear
probe can still gain from more dimensions, so a Geneformer win on the linear
probe could be capacity rather than content. Two controls:

  gf_v2_104m_cls_pca50         Geneformer CLS reduced to 50 PCs (unsupervised,
                               all cells), i.e. Geneformer at the baselines' size
  gf_random_v2_104m_cls_pca50  same for the random-weight control
  pca256                       the classical HVG pipeline with 256 PCs,
                               i.e. the baseline given more room

  python scripts/07_dimension_matched.py
"""
import os, sys
import numpy as np, pandas as pd, anndata as ad, scanpy as sc
from sklearn.decomposition import PCA
sys.path.insert(0, os.path.dirname(__file__))
from common import H5AD, EMB_DIR, CELL_ID, load_embedding

SEED = 0


def save(name, X, ids):
    pd.DataFrame(np.asarray(X, np.float32), index=pd.Index(ids, name=CELL_ID),
                 columns=[f"d{i}" for i in range(X.shape[1])]) \
      .to_parquet(os.path.join(EMB_DIR, f"{name}.parquet"))
    print(f"  {name}: {X.shape}")


a = ad.read_h5ad(H5AD)
ids = a.obs[CELL_ID].astype(str).values

for src in ("gf_v2_104m_cls", "gf_random_v2_104m_cls"):
    if not os.path.exists(os.path.join(EMB_DIR, f"{src}.parquet")):
        print(f"  {src}: missing, skipped"); continue
    X = load_embedding(src, ids)
    p = PCA(50, random_state=SEED).fit(X)
    print(f"  {src}: 50 PCs explain {p.explained_variance_ratio_.sum():.2f} of variance")
    save(f"{src}_pca50", p.transform(X), ids)

# same recipe as 02_baseline_embeddings.py, 256 components
sc.pp.filter_genes(a, min_cells=20)
sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat_v3")
sc.pp.normalize_total(a, target_sum=1e4)
sc.pp.log1p(a)
a = a[:, a.var.highly_variable].copy()
sc.pp.scale(a, max_value=10)
sc.pp.pca(a, n_comps=256, random_state=SEED)
save("pca256", a.obsm["X_pca"], ids)
