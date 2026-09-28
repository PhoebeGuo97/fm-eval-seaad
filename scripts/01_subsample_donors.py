"""Stream the per-donor SEA-AD objects: download, subsample, delete, repeat.

Adapted from project 1 (scdrs-adpd-seaad/scripts/05_subsample_donors.py) with
two changes that matter for this project:

  1. Each donor gets its own RNG seeded from its ID, so which cells are drawn
     does not depend on donor order or on which donors were cached. Project 1
     drew from one shared RNG, so a resumed run was not reproducible.
  2. A stable cell_id column is written. Embeddings are joined back by this ID
     and never by row position.

Keeps raw UMI counts in .X (Geneformer needs counts; the baselines normalize
themselves) and the atlas's obsm['X_scVI'] as a reference embedding.

  python scripts/01_subsample_donors.py --donors config/donors.txt \
      --out data/seaad_mtg_84.h5ad --max-per-stratum 30
"""
import argparse, os, subprocess, sys, shutil, zlib
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

BUCKET = "sea-ad-single-cell-profiling"
PREFIX = "MTG/RNAseq/donors_objects"
REGION = "us-west-2"
HERE = os.path.dirname(os.path.abspath(__file__))

KEEP_OBS = [
    "Donor ID", "Class", "Subclass", "Supertype",
    "Overall AD neuropathological Change", "Braak", "Thal", "CERAD score",
    "Cognitive Status", "Highest Lewy Body Disease", "APOE Genotype",
    "Sex", "Age at Death", "PMI", "Genes detected", "Number of UMIs",
    "Doublet score", "Fraction mitochondrial UMIs", "method",
]


def args_():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--donors", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stamp", default="2026-06-22")
    p.add_argument("--workdir", default="data/stream")
    p.add_argument("--max-per-stratum", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-dir", default=None,
                   help="read donor h5ad files from here instead of S3 (testing)")
    return p.parse_args()


def s3_get(key, dest):
    uri = f"s3://{BUCKET}/{key}"
    if shutil.which("aws"):
        cmd = ["aws", "s3", "cp", uri, dest + "/", "--no-sign-request",
               "--region", REGION, "--only-show-errors"]
    else:
        cmd = [sys.executable, os.path.join(HERE, "s3_public.py"), "get", BUCKET, key, dest]
    subprocess.run(cmd, check=True)
    return os.path.join(dest, os.path.basename(key))


def donor_rng(donor, seed):
    return np.random.default_rng([seed, zlib.crc32(donor.encode())])


def subsample(path, donor, a):
    x = ad.read_h5ad(path)
    ct = x.obs["Subclass"].astype(str).to_numpy()
    rng = donor_rng(donor, a.seed)
    pick = []
    for lvl in sorted(pd.unique(ct)):          # sorted: order-independent draws
        pos = np.flatnonzero(ct == lvl)
        pick.append(rng.choice(pos, size=min(len(pos), a.max_per_stratum), replace=False))
    s = x[np.sort(np.concatenate(pick))].copy()
    X = s.layers["UMIs"] if "UMIs" in s.layers else s.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    if X.nnz and not np.allclose(X.data[:10000], np.round(X.data[:10000])):
        sys.exit(f"{path}: counts are not integers; wrong layer?")
    s.X = X.astype(np.float32)
    s.layers.clear(); s.raw = None
    s.obs = s.obs[[c for c in KEEP_OBS if c in s.obs.columns]].copy()
    s.obs["cell_id"] = s.obs_names.astype(str)
    for k in list(s.obsm):
        if k != "X_scVI":
            del s.obsm[k]
    s.obsp.clear(); s.uns.clear(); s.varm.clear()
    return s


def main():
    a = args_()
    os.makedirs(a.workdir, exist_ok=True)
    donors = [l.strip() for l in open(a.donors) if l.strip() and not l.startswith("#")]
    parts = []
    for i, d in enumerate(donors, 1):
        part = os.path.join(a.workdir, f"{d}.sub.h5ad")
        if os.path.exists(part):
            print(f"[{i}/{len(donors)}] {d}: cached", flush=True)
            parts.append(part); continue
        fname = f"{d}_SEAAD_MTG_RNAseq_final-nuclei.{a.stamp}.h5ad"
        if a.local_dir:
            raw, delete = os.path.join(a.local_dir, fname), False
        else:
            print(f"[{i}/{len(donors)}] {d}: downloading", flush=True)
            raw, delete = s3_get(f"{PREFIX}/{fname}", a.workdir), True
        s = subsample(raw, d, a)
        s.write_h5ad(part, compression="gzip")
        print(f"    kept {s.n_obs:,} cells", flush=True)
        parts.append(part)
        if delete:
            os.remove(raw)

    comb = ad.concat([ad.read_h5ad(p) for p in parts], join="inner",
                     index_unique=None, merge="first")
    if comb.obs_names.has_duplicates:
        comb.obs["cell_id"] = (comb.obs["Donor ID"].astype(str) + ":" +
                               comb.obs["cell_id"].astype(str))
        comb.obs_names = comb.obs["cell_id"].values
        print("note: barcodes collided across donors; cell_id prefixed with donor")
    assert not comb.obs_names.has_duplicates
    for c in comb.obs.select_dtypes("category").columns:
        comb.obs[c] = comb.obs[c].cat.remove_unused_categories()
    if "gene_ids" not in comb.var.columns:
        sys.exit("var has no gene_ids (Ensembl) column; Geneformer needs it")
    comb.write_h5ad(a.out, compression="gzip")
    print(f"\n{comb.n_obs:,} cells x {comb.n_vars:,} genes, "
          f"{comb.obs['Donor ID'].nunique()} donors -> {a.out}")
    print(comb.obs.groupby("Donor ID", observed=True)
          ["Overall AD neuropathological Change"].first().value_counts().to_string())


if __name__ == "__main__":
    main()
