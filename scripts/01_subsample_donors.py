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


def csr_rows(g, idx):
    """Read only the requested rows of an on-disk CSR matrix (h5py group)."""
    indptr = g["indptr"][:]
    starts, ends = indptr[idx], indptr[idx + 1]
    data = np.concatenate([g["data"][s:e] for s, e in zip(starts, ends)])
    ind = np.concatenate([g["indices"][s:e] for s, e in zip(starts, ends)])
    ptr = np.concatenate([[0], np.cumsum(ends - starts)])
    return sp.csr_matrix((data.astype(np.float32), ind, ptr),
                         shape=(len(idx), int(g.attrs["shape"][1])))


def subsample(path, donor, a):
    """Memory-light: obs and var are read whole, but only the chosen rows of
    the counts layer and of X_scVI are read from disk. Loading a whole donor
    object (X, UMIs, obsp graphs, all float64) was OOM-killed at 16 GB."""
    import h5py
    from anndata.experimental import read_elem
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        ct = obs["Subclass"].astype(str).to_numpy()
        rng = donor_rng(donor, a.seed)
        pick = []
        for lvl in sorted(pd.unique(ct)):          # sorted: order-independent draws
            pos = np.flatnonzero(ct == lvl)
            pick.append(rng.choice(pos, size=min(len(pos), a.max_per_stratum), replace=False))
        idx = np.sort(np.concatenate(pick))
        g = f["layers/UMIs"] if "layers" in f and "UMIs" in f["layers"] else f["X"]
        if g.attrs.get("encoding-type") != "csr_matrix":
            sys.exit(f"{path}: counts are {g.attrs.get('encoding-type')}, expected csr_matrix")
        X = csr_rows(g, idx)
        scvi = f["obsm/X_scVI"][idx] if "obsm" in f and "X_scVI" in f["obsm"] else None
    if X.nnz and not np.allclose(X.data[:10000], np.round(X.data[:10000])):
        sys.exit(f"{path}: counts are not integers; wrong layer?")
    o = obs.iloc[idx]
    o = o[[c for c in KEEP_OBS if c in o.columns]].copy()
    o["cell_id"] = o.index.astype(str)
    s = ad.AnnData(X, obs=o, var=var)
    if scvi is not None:
        s.obsm["X_scVI"] = np.asarray(scvi, dtype=np.float32)
    for c in s.obs.select_dtypes("category").columns:
        s.obs[c] = s.obs[c].cat.remove_unused_categories()
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
