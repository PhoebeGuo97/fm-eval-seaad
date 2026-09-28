"""Shared constants and loaders. Every script imports from here so column names
and label encodings are defined once."""
import os
import numpy as np
import pandas as pd

DONOR = "Donor ID"
SUBCLASS = "Subclass"
SUPERTYPE = "Supertype"
ADNC = "Overall AD neuropathological Change"
CHEM = "method"
CELL_ID = "cell_id"

ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]

DATA = os.environ.get("FMEVAL_DATA", "data")
RESULTS = os.environ.get("FMEVAL_RESULTS", "results")
H5AD = os.path.join(DATA, "seaad_mtg_84.h5ad")
EMB_DIR = os.path.join(DATA, "embeddings")
FOLDS = os.path.join(RESULTS, "folds.tsv")


def adnc_code(s):
    """Not AD 0, Low 1, Intermediate 2, High 3. Anything else raises."""
    m = {k: i for i, k in enumerate(ADNC_ORDER)}
    bad = sorted(set(s) - set(m))
    if bad:
        raise ValueError(f"unexpected ADNC values: {bad}")
    return np.array([m[x] for x in s])


def apoe4_dosage(g):
    """'3/4' -> 1, '4/4' -> 2. Missing -> NaN."""
    if not isinstance(g, str) or "/" not in g:
        return np.nan
    return sum(a.strip() == "4" for a in g.split("/"))


def donor_table(obs):
    """One row per donor with the labels and covariates used by the donor task.
    Fails loudly if a donor-level field varies within a donor."""
    cols = [ADNC, "Age at Death", "Sex", "APOE Genotype", "PMI",
            "Braak", "Thal", "CERAD score", "Cognitive Status"]
    cols = [c for c in cols if c in obs.columns]
    g = obs.groupby(DONOR, observed=True)[cols]
    nuniq = g.nunique(dropna=False)
    varying = nuniq.columns[(nuniq > 1).any()].tolist()
    if varying:
        raise ValueError(f"donor-level columns vary within donor: {varying}")
    d = g.first()
    d["adnc"] = adnc_code(d[ADNC].astype(str))
    d["ad_bin"] = (d["adnc"] >= 2).astype(int)   # NIA-AA: Intermediate/High
    d["apoe4"] = d["APOE Genotype"].map(apoe4_dosage) if "APOE Genotype" in d else np.nan
    d["age"] = pd.to_numeric(d["Age at Death"].astype(str).str.replace("90+", "90",
                             regex=False), errors="coerce")
    d["male"] = (d["Sex"].astype(str) == "Male").astype(int)
    d["pmi"] = pd.to_numeric(d["PMI"], errors="coerce") if "PMI" in d else np.nan
    # 10x chemistry is per library, not per donor: most donors mix single-nucleus
    # 10Xv3.1 and 10xMulti libraries. Summarize it as a fraction.
    if CHEM in obs.columns:
        d["frac_multiome"] = (obs[CHEM].astype(str).str.lower().str.contains("multi")
                              .groupby(obs[DONOR], observed=True).mean())
    return d


def load_embedding(name, cell_ids):
    """Read data/embeddings/<name>.parquet (index cell_id) and align to cell_ids.
    Refuses partial coverage rather than silently dropping cells."""
    p = os.path.join(EMB_DIR, f"{name}.parquet")
    e = pd.read_parquet(p)
    if e.index.name != CELL_ID:
        if CELL_ID in e.columns:
            e = e.set_index(CELL_ID)
        else:
            raise ValueError(f"{p}: no {CELL_ID} index or column")
    if e.index.has_duplicates:
        raise ValueError(f"{p}: duplicated cell ids")
    missing = pd.Index(cell_ids).difference(e.index)
    if len(missing):
        raise ValueError(f"{p}: {len(missing)} of {len(cell_ids)} cells missing, "
                         f"e.g. {list(missing[:3])}")
    X = e.loc[cell_ids].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all():
        raise ValueError(f"{p}: non-finite values")
    return X


def list_embeddings():
    if not os.path.isdir(EMB_DIR):
        return []
    return sorted(f[:-8] for f in os.listdir(EMB_DIR) if f.endswith(".parquet"))


def read_obs(path=H5AD):
    import anndata as ad
    a = ad.read_h5ad(path, backed="r")
    obs = a.obs.copy()
    a.file.close()
    if CELL_ID not in obs.columns:
        obs[CELL_ID] = obs.index.astype(str)
    return obs
