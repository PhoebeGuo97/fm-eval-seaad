"""Geneformer zero-shot cell embeddings. Runs on the GPU instance.

  1. Write a tokenizer-ready copy of the object: raw counts, var['ensembl_id'],
     obs['n_counts'], obs['filter_pass'], obs['cell_id'].
  2. Rank-value tokenize with the V2 dictionaries (4096 tokens, CLS/EOS).
  3. Report gene-vocabulary coverage and truncation, because both silently
     change what the model sees.
  4. Extract CLS embeddings from the second-to-last layer (Geneformer's
     recommended general-purpose layer) for every cell, not the default 1000.
  5. The same extraction from a randomly initialized copy of the architecture.
     Same tokenization, same code path, no pretraining. If the pretrained model
     does not beat this, pretraining is not what is doing the work.

Outputs data/embeddings/<name>.parquet indexed by cell_id.

  python scripts/03_geneformer_embed.py --gf-repo ~/Geneformer \
      --models Geneformer-V2-104M --random-control --batch 16
"""
import argparse, os, sys, json, time
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from common import H5AD, EMB_DIR, CELL_ID

p = argparse.ArgumentParser()
p.add_argument("--gf-repo", required=True, help="local clone of huggingface.co/ctheodoris/Geneformer")
p.add_argument("--models", nargs="+", default=["Geneformer-V2-104M"])
p.add_argument("--modes", nargs="+", default=["cls"], choices=["cls", "cell"])
p.add_argument("--random-control", action="store_true")
p.add_argument("--work", default="work/geneformer")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--nproc", type=int, default=4)
p.add_argument("--max-cells", type=int, default=None, help="debug only")
a = p.parse_args()

from geneformer import TranscriptomeTokenizer, EmbExtractor

os.makedirs(a.work, exist_ok=True); os.makedirs(EMB_DIR, exist_ok=True)
tok_in, tok_out = f"{a.work}/tok_in", f"{a.work}/tok_out"
ds_path = f"{tok_out}/seaad.dataset"

# ---- 1-2. tokenize (cached)
if not os.path.exists(ds_path):
    os.makedirs(tok_in, exist_ok=True)
    x = ad.read_h5ad(H5AD)
    if a.max_cells:
        x = x[:a.max_cells].copy()
    X = x.X.tocsr() if sp.issparse(x.X) else sp.csr_matrix(x.X)
    assert np.allclose(X.data[:10000], np.round(X.data[:10000])), "X must be raw counts"
    x.var["ensembl_id"] = x.var["gene_ids"].astype(str).str.split(".").str[0].values
    x.obs["n_counts"] = np.asarray(X.sum(1)).ravel()
    x.obs["filter_pass"] = 1
    x.obs[CELL_ID] = x.obs[CELL_ID].astype(str)
    x.obs = x.obs[[CELL_ID, "n_counts", "filter_pass"]]
    x.obsm.clear()
    x.write_h5ad(f"{tok_in}/seaad.h5ad")
    TranscriptomeTokenizer({CELL_ID: CELL_ID}, nproc=a.nproc, model_version="V2") \
        .tokenize_data(tok_in, tok_out, "seaad", file_format="h5ad")

# ---- 3. coverage report
from datasets import load_from_disk
import pickle, geneformer
ds = load_from_disk(ds_path)
L = np.array(ds["length"])
x = ad.read_h5ad(H5AD, backed="r")
genes = pd.Index(x.var["gene_ids"].astype(str).str.split(".").str[0])
tdict_path = os.path.join(os.path.dirname(geneformer.__file__), "token_dictionary_gc104M.pkl")
report = {"cells_tokenized": int(len(L)), "cells_in_object": int(x.n_obs),
          "median_tokens": float(np.median(L)),
          "frac_at_4096": float((L >= 4096).mean())}
if os.path.exists(tdict_path):
    vocab = pickle.load(open(tdict_path, "rb"))
    report["object_genes_in_vocab"] = int(genes.isin(list(vocab)).sum())
    report["object_genes"] = int(len(genes))
json.dump(report, open(f"{a.work}/tokenization_report.json", "w"), indent=1)
print(json.dumps(report, indent=1))
if len(L) != x.n_obs and not a.max_cells:
    sys.exit("tokenizer dropped cells; check filter_pass / n_counts")

# ---- 5. random-init control: same config, fresh weights
model_dirs = {m: os.path.join(a.gf_repo, m) for m in a.models}
if a.random_control:
    import torch
    from transformers import BertConfig, BertForMaskedLM
    src = model_dirs[a.models[0]]
    rnd = f"{a.work}/random_{a.models[0]}"
    if not os.path.exists(os.path.join(rnd, "config.json")):
        torch.manual_seed(0)
        BertForMaskedLM(BertConfig.from_pretrained(src)).save_pretrained(rnd)
    model_dirs[f"random-{a.models[0]}"] = rnd

# ---- 4. extract
for name, mdir in model_dirs.items():
    for mode in a.modes:
        tag = f"gf_{name}_{mode}".replace("Geneformer-", "").replace("-", "_").lower()
        dest = os.path.join(EMB_DIR, f"{tag}.parquet")
        if os.path.exists(dest):
            print(f"{tag}: exists, skipping"); continue
        t0 = time.time()
        ex = EmbExtractor(model_type="Pretrained", emb_mode=mode, max_ncells=None,
                          emb_layer=-1, emb_label=[CELL_ID],
                          forward_batch_size=a.batch, nproc=a.nproc, model_version="V2")
        embs = ex.extract_embs(mdir, ds_path, f"{a.work}/emb_{tag}", tag)
        embs = embs.set_index(CELL_ID)
        embs.index = embs.index.astype(str); embs.index.name = CELL_ID
        embs.columns = [f"d{c}" for c in range(embs.shape[1])]
        assert not embs.index.has_duplicates and len(embs) == len(L), (len(embs), len(L))
        embs.astype(np.float32).to_parquet(dest)
        print(f"{tag}: {embs.shape} in {(time.time()-t0)/60:.1f} min -> {dest}", flush=True)
