#!/usr/bin/env bash
# End-to-end on synthetic data, minus the GPU step. About a minute.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
T="${SMOKE_DIR:-tests/tmp}"
rm -rf "$T" && mkdir -p "$T"
export FMEVAL_DATA="$T/data" FMEVAL_RESULTS="$T/results"
$PY tests/make_synthetic.py "$T/data"
$PY scripts/02_baseline_embeddings.py
$PY scripts/04_eval_celltype.py --n-perm 1 --min-cells 50 --min-donors 10
$PY scripts/05_eval_donor.py --repeats 2 --n-perm 50 --n-perm-subclass 20 --n-perm-incr 10
$PY scripts/06_batch_retention.py --min-cells 100
$PY - <<'PY'
import os, pandas as pd
s = pd.read_csv(os.environ["FMEVAL_RESULTS"] + "/celltype/summary.tsv", sep="\t")
f1 = s[(s.label == "Subclass") & (s.metric == "macro_f1")].set_index(["embedding", "probe"]).value
assert f1["fm_signal", "logreg"] > 0.9, f1
assert f1["fm_noise", "logreg"] < 0.35, f1
assert f1["pca", "logreg_permuted"] < 0.35, f1
d = pd.read_csv(os.environ["FMEVAL_RESULTS"] + "/donor/summary.tsv", sep="\t")
t2 = d[(d.embedding == "pca") & (d.subclass == "T2")].iloc[0]
assert t2.auc_mean_repeats > 0.8, t2          # planted ADNC signal is found
other = d[(d.embedding == "pca") & (d.model == "per_subclass") & (d.subclass != "T2")]
assert other.auc_mean_repeats.median() < 0.75, other   # and not everywhere
print("SMOKE TEST PASSED")
PY
