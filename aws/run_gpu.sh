#!/usr/bin/env bash
# Runs ON the GPU instance, from the repo root. Geneformer extraction, plus the
# data step if aws/run_data_cpu.sh has not already produced data/seaad_mtg_84.h5ad.
#   bash aws/run_gpu.sh 2>&1 | tee work/gpu_run.log
set -euo pipefail
mkdir -p work
nvidia-smi --query-gpu=name,memory.total --format=csv

# ---- environment
sudo apt-get update -qq && sudo apt-get install -y -qq git-lfs python3-venv >/dev/null
python3 -m venv ~/gf && source ~/gf/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
if [[ ! -d ~/Geneformer ]]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://huggingface.co/ctheodoris/Geneformer ~/Geneformer
fi
( cd ~/Geneformer && git lfs install --local && \
  git lfs pull --include "Geneformer-V2-104M/*,geneformer/*.pkl" )
pip install -q ~/Geneformer
pip install -q -r requirements-gpu.txt
python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"

# ---- data: skipped if the CPU run already produced it and it was scp'd up
if [[ -f data/seaad_mtg_84.h5ad ]]; then
  echo "using existing data/seaad_mtg_84.h5ad"
  [[ -f data/seaad_mtg_84.h5ad.md5 ]] && md5sum -c data/seaad_mtg_84.h5ad.md5
else
  python scripts/00_list_donors.py
  python scripts/01_subsample_donors.py --donors config/donors.txt \
      --out data/seaad_mtg_84.h5ad --max-per-stratum 30
fi

# ---- quick sanity on 200 cells before committing GPU hours
FMEVAL_DATA=data python scripts/03_geneformer_embed.py --gf-repo ~/Geneformer \
    --work work/gf_debug --max-cells 200 --batch 8 || { echo "debug run failed"; exit 1; }
rm -f data/embeddings/gf_*.parquet   # debug output, 200 cells only

# ---- full extraction: pretrained V2-104M and the random-init control
python scripts/03_geneformer_embed.py --gf-repo ~/Geneformer \
    --models Geneformer-V2-104M --random-control --batch "${BATCH:-16}"

tar czf work/fm_eval_outputs.tgz data/seaad_mtg_84.h5ad data/embeddings \
    work/geneformer/tokenization_report.json config/donors.txt
ls -lh work/fm_eval_outputs.tgz
echo "done. copy work/fm_eval_outputs.tgz back, then terminate the instance."
