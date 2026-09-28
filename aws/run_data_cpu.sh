#!/usr/bin/env bash
# Runs ON a CPU instance in us-west-2 (no GPU quota needed), from the repo root.
# Streams the 84 per-donor SEA-AD objects in-region and writes the subsample.
#   bash aws/run_data_cpu.sh 2>&1 | tee work/data_run.log
set -euo pipefail
mkdir -p work data
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv >/dev/null
python3 -m venv ~/fmdata && source ~/fmdata/bin/activate
pip install -q --upgrade pip
pip install -q "numpy==1.26.4" "pandas==2.2.3" "scipy==1.13.1" "anndata==0.10.9" awscli
python scripts/00_list_donors.py
python scripts/01_subsample_donors.py --donors config/donors.txt \
    --out data/seaad_mtg_84.h5ad --max-per-stratum 30
md5sum data/seaad_mtg_84.h5ad | tee data/seaad_mtg_84.h5ad.md5
ls -lh data/seaad_mtg_84.h5ad
echo "done. scp data/seaad_mtg_84.h5ad, its .md5 and config/donors.txt back, then terminate."
