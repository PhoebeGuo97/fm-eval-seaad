"""Write config/donors.txt: every donor with a per-donor MTG snRNA-seq object in
the chosen release. The 20 project-1 donors come first in their original order,
then the rest sorted, so the list is stable and diffable.

  python scripts/00_list_donors.py --stamp 2026-06-22
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from s3_public import list_objects

BUCKET, PREFIX = "sea-ad-single-cell-profiling", "MTG/RNAseq/donors_objects/"

p = argparse.ArgumentParser()
p.add_argument("--stamp", default="2026-06-22")
p.add_argument("--first", default="config/donors_project1.txt")
p.add_argument("--out", default="config/donors.txt")
a = p.parse_args()

rows = list_objects(BUCKET, prefix=PREFIX, suffix=".h5ad")
pat = re.compile(rf"([^/]+)_SEAAD_MTG_RNAseq_final-nuclei\.{re.escape(a.stamp)}\.h5ad$")
found = {}
for key, size in rows:
    m = pat.search(key)
    if m:
        found[m.group(1)] = size
if not found:
    sys.exit(f"no per-donor objects with stamp {a.stamp}; stamps present: "
             f"{sorted({k.rsplit('.', 2)[-2] for k, _ in rows})}")

first = [l.strip() for l in open(a.first) if l.strip()]
absent = [d for d in first if d not in found]
if absent:
    sys.exit(f"project-1 donors not in this release: {absent}")
rest = sorted(set(found) - set(first))
donors = first + rest
with open(a.out, "w") as f:
    f.write("\n".join(donors) + "\n")
gb = sum(found.values()) / 2**30
print(f"{len(donors)} donors ({len(first)} from project 1, {len(rest)} new), "
      f"{gb:.1f} GB total, {sum(found[d] for d in rest)/2**30:.1f} GB new")
print(f"wrote {a.out}")
