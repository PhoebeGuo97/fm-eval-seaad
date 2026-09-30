"""README figures, drawn only from the result tables (no recomputation).

  figures/celltype_supertype.png   Supertype macro-F1, linear probe vs kNN
  figures/donor_adnc.png           donor ADNC AUROC against each model's own null
  figures/depth_vs_adnc.png        how well an embedding encodes donor depth
                                   vs how well it predicts ADNC

  python scripts/09_figures.py
"""
import os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, RD, OUT = "results", "results_dim", "figures"
os.makedirs(OUT, exist_ok=True)

# reference palette, light mode (validated set; slots 1 and 2 only)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, NEUTRAL, NULLBAND = "#2a78d6", "#eb6834", "#8a8984", "#dcdbd6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})

# display order and names; missing embeddings are skipped
ORDER = [
    ("pca", "PCA (50)"), ("pca256", "PCA (256)"), ("pca_harmony", "PCA + Harmony (50)"),
    ("gf_v2_104m_cls", "Geneformer (768)"), ("gf_v2_104m_cls_pca50", "Geneformer, 50 PCs"),
    ("gf_random_v2_104m_cls", "Geneformer, random weights"),
    ("randproj", "Random projection (50)"), ("ref_scvi", "scVI (circular ref.)"),
]


def read(path):
    return pd.read_csv(path, sep="\t") if os.path.exists(path) else None


def both(sub):
    """Main results plus the dimension-matched run; main wins on duplicates."""
    a, b = read(f"{R}/{sub}"), read(f"{RD}/{sub}")
    if b is not None and a is not None:
        b = b[~b.embedding.isin(a.embedding)]
    return pd.concat([x for x in (a, b) if x is not None], ignore_index=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {OUT}/{name}")


# ---------------------------------------------------------------- 1. cell type
s = both("celltype/summary.tsv")
s = s[(s.label == "Supertype") & (s.metric == "macro_f1")]
rows = [(name, s[(s.embedding == e) & (s.probe == "logreg")].value.squeeze(),
         s[(s.embedding == e) & (s.probe == "knn")].value.squeeze())
        for e, name in ORDER if e in set(s.embedding)]
null = s[s.probe == "logreg_permuted"].value.max()
fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(rows) + 1.2))
y = np.arange(len(rows))[::-1]
h = 0.36
for i, (lab, lr, kn) in enumerate(rows):
    ax.barh(y[i] + h / 2 + 0.02, lr, height=h, color=S1, label="Linear probe" if i == 0 else None)
    ax.barh(y[i] - h / 2 - 0.02, kn, height=h, color=S2, label="15-NN" if i == 0 else None)
    ax.text(lr + 0.008, y[i] + h / 2 + 0.02, f"{lr:.3f}", va="center", fontsize=8, color=INK2)
    ax.text(kn + 0.008, y[i] - h / 2 - 0.02, f"{kn:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, [r[0] for r in rows])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 1)
ax.set_xlabel("Supertype macro-F1, donors held out (106 types)")
ax.axvline(null, color=NEUTRAL, lw=1, ls=(0, (3, 3)))
ax.text(null + 0.01, y[-1] - 0.75, f"permuted labels ({null:.3f})", fontsize=8, color=INK2)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
ax.set_title("Cell type: Geneformer's linear-probe lead disappears at 50 dimensions",
             loc="left", fontsize=11, color=INK, pad=26)
save(fig, "celltype_supertype.png")

# ---------------------------------------------------------------- 2. donor ADNC
d = both("donor/summary.tsv")
d = d[d.model == "combined"]
nulls = pd.concat([x for x in (read(f"{R}/donor/null_auroc.tsv"), read(f"{RD}/donor/null_auroc.tsv"))
                   if x is not None], axis=1)
nulls = nulls.loc[:, ~nulls.columns.duplicated()]
cov = read(f"{R}/donor/summary.tsv")
cov_auc = cov[cov.model == "covariates"].auc_mean_repeats.squeeze()
tech_auc = None
if os.path.exists("work/08.log"):
    m = re.search(r"technical metrics alone: AUROC ([0-9.]+)", open("work/08.log").read())
    tech_auc = float(m.group(1)) if m else None

rows = []
for e, name in ORDER:
    if e == "ref_scvi" or e not in set(d.embedding):
        continue
    auc = d[d.embedding == e].auc_mean_repeats.squeeze()
    col = f"{e}|combined"
    lo, hi = (np.percentile(nulls[col].dropna(), [2.5, 97.5]) if col in nulls else (np.nan, np.nan))
    rows.append((name, auc, lo, hi, d[d.embedding == e].perm_p.squeeze()))
fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.4))
y = np.arange(len(rows))[::-1]
for i, (lab, auc, lo, hi, p) in enumerate(rows):
    ax.plot([lo, hi], [y[i], y[i]], color=NULLBAND, lw=7, solid_capstyle="round",
            label="95% permutation null" if i == 0 else None, zorder=1)
    ax.scatter(auc, y[i], s=64, color=S1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="Observed AUROC" if i == 0 else None)
    ax.text(auc + 0.012, y[i], f"{auc:.3f}  (p {p:.3f})", va="center", fontsize=8, color=INK2,
            bbox=dict(boxstyle="square,pad=0.15", fc=SURFACE, ec="none"), zorder=4)
for x, lab, ha, dx in ((cov_auc, "covariates only", "right", -0.006),
                       (tech_auc, "depth & QC metrics only", "left", 0.006)):
    if x is not None and np.isfinite(x):
        ax.axvline(x, color=NEUTRAL, lw=1, ls=(0, (3, 3)), zorder=2)
        ax.text(x + dx, y[0] + 0.75, f"{lab} {x:.3f}", fontsize=8, color=INK2, ha=ha, va="center")
ax.set_yticks(y, [r[0] for r in rows])
ax.tick_params(axis="y", length=0)
ax.set_xlim(0.3, 0.9)
ax.set_ylim(y[-1] - 0.7, y[0] + 1.6)
ax.set_xlabel("AUROC, ADNC Intermediate/High vs Not AD/Low (84 donors, 5x5-fold CV)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
ax.set_title("Donor pathology: Geneformer leads, but depth and QC metrics alone come close",
             loc="left", fontsize=11, color=INK, pad=26)
save(fig, "donor_adnc.png")

# ---------------------------------------------------------------- 3. depth vs ADNC
t = read(f"{R}/technical/embedding_depth_and_adnc.tsv")
if t is not None:
    names = dict(ORDER)
    t = t.merge(d[["embedding", "auc_mean_repeats"]], on="embedding", how="left")
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.grid(True, axis="both")
    ax.scatter(t.depth_r2, t.auc_mean_repeats, s=70, color=S1, edgecolor=SURFACE, linewidth=2, zorder=3)
    for r in t.itertuples():
        ax.annotate(names.get(r.embedding, r.embedding), (r.depth_r2, r.auc_mean_repeats),
                    xytext=(6, -3 if r.embedding != "gf_v2_104m_cls" else 6),
                    textcoords="offset points", fontsize=8, color=INK2,
                    ha="left" if r.depth_r2 < 0.9 else "right")
    if tech_auc:
        ax.axhline(tech_auc, color=NEUTRAL, lw=1, ls=(0, (3, 3)))
        ax.text(0.41, tech_auc + 0.005, f"depth & QC metrics alone ({tech_auc:.3f})", fontsize=8, color=INK2)
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.55, 0.8)
    ax.set_xlabel("Donor median genes detected, predicted from embedding (CV R²)")
    ax.set_ylabel("ADNC AUROC from embedding")
    ax.spines["left"].set_visible(True)
    ax.set_title("The embedding that predicts pathology best\nalso encodes sequencing depth best",
                 loc="left", fontsize=11, color=INK)
    save(fig, "depth_vs_adnc.png")
