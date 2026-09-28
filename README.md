# Do single-cell foundation model embeddings beat PCA on AD cortex?

A donor-stratified benchmark of zero-shot Geneformer embeddings against
classical dimensionality reduction on the SEA-AD middle temporal gyrus atlas,
on two tasks with different difficulty: cell type identity, and donor-level
Alzheimer's neuropathologic change.

**Result.** _Pending the GPU run._ The answer is reported as found, including
if the baseline wins. Kedzierska et al. 2025 (Genome Biology) found that
zero-shot Geneformer and scGPT embeddings generally underperform HVG, scVI and
Harmony baselines, so a baseline win is the prior.

## Design

**Data.** 84 SEA-AD MTG donors (all four ADNC stages), 30 nuclei per donor per
subclass, raw UMI counts. Same release and cohort as
[scdrs-adpd-seaad](https://github.com/PhoebeGuo97/scdrs-adpd-seaad).

**Embeddings, all on the same cells**

| name | what | role |
| --- | --- | --- |
| `pca` | 2000 HVGs (seurat_v3, pooled over donors), log-normalized, scaled, 50 PCs | baseline |
| `pca_harmony` | the same PCs, Harmony-corrected for donor | primary baseline |
| `randproj` | the same HVG matrix, 50-d Gaussian random projection | floor |
| `gf_v2_104m_cls` | Geneformer V2-104M, CLS token, second-to-last layer | test |
| `gf_random_v2_104m_cls` | same architecture and tokenization, random weights | is it the pretraining? |
| `ref_scvi` | the atlas's own scVI latent | reference only: labels were assigned in this space, so it is circular for cell type |

**Task A, cell type held out by donor.** Subclass (24) and Supertype
(types with at least 100 cells in at least 20 donors). Five donor folds,
stratified by ADNC, fixed across embeddings. Linear probe (C tuned by
donor-grouped inner CV) and 15-NN. Accuracy, balanced accuracy, macro-F1, log
loss, Brier, top-label ECE. Per-donor macro-F1 compared to `pca_harmony` with a
paired Wilcoxon test, donors as the unit. Permuted-label null.

**Task B, donor ADNC.** Intermediate/High vs Not AD/Low, plus Spearman against
the 4-level stage. Features are per-subclass mean embeddings; the per-donor
cap removes composition, so this tests cell *state* only. Per-subclass models
and a combined model, each against a covariate-only model (age, sex, APOE e4,
PMI, fraction of nuclei from 10x Multiome libraries). Repeated stratified 5-fold over donors; permutation null
of the identical statistic (1000 permutations, same splits).

**Batch retention.** Within each subclass, how well a 15-NN recovers donor, and
(with donors held out) 10x Multiome vs single-nucleus chemistry.

## Leakage and fairness, stated up front

- Folds are donors, never cells. Scalers, PCA and probes are fit on training donors.
- HVG selection, PCA and Harmony see all donors (unsupervised, transductive);
  no label enters them. Geneformer is equally label-free.
- Geneformer V2 was pretrained on public single-cell corpora that may include
  SEA-AD. Pretraining is unsupervised, so this cannot leak labels, but it is
  noted.
- Harmony corrects toward donor-invariance, which is right for cell type and
  works against the donor task: some of what it removes as "donor effect" is
  pathology. So `pca_harmony` is the primary comparator for task A and
  uncorrected `pca` for task B. Both are reported for both.
- Dimensions differ (50 vs 768). The linear probe is regularized and tuned;
  the kNN probe is dimension-agnostic.

## Run

```bash
conda create -n fmeval -c conda-forge python=3.11 && conda activate fmeval
pip install -r requirements.txt
bash tests/smoke_test.sh                       # synthetic, ~1 min

# GPU steps (data + Geneformer): see aws/README.md, then on the Mac:
python scripts/02_baseline_embeddings.py
python scripts/04_eval_celltype.py
python scripts/05_eval_donor.py
python scripts/06_batch_retention.py
```
