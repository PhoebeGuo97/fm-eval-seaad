# Do single-cell foundation model embeddings beat PCA on AD cortex?

A donor-stratified benchmark of zero-shot Geneformer embeddings against
classical dimensionality reduction on the SEA-AD middle temporal gyrus atlas,
on two tasks with different difficulty: cell type identity, and donor-level
Alzheimer's neuropathologic change.

**Result.** Split by task, and neither half is a clean foundation model win.

- **Cell type: no advantage once dimensions are matched.** Geneformer's 768-d
  CLS embedding beats 50 PCs on Supertype with a linear probe (macro-F1 0.851
  vs 0.801), but cut to 50 PCs it ties PCA (0.800, paired p 0.15), and PCA
  given 256 components closes half the gap (0.825). With kNN Geneformer is
  worse at every size (0.699 vs 0.743). The information is linearly
  recoverable; the neighborhood geometry is not better.
- **Donor pathology: Geneformer is best, but it is also the best depth
  meter.** It predicts ADNC Intermediate/High at AUROC 0.740 against 0.630
  for PCA and 0.678 for 256 PCs, and the gain survives dimension matching
  (0.751 at 50 PCs) and adjustment for donor-level depth and quality metrics
  (conditional permutation p 0.006). But Geneformer reconstructs each donor's
  median genes detected almost exactly (cross-validated R^2 0.955 vs 0.751 for
  PCA), and in this cohort depth itself tracks pathology: AD donors detect
  fewer genes (median 4,815 vs 5,485) and have more mitochondrial reads, and
  those metrics alone reach AUROC 0.727. Whether lower detection in AD tissue
  is technical or biological cannot be separated here.
- **Pretraining is what carries the signal.** The same architecture with
  random weights, fed identical tokens, falls to 0.463 on Supertype and adds
  nothing significant to ADNC beyond covariates (p 0.07 without, 0.11 with
  technical adjustment).

Kedzierska et al. 2025 (Genome Biology) found zero-shot Geneformer and scGPT
generally trail simple baselines; the cell type half agrees with them.

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

## Results

84 donors, 55,461 nuclei. All numbers are out-of-fold with donors held out.

### Cell type (macro-F1, logistic probe / 15-NN)

| embedding | dims | Subclass (24) | Supertype (106) |
| --- | --- | --- | --- |
| pca | 50 | 0.992 / 0.993 | 0.801 / 0.743 |
| pca256 | 256 | 0.994 / 0.962 | 0.825 / 0.724 |
| pca_harmony | 50 | 0.990 / 0.991 | 0.755 / 0.707 |
| Geneformer V2-104M | 768 | 0.993 / 0.990 | **0.851** / 0.699 |
| Geneformer, 50 PCs | 50 | 0.991 / 0.989 | 0.800 / 0.685 |
| Geneformer, random weights | 768 | 0.906 / 0.485 | 0.463 / 0.116 |
| random projection | 50 | 0.901 / 0.868 | 0.607 / 0.454 |
| scVI (circular reference) | 20 | 0.993 / 0.993 | 0.831 / 0.782 |
| permuted labels | | 0.03 to 0.04 | 0.003 to 0.009 |

Subclass is saturated for every real embedding. Harmony costs 4.6 points of
Supertype F1 against uncorrected PCA (73 of 84 donors worse), so PCA, not
PCA plus Harmony, is the baseline to beat. Top-label ECE is at most 0.007 on
Subclass and 0.008 to 0.041 on Supertype.

### Donor ADNC (Intermediate/High vs Not AD/Low, 63 vs 21 donors)

AUROC averaged over 5 repeats of 5-fold CV; p from 1,000 label permutations
on the identical statistic. "Beyond covariates" shuffles embedding rows across
donors while labels and covariates stay fixed.

| model | AUROC | perm p | beyond covariates p | Brier |
| --- | --- | --- | --- | --- |
| covariates (age, sex, APOE e4, PMI, % Multiome) | 0.623 | 0.041 | | 0.181 |
| technical metrics only | 0.727 | 0.002 | | |
| pca | 0.630 | 0.037 | 0.036 | 0.201 |
| pca256 | 0.678 | 0.009 | 0.012 | 0.179 |
| Geneformer | **0.740** | 0.002 | 0.002 | 0.165 |
| Geneformer, 50 PCs | 0.751 | 0.001 | 0.004 | 0.157 |
| Geneformer, random weights | 0.590 | 0.090 | 0.074 | 0.190 |
| scVI | 0.701 | 0.005 | 0.028 | 0.181 |

Adding donor medians of genes detected, log UMIs, mitochondrial fraction and
doublet score to the covariates (script 08): Geneformer AUROC 0.818 vs 0.608
with its embedding shuffled (p 0.006); PCA p 0.050, PCA plus Harmony p 0.082,
random weights p 0.112.

Per subclass, 17 of 24 Geneformer models pass q < 0.05, and so do 15 of 24
for the random-weight model, against 0 for PCA. A network with no training
separating pathology in most cell types is the clearest sign that the rank
encoding itself transmits donor-level technical differences.

### What each embedding retains

| embedding | donor recovery within subclass (x chance) | 10x chemistry, donors held out (balanced acc.) | donor median genes detected (CV R^2) |
| --- | --- | --- | --- |
| Geneformer | 14.0 | 0.608 | 0.955 |
| pca | 12.6 | 0.500 | 0.751 |
| scVI | 7.5 | 0.502 | |
| Geneformer, random weights | 2.1 | 0.500 | 0.660 |
| pca_harmony | 1.3 | 0.500 | 0.548 |

Geneformer is the only embedding that identifies library chemistry in unseen
donors.

## Limitations

- Zero-shot only; no fine-tuning, and one foundation model.
- 54.5% of nuclei exceed Geneformer's 4,096-token input and are truncated to
  their top-ranked genes.
- Cells were capped at 30 per donor per subclass, so composition shifts, the
  main SEA-AD pathology signal, are removed by design.
- Technical adjustment uses donor-level medians. It does not rule out depth
  effects within cell types. The decisive test, downsampling every nucleus to
  a common depth before tokenization, needs another GPU run.
- 21 non-AD donors; AUROC intervals across CV repeats are about +/- 0.04.
- Geneformer's pretraining corpus may include SEA-AD (unsupervised, so no
  label leakage).

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
python scripts/07_dimension_matched.py         # then 04 and 05 with FMEVAL_RESULTS=results_dim
python scripts/08_technical_confounds.py --embeddings pca pca_harmony gf_v2_104m_cls gf_random_v2_104m_cls
```
