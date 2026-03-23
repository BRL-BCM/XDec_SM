# XDec

Python implementation of XDec tumor microenvironment deconvolution algorithm.
XDec resolves bulk RNA-seq data into per-sample cell-type proportions and,
optionally, cell-type-specific gene expression profiles.

## Background

XDec extends the **EDec** alternating NMF algorithm (Boeva et al. 2019) with
single-cell RNA-seq reference integration and epithelial subtype resolution.
It was applied to TCGA BRCA in the original study.

## Requirements

```
numpy >= 1.24
pandas >= 2.0
scipy >= 1.11
quadprog >= 0.1.11   # strongly recommended; falls back to scipy SLSQP without it
```

Install:

```bash
pip install numpy pandas scipy quadprog
```

## Files

| File | Purpose |
|------|---------|
| `xdec.py` | Main tool — all three algorithm stages |
| `preprocess_scrna.py` | Helper — aggregate scRNA-seq cells into pseudo-bulk references |

---

## Quick start (TCGA BRCA / scRNA example)

### Step 1 — Build pseudo-bulk scRNA references

```bash
python preprocess_scrna.py \
    --counts  source/GSE118389_counts_rsem.txt \
    --metadata source/GSE118389_Metadata.txt \
    --outdir  results/
```

Outputs `results/scrna_refs.tsv` (genes × pseudo-bulk samples) and
`results/scrna_metadata.tsv` (Sample_ID / Cell_Type).

### Step 2 — Run XDec end-to-end

```bash
python xdec.py \
    --bulk            source/HiSeqV2_TCGA.txt  --log2-bulk \
    --refs            results/scrna_refs.tsv \
    --ref-metadata    results/scrna_metadata.tsv \
    --p-one-vs-rest   0.05  --n-one-vs-rest 50 \
    --pair-a Tcell    --pair-b Macrophage  --p-pair 0.05  --n-pair 25 \
    --skip-stability  --k 9 \
    --bulk-counts     source/HiSeqV2_TCGA.txt  --log2-counts \
    --metadata        source/BRCA_clinicalMatrix_New.txt \
    --subtype-col     PAM50Call_New \
    --outdir          results/xdec_scrna/
```

Drop `--skip-stability` and add `--k-range 3,4,5,6,7,8,9,10,11,12` to let
XDec estimate the optimal k automatically (adds ~10–30 min depending on data
size).

---

## Algorithm stages

XDec has three sequential stages, each runnable independently via subcommands
(`stage0`, `stage1`, `stage2`) or chained automatically when no subcommand is
given.

### Stage 0 — Probe selection and cell-type number estimation

Selects discriminative genes from the reference dataset using Welch t-tests
(one-vs-rest and an optional pairwise comparison), then estimates the optimal
number of cell types *k* by running stability analyses over a range of *k*
values.  Use `--append-pam50` to additionally include the PAM50 gene panel
(only appropriate for BRCA deconvolutions).

```bash
python xdec.py stage0 \
    --bulk           bulk.tsv          --log2-bulk \
    --refs           refs.tsv \
    --ref-metadata   ref_meta.tsv \
    --k-range        3,4,5,6,7,8,9,10,11,12 \
    --outdir         results/
```

**Outputs:**

| File | Description |
|------|-------------|
| `selected_probes.txt` | One informative gene name per line |
| `stability_scores.tsv` | Stability score for each tested *k* |
| `optimal_k.txt` | Recommended *k* (highest stability) |

### Stage 1 — Deconvolution

Solves **B ≈ P × Mᵀ** by alternating quadratic programming (P) and NNLS (M).
Only the Stage 0 probes are used to update P; M is estimated for all genes.

```bash
python xdec.py stage1 \
    --bulk           bulk.tsv          --log2-bulk \
    --refs           refs.tsv \
    --ref-metadata   ref_meta.tsv \
    --probes         results/selected_probes.txt \
    --k              9 \
    --outdir         results/
```

**Outputs:**

| File | Description |
|------|-------------|
| `proportions.tsv` | Samples × k proportions (rows sum to 1) |
| `cell_type_profiles.tsv` | Genes × k normalised expression profiles (0–1) |
| `correlation.tsv` | Reference cell types × estimated profiles (Spearman) — inspect to assign biological labels |

### Stage 2 — Cell-type-specific expression estimation

Given raw bulk counts and the proportion matrix from Stage 1, estimates
cell-type-specific expression means and standard errors per gene via NNLS and outputs non-normalized counts.
Can be stratified by sample group (e.g. PAM50 subtype).

```bash
python xdec.py stage2 \
    --bulk-counts    bulk.tsv          --log2-counts \
    --proportions    results/proportions.tsv \
    --label-map      labels.json \
    --metadata       clinical.tsv      --subtype-col PAM50 \
    --outdir         results/
```

**Outputs:**

| File | Description |
|------|-------------|
| `stage2_means.tsv` | Genes × (cell_type · group) expression means |
| `stage2_stderrs.tsv` | Genes × (cell_type · group) standard errors |

---

## Full argument reference

### Common arguments (all modes)

| Argument | Default | Description |
|----------|---------|-------------|
| `--outdir DIR` | `xdec_results/` | Output directory (created if absent) |
| `--max-its N` | `2000` | Maximum Stage 1 alternating iterations |
| `--rss-stop F` | `1e-10` | Stage 1 convergence threshold (RSS change) |
| `--seed N` | `12345` | Random seed for M initialisation |
| `--log-level` | `INFO` | Logging verbosity (DEBUG/INFO/WARNING/ERROR) |

### Input data

| Argument | Description |
|----------|-------------|
| `--bulk FILE` | Bulk expression matrix TSV (genes × samples) |
| `--log2-bulk` | Invert log₂(x+1) transform on bulk before normalisation |
| `--refs FILE` | Reference matrix TSV (genes × reference samples) |
| `--log2-refs` | Invert log₂(x+1) on references |
| `--ref-metadata FILE` | Reference sample metadata TSV; must contain columns named by `--sample-col` and `--class-col` |
| `--sample-col COL` | Metadata column for sample IDs (default: `Sample_ID`) |
| `--class-col COL` | Metadata column for cell-type labels (default: `Cell_Type`) |


## `preprocess_scrna.py` arguments

Aggregates raw scRNA-seq count data into pseudo-bulk references for use with
`--refs` / `--ref-metadata`.

```bash
python preprocess_scrna.py \
    --counts    GSE118389_counts_rsem.txt \
    --metadata  GSE118389_Metadata.txt \
    --outdir    results/
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--counts FILE` | required | Raw count matrix TSV (genes × cells) |
| `--metadata FILE` | required | Cell metadata TSV with `Cell_Type` and `Cell_Subtype` columns |
| `--sample-col COL` | `Sample` | Metadata column containing cell IDs |
| `--type-col COL` | `Cell_Type` | Metadata column for broad cell-type labels |
| `--min-lib-size N` | `100000` | Minimum library size to retain a cell |
| `--outdir DIR` | `.` | Output directory |

**Outputs:** `scrna_refs.tsv` and `scrna_metadata.tsv` (ready for `--refs` / `--ref-metadata`).

---

## Input file formats

All matrices are **tab-separated**, with the **first column** as the row index
(gene names) and the **first row** as the header (sample / cell IDs).

**Bulk matrix example** (`bulk.tsv`):

```
Gene    Sample_1    Sample_2    Sample_3
BRCA1   12.3        8.7         15.1
TP53    5.2         6.0         4.8
```

**Reference metadata example** (`ref_meta.tsv`):

```
Sample_ID       Cell_Type
Epithelial.P1   Epithelial
Epithelial.P2   Epithelial
Stroma.P1       Stroma
Tcell.P1        Tcell
```

**Label map example** (`labels.json`):

```json
{
    "Profile_1": "Epithelial.Basal",
    "Profile_2": "T-Cell",
    "Profile_3": "Macrophage"
}
```

---

## Citation

If you use XDec, please cite:
> Deconvolution of cancer cell states by the XDec-SM method, Oscar D. Murillo,Varduhi Petrosyan, Emily L. LaPlante, Lacey E. Dobrolecki, Michael T. Lewis,Aleksandar Milosavljevic, Plos Computational Biology, 2023.


