#!/usr/bin/env python3
"""
xdec.py — XDec: Extended Deconvolution for Tumour Microenvironment Analysis
============================================================================

A Python port of the XDec algorithm (XDec_total_code_vpetrosyan.Rmd).

XDec deconvolves bulk RNA-seq (or DNA methylation) data into per-sample
cell-type proportions and, optionally, cell-type-specific gene expression
profiles.  It extends the EDec algorithm (Boeva et al.) with single-cell
RNA-seq reference integration and epithelial subtype resolution.

ALGORITHM OVERVIEW
------------------
The tool has three sequential stages.

  Stage 0 – Probe/gene selection & cell-type number estimation
      Selects discriminative genes from a reference dataset using Welch
      t-tests (one-vs-rest and optional pairwise comparisons), appends the
      PAM50 gene list, then estimates the optimal number of cell types (k)
      by running stability analyses over a range of k values.

  Stage 1 – Deconvolution: proportions + normalised expression profiles
      Solves a constrained Non-negative Matrix Factorisation (NMF) problem:

          B ≈ P × M^T

      where B (genes × samples) is the normalised bulk matrix, P (samples × k)
      are the cell-type proportions (row-stochastic, ≥ 0), and M (genes × k)
      are the normalised per-cell-type expression profiles (≥ 0).  Solved by
      alternating quadratic programming (QP) for P and non-negative least
      squares (NNLS) for M.  Only the informative probes from Stage 0 are
      used to update P; M is estimated for all genes.

  Stage 2 – Cell-type-specific gene expression estimation
      Given bulk RNA-seq counts B_raw (genes × samples) and the proportion
      matrix P from Stage 1, estimates cell-type-specific expression means
      and standard errors by solving an NNLS problem per gene:

          minimise ‖B_raw[g, :] − P @ m_g‖²   subject to  m_g ≥ 0

      Typically run separately for each sample group (e.g. PAM50 subtype).

INPUT MODES
-----------
  With scRNA / cell-line references (Mode A, recommended):
      References used only for joint quantile normalisation (bulk + refs) and
      for post-hoc Spearman correlation to assign cell-type labels to profiles.
      Stage 1 runs on bulk data alone.

  With cell-line references as anchors (Mode B, --anchor-refs flag):
      Reference samples are appended to the bulk matrix before Stage 1 so
      they act as NMF anchors, guiding the factorisation directly.

  Without references (Mode C, blind):
      Supply --probes (custom gene list) and --k.  Profile labels must be
      assigned manually by the user from correlation.tsv.

  informative_loci optional:
      If --probes is not provided, use --use-all-genes or run Stage 0 first.
      The PAM50 gene list is always appended when probes are selected.

USAGE
-----
  # Run all stages end-to-end (default when no subcommand is given)
  python xdec.py --bulk bulk.tsv --refs refs.tsv \\
                 --ref-metadata meta.tsv --log2-bulk \\
                 --bulk-counts bulk.tsv --log2-counts \\
                 --outdir results/

  # Stage 0: select informative probes, estimate optimal k
  python xdec.py stage0 --bulk bulk.tsv --refs refs.tsv \\
                        --ref-metadata meta.tsv --log2-bulk \\
                        --outdir results/

  # Stage 1: deconvolution
  python xdec.py stage1 --bulk bulk.tsv --refs refs.tsv \\
                        --ref-metadata meta.tsv --log2-bulk \\
                        --probes results/selected_probes.txt --k 9 \\
                        --outdir results/

  # Stage 2: cell-type expression estimation per PAM50 subtype
  python xdec.py stage2 --bulk-counts bulk.tsv --log2-counts \\
                        --proportions results/proportions.tsv \\
                        --label-map labels.json \\
                        --metadata clinical.tsv --subtype-col PAM50 \\
                        --outdir results/

OUTPUTS
-------
  Stage 0:  selected_probes.txt   — one gene name per line
            stability_scores.tsv  — k → stability score
            optimal_k.txt         — recommended k (highest stability)

  Stage 1:  proportions.tsv            — samples × k  (values sum to 1 per row)
            cell_type_profiles.tsv     — genes × k    (normalised, values 0–1)
            correlation.tsv            — reference cell types × estimated profiles
                                         (Spearman; inspect to assign labels)

  Stage 2:  stage2_means.tsv    — genes × (cell_type_group)  expression means
            stage2_stderrs.tsv  — genes × (cell_type_group)  standard errors

DEPENDENCIES
------------
  numpy >= 1.24
  pandas >= 2.0
  scipy >= 1.11
  quadprog >= 0.1.11   (optional but strongly recommended for Stage 1 speed)

REFERENCES
----------
  EDec: Boeva et al., Bioinformatics 2019.
  XDec: Petrosyan et al. (applied to TCGA BRCA; code: XDec_total_code_vpetrosyan.Rmd).
"""

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

# Optional: quadprog provides the same Fortran QP routine as R's quadprog package.
# Without it, a scipy SLSQP fallback is used (correct but slower).
try:
    import quadprog
    _HAS_QUADPROG = True
except ImportError:
    _HAS_QUADPROG = False
    warnings.warn(
        "quadprog not installed.  Stage 1 will use a scipy SLSQP fallback "
        "(correct but slower).  Install with:  pip install quadprog",
        stacklevel=1,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("xdec")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PAM50 gene panel — always appended to the probe list in the R code.
# Source: XDec_total_code_vpetrosyan.Rmd, variable `Pam50.Genes`.
PAM50_GENES: list[str] = [
    "UBE2T", "BIRC5", "NUF2", "CDC6", "CCNB1", "TYMS", "MYBL2", "CEP55",
    "MELK", "NDC80", "RRM2", "UBE2C", "CENPF", "PTTG1", "EXO1", "ANLN",
    "CCNE1", "CDC20", "MKI67", "KIF2C", "ACTR3B", "MYC", "EGFR", "KRT5",
    "PHGDH", "CDH3", "MIA", "KRT17", "FOXC1", "SFRP1", "KRT14", "ESR1",
    "SLC39A6", "BAG1", "MAPT", "PGR", "CXXC5", "MLPH", "BCL2", "MDM2",
    "NAT1", "FOXA1", "BLVRA", "MMP11", "GPR160", "FGFR4", "GRB7",
    "TMEM45B", "ERBB2",
]

# ===========================================================================
# SECTION 1 — Data loading and preprocessing
# ===========================================================================

def load_matrix(path: str, sep: str = "\t") -> pd.DataFrame:
    """Load a gene × sample matrix from a delimited text file.

    Parameters
    ----------
    path : str
        Path to the file.  The first column is used as the row index (gene or
        probe names); the first row is the header (sample IDs).
    sep : str
        Column delimiter (default: tab).

    Returns
    -------
    pd.DataFrame
        Gene × sample matrix with float64 values.
        Rows where the sum across all samples is zero are removed, matching
        the R filter ``df = df[rowSums(df) > 0, ]``.
    """
    df = pd.read_csv(path, sep=sep, index_col=0, header=0)
    df = df.astype(float)
    df = df.loc[df.sum(axis=1) > 0]
    log.info("Loaded %s: %d genes × %d samples", path, *df.shape)
    return df


def inverse_log2_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Invert a log2(x + 1) transform: returns (2 ** df) − 1.

    The TCGA HiSeqV2 bulk RNA-seq file stores values as log2(RPKM + 1).
    XDec operates on the original count/RPKM scale, so this inversion must
    be applied before normalisation.

    R equivalent::

        TCGA.Beta = (2^TCGA.Beta) - 1

    Parameters
    ----------
    df : pd.DataFrame
        Gene × sample matrix in log2(x + 1) space.

    Returns
    -------
    pd.DataFrame
        Gene × sample matrix in count/RPKM space (values ≥ 0).
    """
    return (2.0 ** df) - 1.0


def quantile_normalise(samples_x_genes: pd.DataFrame) -> pd.DataFrame:
    """Quantile-normalise a samples × genes matrix column-wise.

    After normalisation every sample has the same marginal distribution.

    Ports the R helper function from XDec_total_code_vpetrosyan.Rmd::

        quantile_normalisation = function(df){
          df_rank   <- apply(df, 2, rank, ties.method="min")
          df_sorted <- data.frame(apply(df, 2, sort))
          df_mean   <- apply(df_sorted, 1, mean)
          df_final  <- apply(df_rank, 2, index_to_mean, my_mean=df_mean)
          rownames(df_final) <- rownames(df)
          return(df_final)
        }

    Note: the R code transposes the bulk matrix before calling this function,
    so the input convention here is **samples × genes** (rows = samples,
    columns = genes).  Ranking and sorting operate within each column (gene)
    across rows (samples).

    Parameters
    ----------
    samples_x_genes : pd.DataFrame
        Matrix with shape (n_samples, n_genes).

    Returns
    -------
    pd.DataFrame
        Quantile-normalised matrix of the same shape and index.
    """
    values = samples_x_genes.values.astype(float)   # (n_samples, n_genes)

    # Rank each gene column using min ties, 1-based (matches R ties.method="min")
    ranks = np.apply_along_axis(
        lambda col: stats.rankdata(col, method="min"),
        axis=0,
        arr=values,
    ).astype(int)                                    # (n_samples, n_genes)

    # Sort each gene column; compute the target mean per rank position
    sorted_vals = np.sort(values, axis=0)            # (n_samples, n_genes)
    row_means   = sorted_vals.mean(axis=1)           # (n_samples,) — one mean per rank

    # Map every rank to its corresponding mean value (ranks are 1-based)
    result = row_means[ranks - 1]                    # (n_samples, n_genes)

    return pd.DataFrame(
        result,
        index=samples_x_genes.index,
        columns=samples_x_genes.columns,
    )


def logistic_a100(col: np.ndarray) -> np.ndarray:
    """Apply the XDec logistic transform to a 1-D array of gene values.

    Uses the 100th percentile (maximum) of the array as the scale parameter::

        a = 1 / max(x)
        transform(x) = 1 − exp(−a · x)

    Ports the R function::

        logistic.a100 = function(x){
          a = 1/quantile(x, 1.00)
          1 - exp(1)^(-a*x)
        }

    In the R code this is applied per gene across all samples (column of a
    samples × genes matrix).  The transform maps values into [0, 1).

    Parameters
    ----------
    col : np.ndarray
        1-D array of non-negative values for a single gene.

    Returns
    -------
    np.ndarray
        Transformed values in [0, 1).
    """
    q100 = float(np.quantile(col, 1.0))   # = maximum
    if q100 == 0.0:
        return np.zeros_like(col, dtype=float)
    a = 1.0 / q100
    return 1.0 - np.exp(-a * col)


def normalise_bulk(
    bulk_raw: pd.DataFrame,
    refs_raw: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Apply the full XDec normalisation pipeline to a bulk expression matrix.

    Steps (matching XDec_total_code_vpetrosyan.Rmd):

    1. Find shared genes between ``bulk_raw`` and ``refs_raw`` (if provided).
    2. Merge them column-wise to create a joint matrix for quantile
       normalisation.  This ensures that the bulk and reference samples share
       the same marginal distribution — critical for the subsequent
       cell-type assignment via Spearman correlation.
    3. Quantile-normalise the joint (samples × genes) matrix.
    4. Apply ``logistic_a100`` independently to each gene column (i.e. across
       all samples), compressing values into [0, 1).
    5. Remove genes with zero sum across all samples.
    6. Scale the entire matrix by its global maximum → values in [0, 1].
    7. Filter genes whose mean expression in the *bulk portion only* is < 0.01
       (matches ``TCGA.Expressed = TCGA.Expressed[average >= 0.01]``).
    8. Return only the bulk columns.

    Parameters
    ----------
    bulk_raw : pd.DataFrame
        Gene × sample bulk expression matrix (NOT log-transformed).
    refs_raw : pd.DataFrame, optional
        Gene × sample reference matrix (same feature type as bulk).  If
        provided, quantile normalisation is performed jointly on bulk + refs.
        If None, only the bulk matrix is normalised.

    Returns
    -------
    pd.DataFrame
        Normalised bulk matrix (genes × samples), values in [0, 1].
    """
    bulk_cols = list(bulk_raw.columns)

    if refs_raw is not None:
        shared = bulk_raw.index.intersection(refs_raw.index)
        if len(shared) == 0:
            raise ValueError("Bulk and reference matrices share no gene names.")
        log.info(
            "Joint normalisation over %d shared genes (bulk + refs)", len(shared)
        )
        combined = pd.concat(
            [bulk_raw.loc[shared], refs_raw.loc[shared]], axis=1
        )
    else:
        combined = bulk_raw.copy()

    # -- Step 3: quantile normalise (input convention: samples × genes) ------
    combined_t = combined.T                          # (n_samples_total, n_genes)
    qn = quantile_normalise(combined_t)              # (n_samples_total, n_genes)

    # -- Step 4: logistic_a100 per gene column --------------------------------
    logistic_vals = np.apply_along_axis(
        logistic_a100, axis=0, arr=qn.values
    )                                                # (n_samples_total, n_genes)

    # -- Step 5: remove all-zero genes ----------------------------------------
    result = pd.DataFrame(
        logistic_vals.T,                             # (n_genes, n_samples_total)
        index=combined.index,
        columns=combined.columns,
    )
    result = result.loc[result.sum(axis=1) > 0]

    # -- Step 6: global max scaling → [0, 1] ----------------------------------
    gmax = result.values.max()
    if gmax > 0:
        result = result / gmax

    # -- Step 7: filter genes with mean < 0.01 in bulk portion only ----------
    bulk_means = result[bulk_cols].mean(axis=1)
    expressed  = bulk_means[bulk_means >= 0.01].index
    result     = result.loc[expressed]

    log.info(
        "After normalisation: %d expressed genes × %d bulk samples",
        len(expressed), len(bulk_cols),
    )

    # -- Step 8: return only bulk columns -------------------------------------
    return result[bulk_cols]


def normalise_refs_independent(refs_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise reference samples independently (without joint QN with bulk).

    Used in the scRNA approach where references are treated separately from
    bulk for normalisation purposes.  Steps:

    1. Apply ``logistic_a100`` per gene across all reference samples.
    2. Remove genes with zero sum.
    3. Scale by global maximum → [0, 1].

    R equivalent (applied to ``GSE118389.CombinedSums.Norm``)::

        input.SC.trans.QN.logistic = t(apply(t(GSE118389.CombinedSums.Norm), 2, logistic.a100))
        ...
        input.SC.trans.QN.logistic.nonZero.max = (1/max(...)) * ...

    Parameters
    ----------
    refs_raw : pd.DataFrame
        Gene × sample reference matrix (non-negative, library-size normalised
        if scRNA, or beta values if methylation).

    Returns
    -------
    pd.DataFrame
        Normalised reference matrix, values in [0, 1].
    """
    # Apply logistic_a100 per gene (axis=0 over samples × genes transpose)
    vals_t = refs_raw.T.values                       # (n_samples, n_genes)
    logistic_vals = np.apply_along_axis(
        logistic_a100, axis=0, arr=vals_t
    )                                                # (n_samples, n_genes)
    result = pd.DataFrame(
        logistic_vals.T,
        index=refs_raw.index,
        columns=refs_raw.columns,
    )
    result = result.loc[result.sum(axis=1) > 0]
    gmax = result.values.max()
    if gmax > 0:
        result = result / gmax
    return result


# ===========================================================================
# SECTION 2 — Stage 0: probe / gene selection via t-tests
# ===========================================================================

def _t_test_one_vs_rest(
    data: pd.DataFrame,
    classes: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch t-test for one class against all others.

    Parameters
    ----------
    data : pd.DataFrame
        Genes × samples normalised reference matrix.
    classes : np.ndarray
        Cell-type label per sample (aligned with ``data.columns``).
    label : str
        The class to test.

    Returns
    -------
    p_values : np.ndarray, shape (n_genes,)
        Two-sided p-values (Welch's t-test, ``equal_var=False``, matching
        R's ``t.test(..., var.equal=FALSE)``).
    mean_diff : np.ndarray, shape (n_genes,)
        mean(class) − mean(rest) per gene.
    """
    mask = classes == label
    grp  = data.values[:, mask]
    rest = data.values[:, ~mask]
    _, p_vals  = stats.ttest_ind(grp, rest, axis=1, equal_var=False)
    mean_diff  = grp.mean(axis=1) - rest.mean(axis=1)
    return p_vals, mean_diff


def select_probes(
    ref_norm: pd.DataFrame,
    class_vector: np.ndarray,
    one_vs_rest_p: float = 1e-4,
    one_vs_rest_n: int   = 25,
    pair_class_a: Optional[str] = None,
    pair_class_b: Optional[str] = None,
    pair_p: float = 1e-5,
    pair_n: int   = 75,
    append_pam50: bool = True,
) -> list[str]:
    """Select informative genes for deconvolution (Stage 0 probe selection).

    Algorithm (from XDec_total_code_vpetrosyan.Rmd):

    1. **One-vs-rest t-tests** across all cell-type classes:
       For each class C, run a Welch t-test (class C vs all others).
       From the genes that pass ``one_vs_rest_p``, take the top-
       ``one_vs_rest_n`` and bottom-``one_vs_rest_n`` by mean difference.
       Take the union across all classes (preserving first occurrence).

    2. **Optional pairwise t-test** between ``pair_class_a`` and
       ``pair_class_b`` (used in the R code for epithelial vs CAF):
       Take the top-``pair_n`` and bottom-``pair_n`` genes passing
       ``pair_p``.  Union with the one-vs-rest result.

    3. **Append PAM50 genes** (if ``append_pam50`` is True), matching::

           chosenProbes.GEO.7.PAM50 = c(chosenProbes.GEO.7, Pam50.Genes)

    Parameters
    ----------
    ref_norm : pd.DataFrame
        Genes × samples normalised reference matrix.
    class_vector : np.ndarray
        Cell-type label per reference sample (aligned with ``ref_norm.columns``).
    one_vs_rest_p : float
        p-value threshold for one-vs-rest tests.
        R default: 1e-4 (methylation cell-line model), 0.05 (scRNA model).
    one_vs_rest_n : int
        Top/bottom genes per class direction (default 25).
    pair_class_a, pair_class_b : str, optional
        If both are given, run an additional pairwise test between these two
        classes.  In the R code this is used for 'epithelial' vs 'CAF'.
    pair_p : float
        p-value threshold for the pairwise test (default 1e-5).
    pair_n : int
        Top/bottom genes for the pairwise test (default 75).
    append_pam50 : bool
        Append PAM50_GENES to the final list (default True).

    Returns
    -------
    list[str]
        Ordered, deduplicated list of selected gene/probe names.
    """
    gene_names = np.array(ref_norm.index)
    selected: list[str] = []

    # ---- One-vs-rest --------------------------------------------------------
    for label in np.unique(class_vector):
        p_vals, mean_diff = _t_test_one_vs_rest(ref_norm, class_vector, label)
        sig = p_vals < one_vs_rest_p
        if sig.sum() == 0:
            continue
        diff_sig  = mean_diff[sig]
        genes_sig = gene_names[sig]
        order     = np.argsort(diff_sig)
        selected.extend(genes_sig[order[-one_vs_rest_n:]].tolist())   # high
        selected.extend(genes_sig[order[:one_vs_rest_n]].tolist())    # low

    log.info("One-vs-rest: %d genes (before dedup)", len(selected))

    # ---- Optional pairwise --------------------------------------------------
    if pair_class_a and pair_class_b:
        mask_a = class_vector == pair_class_a
        mask_b = class_vector == pair_class_b
        _, p_vals = stats.ttest_ind(
            ref_norm.values[:, mask_a],
            ref_norm.values[:, mask_b],
            axis=1, equal_var=False,
        )
        mean_diff = (
            ref_norm.values[:, mask_a].mean(axis=1)
            - ref_norm.values[:, mask_b].mean(axis=1)
        )
        sig = p_vals < pair_p
        if sig.sum() > 0:
            diff_sig  = mean_diff[sig]
            genes_sig = gene_names[sig]
            order     = np.argsort(diff_sig)
            selected.extend(genes_sig[order[-pair_n:]].tolist())
            selected.extend(genes_sig[order[:pair_n]].tolist())
            log.info(
                "Pairwise (%s vs %s): added more genes (before dedup)",
                pair_class_a, pair_class_b,
            )

    # ---- Deduplicate (preserve first occurrence, matching R's !duplicated) --
    seen: set[str] = set()
    unique: list[str] = []
    for g in selected:
        if g not in seen:
            seen.add(g)
            unique.append(g)

    # ---- Append PAM50 -------------------------------------------------------
    if append_pam50:
        for g in PAM50_GENES:
            if g not in seen:
                seen.add(g)
                unique.append(g)

    log.info("Total selected probes (with PAM50): %d", len(unique))
    return unique


# ===========================================================================
# SECTION 3 — Stage 0: stability estimation
# ===========================================================================

def _stage1_core_inner(
    B_full: np.ndarray,
    probe_idx: np.ndarray,
    k: int,
    max_its: int,
    rss_diff_stop: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner NMF alternating optimisation on numpy arrays (no pandas overhead).

    Separated from ``run_stage1`` so it can be called efficiently from the
    stability loop.

    Parameters
    ----------
    B_full : np.ndarray, shape (n_genes, n_samples)
        Normalised bulk matrix.
    probe_idx : np.ndarray of int
        Row indices of informative probes within B_full.
    k : int
        Number of cell types.
    max_its, rss_diff_stop
        Convergence parameters.
    rng : np.random.Generator
        Seeded generator for M initialisation.

    Returns
    -------
    P : np.ndarray, shape (n_samples, k)
    M : np.ndarray, shape (n_genes, k)
    """
    n_genes, n_samples = B_full.shape
    B_inf = B_full[probe_idx, :]                    # (n_inf, n_samples)

    # Initialise M with uniform random values in (0, 1)
    M = rng.uniform(0.0, 1.0, size=(n_genes, k))

    rss_prev = np.inf

    # Precompute QP constraint constants — identical for every iteration and sample.
    eps_I   = np.eye(k) * 1e-10
    C       = np.zeros((k, k + 1))
    C[:, 0] = 1.0
    np.fill_diagonal(C[:, 1:], 1.0)
    b_c     = np.zeros(k + 1)
    b_c[0]  = 1.0

    for _ in range(max_its):
        M_inf = M[probe_idx, :]                     # (n_inf, k) — current profiles

        # -- Update P --------------------------------------------------------
        # G is the same for every sample in this iteration; compute it once.
        G     = M_inf.T @ M_inf + eps_I             # (k, k)
        # All 'a' vectors via one matrix multiply: A_all[:, s] = M_inf.T @ B_inf[:, s]
        A_all = M_inf.T @ B_inf                     # (k, n_samples)

        P = np.empty((n_samples, k))
        if _HAS_QUADPROG:
            for s in range(n_samples):
                try:
                    p = quadprog.solve_qp(G, A_all[:, s], C, b_c, meq=1)[0]
                    p = np.clip(p, 0.0, None)
                    total = p.sum()
                    P[s]  = p / total if total > 0 else np.ones(k) / k
                except Exception:
                    P[s] = np.ones(k) / k
        else:
            for s in range(n_samples):
                P[s] = _solve_proportions_scipy(M_inf, B_inf[:, s])

        # -- Update M: vectorised CD-NNLS (warm-started from previous M) ----
        # _nnls_batch_cd operates on all genes simultaneously via precomputed
        # normal-equation matrices P^T P and B_full @ P.  Warm-starting from
        # the previous iteration's M typically causes convergence in 1–2 CD
        # passes, making this essentially two BLAS DGEMM calls per outer iter.
        M = _nnls_batch_cd(P, B_full, M_init=M)

        # -- Convergence on informative loci (after both P and M updates) ----
        M_inf_new = M[probe_idx, :]
        residuals = B_inf - M_inf_new @ P.T
        rss = float((residuals ** 2).sum())
        if abs(rss_prev - rss) < rss_diff_stop:
            break
        rss_prev = rss

    return P, M


def estimate_stability(
    bulk_norm: pd.DataFrame,
    probes: list[str],
    k_range: list[int],
    subset_prop: float   = 0.8,
    num_subsets: int     = 3,
    reps_per_subset: int = 3,
    max_its: int         = 2000,
    rss_diff_stop: float = 1e-10,
    seed: int            = 12345,
) -> dict[int, float]:
    """Estimate NMF stability over a range of k values.

    Ports ``EDec::estimate_stability``.  For each k, runs Stage 1 multiple
    times on random sample subsets and measures how consistently it finds the
    same cell-type profiles.  The k with the highest average stability score
    is recommended as the optimal number of cell types.

    Stability is measured as the mean pairwise Spearman correlation between
    corresponding estimated profiles across repetitions within each subset.

    Parameters
    ----------
    bulk_norm : pd.DataFrame
        Normalised bulk matrix (genes × samples).
    probes : list[str]
        Informative probe names (output of ``select_probes``).
    k_range : list[int]
        k values to test, e.g. ``list(range(3, 13))``.
    subset_prop : float
        Fraction of samples per random subset (default 0.8).
    num_subsets : int
        Number of independent random subsets per k (default 3).
    reps_per_subset : int
        Stage 1 repetitions per subset (default 3).
    max_its, rss_diff_stop
        Stage 1 convergence parameters.
    seed : int
        Master random seed (default 12345, matching ``set.seed(12345)`` in R).

    Returns
    -------
    dict[int, float]
        Mapping k → mean stability score (0–1; higher = more stable).
    """
    B_full    = bulk_norm.values.astype(float)
    gene_idx  = {g: i for i, g in enumerate(bulk_norm.index)}
    probe_idx = np.array([gene_idx[p] for p in probes if p in gene_idx], dtype=int)

    n_samples = B_full.shape[1]
    rng       = np.random.default_rng(seed)
    scores: dict[int, float] = {}

    for k in k_range:
        log.info("Stability: testing k=%d …", k)
        n_sub = max(k + 1, int(n_samples * subset_prop))
        corrs: list[float] = []

        for _ in range(num_subsets):
            col_idx = rng.choice(n_samples, size=n_sub, replace=False)
            B_sub   = B_full[:, col_idx]
            # Restrict probe_idx to genes present in the subset (already the
            # same rows, just restrict columns via B_sub)
            profiles_list: list[np.ndarray] = []

            for _ in range(reps_per_subset):
                _, M = _stage1_core_inner(
                    B_sub, probe_idx, k, max_its, rss_diff_stop, rng
                )
                profiles_list.append(M[probe_idx, :])   # (n_inf, k)

            # Average Spearman correlation between pairs of runs
            for i in range(len(profiles_list)):
                for j in range(i + 1, len(profiles_list)):
                    m1, m2 = profiles_list[i], profiles_list[j]
                    # spearmanr on columns: each column is a "variable"
                    r = stats.spearmanr(m1, m2)
                    mat = r.statistic if hasattr(r, "statistic") else r[0]
                    if np.ndim(mat) == 0:
                        corrs.append(float(abs(mat)))
                    else:
                        mat = np.atleast_2d(mat)
                        # Cross-correlation block: profiles of m1 vs profiles of m2
                        cross = mat[:k, k:]            # (k, k)
                        corrs.append(float(cross.max(axis=1).mean()))

        scores[k] = float(np.mean(corrs)) if corrs else 0.0
        log.info("  k=%2d  stability=%.4f", k, scores[k])

    return scores


# ===========================================================================
# SECTION 4 — Stage 1: NMF deconvolution
# ===========================================================================

def _nnls_batch_cd(
    P: np.ndarray,
    B: np.ndarray,
    M_init: Optional[np.ndarray] = None,
    max_iter: int = 2000,
    tol: float = 1e-12,
) -> np.ndarray:
    """Vectorized coordinate-descent NNLS for all genes simultaneously.

    Solves::

        minimise  ‖P @ M.T − B.T‖_F²    subject to  M ≥ 0

    by cycling through the k components of M and applying the optimal
    coordinate update (projected gradient) to all genes at once.

    Using the precomputed normal-equation matrices this is fully vectorised
    with no Python loop over genes:

        P^T P @ M.T = P^T B.T   (normal equations, shared for all genes)
        grad_M[:,j] = M @ PtP[:,j] − BtP[:,j]         (O(k × n_genes))
        M[:,j]  ←  max(M[:,j] − grad / PtP[j,j],  0)  (O(n_genes))

    When ``M_init`` is the result from the previous outer iteration, the
    warm start means convergence typically requires only 1–2 CD passes.

    Parameters
    ----------
    P : np.ndarray, shape (n_samples, k)
        Cell-type proportion matrix.
    B : np.ndarray, shape (n_genes, n_samples)
        Bulk expression matrix (n_samples = columns).
    M_init : np.ndarray, shape (n_genes, k), optional
        Warm-start initialisation.  If None, use zero initialisation.
    max_iter : int
        Maximum coordinate-descent outer iterations (default 2000).
    tol : float
        Convergence tolerance on max absolute change across all M entries
        in a single CD pass (default 1e-12).

    Returns
    -------
    M : np.ndarray, shape (n_genes, k)
        Non-negative gene × cell-type expression profiles.
    """
    k = P.shape[1]
    PtP = P.T @ P                               # (k, k)
    BtP = B @ P                                 # (n_genes, k) — one BLAS call
    diag_PtP = np.maximum(np.diag(PtP), 1e-10) # step sizes per component

    M = np.ascontiguousarray(M_init) if M_init is not None else np.zeros(
        (B.shape[0], k), dtype=float
    )

    for _ in range(max_iter):
        max_change = 0.0
        for j in range(k):
            # Gradient w.r.t. M[:,j]: M @ PtP[:,j] − BtP[:,j]
            grad_j  = M @ PtP[:, j] - BtP[:, j]       # (n_genes,)
            new_col = np.maximum(M[:, j] - grad_j / diag_PtP[j], 0.0)
            diff = np.max(np.abs(new_col - M[:, j]))
            if diff > max_change:
                max_change = diff
            M[:, j] = new_col
        if max_change < tol:
            break

    return M


def _solve_proportions_qp(M_inf: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve for cell-type proportions via quadratic programming (quadprog).

    Solves::

        minimise  ‖b − M_inf @ p‖²
        subject to  sum(p) = 1,  p ≥ 0

    Uses the same Fortran quadprog routine as R's ``quadprog::solve.QP``,
    ensuring maximum numerical compatibility.

    The quadratic objective is::

        0.5 p^T G p − a^T p
        G = M^T M + ε I    (ε = 1e-10 for positive definiteness)
        a = M^T b

    Constraints encoded as ``C^T p >= b_c`` with ``meq=1`` (first constraint
    is an equality)::

        C[:, 0] = 1   →  Σ p_i = 1  (equality, meq=1)
        C[:, 1:] = I  →  p_i ≥ 0   (inequalities)

    Parameters
    ----------
    M_inf : np.ndarray, shape (n_informative, k)
        Cell-type profiles on informative loci only.
    b : np.ndarray, shape (n_informative,)
        Single bulk sample on informative loci.

    Returns
    -------
    np.ndarray, shape (k,)
        Estimated proportions summing to 1, all ≥ 0.
    """
    k = M_inf.shape[1]
    G = M_inf.T @ M_inf + np.eye(k) * 1e-10   # positive definite
    a = M_inf.T @ b

    # Constraint matrix (k rows, k+1 columns)
    C       = np.zeros((k, k + 1))
    C[:, 0] = 1.0              # equality: sum = 1
    np.fill_diagonal(C[:, 1:], 1.0)  # non-negativity: p_i >= 0
    b_c     = np.zeros(k + 1)
    b_c[0]  = 1.0

    try:
        p = quadprog.solve_qp(G, a, C, b_c, meq=1)[0]
        p = np.clip(p, 0.0, None)
        s = p.sum()
        return p / s if s > 0 else np.ones(k) / k
    except Exception as exc:
        log.debug("QP solver failed (%s); returning uniform proportions", exc)
        return np.ones(k) / k


def _solve_proportions_scipy(M_inf: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fallback: solve for proportions via scipy SLSQP (no quadprog required).

    Solves the same constrained least-squares problem as
    ``_solve_proportions_qp`` but uses ``scipy.optimize.minimize`` with the
    SLSQP method.  Numerically equivalent but slower.
    """
    from scipy.optimize import minimize
    k = M_inf.shape[1]

    def obj(p):
        r = b - M_inf @ p
        return 0.5 * float(r @ r)

    def obj_grad(p):
        return M_inf.T @ (M_inf @ p - b)

    result = minimize(
        obj, np.ones(k) / k, jac=obj_grad, method="SLSQP",
        bounds=[(0.0, None)] * k,
        constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    p = np.clip(result.x, 0.0, None)
    s = p.sum()
    return p / s if s > 0 else np.ones(k) / k


def _solve_proportions(M_inf: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dispatch proportion solver to the best available backend."""
    if _HAS_QUADPROG:
        return _solve_proportions_qp(M_inf, b)
    return _solve_proportions_scipy(M_inf, b)


def run_stage1(
    bulk_norm: pd.DataFrame,
    probes: list[str],
    k: int,
    max_its: int         = 2000,
    rss_diff_stop: float = 1e-10,
    seed: int            = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run XDec Stage 1: NMF deconvolution.

    Solves ``B ≈ P × M^T`` by alternating quadratic programming (P) and
    non-negative least squares (M), using only the informative ``probes`` to
    update P.  M is estimated for **all** genes in ``bulk_norm``.

    Ports ``EDec::run_edec_stage_1`` from R.

    Input
    -----
    bulk_norm : pd.DataFrame
        Normalised bulk matrix (genes × samples, values in [0, 1]).
        Produced by ``normalise_bulk``.

        **Mode A** — pass the bulk-only normalised matrix (scRNA reference
        approach; Stage 1 runs on bulk alone).

        **Mode B** — pass a matrix that includes reference samples appended as
        extra columns (``anchor_refs=True`` in the CLI; references act as NMF
        anchors).  After Stage 1, slice off the reference columns from the
        returned ``proportions`` DataFrame.

    probes : list[str]
        Informative gene names from Stage 0 (``selected_probes.txt``).
        Genes absent from ``bulk_norm.index`` are silently ignored.
        Pass ``list(bulk_norm.index)`` to use all genes (slow).

    k : int
        Number of cell types to estimate.

    max_its : int
        Maximum alternating iterations (default 2000).

    rss_diff_stop : float
        Stop when |RSS_prev − RSS_curr| < this value on informative loci
        (default 1e-10).

    seed : int
        Random seed for M initialisation (default 12345).
        Note: R's Mersenne-Twister and numpy's PRNG differ, so exact bit-for-
        bit numerical parity with R is not achievable.  Results are
        functionally equivalent.

    Output
    ------
    proportions : pd.DataFrame, shape (n_samples, k)
        Cell-type proportions per sample (rows sum to 1, all values ≥ 0).
        Index = sample IDs from ``bulk_norm.columns``.
        Columns = "Profile_1" … "Profile_k".

        Assign biological labels (e.g. "Epithelial.Basal") by inspecting
        ``correlation.tsv`` produced by the CLI (Spearman correlation of
        estimated profiles with reference profiles).

    profiles : pd.DataFrame, shape (n_genes, k)
        Normalised cell-type expression profiles for all genes, values in
        [0, 1].  Index = gene names.  Columns = "Profile_1" … "Profile_k".
    """
    log.info(
        "Stage 1: k=%d, max_its=%d, rss_diff_stop=%.1e, seed=%d",
        k, max_its, rss_diff_stop, seed,
    )

    # Resolve probe indices (silently drop probes absent from bulk)
    gene_idx  = {g: i for i, g in enumerate(bulk_norm.index)}
    probe_idx = np.array(
        [gene_idx[p] for p in probes if p in gene_idx], dtype=int
    )
    if len(probe_idx) == 0:
        raise ValueError(
            "None of the provided probes are present in the bulk matrix index."
        )
    log.info(
        "Using %d / %d probes found in bulk matrix", len(probe_idx), len(probes)
    )

    rng = np.random.default_rng(seed)
    P, M = _stage1_core_inner(
        bulk_norm.values.astype(float), probe_idx, k, max_its, rss_diff_stop, rng
    )

    col_names    = [f"Profile_{i + 1}" for i in range(k)]
    proportions  = pd.DataFrame(P, index=bulk_norm.columns,  columns=col_names)
    profiles     = pd.DataFrame(M, index=bulk_norm.index,    columns=col_names)

    return proportions, profiles


# ===========================================================================
# SECTION 5 — Stage 2: cell-type-specific expression estimation
# ===========================================================================

def run_stage2(
    bulk_counts: pd.DataFrame,
    proportions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run XDec Stage 2: estimate cell-type-specific gene expression.

    Given raw bulk RNA-seq counts and known cell-type proportions, solves for
    the expression of each gene in each cell type independently::

        minimise  ‖B[g, :] − P @ m_g‖²   subject to  m_g ≥ 0

    using non-negative least squares (NNLS) per gene.  Standard errors are
    estimated using the OLS covariance approximation on the residuals.

    Ports ``EDec::run_edec_stage_2`` from R.

    Input
    -----
    bulk_counts : pd.DataFrame
        Raw (NOT log-transformed) bulk RNA-seq count matrix, genes × samples.
        Typically ``(2 ** HiSeqV2_values) − 1``.
        Only samples in the intersection of ``bulk_counts.columns`` and
        ``proportions.index`` are used.

    proportions : pd.DataFrame
        Cell-type proportion matrix from Stage 1 (or a subset), shape
        (n_samples, k).  Each row should sum to ≈ 1.

        Before calling this function, aggregate sub-types if needed, e.g.::

            proportions["Epithelial"] = (
                proportions[["Profile_1", "Profile_4", "Profile_7"]].sum(axis=1)
            )

        In the R code, Stage 2 is called **separately** for each sample group
        (PAM50 subtype or XDec subtype) and results are concatenated.

    Output
    ------
    means : pd.DataFrame, shape (n_genes, k)
        Estimated mean expression for each gene in each cell type.
        Index = gene names; columns = cell-type names from ``proportions``.

    std_errors : pd.DataFrame, shape (n_genes, k)
        Standard errors of the mean estimates.
        Computed as ``sqrt(σ² · diag((P^T P)^{−1}))`` where
        ``σ² = ‖residuals‖² / (n_samples − k)``.
        Same shape and column names as ``means``.

    Note: call this function per sample group and concatenate column-wise,
    as in the R code::

        for subtype in ["Basal", "LumA", "LumB", "Her2", "Normal", "Control"]:
            means_sub, stderrs_sub = run_stage2(
                bulk_counts[subtype_samples],
                proportions.loc[subtype_samples]
            )
    """
    shared = bulk_counts.columns.intersection(proportions.index)
    if len(shared) == 0:
        raise ValueError(
            "No shared samples between bulk_counts and proportions.  "
            "Check that sample IDs match."
        )

    B = bulk_counts[shared].values.astype(float)         # (n_genes, n_samples)
    P = proportions.loc[shared].values.astype(float)     # (n_samples, k)
    n_genes, n_samples = B.shape
    k   = P.shape[1]
    dof = max(1, n_samples - k)

    # Pre-compute (P^T P)^{-1} once for all genes (OLS covariance approximation)
    PtP           = P.T @ P
    PtP_inv       = np.linalg.inv(PtP + np.eye(k) * 1e-10)
    var_diag_base = np.diag(PtP_inv)               # (k,) — same for every gene

    # Solve all genes at once via vectorised CD-NNLS (see _nnls_batch_cd).
    # Raw bulk counts can be very large (RPKM ~10^6), so floating-point round-off
    # prevents convergence below ~1e-9 absolute change.  tol=1e-8 is tight enough
    # for biological precision while avoiding the infinite oscillation at 1e-12.
    means_arr = _nnls_batch_cd(P, B, tol=1e-8)    # (n_genes, k), ≥ 0

    # Residuals and standard errors — fully vectorised
    residuals_mat = B - means_arr @ P.T             # (n_genes, n_samples)
    sigma2_vec    = (residuals_mat ** 2).sum(axis=1) / dof  # (n_genes,)
    stderr_arr    = np.sqrt(np.maximum(
        sigma2_vec[:, None] * var_diag_base[None, :], 0.0
    ))                                              # (n_genes, k)

    col_names  = list(proportions.columns)
    gene_names = list(bulk_counts.index)
    means      = pd.DataFrame(means_arr,  index=gene_names, columns=col_names)
    std_errors = pd.DataFrame(stderr_arr, index=gene_names, columns=col_names)

    log.info(
        "Stage 2: %d genes × %d cell types  (%d samples)",
        n_genes, k, n_samples,
    )
    return means, std_errors


# ===========================================================================
# SECTION 6 — CLI
# ===========================================================================

def _save_tsv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame as a tab-separated file and log the event."""
    df.to_csv(path, sep="\t")
    log.info("Saved  %s  (%d × %d)", path, *df.shape)


def _load_probes(path: str) -> list[str]:
    """Load a probe / gene list from a plain-text file (one name per line)."""
    with open(path) as fh:
        probes = [line.strip() for line in fh if line.strip()]
    log.info("Loaded %d probes from %s", len(probes), path)
    return probes


# ---------------------------------------------------------------------------
# Stage 0 command
# ---------------------------------------------------------------------------

def cmd_stage0(args: argparse.Namespace) -> None:
    """Execute Stage 0: probe selection and stability estimation."""
    os.makedirs(args.outdir, exist_ok=True)

    # -- Load bulk -----------------------------------------------------------
    bulk_raw = load_matrix(args.bulk)
    if args.log2_bulk:
        bulk_raw = inverse_log2_transform(bulk_raw)

    # -- Load and normalise references (optional) ----------------------------
    refs_norm: Optional[pd.DataFrame] = None
    refs_raw:  Optional[pd.DataFrame] = None
    if args.refs:
        refs_raw = load_matrix(args.refs)
        if args.log2_refs:
            refs_raw = inverse_log2_transform(refs_raw)
        refs_norm = normalise_refs_independent(refs_raw)

    # -- Normalise bulk (jointly with refs for QN if refs provided) ----------
    bulk_norm = normalise_bulk(bulk_raw, refs_raw)

    # -- Probe selection -----------------------------------------------------
    if args.probes:
        # User supplied a pre-computed probe list; skip t-tests
        probes = _load_probes(args.probes)
    elif refs_norm is not None and args.ref_metadata:
        meta    = pd.read_csv(args.ref_metadata, sep="\t")
        s_col   = args.sample_col
        c_col   = args.class_col
        classes = meta.set_index(s_col).loc[refs_norm.columns, c_col].values
        probes  = select_probes(
            refs_norm, classes,
            one_vs_rest_p = args.p_one_vs_rest,
            one_vs_rest_n = args.n_one_vs_rest,
            pair_class_a  = args.pair_a or None,
            pair_class_b  = args.pair_b or None,
            pair_p        = args.p_pair,
            pair_n        = args.n_pair,
            append_pam50  = not args.no_pam50,
        )
    else:
        log.warning(
            "No reference metadata provided; using all %d bulk genes as probes.",
            len(bulk_norm),
        )
        probes = list(bulk_norm.index)

    # -- Save probe list -----------------------------------------------------
    probe_path = os.path.join(args.outdir, "selected_probes.txt")
    with open(probe_path, "w") as fh:
        fh.write("\n".join(probes) + "\n")
    log.info("Saved %d probes → %s", len(probes), probe_path)

    # -- Stability estimation ------------------------------------------------
    if getattr(args, "skip_stability", False):
        log.info(
            "--skip-stability: skipping stability estimation.  "
            "Probe selection complete.  Use --k to specify k for Stage 1."
        )
        return

    k_range = [int(x) for x in args.k_range.split(",")]
    scores  = estimate_stability(
        bulk_norm, probes, k_range,
        subset_prop   = args.subset_prop,
        num_subsets   = args.num_subsets,
        reps_per_subset = args.reps,
        max_its       = args.max_its,
        rss_diff_stop = args.rss_stop,
        seed          = args.seed,
    )

    stab_df = (
        pd.DataFrame(list(scores.items()), columns=["k", "stability_score"])
        .set_index("k")
    )
    _save_tsv(stab_df, os.path.join(args.outdir, "stability_scores.tsv"))

    optimal_k = max(scores, key=scores.__getitem__)
    with open(os.path.join(args.outdir, "optimal_k.txt"), "w") as fh:
        fh.write(str(optimal_k) + "\n")
    log.info(
        "Recommended k = %d  (stability=%.4f)", optimal_k, scores[optimal_k]
    )


# ---------------------------------------------------------------------------
# Stage 1 command
# ---------------------------------------------------------------------------

def cmd_stage1(args: argparse.Namespace) -> None:
    """Execute Stage 1: NMF deconvolution."""
    os.makedirs(args.outdir, exist_ok=True)

    # -- Load bulk -----------------------------------------------------------
    bulk_raw = load_matrix(args.bulk)
    if args.log2_bulk:
        bulk_raw = inverse_log2_transform(bulk_raw)

    # -- Load references (optional) ------------------------------------------
    refs_raw:  Optional[pd.DataFrame] = None
    refs_norm: Optional[pd.DataFrame] = None
    if args.refs:
        refs_raw = load_matrix(args.refs)
        if args.log2_refs:
            refs_raw = inverse_log2_transform(refs_raw)
        refs_norm = normalise_refs_independent(refs_raw)

    # -- Normalise bulk -------------------------------------------------------
    if args.anchor_refs and refs_raw is not None:
        # Mode B: append reference columns to bulk as NMF anchors
        shared = bulk_raw.index.intersection(refs_raw.index)
        log.info(
            "Mode B (anchor refs): appending %d reference samples",
            refs_raw.shape[1],
        )
        combined_raw  = pd.concat(
            [bulk_raw.loc[shared], refs_raw.loc[shared]], axis=1
        )
        combined_norm = normalise_bulk(combined_raw)
        bulk_cols     = list(bulk_raw.columns)
        bulk_norm     = combined_norm[bulk_cols]
        # The full combined matrix is used for Stage 1 so anchors guide NMF
        stage1_input  = combined_norm
    else:
        # Mode A: joint QN for normalisation only; Stage 1 on bulk alone
        bulk_norm    = normalise_bulk(bulk_raw, refs_raw)
        stage1_input = bulk_norm

    # -- Load probes ---------------------------------------------------------
    if args.probes:
        probes = _load_probes(args.probes)
    elif args.use_all_genes:
        probes = list(stage1_input.index)
        log.info("--use-all-genes: using all %d genes as informative probes", len(probes))
    else:
        sys.exit(
            "ERROR: Supply --probes (output of Stage 0) or --use-all-genes."
        )

    # -- Determine k ---------------------------------------------------------
    if args.k:
        k = args.k
    else:
        k_path = os.path.join(args.outdir, "optimal_k.txt")
        if os.path.exists(k_path):
            with open(k_path) as fh:
                k = int(fh.read().strip())
            log.info("Read k=%d from %s", k, k_path)
        else:
            sys.exit(
                "ERROR: Supply --k or run Stage 0 first to produce optimal_k.txt."
            )

    # -- Run Stage 1 ---------------------------------------------------------
    proportions, profiles = run_stage1(
        stage1_input, probes, k,
        max_its=args.max_its, rss_diff_stop=args.rss_stop, seed=args.seed,
    )

    # Keep only bulk samples in the proportions output
    proportions = proportions.loc[proportions.index.isin(bulk_norm.columns)]

    # -- Correlation with references for label assignment --------------------
    if refs_norm is not None and args.ref_metadata:
        meta    = pd.read_csv(args.ref_metadata, sep="\t")
        s_col   = args.sample_col
        c_col   = args.class_col
        classes = meta.set_index(s_col).loc[refs_norm.columns, c_col].values

        valid_probes   = [p for p in probes if p in profiles.index
                         and p in refs_norm.index]
        # Each column = one sample/profile; each row = one probe (observation).
        # spearmanr(A, B) stacks columns and requires equal row counts.
        ref_sub = refs_norm.loc[valid_probes].values        # (n_probes, n_ref_samples)
        est_sub = profiles.loc[valid_probes].values         # (n_probes, k)

        r   = stats.spearmanr(ref_sub, est_sub)
        mat = r.statistic if hasattr(r, "statistic") else r[0]
        mat = np.atleast_2d(mat)

        unique_types = list(dict.fromkeys(classes))         # ordered unique
        n_ref        = ref_sub.shape[1]                     # number of ref sample columns
        corr_rows: dict[str, np.ndarray] = {}
        for ct in unique_types:
            idx = np.where(classes == ct)[0]
            # Cross-block: ref samples of this cell type (rows) → estimated profiles (cols)
            corr_rows[ct] = mat[np.ix_(idx, np.arange(n_ref, n_ref + k))].mean(axis=0)

        corr_df = pd.DataFrame(
            corr_rows, index=profiles.columns, dtype=float
        ).T
        _save_tsv(corr_df, os.path.join(args.outdir, "correlation.tsv"))
        log.info(
            "Inspect correlation.tsv to assign cell-type labels to Profile_i columns."
        )

    # -- Save ----------------------------------------------------------------
    _save_tsv(proportions, os.path.join(args.outdir, "proportions.tsv"))
    _save_tsv(profiles,    os.path.join(args.outdir, "cell_type_profiles.tsv"))


# ---------------------------------------------------------------------------
# Stage 2 command
# ---------------------------------------------------------------------------

def cmd_stage2(args: argparse.Namespace) -> None:
    """Execute Stage 2: cell-type-specific expression estimation."""
    os.makedirs(args.outdir, exist_ok=True)

    # -- Load bulk counts ----------------------------------------------------
    bulk_raw = load_matrix(args.bulk_counts)
    if args.log2_counts:
        bulk_raw = inverse_log2_transform(bulk_raw)
    bulk_raw = bulk_raw.loc[bulk_raw.sum(axis=1) > 0]

    # -- Load proportions ----------------------------------------------------
    proportions = pd.read_csv(args.proportions, sep="\t", index_col=0)
    log.info("Proportions: %d samples × %d cell types", *proportions.shape)

    # -- Rename profile columns if a label map is supplied -------------------
    #    label_map.json: {"Profile_1": "Epithelial.Basal", "Profile_2": "T-Cell", ...}
    if args.label_map:
        with open(args.label_map) as fh:
            label_map = json.load(fh)
        proportions = proportions.rename(columns=label_map)
        log.info("Renamed profiles using label map.")

    # -- Aggregate proportion columns if requested ---------------------------
    #    aggregate_map.json: {"Epithelial": ["Epithelial.Basal","Epithelial.Her2",...],
    #                         "Immune":     ["T-Cell","Macrophage"],
    #                         "Stromal":    ["Adipocyte","CAF"]}
    #    Matches the R code that sums Epithelial sub-components before Stage 2.
    if args.aggregate_map:
        with open(args.aggregate_map) as fh:
            agg_map: dict[str, list[str]] = json.load(fh)
        agg_props = {}
        for new_name, old_names in agg_map.items():
            present = [c for c in old_names if c in proportions.columns]
            if present:
                agg_props[new_name] = proportions[present].sum(axis=1)
            else:
                log.warning("aggregate_map key '%s': none of %s found.", new_name, old_names)
        proportions = pd.DataFrame(agg_props, index=proportions.index)

    # -- Run Stage 2 per sample group (or on all samples) --------------------
    if args.metadata and args.subtype_col:
        meta = pd.read_csv(args.metadata, sep="\t", index_col=0)
        groups = meta[args.subtype_col].dropna().unique()

        all_means:   list[pd.DataFrame] = []
        all_stderrs: list[pd.DataFrame] = []

        for grp in groups:
            samples = (
                meta[meta[args.subtype_col] == grp].index
                .intersection(bulk_raw.columns)
                .intersection(proportions.index)
                .tolist()
            )
            if len(samples) < 2:
                log.warning("Group '%s': fewer than 2 samples — skipping.", grp)
                continue
            log.info("Stage 2 group '%s': %d samples", grp, len(samples))
            means_g, stderrs_g = run_stage2(bulk_raw[samples], proportions.loc[samples])
            means_g.columns    = [f"{c}_{grp}" for c in means_g.columns]
            stderrs_g.columns  = [f"{c}_{grp}" for c in stderrs_g.columns]
            all_means.append(means_g)
            all_stderrs.append(stderrs_g)

        if not all_means:
            sys.exit("ERROR: No groups had enough samples for Stage 2.")

        means_out   = pd.concat(all_means,   axis=1)
        stderrs_out = pd.concat(all_stderrs, axis=1)

    else:
        # Run on all samples together (no grouping)
        means_out, stderrs_out = run_stage2(bulk_raw, proportions)

    _save_tsv(means_out,   os.path.join(args.outdir, "stage2_means.tsv"))
    _save_tsv(stderrs_out, os.path.join(args.outdir, "stage2_stderrs.tsv"))


# ---------------------------------------------------------------------------
# Run-all command (default when no subcommand is given)
# ---------------------------------------------------------------------------

def cmd_run_all(args: argparse.Namespace) -> None:
    """Run all three stages sequentially, auto-wiring outputs between stages.

    Stage 0 → selected_probes.txt, optimal_k.txt
    Stage 1 → proportions.tsv, cell_type_profiles.tsv
    Stage 2 → stage2_means.tsv, stage2_stderrs.tsv  (skipped if --bulk-counts absent)

    All outputs go to --outdir (default: xdec_results/).
    """
    if not args.bulk:
        sys.exit(
            "ERROR: --bulk is required.\n\n"
            "Run-all usage:\n"
            "  python xdec.py --bulk bulk.tsv [--refs refs.tsv] "
            "[--bulk-counts counts.tsv] ...\n\n"
            "Stage-specific usage:\n"
            "  python xdec.py stage0 --help\n"
            "  python xdec.py stage1 --help\n"
            "  python xdec.py stage2 --help\n"
        )

    # -- Stage 0 -------------------------------------------------------------
    log.info("=== Stage 0: probe selection and stability estimation ===")
    cmd_stage0(args)

    # Wire stage0 → stage1: always use the probe list written by stage0
    probe_path = os.path.join(args.outdir, "selected_probes.txt")
    args.probes = probe_path
    log.info("Stage 0 → Stage 1: probes auto-wired from %s", probe_path)

    # -- Stage 1 -------------------------------------------------------------
    log.info("=== Stage 1: NMF deconvolution ===")
    # If --k was not provided, cmd_stage1 reads optimal_k.txt automatically
    cmd_stage1(args)

    # Wire stage1 → stage2: use the proportions written by stage1
    prop_path = os.path.join(args.outdir, "proportions.tsv")
    args.proportions = prop_path
    log.info("Stage 1 → Stage 2: proportions auto-wired from %s", prop_path)

    # -- Stage 2 (optional) --------------------------------------------------
    if args.bulk_counts:
        log.info("=== Stage 2: cell-type-specific expression estimation ===")
        cmd_stage2(args)
    else:
        log.info(
            "Stage 2 skipped: --bulk-counts not provided.  "
            "Re-run with --bulk-counts <path> to estimate cell-type expression."
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_shared_args(p: argparse.ArgumentParser) -> None:
    """Add arguments shared by all three subcommands."""
    p.add_argument(
        "--outdir", default="xdec_results",
        help="Output directory (created if absent).  Default: xdec_results/",
    )
    p.add_argument(
        "--max-its", type=int, default=2000,
        help="Maximum Stage 1 alternating iterations.  Default: 2000.",
    )
    p.add_argument(
        "--rss-stop", type=float, default=1e-10,
        help="Stage 1 convergence threshold (RSS change).  Default: 1e-10.",
    )
    p.add_argument(
        "--seed", type=int, default=12345,
        help=(
            "Random seed for M initialisation (default 12345, matching "
            "set.seed(12345) in R).  Exact bit-for-bit parity with R is not "
            "achievable due to differing PRNGs, but results are functionally "
            "equivalent."
        ),
    )


def _add_ref_args(p: argparse.ArgumentParser) -> None:
    """Add reference-related arguments shared by Stage 0 and Stage 1."""
    p.add_argument(
        "--refs",
        help=(
            "Reference matrix (TSV, genes × samples).  Each column is one "
            "reference sample (cell line, pseudo-bulk cluster, etc.) of the "
            "same feature type as --bulk.  Optional: if omitted, supply "
            "--probes directly."
        ),
    )
    p.add_argument(
        "--log2-refs", action="store_true",
        help="Invert log2(x+1) transform on references before normalisation.",
    )
    p.add_argument(
        "--ref-metadata",
        help=(
            "TSV file with reference sample metadata.  Required columns: "
            "--sample-col (default Sample_ID) and --class-col (default "
            "Cell_Type).  Optional column: Cell_Subtype."
        ),
    )
    p.add_argument(
        "--class-col", default="Cell_Type",
        help="Metadata column for cell-type class labels.  Default: Cell_Type.",
    )
    p.add_argument(
        "--sample-col", default="Sample_ID",
        help="Metadata column for sample IDs.  Default: Sample_ID.",
    )
    p.add_argument(
        "--no-pam50", action="store_true",
        help="Do not append PAM50 genes to the selected probe list.",
    )


def _add_probe_selection_args(p: argparse.ArgumentParser) -> None:
    """Add Stage 0 probe-selection and stability arguments."""
    p.add_argument(
        "--skip-stability", action="store_true",
        help=(
            "Skip the stability estimation loop in Stage 0.  "
            "Only probe selection is performed (fast).  "
            "Use when k is already known (supply --k for Stage 1)."
        ),
    )
    p.add_argument(
        "--p-one-vs-rest", type=float, default=1e-4,
        help=(
            "p-value threshold for one-vs-rest t-tests.  "
            "R default: 1e-4 (methylation) / 0.05 (scRNA).  Default: 1e-4."
        ),
    )
    p.add_argument(
        "--n-one-vs-rest", type=int, default=25,
        help="Top/bottom genes per class direction for one-vs-rest.  Default: 25.",
    )
    p.add_argument(
        "--pair-a",
        help="First class for optional pairwise t-test (e.g. 'epithelial').",
    )
    p.add_argument(
        "--pair-b",
        help="Second class for optional pairwise t-test (e.g. 'CAF').",
    )
    p.add_argument(
        "--p-pair", type=float, default=1e-5,
        help="p-value threshold for pairwise t-test.  Default: 1e-5.",
    )
    p.add_argument(
        "--n-pair", type=int, default=75,
        help="Top/bottom genes per direction for pairwise test.  Default: 75.",
    )
    p.add_argument(
        "--k-range", default="3,4,5,6,7,8,9,10,11,12",
        help="Comma-separated k values to test (default: 3–12).",
    )
    p.add_argument(
        "--subset-prop", type=float, default=0.8,
        help="Fraction of samples per stability subset.  Default: 0.8.",
    )
    p.add_argument(
        "--num-subsets", type=int, default=3,
        help="Number of random subsets per k.  Default: 3.",
    )
    p.add_argument(
        "--reps", type=int, default=3,
        help="Stage 1 repetitions per subset.  Default: 3.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the XDec command-line argument parser.

    When called without a subcommand (stage0 / stage1 / stage2), xdec runs
    all three stages sequentially, auto-wiring outputs between them.
    """
    parser = argparse.ArgumentParser(
        prog="xdec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.  Default: INFO.",
    )

    # ── Top-level / run-all arguments ────────────────────────────────────────
    # These are used when no subcommand is given.  Subcommands define their
    # own copies of these args (same defaults) so subcommand-specific help
    # text remains accurate.

    parser.add_argument(
        "--bulk",
        help=(
            "Bulk expression matrix (TSV, genes × samples).  "
            "Required for run-all mode.  Use --log2-bulk if values are "
            "log2(x+1)-transformed (e.g. TCGA HiSeqV2)."
        ),
    )
    parser.add_argument(
        "--log2-bulk", action="store_true",
        help="Invert log2(x+1) transform on bulk before normalisation.",
    )
    _add_ref_args(parser)
    _add_shared_args(parser)
    _add_probe_selection_args(parser)

    # Stage 0: pre-computed probe list (skips t-tests if supplied)
    parser.add_argument(
        "--probes",
        help=(
            "Pre-computed probe list (one gene per line).  If supplied, "
            "t-test probe selection is skipped in Stage 0."
        ),
    )

    # Stage 1 specific
    parser.add_argument(
        "--anchor-refs", action="store_true",
        help=(
            "Mode B: append reference samples to bulk before Stage 1 (they "
            "act as NMF anchors).  Default: Mode A (refs used for QN only)."
        ),
    )
    parser.add_argument(
        "--use-all-genes", action="store_true",
        help="Use all genes as informative probes for Stage 1 (slow).",
    )
    parser.add_argument(
        "--k", type=int,
        help=(
            "Number of cell types for Stage 1.  If omitted, reads from "
            "outdir/optimal_k.txt (written by Stage 0)."
        ),
    )

    # Stage 2 specific
    parser.add_argument(
        "--bulk-counts",
        help=(
            "Raw bulk count matrix (TSV, genes × samples) for Stage 2.  "
            "Use --log2-counts if stored as log2(x+1).  "
            "If omitted in run-all mode, Stage 2 is skipped."
        ),
    )
    parser.add_argument(
        "--log2-counts", action="store_true",
        help="Invert log2(x+1) transform on bulk counts for Stage 2.",
    )
    parser.add_argument(
        "--proportions",
        help=(
            "proportions.tsv from Stage 1 (samples × k).  "
            "Auto-wired in run-all mode; required for the stage2 subcommand."
        ),
    )
    parser.add_argument(
        "--label-map",
        help=(
            "JSON file mapping Profile_i → biological cell-type name.  "
            'Example: {"Profile_1": "Epithelial.Basal", "Profile_2": "T-Cell"}.'
        ),
    )
    parser.add_argument(
        "--aggregate-map",
        help=(
            "JSON file describing how to sum proportion columns before Stage 2.  "
            'Example: {"Epithelial": ["Epithelial.Basal", "Epithelial.Her2"], '
            '"Immune": ["T-Cell", "Macrophage"]}.  '
            "Matches the R code that aggregates sub-types before run_edec_stage_2."
        ),
    )
    parser.add_argument(
        "--metadata",
        help="Sample metadata TSV (row index = sample IDs).  Used for Stage 2 grouping.",
    )
    parser.add_argument(
        "--subtype-col",
        help=(
            "Column in --metadata to group samples by for Stage 2 (e.g. 'PAM50').  "
            "Stage 2 is run separately for each group and results are "
            "concatenated column-wise.  If omitted, run on all samples."
        ),
    )

    # ── Subcommands (optional — omit to run all stages) ───────────────────────
    sub = parser.add_subparsers(dest="command")

    # ── Stage 0 ──────────────────────────────────────────────────────────────
    p0 = sub.add_parser(
        "stage0",
        help="Select informative probes and estimate the optimal k.",
        description=(
            "Stage 0 outputs:\n"
            "  selected_probes.txt  — informative gene/probe names\n"
            "  stability_scores.tsv — stability score per k value\n"
            "  optimal_k.txt        — recommended number of cell types\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p0.add_argument(
        "--bulk", required=True,
        help=(
            "Bulk expression matrix (TSV, genes × samples).  Use --log2-bulk "
            "if values are log2(x+1)-transformed (e.g. TCGA HiSeqV2)."
        ),
    )
    p0.add_argument(
        "--log2-bulk", action="store_true",
        help="Invert log2(x+1) transform on bulk before normalisation.",
    )
    _add_ref_args(p0)
    _add_shared_args(p0)
    p0.add_argument(
        "--probes",
        help=(
            "Pre-computed probe list (one gene per line).  If supplied, "
            "t-test probe selection is skipped."
        ),
    )
    _add_probe_selection_args(p0)
    p0.set_defaults(func=cmd_stage0)

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    p1 = sub.add_parser(
        "stage1",
        help="Deconvolve bulk data into proportions and cell-type profiles.",
        description=(
            "Stage 1 outputs:\n"
            "  proportions.tsv          — samples × k proportion matrix\n"
            "  cell_type_profiles.tsv   — genes × k normalised profiles\n"
            "  correlation.tsv          — reference cell types × estimated "
            "profiles\n"
            "                             (Spearman; inspect to assign labels)\n\n"
            "Mode A (default): references used only for joint QN + post-hoc labelling.\n"
            "Mode B (--anchor-refs): reference samples appended as NMF anchors.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p1.add_argument("--bulk", required=True, help="Bulk expression matrix (TSV, genes × samples).")
    p1.add_argument("--log2-bulk", action="store_true", help="Invert log2(x+1) on bulk.")
    _add_ref_args(p1)
    _add_shared_args(p1)
    p1.add_argument(
        "--anchor-refs", action="store_true",
        help=(
            "Mode B: append reference samples to bulk before Stage 1 (they "
            "act as NMF anchors).  Default: Mode A (refs used for QN only)."
        ),
    )
    p1.add_argument(
        "--probes",
        help="Probe list from Stage 0 (selected_probes.txt).  Recommended.",
    )
    p1.add_argument(
        "--use-all-genes", action="store_true",
        help="Use all genes as informative probes (skips --probes).  Slow.",
    )
    p1.add_argument(
        "--k", type=int,
        help="Number of cell types.  If omitted, reads from outdir/optimal_k.txt.",
    )
    p1.set_defaults(func=cmd_stage1)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    p2 = sub.add_parser(
        "stage2",
        help="Estimate cell-type-specific expression given known proportions.",
        description=(
            "Stage 2 outputs:\n"
            "  stage2_means.tsv    — genes × (cell_type_group) expression means\n"
            "  stage2_stderrs.tsv  — genes × (cell_type_group) standard errors\n\n"
            "Provide --metadata + --subtype-col to run separately per sample\n"
            "group (matching the PAM50-stratified R code).\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p2.add_argument(
        "--bulk-counts", required=True,
        help=(
            "Raw bulk count matrix (TSV, genes × samples).  "
            "Use --log2-counts if stored as log2(x+1)."
        ),
    )
    p2.add_argument(
        "--log2-counts", action="store_true",
        help="Invert log2(x+1) transform on bulk counts.",
    )
    p2.add_argument(
        "--proportions", required=True,
        help="proportions.tsv from Stage 1 (samples × k).",
    )
    p2.add_argument(
        "--label-map",
        help=(
            "JSON file mapping Profile_i → biological cell-type name.  "
            'Example: {"Profile_1": "Epithelial.Basal", "Profile_2": "T-Cell"}.'
        ),
    )
    p2.add_argument(
        "--aggregate-map",
        help=(
            "JSON file describing how to sum proportion columns before Stage 2.  "
            'Example: {"Epithelial": ["Epithelial.Basal", "Epithelial.Her2"], '
            '"Immune": ["T-Cell", "Macrophage"], '
            '"Stromal": ["Adipocyte", "CAF"]}.  '
            "Matches the R code that aggregates sub-types before run_edec_stage_2."
        ),
    )
    p2.add_argument(
        "--metadata",
        help="Sample metadata TSV (row index = sample IDs).  Used for grouping.",
    )
    p2.add_argument(
        "--subtype-col",
        help=(
            "Column in --metadata to group samples by (e.g. 'PAM50').  "
            "Stage 2 is run separately for each group and results are "
            "concatenated column-wise.  If omitted, run on all samples."
        ),
    )
    _add_shared_args(p2)
    p2.set_defaults(func=cmd_stage2)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and dispatch to the appropriate stage handler.

    With no subcommand: runs all three stages sequentially (run-all mode).
    With a subcommand (stage0 / stage1 / stage2): runs that stage only.
    """
    parser = build_parser()
    args   = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    if args.command is None:
        cmd_run_all(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
