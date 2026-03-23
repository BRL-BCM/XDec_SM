#!/usr/bin/env python3
"""
preprocess_scrna.py — Prepare scRNA-seq pseudo-bulk references for XDec Stage 0/1.
====================================================================================

Ports the scRNA reference preprocessing chunk from XDec_total_code_vpetrosyan.Rmd
(the "preprocess scRNA references" and "normalize scRNA references" code blocks).

The workflow:
  1. Read raw single-cell RNA-seq count matrix (genes × cells).
  2. For each cell type in the metadata, filter cells by minimum library size,
     sort by library size descending, keep the top N cells, and sum every
     ``group_size`` cells into pseudo-bulk samples.
  3. Library-size normalise every pseudo-bulk column to the total of the first
     Epithelial pseudo-bulk (so all cell types are on a comparable scale).
  4. Write the combined pseudo-bulk matrix as a TSV and generate a metadata TSV
     (Sample_ID, Cell_Type) ready for xdec.py's --refs / --ref-metadata flags.

R equivalents
-------------
  GSE118389.RSEM.Epithelial.Sum   = filter → sort → take top N → sum_every_5
  GSE118389.RSEM.Epithelial.Sum.Norm = library-size normalise
  GSE118389.CombinedSums.Norm     = cbind(Epithelial, Stroma, Tcell, Macrophage)

USAGE
-----
  python preprocess_scrna.py \\
      --counts source/GSE118389_counts_rsem.txt \\
      --metadata source/GSE118389_Metadata.txt \\
      --outdir results/

OUTPUTS
-------
  scrna_refs.tsv       — genes × pseudo-bulk matrix (TSV), input for --refs
  scrna_metadata.tsv   — Sample_ID / Cell_Type table,   input for --ref-metadata

DEPENDENCIES
------------
  numpy >= 1.24
  pandas >= 2.0
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess_scrna")


# ---------------------------------------------------------------------------
# Cell-type configuration (from the R code)
# Each entry: (output_label, filter_column, filter_value, top_n_cells, group_size)
# Epithelial and Stroma use Cell_Type; Tcell and Macrophage use Cell_Subtype.
# Endothelial is computed in R but NOT included in CombinedSums → omit here.
# ---------------------------------------------------------------------------
CELL_TYPE_CONFIG = [
    # (output_label,  filter_column,   filter_value,   top_n,  group_size)
    ("Epithelial",   "Cell_Type",     "Epithelial",    640,    5),
    ("Stroma",       "Cell_Type",     "Stroma",         90,    5),
    ("Tcell",        "Cell_Subtype",  "Tcell",           50,    5),
    ("Macrophage",   "Cell_Subtype",  "macrophage",      60,    5),
]

# Minimum library size per cell (R: Less100K filter)
MIN_LIBRARY_SIZE = 100_000


def pseudobulk_aggregate(
    counts: pd.DataFrame,
    sample_ids: list[str],
    top_n: int,
    group_size: int,
    label: str,
    min_lib_size: int = MIN_LIBRARY_SIZE,
) -> pd.DataFrame:
    """Filter, sort, sub-select, and sum cells into pseudo-bulk samples.

    Ports the R chunk::

        Less100K.X = colSums(X) > 100000
        X = X[, Less100K.X]
        X = X[, names(sort(colSums(X), decreasing = TRUE))]
        X = X[, 1:top_n]
        X.Sum = t(sapply(seq(1, ncol(X), by=group_size), function(i){
            indx <- i:(i+group_size-1)
            rowSums(X[indx[indx <= ncol(X)]])
        }))

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × cells raw count matrix (all cells).
    sample_ids : list[str]
        Cell IDs belonging to this cell type (from metadata).
    top_n : int
        Maximum number of cells to keep (after filtering by library size).
    group_size : int
        Number of cells to sum into each pseudo-bulk sample.
    label : str
        Cell-type label used to build pseudo-bulk column names (e.g. "Epithelial").

    Returns
    -------
    pd.DataFrame
        Genes × n_pseudobulks matrix.  Columns named "<label>.P1", "<label>.P2", …
    """
    # Subset to this cell type's cells (keep only those present in the count matrix)
    present = [s for s in sample_ids if s in counts.columns]
    if not present:
        log.warning("%s: none of %d samples found in count matrix — skipping.", label, len(sample_ids))
        return pd.DataFrame(index=counts.index)

    sub = counts[present]

    # Filter: library size > min_lib_size
    lib_sizes = sub.sum(axis=0)
    sub = sub.loc[:, lib_sizes > min_lib_size]
    log.info("%s: %d / %d cells pass library-size filter (>%d)", label, sub.shape[1], len(present), min_lib_size)

    if sub.empty:
        log.warning("%s: no cells remain after library-size filtering.", label)
        return pd.DataFrame(index=counts.index)

    # Sort by library size descending and take top N
    order = sub.sum(axis=0).sort_values(ascending=False).index
    sub = sub[order].iloc[:, :top_n]
    log.info("%s: using top %d cells (requested %d)", label, sub.shape[1], top_n)

    # Sum every group_size cells → pseudo-bulk samples
    n_cells = sub.shape[1]
    values = sub.values                                   # (n_genes, n_cells)
    pseudobulks = []
    starts = list(range(0, n_cells, group_size))          # 0, group_size, 2*group_size, …
    for i, start in enumerate(starts):
        end = min(start + group_size, n_cells)
        pb = values[:, start:end].sum(axis=1)             # rowSums of the group
        pseudobulks.append(pb)

    pb_array  = np.column_stack(pseudobulks)              # (n_genes, n_pseudobulks)
    col_names = [f"{label}.P{i + 1}" for i in range(len(pseudobulks))]
    result    = pd.DataFrame(pb_array, index=counts.index, columns=col_names)

    log.info("%s: produced %d pseudo-bulk samples", label, len(pseudobulks))
    return result


def library_size_normalise(df: pd.DataFrame, target_colsum: float) -> pd.DataFrame:
    """Scale every column of ``df`` so that its sum equals ``target_colsum``.

    Ports the R normalisation loop::

        for(i in 1:ncol(X)){
            X[i] = X[i] * (target_colsum / colSums(X[i]))
        }

    Parameters
    ----------
    df : pd.DataFrame
        Genes × pseudo-bulk matrix (raw sums, not yet normalised).
    target_colsum : float
        Desired column total for every pseudo-bulk sample.  In the R code
        this is ``colSums(GSE118389.RSEM.Epithelial.Sum.Norm)[1]``, i.e. the
        library size of the first Epithelial pseudo-bulk sample.

    Returns
    -------
    pd.DataFrame
        Library-size normalised matrix (same shape and index).
    """
    col_sums = df.sum(axis=0)
    # Avoid division by zero for empty pseudo-bulks
    scale    = target_colsum / col_sums.where(col_sums > 0, other=1.0)
    return df.multiply(scale, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="preprocess_scrna",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--counts", required=True,
        help="Raw scRNA-seq count matrix TSV (genes × cells). "
             "Row names = genes (first column); column names = cell IDs (header). "
             "Matches GSE118389_counts_rsem.txt format.",
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Cell metadata TSV.  Must have columns: --sample-col (cell ID) and "
             "--type-col (Cell_Type).  Matches GSE118389_Metadata.txt format.",
    )
    parser.add_argument(
        "--sample-col", default="Sample",
        help="Metadata column containing cell IDs.  Default: Sample.",
    )
    parser.add_argument(
        "--type-col", default="Cell_Type",
        help="Metadata column containing broad cell-type labels.  Default: Cell_Type.",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Output directory.  Default: current directory.",
    )
    parser.add_argument(
        "--min-lib-size", type=int, default=MIN_LIBRARY_SIZE,
        help=f"Minimum library size to keep a cell.  Default: {MIN_LIBRARY_SIZE}.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)
    os.makedirs(args.outdir, exist_ok=True)

    min_lib = args.min_lib_size

    # ── Load inputs ──────────────────────────────────────────────────────────
    log.info("Loading count matrix from %s …", args.counts)
    counts = pd.read_csv(args.counts, sep="\t", index_col=0, header=0)
    counts = counts.astype(float)
    counts = counts.loc[counts.sum(axis=1) > 0]
    log.info("Count matrix: %d genes × %d cells", *counts.shape)

    log.info("Loading metadata from %s …", args.metadata)
    meta = pd.read_csv(args.metadata, sep="\t")
    log.info("Metadata: %d rows, columns: %s", len(meta), list(meta.columns))

    # ── Build pseudo-bulks per cell type ─────────────────────────────────────
    pseudobulk_frames: list[pd.DataFrame] = []
    epi_target: float | None = None           # library size of first Epithelial PB

    for label, filter_col, filter_val, top_n, group_size in CELL_TYPE_CONFIG:
        mask     = meta[filter_col] == filter_val
        ids      = meta.loc[mask, args.sample_col].tolist()
        log.info("Cell type '%s' (%s=%s): %d cells in metadata", label, filter_col, filter_val, len(ids))

        pb = pseudobulk_aggregate(counts, ids, top_n, group_size, label, min_lib)
        if pb.empty or pb.shape[1] == 0:
            log.warning("Skipping empty pseudo-bulk for '%s'.", label)
            continue

        # The library-size target is the colSum of the FIRST Epithelial pseudo-bulk
        if epi_target is None and label == "Epithelial":
            epi_target = float(pb.iloc[:, 0].sum())
            log.info("Library-size normalisation target = %.0f (first Epithelial PB)", epi_target)

        pseudobulk_frames.append(pb)

    if not pseudobulk_frames:
        sys.exit("ERROR: No pseudo-bulk samples produced.  Check metadata and counts.")

    if epi_target is None:
        # Fallback: use the first column of whatever came first
        epi_target = float(pseudobulk_frames[0].iloc[:, 0].sum())
        log.warning("Epithelial pseudo-bulks not found; using %.0f as normalisation target.", epi_target)

    # ── Library-size normalise ───────────────────────────────────────────────
    normalised_frames = [library_size_normalise(pb, epi_target) for pb in pseudobulk_frames]

    # ── Combine and filter zero-sum rows ────────────────────────────────────
    combined = pd.concat(normalised_frames, axis=1)
    combined = combined.loc[combined.sum(axis=1) > 0]
    log.info("Combined pseudo-bulk matrix: %d genes × %d samples", *combined.shape)

    # ── Save outputs ─────────────────────────────────────────────────────────
    refs_path = os.path.join(args.outdir, "scrna_refs.tsv")
    combined.to_csv(refs_path, sep="\t")
    log.info("Saved pseudo-bulk matrix → %s", refs_path)

    # Build metadata: Sample_ID = column name, Cell_Type = prefix before ".P"
    cell_types = [col.rsplit(".P", 1)[0] for col in combined.columns]
    meta_out   = pd.DataFrame({"Sample_ID": combined.columns, "Cell_Type": cell_types})
    meta_path  = os.path.join(args.outdir, "scrna_metadata.tsv")
    meta_out.to_csv(meta_path, sep="\t", index=False)
    log.info("Saved metadata → %s", meta_path)
    log.info("Cell types in output: %s", sorted(set(cell_types)))


if __name__ == "__main__":
    main()
