# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optimization-based data selection.

Selects a budget-constrained subset of training items whose aggregate AP
coverage best matches a target (test) distribution from an atlas. Solvers:
greedy (several scoring metrics incl. kl_divergence), LP relaxation + rounding,
direct MIP, and LP-seeded LNS (the latter three use the highspy solver).

Usage:
  python scripts/optimizers.py \\
      --atlas output/atlas/test_frequency.json \\
      --caption-dir output/match_train \\
      --budget 1000 \\
      --optimizer greedy --metric kl_divergence \\
      --exp-idx 0 \\
      -o output/selection
"""

from pathlib import Path
from functools import partial
import json
import argparse
import logging
import math
import hashlib
import sys
import time
import heapq
import random
import numpy as np
from scipy import sparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

try:
    from highspy import Highs, HighsVarType

    HAVE_HIGHSPY = True
except ImportError:
    HAVE_HIGHSPY = False


# ---------------------------------------------------------------------------
# A. Data Loading
# ---------------------------------------------------------------------------

# Process-global for workers: set once via initializer, not re-pickled per task
_worker_triplet_idx = None


def _worker_init(triplet_idx):
    global _worker_triplet_idx
    _worker_triplet_idx = triplet_idx


def _caption_item_key(caption_path):
    """Portable item id for a match file: its stem — the id minted by the
    caption interface (e.g. a navsim frame token or a physical-ai clip UUID).
    Item ids are globally unique by construction of the caption layout, so the
    id is path-free, which keeps selections portable across machines/users."""
    return caption_path.stem


def _process_caption(caption_path, count_once):
    """Parse one caption JSON and return (data_info, counts_dict) or None on failure.

    Returns {col_index: count} using integer keys instead of triplet strings
    to minimize memory (~16 bytes vs ~150 bytes per entry).
    """
    try:
        with open(caption_path, "r") as f:
            caption_data = json.load(f)
        data_info = _caption_item_key(caption_path)
        counts = {}  # {int col_index: count}
        for pair in caption_data["matched_pairs"]:
            for triplet_list in pair["atlas_matches"]:
                triplet = f"({triplet_list[0]},{triplet_list[1]},{triplet_list[2]})"
                idx = _worker_triplet_idx.get(triplet)
                if idx is not None:
                    if not count_once:
                        counts[idx] = counts.get(idx, 0) + 1
                    else:
                        counts[idx] = 1
        return (data_info, counts)
    except Exception as e:
        logging.warning("Failed to process %s: %s", caption_path, e)
        return None


def build_target_vector(
    atlas_dict, budget, reweight, rfs_threshold, num_test_docs, alpha, cap_percentile
):
    """Build the target vector T from atlas triplet frequencies.

    Returns (T, triplet_list, triplet_idx, M, reweight_info).
    """
    triplet_freq = atlas_dict["ap_frequency"]

    if sum(triplet_freq.values()) == 0:
        raise ValueError("All triplet frequencies are zero. Nothing to optimize.")

    if reweight == "repeat-factor" and num_test_docs <= 0:
        raise ValueError("--num-test-docs is required for repeat-factor reweighting.")

    if reweight == "proportional":
        raw_weights = {k: v for k, v in triplet_freq.items()}
    elif reweight == "uniform":
        raw_weights = {k: 1.0 for k in triplet_freq}
    elif reweight == "repeat-factor":
        t = rfs_threshold
        num_docs = num_test_docs
        raw_weights = {}
        for k, count in triplet_freq.items():
            f = count / num_docs
            r = max(1.0, math.sqrt(t / f)) if f > 0 else 1.0
            raw_weights[k] = count * r
    elif reweight == "power-law":
        raw_weights = {
            k: math.pow(v, alpha) if v > 0 else 0 for k, v in triplet_freq.items()
        }
    elif reweight == "power-law-cap":
        power_law_values = {
            k: math.pow(v, alpha) if v > 0 else 0 for k, v in triplet_freq.items()
        }
        cap = np.percentile(list(power_law_values.values()), cap_percentile)
        raw_weights = {k: min(v, cap) for k, v in power_law_values.items()}
    else:
        raw_weights = {k: v for k, v in triplet_freq.items()}

    sum_weights = sum(raw_weights.values())
    triplet_weights = {k: v / sum_weights for k, v in raw_weights.items()}

    # Ordered triplet list and index mapping for array indexing
    triplet_list = list(triplet_weights.keys())
    triplet_idx = {t: i for i, t in enumerate(triplet_list)}
    M = len(triplet_list)

    reweight_info = ""
    if reweight == "repeat-factor":
        reweight_info = f" (rfs_threshold={rfs_threshold})"
    elif reweight == "power-law":
        reweight_info = f" (alpha={alpha})"
    elif reweight == "power-law-cap":
        reweight_info = f" (alpha={alpha}, cap_percentile={cap_percentile})"

    # Target vector T: desired number of occurrences per triplet in the selected set
    T = np.array([triplet_weights[t] * budget for t in triplet_list], dtype=np.float32)

    return T, triplet_list, triplet_idx, M, reweight_info


def build_sparse_matrix(caption_dir, triplet_idx, M, count_once, selected_lists=None):
    """Parse all caption files in parallel and build a sparse CSR matrix.

    Returns (caption_counts, item_ids, N).
    """
    caption_files = list(Path(caption_dir).glob("**/*.json"))
    print(f"Found {len(caption_files)} caption files. Parsing...")
    if selected_lists is not None:
        for selected_list_file in selected_lists:
            with open(selected_list_file, "r") as f:
                selected_list = set(json.load(f))
            caption_files = [
                f for f in caption_files if _caption_item_key(f) not in selected_list
            ]
            print(f"  {len(caption_files)} caption files after ignoring")

    n_workers = cpu_count()

    # Stream results and build sparse matrix in chunks to bound memory.
    CHUNK = 5000
    item_ids = []
    chunk_rows, chunk_cols, chunk_vals = [], [], []
    chunk_row_idx = 0
    sparse_chunks = []
    total_nnz = 0

    with Pool(
        processes=n_workers, initializer=_worker_init, initargs=(triplet_idx,)
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(
                partial(_process_caption, count_once=count_once),
                caption_files,
                chunksize=64,
            ),
            total=len(caption_files),
            desc="Parsing captions",
        ):
            if result is None:
                continue
            data_info, counts = result
            item_ids.append(data_info)
            for col_idx, count in counts.items():
                chunk_rows.append(chunk_row_idx)
                chunk_cols.append(col_idx)
                chunk_vals.append(count)
            chunk_row_idx += 1

            if chunk_row_idx >= CHUNK:
                mat = sparse.csr_matrix(
                    (np.array(chunk_vals, dtype=np.float32), (chunk_rows, chunk_cols)),
                    shape=(chunk_row_idx, M),
                )
                total_nnz += mat.nnz
                sparse_chunks.append(mat)
                chunk_rows, chunk_cols, chunk_vals = [], [], []
                chunk_row_idx = 0

    # Flush remaining
    if chunk_row_idx > 0:
        mat = sparse.csr_matrix(
            (np.array(chunk_vals, dtype=np.float32), (chunk_rows, chunk_cols)),
            shape=(chunk_row_idx, M),
        )
        total_nnz += mat.nnz
        sparse_chunks.append(mat)
        del chunk_rows, chunk_cols, chunk_vals

    N = len(item_ids)
    if N == 0:
        sys.exit(
            "Error: no caption JSON files could be parsed from --caption-dir "
            "(expected stage-8 match outputs)."
        )
    print(f"Successfully parsed {N} captions. Merging {len(sparse_chunks)} chunks...")

    caption_counts = sparse.vstack(sparse_chunks, format="csr")
    del sparse_chunks

    dense_size = N * M * 4 / 1e9
    print(
        f"  Sparse matrix: {N}x{M}, {total_nnz:,} nonzeros "
        f"(density {total_nnz / (N * M):.6f}, saved {dense_size:.1f} GB)"
    )

    return caption_counts, item_ids, N


# ---------------------------------------------------------------------------
# B. Objective & Residual Utilities
# ---------------------------------------------------------------------------


def compute_residual(T, caption_counts, selected_indices):
    """Compute residual r = T - sum_i(c_i) for selected items. Returns dense float64 vector."""
    coverage = np.zeros(len(T), dtype=np.float64)
    for idx in selected_indices:
        coverage += caption_counts[idx].toarray().ravel()
    return T.astype(np.float64) - coverage


def l1_objective(residual):
    """L1 norm of residual vector."""
    return float(np.sum(np.abs(residual)))


def add_delta_sparse_row(row_csc_indices, row_csc_data, residual):
    """Delta in L1 objective from adding an item with given sparse row entries.

    delta = sum over nonzeros j: |r_j - c_ij| - |r_j|
    Negative delta means improvement.
    """
    delta = 0.0
    for j, cij in zip(row_csc_indices, row_csc_data):
        rj = residual[j]
        delta += abs(rj - cij) - abs(rj)
    return delta


def remove_delta_sparse_row(row_csc_indices, row_csc_data, residual):
    """Delta in L1 objective from removing an item with given sparse row entries.

    delta = sum over nonzeros j: |r_j + c_ij| - |r_j|
    Negative delta means improvement (removal helps).
    """
    delta = 0.0
    for j, cij in zip(row_csc_indices, row_csc_data):
        rj = residual[j]
        delta += abs(rj + cij) - abs(rj)
    return delta


def kl_objective(T, coverage):
    """KL(target_dist || coverage_dist). Lower = better."""
    eps = 1e-10
    t_sum = float(T.sum())
    c_sum = float(coverage.sum())
    if t_sum == 0 or c_sum == 0:
        return float("inf")
    p = np.asarray(T, dtype=np.float64) / t_sum
    q = np.maximum(np.asarray(coverage, dtype=np.float64) / c_sum, eps)
    # Only sum where p > 0 to avoid 0*log(0)
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def add_delta_kl(row_indices, row_data, coverage, p_target, coverage_sum):
    """KL change from adding an item. Negative = improvement.

    delta = KL(p || q') - KL(p || q)
          = -sum_j p_j * log((coverage_j + c_ij) / coverage_j)
            + log((coverage_sum + row_sum) / coverage_sum)
    """
    eps = 1e-10
    row_sum = 0.0
    gain = 0.0
    for j, cij in zip(row_indices, row_data):
        cj = max(float(coverage[j]), eps)
        gain += p_target[j] * math.log((cj + cij) / cj)
        row_sum += cij
    # gain is the KL reduction (positive = good), so delta = -gain + norm_penalty
    norm_penalty = math.log((coverage_sum + row_sum) / max(coverage_sum, eps))
    return -gain + norm_penalty  # negative = improvement


def remove_delta_kl(row_indices, row_data, coverage, p_target, coverage_sum):
    """KL change from removing an item. Negative = improvement.

    delta = KL(p || q') - KL(p || q) where q' has item removed.
    """
    eps = 1e-10
    row_sum = 0.0
    gain = 0.0
    for j, cij in zip(row_indices, row_data):
        cj = max(float(coverage[j]), eps)
        new_cj = max(cj - cij, eps)
        gain += p_target[j] * (math.log(new_cj) - math.log(cj))
        row_sum += cij
    # gain is negative when removal hurts (most of the time), positive when it helps
    norm_penalty = math.log(max(coverage_sum - row_sum, eps) / max(coverage_sum, eps))
    return gain - norm_penalty  # = KL_new - KL_old; negative = improvement


def _get_sparse_row(csr_matrix, row_idx):
    """Extract column indices and data values for a single CSR row."""
    start = csr_matrix.indptr[row_idx]
    end = csr_matrix.indptr[row_idx + 1]
    return csr_matrix.indices[start:end], csr_matrix.data[start:end]


# ---------------------------------------------------------------------------
# C. Greedy Optimizer
# ---------------------------------------------------------------------------


def solve_greedy(caption_counts, T, budget, penalty, N, metric="dot_product"):
    """Greedy selection with configurable scoring metric.

    All metrics produce scores where higher = better (compatible with argmax).

    metric:
        "dot_product"      — c_i · effective_gap (residual-weighted)
        "dot_product_norm" — (c_i · effective_gap) / ||c_i||_2 (normalized by item size)
        "l1_delta"         — L1 reduction from adding item i (negated marginal L1 change)
        "kl_divergence"    — marginal KL(target || coverage) reduction from adding item i

    Returns the selected_indices list.
    """
    M = len(T)
    selected_indices = []
    current = np.zeros(M, dtype=np.float32)
    selected_mask = np.zeros(N, dtype=bool)

    # Pre-compute CSR for efficient row access
    csr = (
        caption_counts.tocsr()
        if not sparse.isspmatrix_csr(caption_counts)
        else caption_counts
    )

    # Pre-compute row L2 norms for dot_product_norm (avoids recomputing every step)
    if metric == "dot_product_norm":
        row_norms = sparse.linalg.norm(csr, axis=1)  # shape (N,)
        row_norms[row_norms < 1e-12] = 1.0  # avoid division by zero for empty rows

    # Pre-compute normalized target for kl_divergence (fixed across steps)
    if metric == "kl_divergence":
        t_sum = float(T.sum())
        p_target = (
            (T / t_sum).astype(np.float64)
            if t_sum > 0
            else np.zeros(M, dtype=np.float64)
        )
        # Pre-compute row L1 sums for normalization penalty
        row_l1 = np.asarray(csr.sum(axis=1), dtype=np.float64).ravel()

    KL_EPS = 1e-10  # smoothing to avoid log(0)

    for step in tqdm(range(budget), desc="Greedy selection"):
        residual = T - current  # dense float32

        if metric in ("dot_product", "dot_product_norm"):
            effective_gap = np.maximum(0.0, residual) - penalty * np.maximum(
                0.0, -residual
            )
            # Sparse CSR matvec: O(nnz) instead of O(N*M)
            kp_scores = caption_counts @ effective_gap
            if metric == "dot_product_norm":
                kp_scores = kp_scores / row_norms

        elif metric == "l1_delta":
            # Raw L1 delta: delta_i = sum_j (|r_j - c_ij| - |r_j|)
            # Negative = improvement; selection uses argmin below (not argmax).
            abs_residual = np.abs(residual)
            r_at_cols = abs_residual[csr.indices]
            shifted = np.abs(residual[csr.indices] - csr.data)
            delta_vals = shifted - r_at_cols  # |r_j - c_ij| - |r_j|
            delta_sparse = sparse.csr_matrix(
                (delta_vals, csr.indices, csr.indptr), shape=csr.shape
            )
            kp_scores = np.asarray(delta_sparse.sum(axis=1)).ravel()

        elif metric == "kl_divergence":
            # Marginal KL reduction: KL(p || q) - KL(p || q') where
            #   p = T / ||T||_1 (target distribution, fixed)
            #   q = current / ||current||_1 (current coverage distribution)
            #   q' = (current + c_i) / ||current + c_i||_1 (after adding item i)
            #
            # = sum_j p_j * log(q'_j / q_j)
            # = sum_j p_j * log((current_j + c_ij) / current_j)   [per-feature gain]
            #   - log((sum_current + ||c_i||_1) / sum_current)     [normalization penalty]
            #
            # Higher = more KL reduction = better.
            current_f64 = np.maximum(current.astype(np.float64), KL_EPS)
            sum_current = max(float(current.sum()), KL_EPS)

            # Per-nonzero contribution: p_j * log((current_j + c_ij) / current_j)
            cur_at_cols = current_f64[csr.indices]
            p_at_cols = p_target[csr.indices]
            log_ratios = p_at_cols * np.log(
                (cur_at_cols + csr.data.astype(np.float64)) / cur_at_cols
            )
            gain_sparse = sparse.csr_matrix(
                (log_ratios, csr.indices, csr.indptr), shape=csr.shape
            )
            kl_gains = np.asarray(gain_sparse.sum(axis=1)).ravel()

            # Normalization penalty: log(1 + ||c_i||_1 / sum_current)
            norm_penalty = np.log1p(row_l1 / sum_current)

            kp_scores = kl_gains - norm_penalty

        scores = kp_scores

        if metric == "l1_delta":
            # l1_delta: negative = better, use argmin; mask with +inf
            kp_scores[selected_mask] = np.inf
            best_idx = int(np.argmin(kp_scores))
            if kp_scores[best_idx] == np.inf:
                print(f"Pool exhausted at step {step}. Stopping early.")
                break
        else:
            # dot_product, dot_product_norm, kl_divergence: higher = better, use argmax
            scores[selected_mask] = -np.inf
            best_idx = int(np.argmax(scores))
            if scores[best_idx] == -np.inf:
                print(f"Pool exhausted at step {step}. Stopping early.")
                break

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True
        current += caption_counts[best_idx].toarray().ravel()

    return selected_indices


# ---------------------------------------------------------------------------
# D. HiGHS Model Builder
# ---------------------------------------------------------------------------


def _require_highspy(method_name):
    """Raise a clear error if highspy is not installed."""
    if not HAVE_HIGHSPY:
        raise RuntimeError(
            f"--optimizer {method_name} requires the 'highspy' package "
            f"(a core dependency). Install the project dependencies: uv sync"
        )


def build_highs_l1(
    caption_counts_csc,
    target,
    budget,
    relax_integrality,
    time_limit_sec=300.0,
    mip_gap=0.0,
):
    """Build and solve an L1 minimization LP/MIP using HiGHS from a sparse CSC matrix.

    Formulation:
        min  sum_k t_k
        s.t. sum_i x_i == budget
             sum_i(c_ij * x_i) - t_k <= T[j]     for each active feature j (mapped to k)
            -sum_i(c_ij * x_i) - t_k <= -T[j]    for each active feature j (mapped to k)
             x_i in {0,1} (MIP) or [0,1] (LP)
             t_k >= 0

    Args:
        caption_counts_csc: scipy sparse CSC matrix (N x M), float64
        target: dense float64 array of length M
        budget: int, exact number of items to select
        relax_integrality: bool, True for LP relaxation, False for MIP
        time_limit_sec: wall-clock time limit
        mip_gap: relative MIP gap tolerance

    Returns dict with keys: status, objective, best_bound, gap, x_values,
        fractional_count, runtime_sec, iterations, dead_feature_obj.
    """
    start = time.perf_counter()

    csc = caption_counts_csc
    N_items = csc.shape[0]
    M_features = csc.shape[1]
    if N_items < budget:
        sys.exit(
            f"Error: --budget {budget} exceeds the pool size ({N_items} items); "
            "the budget constraint is an equality, so the model would be "
            "infeasible. Lower --budget or use --optimizer greedy."
        )
    indptr = csc.indptr
    indices = csc.indices
    data = csc.data

    # Identify active features: T[j] > 0 OR column j has nonzeros
    active_features = []
    dead_feature_obj = 0.0
    for j in range(M_features):
        col_nnz = indptr[j + 1] - indptr[j]
        if target[j] != 0.0 or col_nnz > 0:
            active_features.append(j)
        else:
            # T[j]==0 and no items cover it: contributes |T[j]|=0 to objective regardless
            dead_feature_obj += abs(target[j])

    n_active = len(active_features)
    n_cols = N_items + n_active

    # Build model
    model = Highs()
    model.setOptionValue("output_flag", False)
    model.setOptionValue("time_limit", float(time_limit_sec))
    if not relax_integrality:
        model.setOptionValue("mip_rel_gap", float(mip_gap))

    # Add columns: N item variables x_i + n_active slack variables t_k
    costs = np.concatenate(
        [
            np.zeros(N_items, dtype=np.float64),
            np.ones(n_active, dtype=np.float64),
        ]
    )
    lowers = np.zeros(n_cols, dtype=np.float64)
    uppers = np.concatenate(
        [
            np.ones(N_items, dtype=np.float64),
            np.full(n_active, np.inf, dtype=np.float64),
        ]
    )
    empty_int = np.array([], dtype=np.int32)
    empty_float = np.array([], dtype=np.float64)
    model.addCols(n_cols, costs, lowers, uppers, 0, empty_int, empty_int, empty_float)

    if not relax_integrality:
        model.changeColsIntegrality(
            N_items,
            np.arange(N_items, dtype=np.int32),
            np.full(N_items, HighsVarType.kInteger, dtype=np.uint8),
        )

    # Build constraint rows
    # Row 0: budget constraint  sum(x_i) = budget
    # Rows 1+2k, 1+2k+1: absolute value pair for each active feature k
    n_rows = 1 + 2 * n_active
    row_lowers = np.full(n_rows, -np.inf, dtype=np.float64)
    row_uppers = np.empty(n_rows, dtype=np.float64)

    # Pre-compute total nonzeros for row-compressed format
    # Row 0: N_items entries (all x_i with coefficient 1.0)
    # Each feature pair: col_nnz items + 1 slack (x2 for both rows)
    total_nnz_est = N_items  # budget row
    for k, j in enumerate(active_features):
        col_nnz = indptr[j + 1] - indptr[j]
        total_nnz_est += 2 * (
            col_nnz + 1
        )  # two rows per feature, each with items + slack

    all_starts = np.empty(n_rows, dtype=np.int32)
    all_indices = np.empty(total_nnz_est, dtype=np.int32)
    all_values = np.empty(total_nnz_est, dtype=np.float64)

    # Row 0: budget constraint
    all_starts[0] = 0
    row_lowers[0] = float(budget)
    row_uppers[0] = float(budget)
    all_indices[:N_items] = np.arange(N_items, dtype=np.int32)
    all_values[:N_items] = 1.0
    nnz = N_items

    for k, j in enumerate(active_features):
        t_col = N_items + k
        rhs = float(target[j])
        col_start = indptr[j]
        col_end = indptr[j + 1]
        col_item_indices = indices[
            col_start:col_end
        ]  # item indices that cover feature j
        col_item_values = data[col_start:col_end]  # their counts c_ij
        n_col = len(col_item_indices)

        # Row 1+2k: sum_i(c_ij * x_i) - t_k <= T[j]
        row_idx = 1 + 2 * k
        all_starts[row_idx] = nnz
        row_uppers[row_idx] = rhs
        all_indices[nnz : nnz + n_col] = col_item_indices.astype(np.int32)
        all_values[nnz : nnz + n_col] = col_item_values.astype(np.float64)
        nnz += n_col
        all_indices[nnz] = t_col
        all_values[nnz] = -1.0
        nnz += 1

        # Row 1+2k+1: -sum_i(c_ij * x_i) - t_k <= -T[j]
        row_idx2 = 1 + 2 * k + 1
        all_starts[row_idx2] = nnz
        row_uppers[row_idx2] = -rhs
        all_indices[nnz : nnz + n_col] = col_item_indices.astype(np.int32)
        all_values[nnz : nnz + n_col] = -col_item_values.astype(np.float64)
        nnz += n_col
        all_indices[nnz] = t_col
        all_values[nnz] = -1.0
        nnz += 1

    # Trim to actual size (should match total_nnz_est exactly)
    all_indices = all_indices[:nnz]
    all_values = all_values[:nnz]

    model.addRows(
        n_rows,
        row_lowers,
        row_uppers,
        nnz,
        all_starts,
        all_indices,
        all_values,
    )

    model.run()

    info = model.getInfo()
    solution = model.getSolution()
    x_values = np.array(solution.col_value[:N_items], dtype=np.float64)
    runtime = time.perf_counter() - start
    fractional_count = int(np.sum((x_values > 1e-6) & (x_values < 1.0 - 1e-6)))

    return {
        "status": model.modelStatusToString(model.getModelStatus()),
        "objective": info.objective_function_value + dead_feature_obj,
        "best_bound": info.mip_dual_bound + dead_feature_obj,
        "gap": info.mip_gap,
        "x_values": x_values,
        "fractional_count": fractional_count,
        "runtime_sec": runtime,
        "iterations": info.mip_node_count or info.simplex_iteration_count,
        "dead_feature_obj": dead_feature_obj,
    }


# ---------------------------------------------------------------------------
# E. LP Relaxation + Rounding
# ---------------------------------------------------------------------------


def solve_lp_round(caption_counts, T, budget, time_limit_sec=300.0):
    """Solve LP relaxation and round by taking top-B items by decreasing x*.

    Returns (selected_indices, lp_objective, rounded_objective, lp_result).
    """
    _require_highspy("lp_round")

    csc = caption_counts.tocsc().astype(np.float64)
    target = T.astype(np.float64)

    print("Building and solving LP relaxation...")
    lp_result = build_highs_l1(
        csc, target, budget, relax_integrality=True, time_limit_sec=time_limit_sec
    )
    print(
        f"  LP status: {lp_result['status']}, objective: {lp_result['objective']:.4f}, "
        f"fractional: {lp_result['fractional_count']}, time: {lp_result['runtime_sec']:.1f}s"
    )

    # Round: take top-B items by decreasing x*
    x_values = lp_result["x_values"]
    order = np.argsort(-x_values)
    selected_indices = []
    for item_idx in order:
        if len(selected_indices) >= budget:
            break
        if x_values[item_idx] <= 1e-8:
            break
        selected_indices.append(int(item_idx))

    # Compute true L1 objective of rounded solution
    residual = compute_residual(T, caption_counts, selected_indices)
    rounded_obj = l1_objective(residual)
    print(
        f"  Rounded solution: {len(selected_indices)} items, L1 objective: {rounded_obj:.4f}"
    )

    return selected_indices, lp_result["objective"], rounded_obj, lp_result


# ---------------------------------------------------------------------------
# F. Direct MIP
# ---------------------------------------------------------------------------


def solve_mip(caption_counts, T, budget, time_limit_sec=300.0, mip_gap=0.0):
    """Solve full binary MIP with HiGHS.

    Returns (selected_indices, objective, best_bound, gap, mip_result).
    """
    _require_highspy("mip")

    csc = caption_counts.tocsc().astype(np.float64)
    target = T.astype(np.float64)

    print("Building and solving MIP...")
    mip_result = build_highs_l1(
        csc,
        target,
        budget,
        relax_integrality=False,
        time_limit_sec=time_limit_sec,
        mip_gap=mip_gap,
    )
    print(
        f"  MIP status: {mip_result['status']}, objective: {mip_result['objective']:.4f}, "
        f"bound: {mip_result['best_bound']:.4f}, gap: {mip_result['gap']:.6f}, "
        f"time: {mip_result['runtime_sec']:.1f}s"
    )

    # Threshold x at 0.5 to get selection
    x_values = mip_result["x_values"]
    selected_indices = [int(i) for i in range(len(x_values)) if x_values[i] >= 0.5]

    # Compute true L1 objective
    residual = compute_residual(T, caption_counts, selected_indices)
    true_obj = l1_objective(residual)
    print(
        f"  MIP solution: {len(selected_indices)} items, true L1 objective: {true_obj:.4f}"
    )

    return (
        selected_indices,
        true_obj,
        mip_result["best_bound"],
        mip_result["gap"],
        mip_result,
    )


# ---------------------------------------------------------------------------
# G. Sparse Local Search
# ---------------------------------------------------------------------------


def solve_local_search(caption_counts, T, budget, initial_selected, metric="l1"):
    """Local search with lazy heap evaluation.

    metric:
        "l1" — minimize L1 residual (default, used by LNS internally)
        "kl" — minimize KL(target_dist || coverage_dist)

    Both use the same heap machinery; only the delta functions differ.

    Returns (selected_indices, objective, iterations).
    """
    start = time.perf_counter()
    N_items, M = caption_counts.shape
    T64 = T.astype(np.float64)

    # Ensure CSR for efficient row access
    csr = (
        caption_counts.tocsr()
        if not sparse.isspmatrix_csr(caption_counts)
        else caption_counts
    )

    # Initialize selection
    selected = np.zeros(N_items, dtype=bool)
    initial_set = set()
    if initial_selected is not None:
        for idx in initial_selected:
            if 0 <= idx < N_items and not selected[idx]:
                selected[idx] = True
                initial_set.add(idx)
        # Trim to budget if needed
        if len(initial_set) > budget:
            kept = sorted(initial_set)[:budget]
            selected[:] = False
            initial_set = set(kept)
            for idx in kept:
                selected[idx] = True

    # Compute initial coverage (needed for both metrics)
    coverage = np.zeros(M, dtype=np.float64)
    for idx in initial_set:
        row_indices, row_data = _get_sparse_row(csr, idx)
        for j, v in zip(row_indices, row_data):
            coverage[j] += v

    selected_count = len(initial_set)

    # Set up metric-specific delta functions and objective
    use_kl = metric == "kl"
    if use_kl:
        t_sum = float(T64.sum())
        p_target = (T64 / t_sum) if t_sum > 0 else np.zeros(M, dtype=np.float64)
        coverage_sum = float(coverage.sum())
        objective = kl_objective(T64, coverage)

        def _add_delta(row_indices, row_data):
            return add_delta_kl(row_indices, row_data, coverage, p_target, coverage_sum)

        def _remove_delta(row_indices, row_data):
            return remove_delta_kl(
                row_indices, row_data, coverage, p_target, coverage_sum
            )
    else:
        residual = T64 - coverage
        objective = float(np.sum(np.abs(residual)))

        def _add_delta(row_indices, row_data):
            return add_delta_sparse_row(row_indices, row_data, residual)

        def _remove_delta(row_indices, row_data):
            return remove_delta_sparse_row(row_indices, row_data, residual)

    iterations = 0

    # Generation counters for lazy evaluation
    add_gen = np.zeros(N_items, dtype=np.int64)
    remove_gen = np.zeros(N_items, dtype=np.int64)
    global_gen = np.int64(0)

    # Build initial heaps: (delta, generation, item_idx)
    add_heap = []
    remove_heap = []

    for i in range(N_items):
        row_indices, row_data = _get_sparse_row(csr, i)
        if selected[i]:
            delta = _remove_delta(row_indices, row_data)
            remove_gen[i] = global_gen
            heapq.heappush(remove_heap, (delta, int(global_gen), i))
        else:
            delta = _add_delta(row_indices, row_data)
            add_gen[i] = global_gen
            heapq.heappush(add_heap, (delta, int(global_gen), i))

    def clean_add_heap():
        """Pop the best valid add move, recomputing if stale."""
        while add_heap:
            delta, gen, idx = heapq.heappop(add_heap)
            if selected[idx]:
                continue
            if gen == add_gen[idx]:
                return delta, idx
            row_indices, row_data = _get_sparse_row(csr, idx)
            true_delta = _add_delta(row_indices, row_data)
            add_gen[idx] = global_gen
            heapq.heappush(add_heap, (true_delta, int(global_gen), idx))
        return None

    def clean_remove_heap():
        """Pop the best valid remove move, recomputing if stale."""
        while remove_heap:
            delta, gen, idx = heapq.heappop(remove_heap)
            if not selected[idx]:
                continue
            if gen == remove_gen[idx]:
                return delta, idx
            row_indices, row_data = _get_sparse_row(csr, idx)
            true_delta = _remove_delta(row_indices, row_data)
            remove_gen[idx] = global_gen
            heapq.heappush(remove_heap, (true_delta, int(global_gen), idx))
        return None

    while True:
        best_add = clean_add_heap()
        best_remove = clean_remove_heap()

        add_move = (
            best_add is not None and best_add[0] < -1e-12 and selected_count < budget
        )
        remove_move = best_remove is not None and best_remove[0] < -1e-12

        if not add_move and not remove_move:
            break

        use_add = False
        if add_move and remove_move:
            use_add = best_add[0] <= best_remove[0]
        elif add_move:
            use_add = True

        global_gen += 1

        if use_add:
            delta, idx = best_add
            selected[idx] = True
            initial_set.add(idx)
            selected_count += 1
            objective += delta

            row_indices, row_data = _get_sparse_row(csr, idx)
            if use_kl:
                for j, v in zip(row_indices, row_data):
                    coverage[j] += v
                coverage_sum += float(row_data.sum())
            else:
                for j, v in zip(row_indices, row_data):
                    residual[j] -= v

            rm_delta = _remove_delta(row_indices, row_data)
            remove_gen[idx] = global_gen
            heapq.heappush(remove_heap, (rm_delta, int(global_gen), idx))
        else:
            delta, idx = best_remove
            selected[idx] = False
            initial_set.discard(idx)
            selected_count -= 1
            objective += delta

            row_indices, row_data = _get_sparse_row(csr, idx)
            if use_kl:
                for j, v in zip(row_indices, row_data):
                    coverage[j] -= v
                coverage_sum -= float(row_data.sum())
            else:
                for j, v in zip(row_indices, row_data):
                    residual[j] += v

            ad_delta = _add_delta(row_indices, row_data)
            add_gen[idx] = global_gen
            heapq.heappush(add_heap, (ad_delta, int(global_gen), idx))

        iterations += 1

    runtime = time.perf_counter() - start
    selected_indices = sorted(initial_set)
    metric_label = "KL" if use_kl else "L1"
    print(
        f"  Local search ({metric_label}): {iterations} iterations, objective: {objective:.6f}, "
        f"time: {runtime:.1f}s"
    )
    return selected_indices, objective, iterations


# ---------------------------------------------------------------------------
# H. LNS Optimizer
# ---------------------------------------------------------------------------


def solve_lns(caption_counts, T, budget, args):
    """LP-seeded LNS: LP round -> local search polish -> destroy-repair loop.

    Returns (selected_indices, objective, lns_metrics).
    """
    _require_highspy("lns")

    rng = random.Random(args.lns_seed)
    start = time.perf_counter()

    # Step 1: Seed with LP rounding
    print("LNS Step 1: LP seed...")
    lp_selected, lp_obj, rounded_obj, lp_result = solve_lp_round(
        caption_counts, T, budget, time_limit_sec=args.time_limit
    )

    # Step 2: Polish with local search
    print("LNS Step 2: Local search polish...")
    current_selected, current_objective, ls_iters = solve_local_search(
        caption_counts, T, budget, initial_selected=lp_selected
    )
    seed_objective = current_objective
    seed_selected_count = len(current_selected)
    print(
        f"  Seed objective after polish: {current_objective:.4f} ({seed_selected_count} items)"
    )

    accepted_moves = 0
    neighborhood_sizes = []
    repair_runtimes = []

    csr = (
        caption_counts.tocsr()
        if not sparse.isspmatrix_csr(caption_counts)
        else caption_counts
    )
    N_items = caption_counts.shape[0]

    destroy_size = max(1, int(round(args.lns_destroy_fraction * budget)))
    add_pool_size = max(1, int(round(args.lns_add_pool_multiplier * destroy_size)))

    for iteration in range(args.lns_iterations):
        print(f"LNS iteration {iteration + 1}/{args.lns_iterations}...")
        current_set = set(current_selected)
        if not current_set:
            break

        # Compute residual for current solution
        residual = compute_residual(T, caption_counts, current_selected)

        # Rank selected items by remove_delta (worst = highest delta, most damaging to remove)
        selected_list = list(current_selected)
        remove_ranked = []
        for item_idx in selected_list:
            row_indices, row_data = _get_sparse_row(csr, item_idx)
            rd = remove_delta_sparse_row(row_indices, row_data, residual)
            remove_ranked.append((rd, item_idx))
        remove_ranked.sort(key=lambda pair: (pair[0], pair[1]))

        # Destroy: worst 70% of destroy_size + random 30%
        destroy_size_eff = min(destroy_size, len(selected_list))
        greedy_take = max(1, int(round(0.7 * destroy_size_eff)))
        destroy = {item_idx for _, item_idx in remove_ranked[:greedy_take]}
        remaining_selected = [item_idx for _, item_idx in remove_ranked[greedy_take:]]
        if len(destroy) < destroy_size_eff and remaining_selected:
            extra = min(destroy_size_eff - len(destroy), len(remaining_selected))
            destroy.update(rng.sample(remaining_selected, extra))

        fixed_selected = current_set - destroy

        # Compute repair residual (residual after removing destroyed items)
        repair_residual = residual.copy()
        for item_idx in destroy:
            row_indices, row_data = _get_sparse_row(csr, item_idx)
            for j, v in zip(row_indices, row_data):
                repair_residual[j] += v

        # Add candidates: best (1-rho)*add_pool_size by add_delta + random rho fraction
        add_scores = []
        for item_idx in range(N_items):
            if item_idx in current_set:
                continue
            row_indices, row_data = _get_sparse_row(csr, item_idx)
            ad = add_delta_sparse_row(row_indices, row_data, repair_residual)
            add_scores.append((ad, item_idx))
        add_scores.sort(key=lambda pair: (pair[0], pair[1]))

        random_take = int(round(add_pool_size * args.lns_random_add_fraction))
        greedy_take_add = max(0, add_pool_size - random_take)
        add_candidates = [item_idx for _, item_idx in add_scores[:greedy_take_add]]
        random_pool = [
            item_idx
            for _, item_idx in add_scores[
                greedy_take_add : greedy_take_add + 4 * max(1, random_take)
            ]
        ]
        if random_take > 0 and random_pool:
            add_candidates.extend(
                rng.sample(random_pool, min(random_take, len(random_pool)))
            )

        neighborhood_items = sorted(destroy | set(add_candidates))
        neighborhood_sizes.append(len(neighborhood_items))
        if not neighborhood_items:
            continue

        # Build compressed repair subproblem
        subproblem_budget = max(0, budget - len(fixed_selected))

        # Build a submatrix for the neighborhood items
        neighborhood_list = list(neighborhood_items)
        n_sub = len(neighborhood_list)
        sub_matrix = caption_counts[neighborhood_list, :]

        # Identify active features in this subproblem
        sub_csc = sub_matrix.tocsc().astype(np.float64)
        repair_target = repair_residual.astype(np.float64)

        repair_start = time.perf_counter()
        repair_result = build_highs_l1(
            sub_csc,
            repair_target,
            subproblem_budget,
            relax_integrality=False,
            time_limit_sec=args.lns_repair_time_limit,
            mip_gap=0.0,
        )
        repair_runtimes.append(time.perf_counter() - repair_start)

        # Lift solution back to full problem
        repaired_selection = set()
        for local_idx in range(n_sub):
            if repair_result["x_values"][local_idx] >= 0.5:
                repaired_selection.add(neighborhood_list[local_idx])

        candidate_selected = sorted(fixed_selected | repaired_selection)

        # Polish with local search
        polished_selected, polished_obj, _ = solve_local_search(
            caption_counts, T, budget, initial_selected=candidate_selected
        )

        if polished_obj < current_objective - 1e-12:
            print(f"  Accepted: {current_objective:.4f} -> {polished_obj:.4f}")
            current_objective = polished_obj
            current_selected = polished_selected
            accepted_moves += 1
        else:
            print(f"  Rejected: {polished_obj:.4f} >= {current_objective:.4f}")

    runtime = time.perf_counter() - start
    lns_metrics = {
        "seed_objective": seed_objective,
        "seed_selected_count": seed_selected_count,
        "accepted_moves": accepted_moves,
        "lns_iterations": args.lns_iterations,
        "avg_neighborhood_size": (sum(neighborhood_sizes) / len(neighborhood_sizes))
        if neighborhood_sizes
        else 0.0,
        "avg_repair_runtime_sec": (sum(repair_runtimes) / len(repair_runtimes))
        if repair_runtimes
        else 0.0,
        "total_runtime_sec": runtime,
        "lp_bound": lp_result["objective"],
    }
    print(
        f"LNS complete: objective={current_objective:.4f}, accepted={accepted_moves}/{args.lns_iterations}, "
        f"time={runtime:.1f}s"
    )

    return current_selected, current_objective, lns_metrics


# ---------------------------------------------------------------------------
# I. CLI Args & Output
# ---------------------------------------------------------------------------


def argparser():
    parser = argparse.ArgumentParser(
        description="Optimization-based caption sampling with multiple solvers"
    )
    parser.add_argument(
        "--atlas",
        type=str,
        required=True,
        help="Aggregated target-frequency JSON from aggregate_match_result.py "
        "(must contain ap_frequency; not the atlas.json itself)",
    )
    parser.add_argument(
        "--budget", type=int, required=True, help="Number of captions to select"
    )
    parser.add_argument(
        "--caption-dir",
        type=str,
        required=True,
        help="Directory with per-item match JSONs from "
        "match_data_to_k_atlas_faiss.py (the training pool)",
    )
    parser.add_argument(
        "--selected-lists",
        nargs="+",
        default=None,
        help="Optional prior-run selection JSON files (the "
        "*_selection.json written by this script) whose ids "
        "are prepended to this run's selection",
    )
    parser.add_argument(
        "--count-once",
        action="store_true",
        help="Count each AP only once per caption",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, required=True, help="Output folder"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="Penalty weight for adding triplets already at or over target; only "
        "affects the greedy dot_product/dot_product_norm metrics (default: 1.0)",
    )
    parser.add_argument(
        "--reweight",
        type=str,
        default="proportional",
        choices=[
            "proportional",
            "uniform",
            "repeat-factor",
            "power-law",
            "power-law-cap",
        ],
        help="Triplet reweighting strategy (default: proportional)",
    )
    parser.add_argument(
        "--rfs-threshold",
        type=float,
        default=0.01,
        help="Frequency threshold for repeat-factor sampling (default: 0.01)",
    )
    parser.add_argument(
        "--num-test-docs",
        type=int,
        default=0,
        help="Number of test documents used to build the atlas. "
        "Used as denominator for RFS frequency (f = count / num_test_docs). "
        "Required when --reweight=repeat-factor.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Exponent for power-law reweighting (default: 0.5, 1.0=proportional, 0.0=uniform)",
    )
    parser.add_argument(
        "--cap-percentile",
        type=float,
        default=75.0,
        help="Percentile (0-100) of freq^alpha distribution used as hard cap for power-law-cap mode (default: 75)",
    )
    parser.add_argument("--exp-idx", type=int, required=True, help="Experiment index")
    parser.add_argument(
        "--prefix", type=str, default="", help="Prefix of the output directory"
    )
    # New optimizer args
    parser.add_argument(
        "--optimizer",
        type=str,
        default="greedy",
        choices=["greedy", "lp_round", "mip", "lns"],
        help="Selection optimizer to use (default: greedy)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="dot_product",
        choices=["dot_product", "dot_product_norm", "l1_delta", "kl_divergence"],
        help="Greedy scoring metric: dot_product (residual-weighted), "
        "dot_product_norm (divided by row L2 norm — prevents dense items from dominating), "
        "l1_delta (exact L1 marginal reduction), or "
        "kl_divergence (marginal KL(target||coverage) reduction). "
        "Only affects --optimizer greedy. (default: dot_product)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="MIP/LP wall-clock time limit in seconds (default: 300.0)",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="MIP relative gap tolerance (default: 0.0)",
    )
    parser.add_argument(
        "--lns-iterations",
        type=int,
        default=8,
        help="Number of LNS destroy-repair iterations (default: 8)",
    )
    parser.add_argument(
        "--lns-destroy-fraction",
        type=float,
        default=0.1,
        help="Fraction of budget to destroy each LNS iteration (default: 0.1)",
    )
    parser.add_argument(
        "--lns-add-pool-multiplier",
        type=float,
        default=3.0,
        help="Multiplier on destroy_size for add candidate pool (default: 3.0)",
    )
    parser.add_argument(
        "--lns-repair-time-limit",
        type=float,
        default=5.0,
        help="Per-repair MIP time limit in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--lns-seed", type=int, default=42, help="Random seed for LNS (default: 42)"
    )
    parser.add_argument(
        "--lns-random-add-fraction",
        type=float,
        default=0.2,
        help="Fraction of add pool filled randomly vs greedily (default: 0.2)",
    )
    parser.add_argument(
        "--polish",
        type=str,
        default="none",
        choices=["none", "l1", "kl"],
        help="Post-optimization local search polish: none (skip), l1 (minimize L1 residual), "
        "or kl (minimize KL divergence). Applied to the output of any optimizer. (default: none)",
    )
    return parser.parse_args()


def main(args):
    # Per-file parse warnings go to a log inside the output folder.
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / "optimizers.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print(f"Warnings are logged to {log_path}")

    # --- Load atlas and build target distribution ---
    with open(args.atlas, "r") as f:
        atlas_dict = json.load(f)

    T, triplet_list, triplet_idx, M, reweight_info = build_target_vector(
        atlas_dict,
        args.budget,
        args.reweight,
        args.rfs_threshold,
        args.num_test_docs,
        args.alpha,
        args.cap_percentile,
    )
    print(f"Triplet space: {M} triplets | Reweight: {args.reweight}{reweight_info}")

    # --- Parse all caption files in parallel ---
    caption_counts, item_ids, N = build_sparse_matrix(
        args.caption_dir,
        triplet_idx,
        M,
        args.count_once,
        selected_lists=args.selected_lists,
    )

    # --- Run selected optimizer ---
    print(f"Starting optimization with --optimizer {args.optimizer}...")
    method_metrics = {}

    if args.optimizer == "greedy":
        selected_indices = solve_greedy(
            caption_counts,
            T,
            args.budget,
            args.penalty,
            N,
            metric=args.metric,
        )
        residual = compute_residual(T, caption_counts, selected_indices)
        obj = l1_objective(residual)
        method_metrics["l1_objective"] = obj
        print(
            f"Greedy selection complete: {len(selected_indices)} items, L1 objective: {obj:.4f}"
        )

    elif args.optimizer == "lp_round":
        selected_indices, lp_obj, rounded_obj, lp_result = solve_lp_round(
            caption_counts, T, args.budget, time_limit_sec=args.time_limit
        )
        method_metrics["lp_bound"] = lp_obj
        method_metrics["rounded_objective"] = rounded_obj
        method_metrics["lp_status"] = lp_result["status"]
        method_metrics["lp_fractional_count"] = lp_result["fractional_count"]
        method_metrics["lp_runtime_sec"] = lp_result["runtime_sec"]

    elif args.optimizer == "mip":
        selected_indices, true_obj, best_bound, gap, mip_result = solve_mip(
            caption_counts,
            T,
            args.budget,
            time_limit_sec=args.time_limit,
            mip_gap=args.mip_gap,
        )
        method_metrics["mip_objective"] = true_obj
        method_metrics["mip_best_bound"] = best_bound
        method_metrics["mip_gap"] = gap
        method_metrics["mip_status"] = mip_result["status"]
        method_metrics["mip_runtime_sec"] = mip_result["runtime_sec"]

    elif args.optimizer == "lns":
        selected_indices, lns_obj, lns_metrics = solve_lns(
            caption_counts, T, args.budget, args
        )
        method_metrics.update(lns_metrics)
        method_metrics["lns_objective"] = lns_obj

    # --- Optional polish ---
    if args.polish != "none":
        print(f"\nPolishing with --polish {args.polish} local search...")
        selected_indices, polish_obj, polish_iters = solve_local_search(
            caption_counts,
            T,
            args.budget,
            initial_selected=selected_indices,
            metric=args.polish,
        )
        method_metrics["polish"] = args.polish
        method_metrics["polish_objective"] = polish_obj
        method_metrics["polish_iterations"] = polish_iters
        print(f"Polish complete: {len(selected_indices)} items")

    # --- Save output ---
    selected_data = [item_ids[i] for i in selected_indices]
    print(f"Selected {len(selected_data)} captions.")
    if args.selected_lists is not None:
        all_pre_selected_list = []
        for selected_list_file in args.selected_lists:
            with open(selected_list_file, "r") as f:
                pre_selected_list = json.load(f)
            all_pre_selected_list.extend(pre_selected_list)
        selected_data = all_pre_selected_list + selected_data
        print(
            f"  {len(all_pre_selected_list)} already selected, {len(selected_data)} total"
        )
        print("  Checking uniqueness...")
        print(f"  {len(set(selected_data))} unique selections")

    output_dir = Path(args.output_dir)
    if args.reweight == "repeat-factor":
        suffix = f"rfs_t{args.rfs_threshold}"
    elif args.reweight == "power-law":
        suffix = f"powerlaw_a{args.alpha}"
    elif args.reweight == "power-law-cap":
        suffix = f"powerlawcap_a{args.alpha}_p{args.cap_percentile}"
    elif args.reweight == "uniform":
        suffix = "uniform"
    else:
        suffix = "proportional"
    if args.prefix == "":
        output_dir = (
            output_dir
            / f"exp{int(args.exp_idx):02d}_{suffix}_p{args.penalty}_b{args.budget}_total_{len(selected_data)}"
        )
    else:
        output_dir = (
            output_dir
            / f"{args.prefix}{int(args.exp_idx):02d}_{suffix}_p{args.penalty}_b{args.budget}_total_{len(selected_data)}"
        )
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{args.optimizer}_selection.json"
    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(selected_data, f)

    # Save config.json provenance
    atlas_path = Path(args.atlas)
    atlas_hash = hashlib.sha256(atlas_path.read_bytes()).hexdigest()
    config = {
        "atlas": str(atlas_path.resolve()),
        "atlas_sha256": atlas_hash,
        "caption_dir": str(Path(args.caption_dir).resolve()),
        "budget": args.budget,
        "penalty": args.penalty,
        "reweight": args.reweight,
        "rfs_threshold": args.rfs_threshold,
        "alpha": args.alpha,
        "cap_percentile": args.cap_percentile,
        "selected_count": len(selected_data),
        "ap_space_size": M,
        "pool_size": N,
        "optimizer": args.optimizer,
        "metric": args.metric,
        "polish": args.polish,
    }
    # Add method-specific metrics
    config.update(method_metrics)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    args = argparser()
    main(args)
