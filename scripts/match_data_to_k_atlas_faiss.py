# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FAISS-accelerated matching of data KGs against an atlas KG.

Matches every per-item KG against the atlas KG using precomputed
embeddings and a one-shot FAISS approach:
  1. Collect all unique predicate/subject/object strings across all items.
  2. One-shot FAISS range_search per position → {string: matching atlas indices}.
  3. Per-item matching is pure dict lookups + set intersections (no matmul).

Usage:
  python match_data_to_k_atlas_faiss.py \
      output/atlas/atlas.json \
      output/data_kgs \
      -o output/match_results/entity_relation \
      --threshold 0.85 \
      --embedding-cache-path output/embedding_cache.pkl \
      --ap-match entity-relation
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

try:
    import faiss
except ImportError as exc:
    raise ImportError(
        "faiss is required for atlas matching (a core dependency). "
        "Install the project dependencies: uv sync"
    ) from exc
import numpy as np
import tqdm

import ap_embeddings
from common import acquire_lock, load_kg, normalize, write_json_atomic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_item_paths(items_folder: Path, glob: str = "graph.json") -> list[Path]:
    paths = sorted(items_folder.rglob(glob))
    if not paths:
        print(f"  Warning: no {glob} files found under {items_folder}")
    return paths


def output_path_for(item_path: Path, items_folder: Path, output_folder: Path) -> Path:
    rel = item_path.relative_to(items_folder)
    if rel.parent == Path("."):
        return output_folder / rel.name
    if item_path.name != "graph.json":
        return output_folder / rel
    return output_folder / rel.parent.with_suffix(".json")


def save_result(
    out_file, atlas_path, item_path, args, atlas_triplets, item_triplets, results
):
    total2 = len(item_triplets)
    data = {
        "atlas_path": str(atlas_path),
        "item_path": str(item_path),
        "match_mode": "embedding",
        "ap_match": args.ap_match,
        "atlas_ap_count": len(atlas_triplets),
        "item_ap_count": total2,
        "item_matched_count": results["item_matched_count"],
        "item_coverage": results["item_matched_count"] / total2 if total2 else 0.0,
        "matched_pairs": [
            {"item_ap": p["item_ap"], "atlas_matches": p["atlas_matches"]}
            for p in results["matched_pairs"]
        ],
        "ap_frequency": dict(results["ap_counter"].most_common()),
    }
    write_json_atomic(out_file, data)


# ---------------------------------------------------------------------------
# FAISS bulk precomputation
# ---------------------------------------------------------------------------


def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(matrix.astype(np.float32)))
    return index


def faiss_range_search_to_dict(
    query_strings: list[str],
    query_matrix: np.ndarray,
    index: faiss.IndexFlatIP,
    threshold: float,
) -> dict[str, np.ndarray]:
    """Run range_search and return {normalized_string: array of matching atlas indices}."""
    lims, _, neighbor_ids = index.range_search(
        np.ascontiguousarray(query_matrix.astype(np.float32)), threshold
    )
    result = {}
    for i, s in enumerate(query_strings):
        indices = neighbor_ids[lims[i] : lims[i + 1]]
        if len(indices) > 0:
            result[s] = indices
    return result


def precompute_matches(
    emb_cache: dict,
    atlas_triplets: list,
    all_item_triplets: list,
    threshold: float,
    criterion: str,
) -> tuple:
    """Precompute all matching atlas indices for every unique item string.

    Returns (subj_matches, pred_matches, obj_matches) dicts mapping
    normalized string → array of atlas indices where that position matches.
    """

    def get_emb(s):
        e = ap_embeddings.cache_lookup(emb_cache, s)
        if e is not None:
            return e
        sys.exit(
            f"Error: no embedding found in cache for: '{normalize(s)}'. The "
            "cache must cover the atlas and every data KG being matched — "
            "re-run ap_embeddings.py with this input included."
        )

    # Build atlas matrices and indices
    print("Building atlas embedding matrices...")
    atlas_subj = np.stack([get_emb(t[0]) for t in atlas_triplets]).astype(np.float32)
    atlas_pred = np.stack([get_emb(t[1]) for t in atlas_triplets]).astype(np.float32)
    atlas_obj = np.stack([get_emb(t[2]) for t in atlas_triplets]).astype(np.float32)

    print("Building FAISS indices...")
    subj_index = build_faiss_index(atlas_subj)
    pred_index = build_faiss_index(atlas_pred)
    obj_index = build_faiss_index(atlas_obj)

    # Collect unique item strings per position
    unique_subj = list({normalize(t[0]) for t in all_item_triplets})
    unique_pred = list({normalize(t[1]) for t in all_item_triplets})
    unique_obj = list({normalize(t[2]) for t in all_item_triplets})
    print(
        f"  Unique item strings: {len(unique_subj)} subjects, "
        f"{len(unique_pred)} predicates, {len(unique_obj)} objects"
    )

    # Build query matrices
    subj_qmat = np.stack([get_emb(s) for s in unique_subj]).astype(np.float32)
    pred_qmat = np.stack([get_emb(s) for s in unique_pred]).astype(np.float32)
    obj_qmat = np.stack([get_emb(s) for s in unique_obj]).astype(np.float32)

    # One-shot FAISS range_search per position
    print("Running FAISS range_search (one-shot for all item strings)...")

    if criterion in ("entity-relation", "full"):
        # Positional: each item position searched against same atlas position
        pred_matches = faiss_range_search_to_dict(
            unique_pred, pred_qmat, pred_index, threshold
        )
        subj_matches = faiss_range_search_to_dict(
            unique_subj, subj_qmat, subj_index, threshold
        )
        obj_matches = faiss_range_search_to_dict(
            unique_obj, obj_qmat, obj_index, threshold
        )
    else:  # any-item — cross-position: each item string vs all 3 atlas indices
        all_unique = list({normalize(s) for t in all_item_triplets for s in t})
        all_qmat = np.stack([get_emb(s) for s in all_unique]).astype(np.float32)
        subj_matches = faiss_range_search_to_dict(
            all_unique, all_qmat, subj_index, threshold
        )
        pred_matches = faiss_range_search_to_dict(
            all_unique, all_qmat, pred_index, threshold
        )
        obj_matches = faiss_range_search_to_dict(
            all_unique, all_qmat, obj_index, threshold
        )

    print(
        f"  Predicate hits: {len(pred_matches)}, Subject hits: {len(subj_matches)}, "
        f"Object hits: {len(obj_matches)}"
    )

    return subj_matches, pred_matches, obj_matches


# ---------------------------------------------------------------------------
# Per-item matching (pure lookups, no matmul)
# ---------------------------------------------------------------------------


def match_item(
    item_triplets: list,
    atlas_triplets: list,
    subj_matches: dict,
    pred_matches: dict,
    obj_matches: dict,
    criterion: str,
) -> dict:
    matched_pairs = []
    ap_counter = Counter()
    item_matched_count = 0

    for t2 in item_triplets:
        s2 = normalize(t2[0])
        p2 = normalize(t2[1])
        o2 = normalize(t2[2])

        if criterion == "entity-relation":
            pred_cands = pred_matches.get(p2)
            if pred_cands is None:
                continue
            pred_set = set(pred_cands.tolist())

            subj_cands = subj_matches.get(s2)
            obj_cands = obj_matches.get(o2)
            subj_set = set(subj_cands.tolist()) if subj_cands is not None else set()
            obj_set = set(obj_cands.tolist()) if obj_cands is not None else set()

            # (subject OR object) AND predicate
            matched_indices = (subj_set | obj_set) & pred_set
            if not matched_indices:
                continue

            matches_in_atlas = [atlas_triplets[i] for i in sorted(matched_indices)]
            item_matched_count += 1
            matched_pairs.append({"item_ap": t2, "atlas_matches": matches_in_atlas})

            for i in sorted(matched_indices):
                t1 = atlas_triplets[i]
                ap_counter[f"({t1[0]},{t1[1]},{t1[2]})"] += 1

        elif criterion == "full":
            pred_cands = pred_matches.get(p2)
            subj_cands = subj_matches.get(s2)
            obj_cands = obj_matches.get(o2)
            if pred_cands is None or subj_cands is None or obj_cands is None:
                continue

            matched_indices = (
                set(subj_cands.tolist())
                & set(pred_cands.tolist())
                & set(obj_cands.tolist())
            )
            if not matched_indices:
                continue

            matches_in_atlas = [atlas_triplets[i] for i in sorted(matched_indices)]
            item_matched_count += 1
            matched_pairs.append({"item_ap": t2, "atlas_matches": matches_in_atlas})

            for i in sorted(matched_indices):
                t1 = atlas_triplets[i]
                ap_counter[f"({t1[0]},{t1[1]},{t1[2]})"] += 1

        else:  # any-item — cross-position
            # Each item slot can match any atlas slot
            item_strings = [s2, p2, o2]
            # atlas subj matched if any item string hits subj_index
            subj_hit = set()
            pred_hit = set()
            obj_hit = set()
            for qs in item_strings:
                s_cands = subj_matches.get(qs)
                if s_cands is not None:
                    subj_hit.update(s_cands.tolist())
                p_cands = pred_matches.get(qs)
                if p_cands is not None:
                    pred_hit.update(p_cands.tolist())
                o_cands = obj_matches.get(qs)
                if o_cands is not None:
                    obj_hit.update(o_cands.tolist())

            all_matched = subj_hit | pred_hit | obj_hit
            if not all_matched:
                continue

            matches_in_atlas = [atlas_triplets[i] for i in sorted(all_matched)]
            item_matched_count += 1
            matched_pairs.append({"item_ap": t2, "atlas_matches": matches_in_atlas})

            for i in sorted(all_matched):
                t1 = atlas_triplets[i]
                ap_counter[f"({t1[0]},{t1[1]},{t1[2]})"] += 1

    return {
        "item_matched_count": item_matched_count,
        "matched_pairs": matched_pairs,
        "ap_counter": ap_counter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="FAISS-accelerated KG matching against an atlas."
    )
    parser.add_argument("atlas_path", help="Path to the atlas graph.json")
    parser.add_argument(
        "items_folder", help="Folder of per-item KGs to match (searched recursively)"
    )
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Cosine similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--embedding-cache-path",
        required=True,
        help="Path to the {text: embedding} pickle cache generated by ap_embeddings.py",
    )
    parser.add_argument(
        "--ap-match",
        choices=["any-item", "entity-relation", "full"],
        default="entity-relation",
        help="Match criterion (default: entity-relation)",
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Reclaim .inprogress lock files older than this many seconds",
    )
    parser.add_argument(
        "--item-glob",
        default="graph.json",
        help="Filename glob for discovering per-item KGs under items_folder "
        "(default: graph.json; use '*.json' for arbitrary layouts)",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas_path)
    items_folder = Path(args.items_folder)
    output_folder = Path(args.output)

    # Load the atlas
    print(f"Loading atlas from {atlas_path}...")
    atlas_triplets = load_kg(atlas_path)
    print(f"  Atlas: {len(atlas_triplets)} APs")

    # Load embedding cache
    cache_path = Path(args.embedding_cache_path)
    if not cache_path.exists():
        sys.exit(f"Error: embedding cache not found: {cache_path}")
    print(f"Loading embedding cache from {cache_path}...")
    emb_cache = ap_embeddings.load_embedding_cache(cache_path)
    print(f"  {len(emb_cache):,} entries loaded")

    # Discover items and filter already-done
    item_paths = resolve_item_paths(items_folder, args.item_glob)
    if not item_paths:
        sys.exit(f"No {args.item_glob} item files found.")
    print(f"Found {len(item_paths)} item KG(s) under {items_folder}")

    pending = []
    for p in item_paths:
        out_file = output_path_for(p, items_folder, output_folder)
        if out_file.exists():
            continue
        pending.append(p)
    print(f"  {len(pending)} remaining after skipping existing outputs")
    if not pending:
        return

    # Load all item KGs upfront
    print("Loading all item graphs...")
    item_map: dict[Path, list] = {}
    for p in tqdm.tqdm(pending, desc="Loading item KGs"):
        item_map[p] = load_kg(p)

    all_item_triplets = [t for triplets in item_map.values() for t in triplets]
    print(f"  Total item APs: {len(all_item_triplets)}")

    # Precompute ALL matches in one shot
    subj_matches, pred_matches, obj_matches = precompute_matches(
        emb_cache,
        atlas_triplets,
        all_item_triplets,
        args.threshold,
        args.ap_match,
    )

    # Free embedding cache
    del emb_cache, all_item_triplets

    # Per-item matching (pure lookups, no matmul)
    output_folder.mkdir(parents=True, exist_ok=True)
    for item_path in tqdm.tqdm(pending, desc="Matching items"):
        out_file = output_path_for(item_path, items_folder, output_folder)
        lock_file = out_file.with_suffix(".inprogress")
        if out_file.exists():
            continue
        if not acquire_lock(lock_file, args.stale_lock_seconds):
            print(f"  Skipping locked item (in progress elsewhere): {item_path}")
            continue

        try:
            item_triplets = item_map[item_path]
            results = match_item(
                item_triplets,
                atlas_triplets,
                subj_matches,
                pred_matches,
                obj_matches,
                args.ap_match,
            )
            save_result(
                out_file,
                atlas_path,
                item_path,
                args,
                atlas_triplets,
                item_triplets,
                results,
            )
        finally:
            lock_file.unlink(missing_ok=True)

    print(f"Results saved to {output_folder}")


if __name__ == "__main__":
    main()
