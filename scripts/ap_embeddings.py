# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AP embedding cache: shared load/lookup helpers and the stage-6 generator.

Everything embedding-related lives here. As a library it provides the
{text: embedding} cache used across the pipeline — written by this script
(stage 6), extended in place by clean_atlas.py's embedding prerequisite
(stage 7), and read for FAISS matching by match_data_to_k_atlas_faiss.py
(stage 8). As a CLI it is stage 6 itself: it reads one or more proposition
graphs (KGs) and encodes every unique subject/predicate/object string into a
single cache. Pass the atlas AND the data-KG folders to be matched so the
cache covers both.

Cache format: keys are common.normalize(text) (lowercased, stripped —
trailing commas are kept; cache_lookup tries the comma variants), values are
L2-normalized embedding vectors, and the whole dict is saved with dill.

Example:
  python scripts/ap_embeddings.py \
      output/atlas/raw_atlas.json output/propositions_test_cleaned \
      --output-file output/atlas/ap_embeddings.pkl --batch-size 256 --checkpoint-every 20
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import dill
import numpy as np
import tqdm

import common


# ---------------------------------------------------------------------------
# Cache IO and lookup (shared by stages 6, 7, and 8)
# ---------------------------------------------------------------------------


def load_embedding_cache(path: Path) -> dict[str, np.ndarray]:
    """Load a {text: embedding} dill cache."""
    with open(path, "rb") as f:
        data = dill.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict embedding cache: {path}")
    return {
        str(key): np.asarray(value, dtype=np.float32) for key, value in data.items()
    }


def cache_lookup(cache: dict[str, np.ndarray], text: str) -> np.ndarray | None:
    """Look up a string in the embedding cache; None when absent.

    Cache keys keep trailing commas (common.normalize) while the atlas
    cleaners strip them (common.normalize_ap), so try the normalized key and
    both comma variants.
    """
    key = common.normalize(str(text))
    stripped = key.rstrip(",")
    for candidate in (key, stripped, stripped + ","):
        value = cache.get(candidate)
        if value is not None:
            return value
    return None


def save_embedding_cache_atomic(path: Path, cache: dict[str, np.ndarray]) -> None:
    """Atomically persist a {text: embedding} dill cache."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "wb") as f:
        dill.dump(cache, f)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Cache generation
# ---------------------------------------------------------------------------


def load_sentence_transformer(model_name: str):
    """Lazily import sentence-transformers and load the embedding model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required to compute embeddings "
            "(a core dependency).\n"
            "Install the project dependencies: uv sync"
        ) from exc
    return SentenceTransformer(model_name)


def ensure_embeddings(
    cache_path: Path,
    required: Iterable[str],
    model_name: str,
    batch_size: int,
    checkpoint_every: int,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load the embedding cache and compute any missing required strings.

    The cache is loaded if it exists, missing strings are encoded in batches
    (checkpointed every `checkpoint_every` batches; 0 saves only at the end),
    and the updated cache is written back atomically.

    Returns (cache, stats) where stats counts required/cached/computed strings.
    """
    cache_path = Path(cache_path)
    cache = load_embedding_cache(cache_path) if cache_path.exists() else {}
    missing: list[str] = []
    seen: set[str] = set()
    for value in required:
        key = common.normalize(str(value))
        if not key or key in seen:
            continue
        seen.add(key)
        if cache_lookup(cache, key) is None:
            missing.append(key)
    stats = {
        "cache_path": str(cache_path),
        "required_count": len(seen),
        "cached_count": len(seen) - len(missing),
        "computed_count": len(missing),
    }
    print(
        f"Embedding prerequisite: {len(seen)} required strings, "
        f"{stats['cached_count']} already cached, {len(missing)} to compute"
    )
    if missing:
        print(f"Encoding {len(missing)} strings with '{model_name}'...")
        model = load_sentence_transformer(model_name)
        batches = range(0, len(missing), batch_size)
        for batch_idx, start in enumerate(
            tqdm.tqdm(batches, desc="Encoding batches"), start=1
        ):
            batch = missing[start : start + batch_size]
            embeddings = model.encode(
                batch,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for text, embedding in zip(batch, np.asarray(embeddings)):
                cache[text] = np.asarray(embedding, dtype=np.float32)
            if checkpoint_every and batch_idx % checkpoint_every == 0:
                save_embedding_cache_atomic(cache_path, cache)
        save_embedding_cache_atomic(cache_path, cache)
        print(f"Saved {len(cache)} embeddings to {cache_path}")
    elif not cache_path.exists():
        save_embedding_cache_atomic(cache_path, cache)
    return cache, stats


# ---------------------------------------------------------------------------
# Stage-6 CLI
# ---------------------------------------------------------------------------


def resolve_kg_paths(inputs: list[str]) -> list[Path]:
    """Each input is a KG .json file or a folder searched recursively for graph.json."""
    paths = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("graph.json"))
            if not found:
                print(f"  Warning: no graph.json files found under {p}")
            paths.extend(found)
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f"Input not found: {p}")
    return paths


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precalculate embeddings for all AP components (entities and "
        "predicates) across one or more KGs into a single cache"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="KG .json files and/or folders (folders are searched "
        "recursively for graph.json). Pass the atlas AND the "
        "data-KG folders to be matched so the cache covers both.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Output dill cache file; an existing partial cache is resumed",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Sentence-transformer model (default: Qwen/Qwen3-Embedding-8B)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Encoding batch size (default: 256)"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a checkpoint every N encoded batches; 0 saves only at the end",
    )
    return parser.parse_args()


def main():
    args = arg_parser()

    all_triplets = []
    for kg_path in resolve_kg_paths(args.inputs):
        all_triplets.extend(common.load_kg(kg_path))
    print(f"Found {len(all_triplets)} triplets")

    required = [str(part) for triplet in all_triplets for part in triplet]
    cache, _stats = ensure_embeddings(
        cache_path=args.output_file,
        required=required,
        model_name=args.model,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Cache at {args.output_file} holds {len(cache)} embeddings")


if __name__ == "__main__":
    main()
