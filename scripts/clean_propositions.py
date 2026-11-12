# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Nuisance filtering for proposition graphs.

Uses an LLM to strip nuisance descriptors from proposition-graph entities
(e.g. color, size, shape, and brand descriptors on vehicles; quantity words
and outfit descriptions on pedestrians/cyclists) while preserving meaningful
descriptors such as traffic-light states.

Two-phase workflow to minimize LLM cost:
  Phase 1 (collect): Gather all unique normalized entities from every
                     graph.json, batch-call the LLM, and save an
                     {original: cleaned} mapping JSON.
  Phase 2 (apply):   Load the mapping, apply it to every graph.json,
                     deduplicate entities/edges/relations, and write the
                     cleaned graphs to the output directory.

Usage:
  # Both phases (default)
  python clean_propositions.py \\
      --input-dir <propositions-dir> \\
      --output-dir <cleaned-propositions-dir> \\
      --entity-cache <entity_cleaning_cache.json>

  # Phase 1 only (build cache)
  python clean_propositions.py --input-dir ... --entity-cache ... --phase collect

  # Phase 2 only (apply existing cache)
  python clean_propositions.py --input-dir ... --output-dir ... --entity-cache ... --phase apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing
import re
import sys
from pathlib import Path

import httpx
import tqdm

import common


# ---------------------------------------------------------------------------
# Graph editing helpers
# ---------------------------------------------------------------------------


def replace_in_str(s: str, find: str, replace: str) -> str:
    """Case-insensitive substring replacement (both strings taken literally)."""
    # The lambda keeps backslashes / group references in `replace` literal.
    return re.sub(re.escape(find), lambda _: replace, s, flags=re.IGNORECASE)


def apply_to_graph(graph: dict, find: str, replace: str) -> None:
    """In-place string replacement across entities, edges, and relations.

    Replacement is plain substring matching (no word boundaries) — mappings
    are applied longest-original-first by the caller to limit collisions, but
    a mapping whose key is a substring of an unrelated word will still rewrite
    it. Keep mapping keys specific.
    """
    graph["entities"] = [replace_in_str(e, find, replace) for e in graph["entities"]]
    graph["edges"] = [replace_in_str(e, find, replace) for e in graph["edges"]]
    graph["relations"] = [
        [replace_in_str(s, find, replace) for s in r] for r in graph["relations"]
    ]


# ---------------------------------------------------------------------------
# Phase 1: collect entities and call LLM
# ---------------------------------------------------------------------------


def collect_entities(graph_paths: list[Path]) -> set[str]:
    all_entities: set[str] = set()
    for p in tqdm.tqdm(graph_paths, desc="Collecting entities"):
        try:
            with open(p) as f:
                data = json.load(f)
            for e in data.get("entities", []):
                all_entities.add(common.normalize(e))
        except Exception as ex:
            print(f"  Warning: failed to read {p}: {ex}")
    return all_entities


async def _call_llm_batch(
    batch: list[str],
    client: httpx.AsyncClient,
    model: str,
    prompt_template: str,
    semaphore: asyncio.Semaphore,
    batch_idx: int,
    reasoning_effort: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Returns (mapping or None on failure, API usage block or None)."""
    prompt = prompt_template + json.dumps(batch, ensure_ascii=False)
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.post(
                    "/chat/completions",
                    json=request,
                )
                response.raise_for_status()
                body = response.json()
                raw = body["choices"][0]["message"]["content"] or ""
                result = json.loads(common.strip_markdown_json(raw))
                # The model may wrap the mapping under a key; unwrap if needed
                if isinstance(result, dict):
                    # flatten one level of nesting if the value is also a dict
                    for v in result.values():
                        if isinstance(v, dict):
                            result = v
                            break
                return result, body.get("usage")
            except Exception as ex:
                print(f"  Batch {batch_idx} attempt {attempt + 1} failed: {ex}")
                await asyncio.sleep(2**attempt)
    print(f"  Batch {batch_idx} failed after 3 attempts, skipping.")
    return None, None


async def run_collect_phase(
    entities: set[str],
    model: str,
    api_key: str,
    base_url: str,
    batch_size: int,
    workers: int,
    prompt_template: str,
    reasoning_effort: str | None = None,
) -> tuple[dict, set[str], dict]:
    """Query the LLM for the entity mapping in batches.

    Returns ``(mapping, failed_entities, api_usage_total)``: the merged mapping
    from successful batches, the entities whose batch failed after retries,
    and the summed API usage.
    """
    entity_list = sorted(entities)
    batches = [
        entity_list[i : i + batch_size] for i in range(0, len(entity_list), batch_size)
    ]
    print(
        f"  {len(entity_list):,} unique entities -> {len(batches)} batches of <={batch_size}"
    )

    client = httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    )
    semaphore = asyncio.Semaphore(workers)

    tasks = [
        _call_llm_batch(
            batch, client, model, prompt_template, semaphore, i, reasoning_effort
        )
        for i, batch in enumerate(batches)
    ]

    mapping: dict = {}
    failed_entities: set[str] = set()
    usage_total: dict = {}
    results = await asyncio.gather(*tasks)
    await client.aclose()
    for batch, (r, usage) in zip(batches, results):
        common.merge_usage(usage_total, usage)
        if r is None:
            failed_entities.update(batch)
        else:
            mapping.update(r)

    return mapping, failed_entities, usage_total


# ---------------------------------------------------------------------------
# Phase 2: apply mapping to each graph
# ---------------------------------------------------------------------------


def apply_mapping_to_graph(graph: dict, mapping: dict) -> dict:
    """Normalize all strings, apply the LLM mapping, deduplicate everything."""
    graph["entities"] = [
        e for e in graph.get("entities", []) if isinstance(e, str) and e.strip()
    ]
    graph["edges"] = [
        e for e in graph.get("edges", []) if isinstance(e, str) and e.strip()
    ]
    graph["relations"] = [
        r
        for r in graph.get("relations", [])
        if isinstance(r, list)
        and len(r) == 3
        and all(isinstance(s, str) and s.strip() for s in r)
    ]

    # Normalize
    graph["entities"] = [common.normalize(e) for e in graph["entities"]]
    graph["edges"] = [common.normalize(e) for e in graph["edges"]]
    graph["relations"] = [[common.normalize(s) for s in r] for r in graph["relations"]]

    # Apply replacements (longest originals first to avoid partial-match conflicts)
    for original, cleaned in sorted(
        mapping.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if original != cleaned:
            apply_to_graph(graph, original, cleaned)

    # Deduplicate entities and edges (preserve order of first occurrence)
    graph["entities"] = list(dict.fromkeys(graph["entities"]))
    graph["edges"] = list(dict.fromkeys(graph["edges"]))

    # Deduplicate relations (tuples for hashability, then back to lists)
    seen = set()
    unique_relations = []
    for r in graph["relations"]:
        key = tuple(r)
        if key not in seen:
            seen.add(key)
            unique_relations.append(r)
    graph["relations"] = unique_relations

    return graph


def _apply_single(args: tuple) -> str | None:
    """Worker function for multiprocessing. Returns error message or None."""
    graph_path_str, input_dir_str, output_dir_str, mapping = args
    graph_path = Path(graph_path_str)
    out_path = Path(output_dir_str) / graph_path.relative_to(input_dir_str)
    if out_path.exists():
        return None
    try:
        with open(graph_path) as f:
            graph = json.load(f)
        graph = apply_mapping_to_graph(graph, mapping)
        common.write_json_atomic(out_path, graph)
    except Exception as ex:
        return f"  Warning: failed to process {graph_path}: {ex}"
    return None


def run_apply_phase(
    graph_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    mapping: dict,
    num_workers: int = 8,
) -> None:
    work_items = [
        (str(gp), str(input_dir), str(output_dir), mapping) for gp in graph_paths
    ]
    with multiprocessing.Pool(processes=num_workers) as pool:
        for err in tqdm.tqdm(
            pool.imap_unordered(_apply_single, work_items, chunksize=64),
            total=len(work_items),
            desc="Applying mapping",
        ):
            if err:
                print(err)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Nuisance filtering: clean noisy entity descriptors from "
        "proposition graphs using an LLM."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing graph.json files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root for cleaned graphs (required for apply phase)",
    )
    parser.add_argument(
        "--entity-cache",
        type=Path,
        default=None,
        help="Path to save/load the {original: cleaned} mapping JSON",
    )
    parser.add_argument(
        "--phase",
        choices=["collect", "apply", "both"],
        default="both",
        help="Which phase to run (default: both)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=common.PROMPTS_DIR / "clean_entities.txt",
        help="Prompt template file (default: prompts/clean_entities.txt)",
    )
    common.add_llm_args(parser)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Entities per LLM call (default: 5000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Concurrent async LLM calls in the collect phase (default: 16)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel worker processes for the apply phase (default: 8)",
    )
    args = parser.parse_args()
    args.reasoning_effort = common.effective_reasoning_effort(args.reasoning_effort)

    if not args.input_dir.is_dir():
        sys.exit(f"Error: --input-dir not found: {args.input_dir}")

    cache_path = args.entity_cache
    do_collect = args.phase in ("collect", "both")
    do_apply = args.phase in ("apply", "both")

    if do_apply and not args.output_dir:
        sys.exit("Error: --output-dir is required for the apply phase")

    if not args.prompt_file.exists():
        sys.exit(f"Error: prompt file not found: {args.prompt_file}")
    prompt_template = args.prompt_file.read_text()

    # Discover graphs
    print(f"Scanning {args.input_dir} for graph.json files...")
    graph_paths = sorted(args.input_dir.rglob("graph.json"))
    print(f"  Found {len(graph_paths):,} graph(s)")
    if not graph_paths:
        sys.exit("Error: no graph.json files found")

    # Phase 1: collect & call LLM. The cache is incremental: entities already
    # in it (including identity entries for previously-queried entities the
    # LLM left unchanged) are not re-queried, so reusing one cache across the
    # target and training-pool runs canonicalizes both sides identically.
    mapping: dict = {}
    if do_collect:
        if cache_path and cache_path.exists():
            print(f"Loading existing entity cache from {cache_path}...")
            with open(cache_path) as f:
                mapping = json.load(f)
            print(f"  {len(mapping):,} entries loaded")

        print("Phase 1: collecting unique entities...")
        entities = collect_entities(graph_paths)
        pending = [entity for entity in entities if entity not in mapping]
        print(
            f"  {len(entities):,} unique normalized entities "
            f"({len(pending):,} not in cache)"
        )

        if pending:
            api_key = common.load_api_key(args.api_key_file)

            print(
                f"Phase 1: calling {args.model} ({args.workers} concurrent workers)..."
            )
            new_mapping, failed_entities, usage_total = asyncio.run(
                run_collect_phase(
                    entities=pending,
                    model=args.model,
                    api_key=api_key,
                    base_url=args.base_url,
                    batch_size=args.batch_size,
                    workers=args.workers,
                    prompt_template=prompt_template,
                    reasoning_effort=args.reasoning_effort,
                )
            )
            print(f"  Mapping: {len(new_mapping):,} entities will be modified")
            print(f"  api_usage_total: {json.dumps(usage_total)}")
            if failed_entities:
                print(
                    f"  WARNING: {len(failed_entities):,} entities were in failed "
                    "batches; they are left out of the cache and will be "
                    "re-queried on the next run."
                )

            # Record identity entries for successfully-queried-but-unmodified
            # entities so they are never re-queried on later runs. Entities
            # from failed batches are deliberately NOT recorded, so a re-run
            # retries them instead of treating them as clean.
            for entity in pending:
                if entity not in failed_entities:
                    mapping[entity] = new_mapping.get(entity, entity)

            if cache_path:
                common.write_json_atomic(cache_path, mapping)
                print(f"  Saved mapping cache to {cache_path}")

    # Phase 2: apply
    if do_apply:
        if not mapping and cache_path and cache_path.exists():
            print(f"Loading entity cache from {cache_path}...")
            with open(cache_path) as f:
                mapping = json.load(f)
            print(f"  {len(mapping):,} entries loaded")

        if not mapping:
            sys.exit(
                "Error: no mapping available -- run collect phase first or provide --entity-cache"
            )

        print(
            f"Phase 2: applying mapping to {len(graph_paths):,} graphs -> "
            f"{args.output_dir} ({args.num_workers} workers)"
        )
        run_apply_phase(
            graph_paths,
            args.input_dir,
            args.output_dir,
            mapping,
            num_workers=args.num_workers,
        )
        print("Done.")


if __name__ == "__main__":
    main()
