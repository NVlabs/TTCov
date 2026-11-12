# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine multiple proposition graphs (KGs) into one raw atlas JSON.

Examples:
  python combine_propositions.py path/a/graph.json path/b/graph.json -o raw_atlas.json
  python combine_propositions.py output/proposition_graphs/ -o raw_atlas.json
  python combine_propositions.py output/folder1/ path/b/graph.json -o raw_atlas.json

Aggregates entities, edges, and relations across the inputs with exact-string
deduplication; semantic deduplication is the atlas-unification stage's job
(see clean_atlas.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    """Expand a mix of files and folders into a flat list of graph.json paths."""
    paths = []
    for item in inputs:
        p = Path(item)
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


def _append_unique(items: list[str], seen: set[str], value) -> None:
    if not isinstance(value, str):
        return
    value = value.strip()
    if value and value not in seen:
        seen.add(value)
        items.append(value)


def _append_unique_relation(
    relations: list[list[str]],
    seen: set[tuple[str, str, str]],
    relation,
) -> tuple[str, str, str] | None:
    if not isinstance(relation, (list, tuple)) or len(relation) != 3:
        return None
    normalized = tuple(str(part).strip() for part in relation)
    if not all(normalized):
        return None
    if normalized not in seen:
        seen.add(normalized)
        relations.append(list(normalized))
    return normalized


def load_graph_dict(path: Path) -> dict:
    with open(path) as f:
        graph = json.load(f)
    if not isinstance(graph, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return graph


def combine_graph_dicts(graph_paths: list[Path]) -> dict:
    """Combine graph JSON files without requiring model dependencies."""
    entities: list[str] = []
    edges: list[str] = []
    relations: list[list[str]] = []
    entity_seen: set[str] = set()
    edge_seen: set[str] = set()
    relation_seen: set[tuple[str, str, str]] = set()

    for path in graph_paths:
        print(f"Loading {path}...")
        graph = load_graph_dict(path)
        for entity in graph.get("entities", []):
            _append_unique(entities, entity_seen, entity)
        for edge in graph.get("edges", []):
            _append_unique(edges, edge_seen, edge)
        for relation in graph.get("relations", []):
            normalized = _append_unique_relation(relations, relation_seen, relation)
            if normalized is None:
                continue
            subject, predicate, obj = normalized
            _append_unique(entities, entity_seen, subject)
            _append_unique(edges, edge_seen, predicate)
            _append_unique(entities, entity_seen, obj)

    return {"entities": entities, "edges": edges, "relations": relations}


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple proposition graphs (graph.json) into one raw atlas."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="graph.json files and/or folders searched recursively for graph.json",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output path for the combined raw atlas JSON",
    )
    args = parser.parse_args()

    graph_paths = resolve_input_paths(args.inputs)
    if not graph_paths:
        raise SystemExit("No graph.json files found in the provided inputs.")
    print(f"Found {len(graph_paths)} graph(s) to combine:")
    for p in graph_paths:
        print(f"  {p}")

    out_path = Path(args.output)
    print("Aggregating graphs...")
    combined_graph = combine_graph_dicts(graph_paths)
    print(
        f"  Combined: {len(combined_graph['entities'])} entities, "
        f"{len(combined_graph['edges'])} edge types, "
        f"{len(combined_graph['relations'])} relations"
    )
    common.write_json_atomic(out_path, combined_graph)
    print(f"Saved combined graph to {out_path}")


if __name__ == "__main__":
    main()
