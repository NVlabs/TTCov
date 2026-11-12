# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert distilled sentences to propositions.

Reads per-item breakdown JSON files produced by text_distillation.py, joins
the atomic sentences back into a text block, and extracts atomic-proposition
(AP) [subject, predicate, object] triplets, writing one proposition graph per
item to <output-dir>/<item_id[:4]>/<item_id>/graph.json.

Extraction is two-pass and grounded: pass 1 asks the LLM for a thorough list
of key entities, and pass 2 extracts [subject, predicate, object] triplets
whose subject and object must come from that entity list. Grounding the
triplets in a fixed entity list suppresses hallucinated participants and
inconsistent entity phrasings.

The script talks to any OpenAI-compatible chat-completions endpoint
(including a local Ollama server's /v1 API) via --base-url.

Runs are resumable: existing graphs are skipped (unless --overwrite) and
in-flight items are protected by lock files.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import socket
import time
from pathlib import Path

import httpx

import common


# The extraction mechanics below are part of the method and stay in code; the
# only domain-specific piece is the {context} fragment, read from a small text
# file (--context-file, default prompts/context.txt) so adapting TTCov to a
# new domain never requires editing this script.
ENTITY_PROMPT_TEMPLATE = (
    "You are extracting key entities from {context}.\n"
    "The input is a newline-separated list of atomic subject-verb-object sentences.\n"
    "Extract a THOROUGH list of the key entities: every subject and every object "
    "that appears in the sentences. Use concise entity phrases taken from the "
    "input, phrase each entity consistently, and do not include predicates, "
    "actions, or full sentences.\n"
    'Return only JSON with this exact structure: {{"entities": ["entity"]}}.\n\n'
    "Input sentences:\n{text}"
)

RELATION_PROMPT_TEMPLATE = (
    "You are extracting [subject, predicate, object] proposition triplets from "
    "{context}.\n"
    "The input is a newline-separated list of atomic subject-verb-object sentences "
    "plus a fixed list of entities.\n"
    "Extract triplets that thoroughly cover the sentences. The subject and object "
    "of every triplet MUST be strings copied exactly from the entity list; the "
    "predicate is a short free-form phrase.\n"
    "Return only JSON with this exact structure: "
    '{{"relations": [["subject", "predicate", "object"]]}}. '
    "Every relation must have exactly three strings.\n\n"
    "Entity list:\n{entities}\n\n"
    "Input sentences:\n{text}"
)

RELATION_FIX_PROMPT_TEMPLATE = (
    "You are fixing [subject, predicate, object] proposition triplets extracted "
    "from {context}.\n"
    "The triplets below were extracted from the input sentences, but their "
    "subject or object is not copied exactly from the entity list. Rewrite each "
    "triplet so that BOTH the subject and the object are strings copied exactly "
    "from the entity list, preserving the triplet's meaning. Drop a triplet "
    "only when no entity in the list can express its subject or object.\n"
    "Return only JSON with this exact structure: "
    '{{"relations": [["subject", "predicate", "object"]]}}. '
    "Every relation must have exactly three strings. Return only the corrected "
    "triplets, nothing else.\n\n"
    "Entity list:\n{entities}\n\n"
    "Input sentences:\n{text}\n\n"
    "Triplets to fix:\n{invalid}"
)


def load_context(path: Path) -> str:
    """Read the domain-context fragment inserted into both extraction prompts."""
    context = path.read_text(encoding="utf-8").strip()
    if not context:
        raise ValueError(f"Context file is empty: {path}")
    return context


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert distilled sentences to propositions (one graph.json per item)."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of breakdown JSON files from text_distillation.py",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for per-item proposition graphs",
    )
    common.add_llm_args(parser)
    parser.add_argument(
        "--context-file",
        type=Path,
        default=common.PROMPTS_DIR / "context.txt",
        help="Text file with the domain-context fragment inserted into both "
        "extraction prompts, completing 'You are extracting ... from <context>.' "
        "(default: prompts/context.txt)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--max-retries",
        type=lambda v: max(1, int(v)),
        default=3,
        help="LLM attempts per call, minimum 1 (default: 3)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel worker processes; calls are API-bound, so "
        "raise this until the endpoint rate-limits (default: 16)",
    )
    common.add_selection_args(parser)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifier for the run manifest (default: auto-generated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process items that already have a graph.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected work without calling the LLM",
    )
    parser.add_argument(
        "--save-input-text",
        action="store_true",
        help="Save the newline-separated atomic sentences as input.txt next to each graph.",
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Reclaim lock files older than this many seconds",
    )
    return parser


def breakdown_to_text(path: Path) -> tuple[str, str]:
    """Return (item_id, newline-joined atomic sentences) for one breakdown JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    item_id = data.get("item_id") or path.stem
    breakdown = data.get("breakdown", data)
    sentences = list(common.iter_basic_sentences(breakdown))
    if not sentences:
        raise ValueError(f"No basic_sentences found in {path}")
    return item_id, "\n".join(sentences)


def _chat_json(
    client: httpx.Client,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> tuple[dict, dict | None]:
    """Send one chat request; returns (parsed JSON object, API usage or None)."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.post(
                "/chat/completions",
                json=request,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"] or ""
            return json.loads(common.strip_markdown_json(content)), body.get("usage")
        except Exception as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def _parse_entities(payload: dict) -> list[str]:
    entities = payload.get("entities")
    if not isinstance(entities, list):
        raise ValueError("Entity response missing 'entities' list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        if not isinstance(entity, str) or not entity.strip():
            raise ValueError(
                f"Entity response contained a non-string entity: {entity!r}"
            )
        entity = entity.strip()
        key = entity.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(entity)
    if not cleaned:
        raise ValueError("Entity response contained no entities")
    return cleaned


def _validate_relations(
    payload: dict,
    entity_lookup: dict[str, str],
) -> tuple[list[list[str]], list]:
    """Split the response relations into (valid, invalid) against the entity list.

    Valid triplets have their subject/object mapped back to the pass-1 entity
    casing; comparison is case-insensitive on stripped strings.
    """
    relations = payload.get("relations")
    if not isinstance(relations, list):
        raise ValueError("Relation response missing 'relations' list")
    valid: list[list[str]] = []
    invalid: list = []
    for relation in relations:
        if (
            isinstance(relation, list)
            and len(relation) == 3
            and all(isinstance(part, str) and part.strip() for part in relation)
        ):
            subject, predicate, obj = (part.strip() for part in relation)
            subject_key, object_key = subject.lower(), obj.lower()
            if subject_key in entity_lookup and object_key in entity_lookup:
                valid.append(
                    [entity_lookup[subject_key], predicate, entity_lookup[object_key]]
                )
                continue
        invalid.append(relation)
    return valid, invalid


def _run_two_pass(
    text: str,
    output_dir: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    context: str,
    usage_total: dict,
    reasoning_effort: str | None = None,
) -> dict:
    """Two-pass grounded extraction: entities first, then entity-constrained triplets.

    Returns per-item stats (entity/relation counts and dropped-triplet counts)
    for the run manifest.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(
        base_url=base_url, headers=headers, timeout=request_timeout
    ) as client:
        # Pass 1: thorough list of key entities (subjects/objects).
        entity_prompt = ENTITY_PROMPT_TEMPLATE.format(context=context, text=text)
        entity_payload, usage = _chat_json(
            client, model, entity_prompt, temperature, max_retries, reasoning_effort
        )
        common.merge_usage(usage_total, usage)
        entities = _parse_entities(entity_payload)
        entity_lookup = {entity.lower(): entity for entity in entities}
        entities_json = json.dumps(entities, ensure_ascii=False)

        # Pass 2: triplets whose subject and object come from the entity list.
        relation_prompt = RELATION_PROMPT_TEMPLATE.format(
            context=context, entities=entities_json, text=text
        )
        relation_payload, usage = _chat_json(
            client, model, relation_prompt, temperature, max_retries, reasoning_effort
        )
        common.merge_usage(usage_total, usage)
        relations, invalid = _validate_relations(relation_payload, entity_lookup)
        num_dropped_initial = len(invalid)

        # One corrective fix pass when triplets were dropped: send back only
        # the invalid triplets for remapping onto the entity list — the output
        # cost is proportional to the invalid count instead of a full second
        # extraction — then keep the union of everything that validates.
        num_dropped_final = num_dropped_initial
        num_fixed = 0
        if invalid:
            fix_prompt = RELATION_FIX_PROMPT_TEMPLATE.format(
                context=context,
                entities=entities_json,
                text=text,
                invalid=json.dumps(invalid, ensure_ascii=False),
            )
            retry_payload, usage = _chat_json(
                client, model, fix_prompt, temperature, max_retries, reasoning_effort
            )
            common.merge_usage(usage_total, usage)
            retry_relations, _ = _validate_relations(retry_payload, entity_lookup)
            seen = {tuple(relation) for relation in relations}
            for relation in retry_relations:
                if tuple(relation) not in seen:
                    seen.add(tuple(relation))
                    relations.append(relation)
            num_fixed = len(retry_relations)
            num_dropped_final = max(0, num_dropped_initial - num_fixed)

    edges: list[str] = []
    seen_edges: set[str] = set()
    for _subject, predicate, _obj in relations:
        if predicate not in seen_edges:
            seen_edges.add(predicate)
            edges.append(predicate)

    graph = {"entities": entities, "edges": edges, "relations": relations}
    common.write_json_atomic(output_dir / "graph.json", graph)
    return {
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_dropped_relations_initial": num_dropped_initial,
        "num_relations_fixed_by_retry": num_fixed,
        "num_dropped_relations_final": num_dropped_final,
    }


def _process_one(task: dict) -> dict:
    input_path = Path(task["input_path"])
    output_root = Path(task["output_dir"])
    overwrite = task["overwrite"]
    dry_run = task["dry_run"]
    # Mutable accumulator: keeps tokens from successful passes countable even
    # when a later pass raises and the item lands on the error path.
    usage_total: dict = {}

    try:
        item_id, input_text = breakdown_to_text(input_path)
        out_dir = common.output_graph_dir(output_root, item_id)
        graph_path = out_dir / "graph.json"
        lock_path = out_dir / ".inprogress"

        if graph_path.exists() and not overwrite:
            return {"item_id": item_id, "status": "skipped_existing"}

        if not common.acquire_lock(lock_path, task["stale_lock_seconds"]):
            return {"item_id": item_id, "status": "skipped_locked"}

        try:
            if dry_run:
                return {
                    "item_id": item_id,
                    "status": "dry_run",
                    "input_path": str(input_path),
                    "num_basic_sentences": input_text.count("\n") + 1,
                }

            out_dir.mkdir(parents=True, exist_ok=True)
            if task["save_input_text"]:
                common.write_text_atomic(out_dir / "input.txt", input_text)
            result = {"item_id": item_id, "status": "done"}
            stats = _run_two_pass(
                text=input_text,
                output_dir=out_dir,
                model=task["model"],
                api_key=task["api_key"],
                base_url=task["base_url"],
                temperature=task["temperature"],
                request_timeout=task["request_timeout"],
                max_retries=task["max_retries"],
                context=task["context"],
                usage_total=usage_total,
                reasoning_effort=task["reasoning_effort"],
            )
            result.update(stats)
            result["usage"] = usage_total
            return result
        finally:
            lock_path.unlink(missing_ok=True)
    except Exception as exc:
        return {
            "item_id": input_path.stem,
            "status": "error",
            "input_path": str(input_path),
            "error": repr(exc),
            "usage": usage_total,
        }


def main() -> None:
    args = argparser().parse_args()
    args.reasoning_effort = common.effective_reasoning_effort(args.reasoning_effort)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_id is None:
        args.run_id = f"{common.utc_now().replace(':', '').replace('-', '')}_{socket.gethostname()}_{os.getpid()}"

    input_files = common.discover_json_items(args.input_dir)
    selected_files = common.select_by_common_args(
        input_files,
        item_ids=common.load_item_ids(args.item_ids),
        offset=args.offset,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    api_key = None
    if not args.dry_run:
        api_key = common.load_api_key(args.api_key_file)

    context = load_context(args.context_file)

    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "created_at": common.utc_now(),
        "finished_at": None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "output_index": "{output_dir}/{item_id[:4]}/{item_id}/graph.json",
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "context_file": str(args.context_file),
        "context_sha256": common.sha256_text(context),
        "temperature": args.temperature,
        "request_timeout": args.request_timeout,
        "num_workers": args.num_workers,
        "num_input_files": len(input_files),
        "num_selected": len(selected_files),
        "limit": args.limit,
        "offset": args.offset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "dry_run": args.dry_run,
        "save_input_text": args.save_input_text,
        "counts": {},
        "items": [],
    }
    manifest_path = args.output_dir / "_manifests" / f"{args.run_id}.json"
    common.write_json_atomic(manifest_path, manifest)

    tasks = [
        {
            "input_path": str(path),
            "output_dir": str(args.output_dir),
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "save_input_text": args.save_input_text,
            "stale_lock_seconds": args.stale_lock_seconds,
            "context": context,
            "model": args.model,
            "api_key": api_key,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "request_timeout": args.request_timeout,
            "max_retries": args.max_retries,
            "reasoning_effort": args.reasoning_effort,
        }
        for path in selected_files
    ]

    counts: dict[str, int] = {}
    results: list[dict] = []
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        for result in pool.imap_unordered(_process_one, tasks):
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            results.append(result)
            print(
                json.dumps({"item_id": result["item_id"], "status": result["status"]}),
                flush=True,
            )

    usage_total: dict = {}
    for result in results:
        common.merge_usage(usage_total, result.pop("usage", None))

    manifest["status"] = "finished"
    manifest["finished_at"] = common.utc_now()
    manifest["counts"] = counts
    manifest["api_usage_total"] = usage_total
    manifest["items"] = sorted(results, key=lambda item: item["item_id"])
    common.write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {"run_id": args.run_id, "counts": counts, "api_usage_total": usage_total},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
