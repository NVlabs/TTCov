# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atlas unification (pipeline stage 7): embedding prerequisite + leader cleaning.

One self-contained stage that owns its embedding prerequisite and cleans the
raw atlas with the shared leader-find algorithm (FAISS nearest-neighbor
candidate groups at a cosine threshold, a frequency-ranked leader per group,
leader-anchored LLM verification, only verified candidates merged) at one of
two granularities:

  Default (whole-AP): deduplicate WHOLE AP triplets. Each unique AP is
  embedded natively as one "subject predicate object" string; candidate
  groups come from an exhaustive FAISS range search at --threshold (default
  0.80); the LLM judge decides which candidate APs state the same fact as the
  leader; verified members are replaced by their leader triplet, then
  relations are deduplicated, filtered, and the entity/edge lists rebuilt.

  --component: deduplicate entity and predicate COMPONENTS. The LLM verifies
  which candidate names are true aliases of the leader (--threshold default
  0.85) before relations are rewritten, deduplicated, and filtered.

Embedding prerequisite (both modes): on startup the persistent dill cache at
--embedding-cache is loaded if it exists, embeddings are computed for whatever
required strings are missing (whole-AP mode: the joined "s p o" string of
every unique normalized relation; --component mode: every unique normalized
entity/predicate string), and the updated cache is persisted with resumable
checkpointing (--batch-size, --checkpoint-every, --embedding-model). The cache
format matches scripts/ap_embeddings.py (stage 6), so in
--component mode the stage-6 pickle can be reused directly with zero embedding
work. --embed-only runs just this prerequisite and exits, so the GPU-bound
part can run on a different node than the LLM-verification part.

LLM verification is integral to the method: candidate groups are proposed
generously by embedding similarity, and only judge-approved merges are
applied. Verification is iterative (--max-verify-rounds): judge-rejected
candidates re-pool and are re-clustered among themselves in later rounds,
so near-duplicates that were each rejected from some other leader's group
still merge with each other. Use --dry-run to preview candidate groups and
tune --threshold
without an API key; it writes logs and a summary but never the cleaned atlas.
Groups whose verification fails after retries are left unmerged and counted
in the summary; re-running retries only those (verified results are cached).

Usage:
  # whole-AP (default)
  python clean_atlas.py \
      --input output/raw_atlas.json --output output/atlas.json \
      --log-dir output/atlas_clean_logs \
      --embedding-cache output/atlas_ap_embeddings.pkl \
      --base-url <endpoint> --model <model>

  # component-level, reusing the stage-6 cache
  python clean_atlas.py --component \
      --input output/raw_atlas.json --output output/atlas.json \
      --log-dir output/atlas_clean_logs \
      --embedding-cache output/ap_embeddings.pkl \
      --base-url <endpoint> --model <model>
"""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from typing import Callable, Iterable

import numpy as np
import tqdm
from openai import AsyncOpenAI

import ap_embeddings
import common


# Threshold defaults per mode: native sentence-level AP similarity runs on a
# different scale than component similarity, and whole-AP mode can run looser
# because the LLM judge filters candidate merges.
DEFAULT_THRESHOLD_WHOLE_AP = 0.80
DEFAULT_THRESHOLD_COMPONENT = 0.85

AP_VERIFY_PROMPT = """\
You are cleaning autonomous-driving knowledge-graph propositions. Each
proposition is a subject-predicate-object triplet written as one line.

The leader is the canonical proposition for this small local cluster. Decide
which candidate propositions state the same fact as the leader. Keep
propositions separate when they differ in actor, maneuver, direction, road
location, traffic-control state, or safety implication -- merge only true
restatements of the same fact.

Return JSON only:
{{
  "same": ["propositions that state the same fact as the leader"],
  "different": ["propositions that state a different fact"]
}}

Leader: {leader_json}
Candidates, including the leader:
{group_json}
"""

COMPONENT_VERIFY_PROMPT = """\
You are cleaning autonomous-driving knowledge-graph {label} names.

The leader is the canonical candidate for this small local cluster. Decide which
candidate names mean the same thing as the leader in an autonomous-driving KG.
Keep broader/narrower concepts, different actors, different maneuvers, road
locations, traffic-control states, and safety concepts separate unless they are
true aliases.

Return JSON only:
{{
  "same": ["names that mean the same as the leader"],
  "different": ["names that do not mean the same as the leader"]
}}

Leader: {leader_json}
Candidates, including the leader:
{group_json}
"""


AP_VERIFY_BATCH_PROMPT = """\
You are cleaning autonomous-driving knowledge-graph propositions. Each
proposition is a subject-predicate-object triplet written as one line.

Below are {count} independent clusters. In each cluster, the leader is the
canonical proposition and the candidates include the leader itself. For each
cluster, decide which candidate propositions state the same fact as that
cluster's leader. Keep propositions separate when they differ in actor,
maneuver, direction, road location, traffic-control state, or safety
implication -- merge only true restatements of the same fact. Judge every
cluster independently of the others.

Return JSON only:
{{
  "groups": [
    {{"id": 0,
      "same": ["propositions that state the same fact as the leader"],
      "different": ["propositions that state a different fact"]}}
  ]
}}
Include exactly one entry for every cluster id from 0 to {last}.

Clusters:
{clusters_json}
"""

COMPONENT_VERIFY_BATCH_PROMPT = """\
You are cleaning autonomous-driving knowledge-graph {label} names.

Below are {count} independent clusters. In each cluster, the leader is the
canonical candidate and the candidates include the leader itself. For each
cluster, decide which candidate names mean the same thing as that cluster's
leader in an autonomous-driving KG. Keep broader/narrower concepts, different
actors, different maneuvers, road locations, traffic-control states, and
safety concepts separate unless they are true aliases. Judge every cluster
independently of the others.

Return JSON only:
{{
  "groups": [
    {{"id": 0,
      "same": ["names that mean the same as the leader"],
      "different": ["names that do not mean the same as the leader"]}}
  ]
}}
Include exactly one entry for every cluster id from 0 to {last}.

Clusters:
{clusters_json}
"""


def build_ap_prompt(task: dict) -> str:
    return AP_VERIFY_PROMPT.format(
        leader_json=json.dumps(task["leader"], ensure_ascii=False),
        group_json=json.dumps(task["group"], ensure_ascii=False),
    )


def build_component_prompt(label: str, task: dict) -> str:
    return COMPONENT_VERIFY_PROMPT.format(
        label=label,
        leader_json=json.dumps(task["leader"], ensure_ascii=False),
        group_json=json.dumps(task["group"], ensure_ascii=False),
    )


def _batch_clusters_json(tasks: list[dict]) -> str:
    return json.dumps(
        [
            {"id": idx, "leader": task["leader"], "candidates": task["group"]}
            for idx, task in enumerate(tasks)
        ],
        ensure_ascii=False,
    )


def build_ap_batch_prompt(tasks: list[dict]) -> str:
    return AP_VERIFY_BATCH_PROMPT.format(
        count=len(tasks),
        last=len(tasks) - 1,
        clusters_json=_batch_clusters_json(tasks),
    )


def build_component_batch_prompt(label: str, tasks: list[dict]) -> str:
    return COMPONENT_VERIFY_BATCH_PROMPT.format(
        label=label,
        count=len(tasks),
        last=len(tasks) - 1,
        clusters_json=_batch_clusters_json(tasks),
    )


# ---------------------------------------------------------------------------
# Leader-find / LLM-verify engine: verify cache
# ---------------------------------------------------------------------------


def verify_cache_key(label: str, group: list[str]) -> str:
    """Stable cache key for one LLM verification group."""
    payload = json.dumps(
        [common.normalize_ap(value) for value in group], ensure_ascii=False
    )
    return f"{label}:{common.sha256_text(payload)}"


def load_verify_cache(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict LLM verify cache: {path}")
    return data


# ---------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------


def relation_tuple(relation) -> tuple[str, str, str] | None:
    """Normalize a [s, p, o] relation; None when malformed or partly empty."""
    if not isinstance(relation, (list, tuple)) or len(relation) != 3:
        return None
    parts = tuple(common.normalize_ap(part) for part in relation)
    if not all(parts):
        return None
    return parts


# ---------------------------------------------------------------------------
# Leader clustering (shared by both cleaners; the unit — component string or
# whole AP — is whatever `strings`/`vectors` encode)
# ---------------------------------------------------------------------------


def build_index(vectors: np.ndarray):
    """Build a FAISS inner-product index over L2-normalized vectors."""
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is required for nearest-neighbor candidate search "
            "(a core dependency).\n"
            "Install the project dependencies: uv sync"
        ) from exc

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def leader_clusters(
    strings: list[str],
    vectors: np.ndarray,
    frequencies: Counter,
    threshold: float,
    min_frequency: int,
    top_k_neighbors: int | None = None,
) -> tuple[list[dict], dict]:
    """Greedy leader-first clustering over an inner-product FAISS index.

    Visits strings by (-frequency, len, text); each unvisited string becomes a
    leader and claims its still-unassigned neighbors at or above `threshold`.
    With `top_k_neighbors` set, neighbors come from a bounded top-k search;
    with None, an exhaustive range search at `threshold` is used instead.
    """
    n = len(strings)
    index = build_index(vectors)
    assigned = np.zeros(n, dtype=bool)
    order = sorted(
        range(n),
        key=lambda idx: (-frequencies[strings[idx]], len(strings[idx]), strings[idx]),
    )
    clusters = []
    max_neighbor_similarity = -1.0
    searched_k = None if top_k_neighbors is None else min(max(top_k_neighbors, 2), n)

    for idx in tqdm.tqdm(order, desc="Leader clustering"):
        if assigned[idx]:
            continue
        leader = strings[idx]
        if frequencies[leader] < min_frequency:
            assigned[idx] = True
            continue
        if searched_k is None:
            _, sims, neighbors = index.range_search(vectors[idx : idx + 1], threshold)
        else:
            sims, neighbors = index.search(vectors[idx : idx + 1], searched_k)
            sims, neighbors = sims[0], neighbors[0]
        members = []
        for sim, neighbor in zip(sims, neighbors):
            neighbor = int(neighbor)
            if neighbor < 0 or assigned[neighbor]:
                continue
            sim = float(sim)
            if neighbor != idx:
                max_neighbor_similarity = max(max_neighbor_similarity, sim)
            if sim >= threshold:
                members.append(
                    {
                        "index": neighbor,
                        "text": strings[neighbor],
                        "frequency": int(frequencies[strings[neighbor]]),
                        "leader_similarity": sim,
                    }
                )
        if not any(member["index"] == idx for member in members):
            members.append(
                {
                    "index": idx,
                    "text": leader,
                    "frequency": int(frequencies[leader]),
                    "leader_similarity": 1.0,
                }
            )
        members = sorted(
            members,
            key=lambda item: (
                item["index"] != idx,
                -item["leader_similarity"],
                -item["frequency"],
                item["text"],
            ),
        )
        for member in members:
            assigned[member["index"]] = True
        if len(members) > 1:
            clusters.append(
                {
                    "cluster_id": f"cl{len(clusters):06d}",
                    "leader": leader,
                    "leader_frequency": int(frequencies[leader]),
                    "member_count": len(members),
                    "total_frequency": int(
                        sum(member["frequency"] for member in members)
                    ),
                    "members": members,
                }
            )
        else:
            assigned[idx] = True

    meta = {
        "cluster_mode": "leader",
        "search_mode": "range" if searched_k is None else "top_k",
        "threshold": threshold,
        "top_k_neighbors": top_k_neighbors,
        "searched_k": searched_k,
        "candidate_cluster_count": len(clusters),
        "max_neighbor_similarity": max_neighbor_similarity,
    }
    return clusters, meta


def verification_tasks(clusters: list[dict], max_items: int) -> list[dict]:
    """Split candidate clusters into bounded leader-anchored LLM groups."""
    tasks = []
    max_items = max(max_items, 2)
    for cluster in sorted(
        clusters, key=lambda item: (-item["total_frequency"], item["leader"])
    ):
        leader = cluster["leader"]
        candidates = [
            member for member in cluster["members"] if member["text"] != leader
        ]
        for start in range(0, len(candidates), max_items - 1):
            chunk = candidates[start : start + max_items - 1]
            group = [leader] + [member["text"] for member in chunk]
            sim_by_text = {
                member["text"]: member["leader_similarity"] for member in chunk
            }
            sim_by_text[leader] = 1.0
            tasks.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "leader": leader,
                    "group": group,
                    "similarity": sim_by_text,
                    "partial": len(candidates) > max_items - 1,
                }
            )
    return tasks


async def verify_one(
    task: dict,
    label: str,
    prompt: str,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
    fallback_auto_threshold: float,
    reasoning_effort: str | None = None,
) -> dict:
    """Run one leader-anchored verification group through the LLM."""
    leader = task["leader"]
    group = task["group"]
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    async with semaphore:
        for attempt in range(3):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**request),
                    timeout=timeout + 5,
                )
                data = json.loads(
                    common.strip_markdown_json(
                        response.choices[0].message.content or ""
                    )
                )
                if not isinstance(data.get("same"), list):
                    raise ValueError("LLM JSON did not contain same list")
                if not isinstance(data.get("different"), list):
                    data["different"] = []
                usage = response.usage.model_dump() if response.usage else None
                return {"result": data, "error": None, "source": "llm", "usage": usage}
            except Exception as exc:
                print(
                    f"  LLM verify failed attempt {attempt + 1} for {label} "
                    f"cluster={task['cluster_id']} size={len(group)} leader={leader!r}: {exc}",
                    flush=True,
                )
                await asyncio.sleep(2**attempt)

    if fallback_auto_threshold > 0:
        same = [
            value
            for value in group
            if float(task["similarity"].get(value, 0.0)) >= fallback_auto_threshold
        ]
    else:
        same = []
    return {
        "result": {
            "same": same,
            "different": [value for value in group if value not in same],
        },
        "error": "failed_after_retries",
        "source": "fallback_auto" if same else "fallback_skip",
    }


async def verify_batch_call(
    tasks: list[dict],
    label: str,
    prompt: str,
    client: AsyncOpenAI,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
    reasoning_effort: str | None,
) -> tuple[dict[int, dict] | None, dict | None]:
    """Run one packed request covering several verification groups.

    Returns ({cluster index -> {"same", "different"}}, usage). None results
    (whole-call failure) or missing/invalid per-cluster entries are handled
    by the caller, which falls back to single-group calls, so packing can
    never lose groups relative to the unpacked path.
    """
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        request["reasoning_effort"] = reasoning_effort
    async with semaphore:
        for attempt in range(2):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(**request),
                    timeout=timeout + 5,
                )
                data = json.loads(
                    common.strip_markdown_json(
                        response.choices[0].message.content or ""
                    )
                )
                groups = data.get("groups")
                if not isinstance(groups, list):
                    raise ValueError("batched LLM JSON did not contain groups list")
                parsed: dict[int, dict] = {}
                for entry in groups:
                    if not isinstance(entry, dict):
                        continue
                    try:
                        idx = int(entry.get("id"))
                    except (TypeError, ValueError):
                        continue
                    if idx < 0 or idx >= len(tasks) or idx in parsed:
                        continue
                    if not isinstance(entry.get("same"), list):
                        continue
                    different = entry.get("different")
                    parsed[idx] = {
                        "same": entry["same"],
                        "different": different if isinstance(different, list) else [],
                    }
                usage = response.usage.model_dump() if response.usage else None
                return parsed, usage
            except Exception as exc:
                print(
                    f"  LLM batched verify failed attempt {attempt + 1} for {label} "
                    f"({len(tasks)} groups): {exc}",
                    flush=True,
                )
                await asyncio.sleep(2**attempt)
    return None, None


def pack_verification_tasks(
    pending: list[tuple[str, dict]],
    groups_per_request: int,
    max_strings_per_request: int,
) -> tuple[list[list[tuple[str, dict]]], list[tuple[str, dict]]]:
    """Greedily pack pending (key, task) pairs into multi-group requests.

    Returns (batches of 2+ groups, leftovers routed to single-group calls).
    """
    if groups_per_request <= 1:
        return [], list(pending)
    batches: list[list[tuple[str, dict]]] = []
    singles: list[tuple[str, dict]] = []
    current: list[tuple[str, dict]] = []
    current_strings = 0
    for key, task in pending:
        size = len(task["group"])
        if current and (
            len(current) >= groups_per_request
            or current_strings + size > max_strings_per_request
        ):
            if len(current) > 1:
                batches.append(current)
            else:
                singles.extend(current)
            current, current_strings = [], 0
        current.append((key, task))
        current_strings += size
    if len(current) > 1:
        batches.append(current)
    else:
        singles.extend(current)
    return batches, singles


async def verify_tasks(
    tasks: list[dict],
    label: str,
    prompt_builder: Callable[[dict], str],
    api_key: str,
    base_url: str,
    model: str,
    workers: int,
    timeout: float,
    cache_path: Path,
    fallback_auto_threshold: float = 0.0,
    max_verify_groups: int | None = None,
    reasoning_effort: str | None = None,
    batch_prompt_builder: Callable[[list[dict]], str] | None = None,
    groups_per_request: int = 1,
    max_strings_per_request: int = 64,
) -> tuple[dict[str, dict], dict]:
    """Verify all groups through the LLM with a persistent JSON cache.

    With `batch_prompt_builder` and `groups_per_request` > 1, cache-miss
    groups are packed several-per-request (per-group cache keys unchanged);
    groups the packed response misses fall back to single-group calls.

    Returns (results by cache key, API usage summed over the freshly-run
    groups — cache hits cost nothing and are excluded).
    """
    if max_verify_groups is not None:
        tasks = tasks[:max_verify_groups]
    cache = load_verify_cache(cache_path)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    semaphore = asyncio.Semaphore(workers)
    results = {}
    usage_total: dict = {}
    pending = []
    retried = 0
    for task in tasks:
        key = verify_cache_key(label, task["group"])
        cached = cache.get(key)
        # A cached failure is not a hit: re-runs retry exactly those groups.
        if isinstance(cached, dict) and not cached.get("error"):
            results[key] = cached
        else:
            if isinstance(cached, dict):
                retried += 1
            pending.append((key, task))

    batches, singles = (
        pack_verification_tasks(pending, groups_per_request, max_strings_per_request)
        if batch_prompt_builder is not None
        else ([], list(pending))
    )
    print(
        f"  LLM verify {label}: {len(results)} cache hits, {len(pending)} misses"
        + (f" ({retried} cached failures being retried)" if retried else "")
        + (
            f"; packed into {len(batches)} multi-group + {len(singles)} single requests"
            if batch_prompt_builder is not None
            else ""
        )
    )

    async def run_one(key: str, task: dict):
        result = await verify_one(
            task=task,
            label=label,
            prompt=prompt_builder(task),
            client=client,
            model=model,
            semaphore=semaphore,
            timeout=timeout,
            fallback_auto_threshold=fallback_auto_threshold,
            reasoning_effort=reasoning_effort,
        )
        payload = {**task, **result}
        common.merge_usage(usage_total, payload.pop("usage", None))
        return key, payload

    async def run_batch(batch: list[tuple[str, dict]]):
        assert batch_prompt_builder is not None
        parsed, usage = await verify_batch_call(
            tasks=[task for _, task in batch],
            label=label,
            prompt=batch_prompt_builder([task for _, task in batch]),
            client=client,
            model=model,
            semaphore=semaphore,
            timeout=timeout,
            reasoning_effort=reasoning_effort,
        )
        common.merge_usage(usage_total, usage)
        out = []
        fallback: list[tuple[str, dict]] = []
        for idx, (key, task) in enumerate(batch):
            entry = None if parsed is None else parsed.get(idx)
            if entry is None:
                fallback.append((key, task))
            else:
                out.append(
                    (key, {**task, "result": entry, "error": None, "source": "llm_batched"})
                )
        for key, task in fallback:
            out.append(await run_one(key, task))
        return out

    try:
        coros = [run_batch(batch) for batch in batches] + [
            run_one(key, task) for key, task in singles
        ]
        completed_since_save = 0
        for coro in tqdm.tqdm(
            asyncio.as_completed(coros),
            total=len(coros),
            desc=f"LLM verifying {label}",
        ):
            outcome = await coro
            payloads = outcome if isinstance(outcome, list) else [outcome]
            for key, payload in payloads:
                cache[key] = payload
                results[key] = payload
                completed_since_save += 1
            if completed_since_save >= 20:
                common.write_json_atomic(cache_path, cache)
                completed_since_save = 0
        if pending:
            common.write_json_atomic(cache_path, cache)
    finally:
        await client.close()
    return results, usage_total


def build_replacements(
    tasks: list[dict],
    verify_results: dict[str, dict],
    label: str,
) -> tuple[dict[str, str], dict[str, list[str]], dict]:
    """Turn verified groups into a member -> leader replacement map."""
    replacement: dict[str, str] = {}
    removal_log: dict[str, list[str]] = defaultdict(list)
    stats = Counter()
    for task in tasks:
        key = verify_cache_key(label, task["group"])
        payload = verify_results.get(key)
        if not payload:
            stats["unverified_tasks"] += 1
            continue
        stats[f"source_{payload.get('source', 'unknown')}"] += 1
        if payload.get("error"):
            stats["failed_tasks"] += 1
        leader = common.normalize_ap(task["leader"])
        group_set = {common.normalize_ap(value) for value in task["group"]}
        same = []
        for value in payload.get("result", {}).get("same", []):
            norm = common.normalize_ap(value)
            if norm in group_set and norm not in same:
                same.append(norm)
        if same and leader not in same:
            same.insert(0, leader)
        if len(same) < 2:
            continue
        for value in same:
            if value == leader:
                continue
            current = replacement.get(value)
            if current is not None and current != leader:
                stats["conflicting_replacements"] += 1
                continue
            replacement[value] = leader
            removal_log[leader].append(value)
            stats["removed_strings"] += 1
    return replacement, dict(removal_log), dict(stats)


def run_leader_verify_rounds(
    pool: list[str],
    get_vectors: Callable[[list[str]], np.ndarray],
    frequencies: Counter,
    threshold: float,
    min_frequency: int,
    top_k_neighbors: int | None,
    max_llm_items: int,
    verify: Callable[[list[dict]], tuple[dict, dict]],
    label: str,
    max_rounds: int,
) -> dict:
    """Iterative leader-find/verify with re-pooling (Algorithm 1 semantics).

    Each round clusters the pool, verifies the candidate groups, and applies
    the approved merges. Verified leaders, merged members, and unclustered
    singletons then leave the pool; judge-REJECTED candidates re-pool and may
    lead or join new groups in the next round — so two near-duplicates that
    were both rejected from a third string's group still get merged with each
    other. Stops when a round adds no merges, no candidate clusters form, the
    pool is exhausted, or `max_rounds` is reached. Every leader is final once
    it heads a verified group, so the combined member->leader map is flat
    (no chains).

    Returns {replacement, removal_log, stats, rounds, first_round_meta,
    first_round_clusters, usage}.
    """
    replacement_all: dict[str, str] = {}
    removal_all: dict[str, list[str]] = defaultdict(list)
    stats_all: Counter = Counter()
    usage_all: dict = {}
    rounds_meta: list[dict] = []
    first_round_meta: dict = {}
    first_round_clusters: list[dict] = []

    pool = list(pool)
    for round_index in range(1, max_rounds + 1):
        if len(pool) < 2:
            break
        vectors = get_vectors(pool)
        clusters, cluster_meta = leader_clusters(
            strings=pool,
            vectors=vectors,
            frequencies=frequencies,
            threshold=threshold,
            min_frequency=min_frequency,
            top_k_neighbors=top_k_neighbors,
        )
        if round_index == 1:
            first_round_meta = cluster_meta
            first_round_clusters = clusters
        if not clusters:
            break
        tasks = verification_tasks(clusters, max_llm_items)
        print(
            f"  {label}: round {round_index}: pool {len(pool)} -> "
            f"{len(clusters)} candidate clusters, {len(tasks)} verification groups"
        )
        verify_results, usage = verify(tasks)
        common.merge_usage(usage_all, usage)
        replacement, removal_log, stats = build_replacements(
            tasks=tasks,
            verify_results=verify_results,
            label=label,
        )
        new_merges = 0
        for member, leader in replacement.items():
            if member in replacement_all or member == leader:
                continue
            replacement_all[member] = leader
            removal_all[leader].append(member)
            new_merges += 1
        for key, value in stats.items():
            stats_all[key] += value
        rounds_meta.append(
            {
                "round": round_index,
                "pool_size": len(pool),
                "candidate_clusters": len(clusters),
                "verification_groups": len(tasks),
                "new_merges": new_merges,
            }
        )
        print(f"  {label}: round {round_index}: {new_merges} new merges")
        if new_merges == 0:
            break
        # Re-pool only the judge-rejected candidates: verified leaders and
        # merged members are final, unclustered singletons cannot gain
        # neighbors in a shrinking pool.
        clustered = {member["text"] for c in clusters for member in c["members"]}
        leaders = {c["leader"] for c in clusters}
        pool = sorted(clustered - leaders - set(replacement_all))

    return {
        "replacement": replacement_all,
        "removal_log": dict(removal_all),
        "stats": dict(stats_all),
        "rounds": rounds_meta,
        "first_round_meta": first_round_meta,
        "first_round_clusters": first_round_clusters,
        "usage": usage_all,
    }


# ---------------------------------------------------------------------------
# Relation filtering
# ---------------------------------------------------------------------------


def is_relation_kept(relation: list[str]) -> bool:
    """Filter relations with TECHNICAL META-DATA and TAXONOMIC & REDUNDANCY filters."""
    if len(relation) != 3:
        return False
    s, p, o = [str(x).lower().strip() for x in relation]

    # TECHNICAL META-DATA FILTER
    # Deliberately aggressive plain-substring matching: it also drops words
    # that merely contain a keyword (e.g. "review" contains "view"). Tune this
    # list to your captions if too much is filtered.
    meta = [
        "frame",
        "blur",
        "sequence",
        "video",
        "background",
        "foreground",
        "image",
        "occlusion",
        "presence",
        "view",
        "provided",
    ]
    if any(m in s or m in p or m in o for m in meta):
        return False

    # TAXONOMIC & REDUNDANCY FILTER
    vehicle_classes = {
        "vehicle",
        "vehicles",
        "car",
        "cars",
        "object",
        "objects",
        "entity",
    }
    if p in {"is", "are", "include", "includes", "is a"}:
        if o in vehicle_classes:
            return False
        if s == o or s.split()[-1] == o:
            return False

    return True


# ---------------------------------------------------------------------------
# Whole-AP mode helpers
# ---------------------------------------------------------------------------


def ap_string(triplet: tuple[str, str, str]) -> str:
    return " ".join(triplet)


def collect_aps(relations: list) -> tuple[list[str], dict[str, list[str]], Counter]:
    """Unique AP strings (first-appearance order), string->triplet, frequency."""
    unique: list[str] = []
    triplet_by_ap: dict[str, list[str]] = {}
    frequencies: Counter[str] = Counter()
    for relation in relations:
        triplet = relation_tuple(relation)
        if triplet is None:
            continue
        ap = ap_string(triplet)
        if ap not in triplet_by_ap:
            unique.append(ap)
            triplet_by_ap[ap] = list(triplet)
        frequencies[ap] += 1
    return unique, triplet_by_ap, frequencies


def ap_vectors(aps: list[str], cache: dict[str, np.ndarray]) -> np.ndarray:
    """Native joined-string AP vectors from the cache, L2-normalized.

    The embedding prerequisite runs first, so every AP should be present;
    hard-error listing any string that is still missing (e.g. a zero vector).
    """
    missing: list[str] = []
    vectors = []
    for ap in aps:
        emb = ap_embeddings.cache_lookup(cache, ap)
        if emb is None:
            missing.append(ap)
            continue
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            missing.append(ap)
            continue
        vectors.append(emb / norm)
    if missing:
        listing = "\n".join(f"  {value!r}" for value in missing)
        raise SystemExit(
            f"Embedding cache is missing {len(missing)} AP string(s) after the "
            f"embedding prerequisite; the cache may be corrupt:\n{listing}"
        )
    return np.ascontiguousarray(np.stack(vectors), dtype=np.float32)


# ---------------------------------------------------------------------------
# Component mode helpers
# ---------------------------------------------------------------------------


def unique_normalized(strings: Iterable[str]) -> list[str]:
    values = []
    seen = set()
    for value in strings:
        norm = common.normalize_ap(value)
        if norm and norm not in seen:
            seen.add(norm)
            values.append(norm)
    return values


def component_frequency(relations: list, label: str) -> Counter:
    counts: Counter[str] = Counter()
    for relation in relations:
        triplet = relation_tuple(relation)
        if triplet is None:
            continue
        s, p, o = triplet
        if label == "entities":
            counts[s] += 1
            counts[o] += 1
        else:
            counts[p] += 1
    return counts


def resolve_embeddings(
    strings: list[str],
    cache: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, list[str]]:
    kept = []
    missing = []
    embeddings = []
    for value in strings:
        emb = ap_embeddings.cache_lookup(cache, value)
        if emb is None:
            missing.append(value)
            continue
        emb = np.asarray(emb, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm == 0:
            missing.append(value)
            continue
        kept.append(value)
        embeddings.append(emb / norm)
    if not embeddings:
        return kept, np.empty((0, 0), dtype=np.float32), missing
    return kept, np.ascontiguousarray(np.stack(embeddings), dtype=np.float32), missing


def dedup_components(
    strings: list[str],
    relations: list,
    label: str,
    cache: dict[str, np.ndarray],
    args: argparse.Namespace,
    threshold: float,
    api_key: str,
    verify_cache_path: Path,
) -> tuple[list[str], dict[str, str], dict, dict]:
    unique = unique_normalized(strings)
    frequencies = component_frequency(relations, label)
    for value in unique:
        frequencies[value] += 0
    print(f"  {label}: {len(strings)} raw -> {len(unique)} unique after normalization")
    embedded_strings, vectors, missing = resolve_embeddings(unique, cache)
    print(f"  {label}: {len(embedded_strings)} embedding hits, {len(missing)} misses")
    if len(embedded_strings) == 0:
        return unique, {}, {}, {"missing_examples": missing[:20]}

    vector_by_string = {s: vectors[i] for i, s in enumerate(embedded_strings)}

    def get_vectors(strings: list[str]) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack([vector_by_string[s] for s in strings]), dtype=np.float32
        )

    def verify(tasks: list[dict]) -> tuple[dict, dict]:
        if args.dry_run:
            return {}, {}
        return asyncio.run(
            verify_tasks(
                tasks=tasks,
                label=label,
                prompt_builder=lambda task: build_component_prompt(label, task),
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                workers=args.workers,
                timeout=args.llm_timeout,
                cache_path=verify_cache_path,
                fallback_auto_threshold=args.fallback_auto_threshold,
                max_verify_groups=args.max_verify_groups,
                reasoning_effort=args.reasoning_effort,
                batch_prompt_builder=lambda tasks_batch: build_component_batch_prompt(
                    label, tasks_batch
                ),
                groups_per_request=args.verify_groups_per_request,
                max_strings_per_request=args.max_strings_per_request,
            )
        )

    outcome = run_leader_verify_rounds(
        pool=embedded_strings,
        get_vectors=get_vectors,
        frequencies=frequencies,
        threshold=threshold,
        min_frequency=args.min_component_frequency,
        top_k_neighbors=args.top_k_neighbors,
        max_llm_items=args.max_llm_items,
        verify=verify,
        label=label,
        max_rounds=args.max_verify_rounds,
    )
    replacement = outcome["replacement"]
    removal_log = outcome["removal_log"]
    replacement_stats = outcome["stats"]
    verify_usage = outcome["usage"]
    clusters = outcome["first_round_clusters"]
    cluster_meta = outcome["first_round_meta"]
    failed_tasks = replacement_stats.get("failed_tasks", 0)
    if failed_tasks and not args.dry_run:
        print(
            f"  WARNING: {label}: {failed_tasks} verification group(s) failed after retries "
            f"and were left unmerged; re-run to retry them (verified results are cached "
            f"at {verify_cache_path})"
        )
    removed = set(replacement)
    deduped = [value for value in unique if value not in removed]
    print(
        f"  {label}: {len(unique)} -> {len(deduped)} after leader dedup ({len(removed)} removed)"
    )
    summary = {
        **cluster_meta,
        "raw_count": len(strings),
        "unique_count": len(unique),
        "embedded_count": len(embedded_strings),
        "missing_embedding_count": len(missing),
        "missing_embedding_examples": missing[:20],
        "verification_group_count": sum(
            r["verification_groups"] for r in outcome["rounds"]
        ),
        "verify_rounds": outcome["rounds"],
        "api_usage_total": verify_usage,
        "deduped_count": len(deduped),
        "removed_count": len(removed),
        "replacement_stats": replacement_stats,
        "sample_clusters": [
            {
                **{key: value for key, value in cluster.items() if key != "members"},
                "members": cluster["members"][: args.max_log_members],
            }
            for cluster in clusters[: args.max_log_clusters]
        ],
    }
    return deduped, replacement, removal_log, summary


# ---------------------------------------------------------------------------
# Shared relation rebuild
# ---------------------------------------------------------------------------


def dedup_and_filter_relations(
    updated_relations: list[list[str]],
) -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Deduplicate rewritten relations, then apply is_relation_kept."""
    seen = set()
    relations_deduped = []
    for relation in updated_relations:
        key = tuple(relation)
        if key in seen:
            continue
        seen.add(key)
        relations_deduped.append(relation)

    relations_kept = []
    relations_removed = []
    for relation in relations_deduped:
        if is_relation_kept(relation):
            relations_kept.append(relation)
        else:
            relations_removed.append(relation)
    return relations_deduped, relations_kept, relations_removed


def write_output(
    args: argparse.Namespace,
    kg: dict,
    entities,
    edges,
    relations,
    counts_before: tuple[int, int, int],
    summary: dict,
) -> None:
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("[dry-run] output graph was not written")
        return
    output = dict(kg)
    output["entities"] = entities
    output["edges"] = edges
    output["relations"] = relations
    common.write_json_atomic(args.output, output)
    print(f"\nSaved cleaned atlas to {args.output}")
    print(f"Saved logs to {args.log_dir}/")
    print("\n" + "=" * 40)
    print(f"  Entities:  {counts_before[0]:>6} -> {len(entities):>6}")
    print(f"  Edges:     {counts_before[1]:>6} -> {len(edges):>6}")
    print(f"  Relations: {counts_before[2]:>6} -> {len(relations):>6}")
    print("=" * 40)


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_whole_ap(
    args: argparse.Namespace,
    kg: dict,
    threshold: float,
    cache: dict[str, np.ndarray],
    embed_stats: dict,
    aps: list[str],
    triplet_by_ap: dict[str, list[str]],
    frequencies: Counter,
    api_key: str,
) -> None:
    entities_raw = kg.get("entities", []) or []
    edges_raw = kg.get("edges", []) or []
    relations_raw = kg.get("relations", []) or []
    verify_cache_path = args.verify_cache or (
        args.log_dir / "ap_leader_llm_verify_cache.json"
    )

    print("\n=== Leader dedup APs ===")
    print(
        f"  APs: {len(relations_raw)} relations -> {len(aps)} unique APs after normalization"
    )

    def verify(tasks: list[dict]) -> tuple[dict, dict]:
        if args.dry_run:
            return {}, {}
        return asyncio.run(
            verify_tasks(
                tasks=tasks,
                label="aps",
                prompt_builder=build_ap_prompt,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                workers=args.workers,
                timeout=args.llm_timeout,
                cache_path=verify_cache_path,
                reasoning_effort=args.reasoning_effort,
                batch_prompt_builder=build_ap_batch_prompt,
                groups_per_request=args.verify_groups_per_request,
                max_strings_per_request=args.max_strings_per_request,
            )
        )

    outcome = run_leader_verify_rounds(
        pool=aps,
        get_vectors=lambda strings: ap_vectors(strings, cache),
        frequencies=frequencies,
        threshold=threshold,
        min_frequency=1,
        top_k_neighbors=None,  # exhaustive range search at the threshold
        max_llm_items=args.max_llm_items,
        verify=verify,
        label="aps",
        max_rounds=args.max_verify_rounds,
    )
    replacement = outcome["replacement"]
    removal_log = outcome["removal_log"]
    replacement_stats = outcome["stats"]
    verify_usage = outcome["usage"]
    failed_tasks = replacement_stats.get("failed_tasks", 0)
    if failed_tasks and not args.dry_run:
        print(
            f"  WARNING: {failed_tasks} verification group(s) failed after retries and were "
            f"left unmerged; re-run to retry them (verified results are cached "
            f"at {verify_cache_path})"
        )
    print(
        f"  APs: {len(aps)} -> {len(aps) - len(replacement)} after leader dedup "
        f"({len(replacement)} merged into their leaders)"
    )

    print("\n=== Updating relations ===")
    updated_relations = []
    for relation in relations_raw:
        triplet = relation_tuple(relation)
        if triplet is None:
            continue
        leader = replacement.get(ap_string(triplet))
        updated_relations.append(
            list(triplet_by_ap[leader]) if leader else list(triplet)
        )
    relations_deduped, relations_kept, relations_removed = dedup_and_filter_relations(
        updated_relations
    )
    print(
        f"  Relations: {len(relations_raw)} -> {len(relations_deduped)} after replacement & dedup"
    )
    print(
        f"  Relations: {len(relations_deduped)} -> {len(relations_kept)} after filtering "
        f"({len(relations_removed)} removed)"
    )

    entities_out: list[str] = []
    edges_out: list[str] = []
    seen_entities: set[str] = set()
    seen_edges: set[str] = set()
    for s, p, o in relations_kept:
        for entity in (s, o):
            if entity not in seen_entities:
                seen_entities.add(entity)
                entities_out.append(entity)
        if p not in seen_edges:
            seen_edges.add(p)
            edges_out.append(p)

    summary = {
        "mode": "whole-ap",
        "cleaning_scope": "whole_aps_not_components",
        "parameters": {
            "threshold": threshold,
            "max_llm_items": args.max_llm_items,
            "llm_model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "verify_groups_per_request": args.verify_groups_per_request,
            "max_strings_per_request": args.max_strings_per_request,
            "workers": args.workers,
            "llm_timeout": args.llm_timeout,
            "dry_run": args.dry_run,
        },
        "embedding": embed_stats,
        "before": {
            "entities": len(entities_raw),
            "edges": len(edges_raw),
            "relations": len(relations_raw),
        },
        "after": {
            "entities": len(entities_out),
            "edges": len(edges_out),
            "relations": len(relations_kept),
        },
        "aps": {
            **outcome["first_round_meta"],
            "unique_count": len(aps),
            "verification_group_count": sum(
                r["verification_groups"] for r in outcome["rounds"]
            ),
            "verify_rounds": outcome["rounds"],
            "api_usage_total": verify_usage,
            "merged_count": len(replacement),
            "replacement_stats": replacement_stats,
        },
        "relations_after_replacement_dedup": len(relations_deduped),
        "relations_removed_by_filter": len(relations_removed),
    }

    common.write_json_atomic(args.log_dir / "ap_dedup_log.json", removal_log)
    common.write_json_atomic(
        args.log_dir / "relation_filter_log.json", relations_removed
    )
    common.write_json_atomic(args.log_dir / "atlas_clean_summary.json", summary)
    write_output(
        args,
        kg,
        entities_out,
        edges_out,
        relations_kept,
        (len(entities_raw), len(edges_raw), len(relations_raw)),
        summary,
    )


def run_component(
    args: argparse.Namespace,
    kg: dict,
    threshold: float,
    cache: dict[str, np.ndarray],
    embed_stats: dict,
    api_key: str,
) -> None:
    entities_raw = kg.get("entities", []) or []
    edges_raw = kg.get("edges", []) or []
    relations_raw = kg.get("relations", []) or []
    verify_cache_path = args.verify_cache or (
        args.log_dir / "component_leader_llm_verify_cache.json"
    )

    print("\n=== Leader dedup entities ===")
    entities_deduped, entity_replace, entity_log, entity_summary = dedup_components(
        strings=entities_raw,
        relations=relations_raw,
        label="entities",
        cache=cache,
        args=args,
        threshold=threshold,
        api_key=api_key,
        verify_cache_path=verify_cache_path,
    )

    print("\n=== Leader dedup edges ===")
    edges_deduped, edge_replace, edge_log, edge_summary = dedup_components(
        strings=edges_raw,
        relations=relations_raw,
        label="edges",
        cache=cache,
        args=args,
        threshold=threshold,
        api_key=api_key,
        verify_cache_path=verify_cache_path,
    )

    print("\n=== Updating relations ===")
    updated_relations = []
    for relation in relations_raw:
        triplet = relation_tuple(relation)
        if triplet is None:
            continue
        s, p, o = triplet
        updated_relations.append(
            [
                entity_replace.get(s, s),
                edge_replace.get(p, p),
                entity_replace.get(o, o),
            ]
        )
    relations_deduped, relations_kept, relations_removed = dedup_and_filter_relations(
        updated_relations
    )
    print(
        f"  Relations: {len(relations_raw)} -> {len(relations_deduped)} after replacement & dedup"
    )
    print(
        f"  Relations: {len(relations_deduped)} -> {len(relations_kept)} after filtering "
        f"({len(relations_removed)} removed)"
    )

    summary = {
        "mode": "component",
        "cleaning_scope": "triplet_components_not_whole_aps",
        "parameters": {
            "threshold": threshold,
            "top_k_neighbors": args.top_k_neighbors,
            "min_component_frequency": args.min_component_frequency,
            "max_llm_items": args.max_llm_items,
            "llm_model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "verify_groups_per_request": args.verify_groups_per_request,
            "max_strings_per_request": args.max_strings_per_request,
            "workers": args.workers,
            "llm_timeout": args.llm_timeout,
            "fallback_auto_threshold": args.fallback_auto_threshold,
            "max_verify_groups": args.max_verify_groups,
            "dry_run": args.dry_run,
        },
        "embedding": embed_stats,
        "before": {
            "entities": len(entities_raw),
            "edges": len(edges_raw),
            "relations": len(relations_raw),
        },
        "after": {
            "entities": len(entities_deduped),
            "edges": len(edges_deduped),
            "relations": len(relations_kept),
        },
        "entities": entity_summary,
        "edges": edge_summary,
        "relations_after_replacement_dedup": len(relations_deduped),
        "relations_removed_by_filter": len(relations_removed),
    }

    common.write_json_atomic(args.log_dir / "entity_dedup_log.json", entity_log)
    common.write_json_atomic(args.log_dir / "edge_dedup_log.json", edge_log)
    common.write_json_atomic(
        args.log_dir / "relation_filter_log.json", relations_removed
    )
    common.write_json_atomic(args.log_dir / "atlas_clean_summary.json", summary)
    write_output(
        args,
        kg,
        entities_deduped,
        edges_deduped,
        relations_kept,
        (len(entities_raw), len(edges_raw), len(relations_raw)),
        summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atlas unification: precompute and persist the embeddings this stage "
            "needs, then dedup the atlas with FAISS nearest-neighbor candidate "
            "groups, leader election, and LLM verification. The default cleans "
            "whole AP triplets; --component cleans entity/predicate components."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input raw atlas JSON file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output cleaned atlas JSON file"
    )
    parser.add_argument(
        "--log-dir",
        required=True,
        type=Path,
        help="Directory for dedup/filter/summary log JSONs",
    )
    parser.add_argument(
        "--component",
        action="store_true",
        help="Clean entity/predicate COMPONENTS instead of whole AP triplets",
    )

    embed = parser.add_argument_group("embedding prerequisite")
    embed.add_argument(
        "--embedding-cache",
        required=True,
        type=Path,
        help="Persistent {text: embedding} dill cache; loaded if it "
        "exists, extended with any missing required strings, and "
        "persisted. Same format as the stage-6 cache from "
        "scripts/ap_embeddings.py, so in "
        "--component mode you can point this at the stage-6 "
        "pickle and no embedding work is needed",
    )
    embed.add_argument(
        "--embed-only",
        action="store_true",
        help="Run only the embedding prerequisite and persist the cache, "
        "then exit before any cleaning/LLM work (e.g. to run the "
        "GPU part on a different node)",
    )
    embed.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="Sentence-transformer model for missing embeddings "
        "(default: Qwen/Qwen3-Embedding-8B)",
    )
    embed.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256)",
    )
    embed.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an embedding-cache checkpoint every N encoded "
        "batches; 0 saves only at the end",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for candidate groups "
        f"(default: {DEFAULT_THRESHOLD_WHOLE_AP} in whole-AP mode, "
        f"{DEFAULT_THRESHOLD_COMPONENT} in --component mode)",
    )
    parser.add_argument(
        "--max-llm-items",
        type=int,
        default=16,
        help="Maximum strings per LLM verification group (default: 16)",
    )
    parser.add_argument(
        "--max-verify-rounds",
        type=int,
        default=8,
        help="Iterative re-pooling rounds: after each verification round, "
        "judge-rejected candidates return to the pool and are re-clustered "
        "among themselves (verified leaders and merged members leave the "
        "pool), so near-duplicates rejected from a third string's group "
        "still merge with each other. Stops early when a round adds no "
        "merges; 1 restores single-pass behavior. Default 8 = the merge "
        "plateau measured on the e2e atlas (last round ~2%% of merges; "
        "rounds 9-10 add only noise-level merges)",
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="Concurrent LLM calls (default: 16)"
    )
    parser.add_argument(
        "--verify-groups-per-request",
        type=int,
        default=4,
        help="Pack up to this many verification groups into one LLM request "
        "(most clusters are 2-3 strings, so per-request prompt and "
        "thinking overhead dominates unpacked calls); groups the packed "
        "response misses fall back to single-group calls. Larger packs "
        "are cheaper but drift toward slightly more merging; 1 disables "
        "packing (default: 4)",
    )
    parser.add_argument(
        "--max-strings-per-request",
        type=int,
        default=64,
        help="Cap on total candidate strings across the groups packed into "
        "one request (default: 64)",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=60.0,
        help="Per-request LLM timeout in seconds (default: 60)",
    )
    parser.add_argument(
        "--verify-cache",
        type=Path,
        default=None,
        help="JSON cache for LLM verification results (default: "
        "<log-dir>/ap_leader_llm_verify_cache.json, or "
        "<log-dir>/component_leader_llm_verify_cache.json "
        "in --component mode)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview candidate groups for threshold tuning without an API "
        "key: write logs and a summary but not the cleaned atlas",
    )

    component = parser.add_argument_group("--component mode options")
    component.add_argument(
        "--top-k-neighbors",
        type=int,
        default=64,
        help="FAISS neighbors searched per leader (default: 64; "
        "whole-AP mode always uses an exhaustive range search)",
    )
    component.add_argument(
        "--min-component-frequency",
        type=int,
        default=1,
        help="Minimum leader frequency to form a cluster (default: 1)",
    )
    component.add_argument(
        "--fallback-auto-threshold",
        type=float,
        default=0.0,
        help="On LLM failure, auto-accept members at or above this "
        "leader similarity; 0 disables the fallback (default: 0)",
    )
    component.add_argument(
        "--max-verify-groups",
        type=int,
        default=None,
        help="Verify at most this many groups (default: all)",
    )
    component.add_argument(
        "--max-log-clusters",
        type=int,
        default=200,
        help="Sample clusters kept in the summary log (default: 200)",
    )
    component.add_argument(
        "--max-log-members",
        type=int,
        default=20,
        help="Members kept per sample cluster in the summary log (default: 20)",
    )

    common.add_llm_args(parser, endpoint_optional=True)
    return parser


def main() -> None:
    args = argparser().parse_args()
    args.reasoning_effort = common.effective_reasoning_effort(args.reasoning_effort)
    mode = "component" if args.component else "whole-ap"
    threshold = (
        args.threshold
        if args.threshold is not None
        else (
            DEFAULT_THRESHOLD_COMPONENT
            if args.component
            else DEFAULT_THRESHOLD_WHOLE_AP
        )
    )
    print(f"Mode: {mode}")
    print(
        f"Similarity threshold: {threshold}"
        + ("" if args.threshold is not None else " (mode default)")
    )

    if not args.input.is_file():
        sys.exit(f"Error: --input atlas not found: {args.input}")
    try:
        kg = json.loads(args.input.read_text())
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: --input is not valid JSON ({args.input}): {exc}")
    entities_raw = kg.get("entities", []) or []
    edges_raw = kg.get("edges", []) or []
    relations_raw = kg.get("relations", []) or []
    print(
        f"Loading atlas from {args.input}: "
        f"{len(entities_raw)} entities, {len(edges_raw)} edges, {len(relations_raw)} relations"
    )

    # Embedding prerequisite: compute + persist whatever this mode needs.
    if args.component:
        print(
            "Cleaning scope: triplet components (entities and predicates), not whole APs"
        )
        required = unique_normalized(list(entities_raw) + list(edges_raw))
        aps, triplet_by_ap, frequencies = [], {}, Counter()
    else:
        print("Cleaning scope: whole AP triplets (not individual components)")
        aps, triplet_by_ap, frequencies = collect_aps(relations_raw)
        required = aps
    cache, embed_stats = ap_embeddings.ensure_embeddings(
        cache_path=args.embedding_cache,
        required=required,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
    )
    if args.embed_only:
        print("--embed-only: embedding prerequisite done, skipping cleaning/LLM work")
        return

    if not args.dry_run:
        common.require_endpoint(args)
    api_key = "" if args.dry_run else common.load_api_key(args.api_key_file)
    if args.component:
        run_component(args, kg, threshold, cache, embed_stats, api_key)
    else:
        run_whole_ap(
            args,
            kg,
            threshold,
            cache,
            embed_stats,
            aps,
            triplet_by_ap,
            frequencies,
            api_key,
        )


if __name__ == "__main__":
    main()
