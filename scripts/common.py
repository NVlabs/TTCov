# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the TTCov pipeline scripts.

Covers LLM endpoint/key resolution, atomic file IO, resumable locks,
work-item selection/sharding, and KG (proposition graph) loading.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


# Any OpenAI-compatible chat-completions endpoint works. There is no built-in
# endpoint or model default: supply --base-url and --model on each script, or
# set the TTCOV_API_BASE / TTCOV_MODEL environment variables. Both are None when
# unset; add_llm_args then makes the flags required (see require_endpoint).
DEFAULT_API_BASE = os.environ.get("TTCOV_API_BASE")
DEFAULT_MODEL = os.environ.get("TTCOV_MODEL")

# Repo root (this file lives in scripts/).
REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "prompts"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# API key / endpoint
# ---------------------------------------------------------------------------


def load_api_key(explicit_path: Path | str | None = None) -> str:
    """Resolve the LLM API key: --api-key-file wins, then TTCOV_API_KEY."""
    if explicit_path is not None:
        path = Path(explicit_path)
        key = path.read_text().strip()
        if not key:
            raise ValueError(f"API key file is empty: {path}")
        return key
    env_key = os.environ.get("TTCOV_API_KEY")
    if env_key:
        return env_key.strip()
    raise FileNotFoundError(
        "No API key found. Pass --api-key-file <path> or set TTCOV_API_KEY."
    )


def effective_reasoning_effort(value: str | None) -> str | None:
    """Map the --reasoning-effort CLI value to the request field.

    'default' (and None) mean "omit the field", leaving the provider's own
    thinking behavior in effect.
    """
    return None if value in (None, "default") else value


def merge_usage(total: dict, usage: dict | None) -> dict:
    """Accumulate an API-response `usage` block into a running total (in place).

    Sums numeric fields recursively, so nested detail blocks (e.g.
    completion_tokens_details) are accumulated too. None is a no-op, letting
    callers pass usage straight from responses that omit it.
    """
    if usage is None:
        return total
    for key, value in usage.items():
        if isinstance(value, dict):
            existing = total.get(key)
            if not isinstance(existing, dict):
                existing = {}
                total[key] = existing
            merge_usage(existing, value)
        elif isinstance(value, (int, float)):
            if isinstance(total.get(key), dict):
                continue  # scalar/dict schema mismatch across responses; keep the dict
            total[key] = total.get(key, 0) + value
    return total


def add_llm_args(parser, endpoint_optional: bool = False) -> None:
    """Attach the standard LLM connection flags to an argparse parser.

    ``--model`` and ``--base-url`` have no built-in default. They are required
    unless supplied via the TTCOV_MODEL / TTCOV_API_BASE environment variables.
    Scripts with a no-API mode (e.g. clean_atlas ``--dry-run``) pass
    ``endpoint_optional=True`` so the flags stay optional, and validate them at
    the point of an actual API call via :func:`require_endpoint`.
    """
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        required=DEFAULT_MODEL is None and not endpoint_optional,
        help="Model name served by --base-url (pick the two together). "
        "No default; required unless set via the TTCOV_MODEL env var.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_API_BASE,
        required=DEFAULT_API_BASE is None and not endpoint_optional,
        help="OpenAI-compatible chat-completions base URL. No default; "
        "required unless set via the TTCOV_API_BASE env var.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="File containing the API key (default: TTCOV_API_KEY env var)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["default", "none", "low", "medium", "high"],
        default="low",
        help="Reasoning/thinking effort forwarded to the endpoint. Thinking "
        "models such as Gemini 2.5 otherwise think with an unbounded dynamic "
        "budget and bill those tokens at the output rate; 'low' (the default) "
        "caps the budget with output quality equivalent to unbounded thinking "
        "on this pipeline's tasks, 'none' disables thinking where the model "
        "allows it, and 'default' omits the field so the provider's dynamic "
        "budget applies.",
    )


def require_endpoint(args) -> None:
    """Ensure an LLM endpoint and model are set before making an API call.

    Scripts that attach the LLM flags with ``endpoint_optional=True`` (because
    they have a no-API mode) call this only when they actually need the API.
    Raises SystemExit with a clear message if either is missing.
    """
    missing = [
        flag
        for flag, value in (("--base-url", args.base_url), ("--model", args.model))
        if not value
    ]
    if missing:
        raise SystemExit(
            "Missing required LLM setting(s): "
            + ", ".join(missing)
            + ". Pass them on the command line or set TTCOV_API_BASE / TTCOV_MODEL."
        )


# ---------------------------------------------------------------------------
# Atomic IO and locks
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp_path.write_text(text.rstrip() + "\n", encoding="utf-8")
    tmp_path.replace(path)


def acquire_lock(path: Path, stale_lock_seconds: float) -> bool:
    """Create an exclusive lock file; reclaim it when older than the threshold.

    Returns False whenever another process holds (or wins) the lock — races
    with a concurrent release or reclaim are treated as "not acquired" rather
    than raised, so callers can simply skip the item.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            age = time.time() - path.stat().st_mtime
        except FileNotFoundError:
            return False  # holder released between open and stat
        if age <= stale_lock_seconds:
            return False
        path.unlink(missing_ok=True)
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False  # another reclaimer won the race

    with os.fdopen(fd, "w") as f:
        json.dump(
            {"pid": os.getpid(), "host": socket.gethostname(), "created_at": utc_now()},
            f,
        )
    return True


# ---------------------------------------------------------------------------
# Work-item discovery, selection, and sharding
# ---------------------------------------------------------------------------


def load_item_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def select_by_common_args(
    items: list,
    item_ids: set[str] | None,
    offset: int,
    limit: int | None,
    num_shards: int,
    shard_index: int,
    key=lambda item: item.stem,
) -> list:
    """Filter items by id set, shard them round-robin, then apply offset/limit."""
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")

    if item_ids is not None:
        items = [item for item in items if key(item) in item_ids]
    items = [item for idx, item in enumerate(items) if idx % num_shards == shard_index]
    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


def add_selection_args(parser) -> None:
    """Attach the standard resumable-run selection flags."""
    parser.add_argument(
        "--item-ids",
        type=Path,
        default=None,
        help="Optional file with one item/clip id per line to process",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip this many items after id filtering and sharding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many items (default: all)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the items round-robin over this many parallel workers",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard this worker processes, in [0, --num-shards)",
    )


def discover_caption_files(root: Path) -> list[Path]:
    """Find caption .txt files under ``root`` (or ``root/captions`` when present).

    Accepts either a stage-1 run root or the captions directory itself, so
    downstream stages can be pointed at either.
    """
    base = root / "captions" if (root / "captions").is_dir() else root
    return sorted(
        path
        for path in base.rglob("*.txt")
        if path.is_file()
        and path.stem
        # hidden-dir filter applies below the captions base only, so a run
        # rooted inside a dot-directory still discovers its captions
        and not any(part.startswith(".") for part in path.relative_to(base).parts)
    )


def discover_json_items(root: Path) -> list[Path]:
    """Find per-item output JSONs under ``root``, skipping run bookkeeping
    (``_manifests/``, ``.locks/``, ``manifest.json``)."""
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.is_file()
        and "_manifests" not in path.parts
        and ".locks" not in path.parts
        and path.name != "manifest.json"
    )


def output_json_path(output_dir: Path, item_id: str) -> Path:
    """Fan-out path for one item's JSON output: <output_dir>/<id[:4]>/<id>.json."""
    return output_dir / item_id[:4] / f"{item_id}.json"


def output_graph_dir(output_dir: Path, item_id: str) -> Path:
    """Fan-out directory for one item's graph.json: <output_dir>/<id[:4]>/<id>/."""
    return output_dir / item_id[:4] / item_id


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------


def strip_markdown_json(content: str) -> str:
    """Extract the JSON object from a possibly markdown-fenced LLM response."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
    if content.startswith("json"):
        content = content[4:].strip()
    if not content.startswith("{"):
        brace_index = content.find("{")
        if brace_index < 0:
            raise ValueError("LLM response did not contain a JSON object")
        content = content[brace_index:]
    end_index = content.rfind("}")
    if end_index >= 0:
        content = content[: end_index + 1]
    return content.strip()


def render_prompt(template: str, input_text: str) -> str:
    """Insert the input into a prompt template.

    Templates may mark the insertion point with the literal placeholder
    ``<insert input here>``; without one, the input is appended under an
    ``Input Text:`` header.
    """
    placeholder = "<insert input here>"
    if placeholder in template:
        return template.replace(placeholder, input_text)
    return f"{template.rstrip()}\n\nInput Text:\n{input_text}"


def iter_basic_sentences(breakdown: dict) -> Iterable[str]:
    """Yield the atomic sentences from a stage-2 distillation JSON,
    normalized to end with sentence punctuation."""
    for item in breakdown.get("paragraph_breakdown", []):
        for sentence in item.get("basic_sentences", []):
            sentence = str(sentence).strip()
            if not sentence:
                continue
            if sentence[-1] not in ".!?":
                sentence += "."
            yield sentence


# ---------------------------------------------------------------------------
# Proposition graphs (KGs)
# ---------------------------------------------------------------------------


def normalize(s: str) -> str:
    """Canonical string form used for AP comparison: lowercased and stripped."""
    return s.lower().strip()


def normalize_ap(text: str) -> str:
    """normalize plus trailing-comma stripping (the atlas cleaners' AP key)."""
    return normalize(str(text)).rstrip(",")


def load_kg(path: Path | str) -> list:
    """Load the [subject, predicate, object] triplet list from a graph JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    relations = data.get("relations", [])
    return [r for r in relations if isinstance(r, (list, tuple)) and len(r) == 3]
