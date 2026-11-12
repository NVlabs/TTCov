# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Distill captions into atomic sentences.

Reads plain-text caption files (captions/<item_id[:4]>/<item_id>.txt), asks an
LLM to break each caption into atomic subject-verb-object sentences, and writes
one breakdown JSON per item to <output-dir>/<item_id[:4]>/<item_id>.json.

Runs are resumable: existing outputs are skipped (unless --overwrite) and
in-flight items are protected by lock files, so multiple instances can share
the same output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
from pathlib import Path

import httpx

import common


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Distill captions into atomic sentences (one breakdown JSON per item)."
    )
    parser.add_argument(
        "-i",
        "--captions-dir",
        type=Path,
        required=True,
        help="Directory containing caption .txt files (searched recursively)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for per-item breakdown JSON files",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=common.PROMPTS_DIR / "distillation.txt",
        help="Prompt template file (default: prompts/distillation.txt)",
    )
    common.add_llm_args(parser)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max simultaneous LLM requests (default: 16)",
    )
    parser.add_argument(
        "--max-retries",
        type=lambda v: max(1, int(v)),
        default=3,
        help="LLM attempts per item, minimum 1 (default: 3)",
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
        help="Re-process items that already have an output file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected work without calling the LLM",
    )
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Reclaim lock files older than this many seconds",
    )
    return parser


async def call_llm_json(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_retries: int,
    reasoning_effort: str | None = None,
) -> tuple[dict, dict | None]:
    """Send one chat request; returns (parsed JSON, API usage block or None)."""
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
            response = await client.post(
                "/chat/completions",
                json=request,
                timeout=request_timeout,
            )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"] or ""
            return json.loads(common.strip_markdown_json(content)), body.get("usage")
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Distillation failed after {max_retries} attempts: {last_error}"
    )


async def process_caption(
    caption_path: Path,
    args: argparse.Namespace,
    template: str,
    prompt_hash: str,
    client: httpx.AsyncClient | None,
) -> dict:
    item_id = caption_path.stem
    out_path = common.output_json_path(args.output_dir, item_id)
    lock_path = args.output_dir / ".locks" / item_id[:4] / f"{item_id}.lock"

    if out_path.exists() and not args.overwrite:
        return {"item_id": item_id, "status": "skipped_existing"}

    if not common.acquire_lock(lock_path, args.stale_lock_seconds):
        return {"item_id": item_id, "status": "skipped_locked"}

    try:
        caption = caption_path.read_text(encoding="utf-8").strip()
        if args.dry_run:
            return {
                "item_id": item_id,
                "status": "dry_run",
                "caption_path": str(caption_path),
                "caption_chars": len(caption),
            }

        assert client is not None
        prompt = common.render_prompt(template, caption)
        breakdown, usage = await call_llm_json(
            client=client,
            model=args.model,
            prompt=prompt,
            temperature=args.temperature,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            reasoning_effort=args.reasoning_effort,
        )

        payload = {
            "schema_version": 1,
            "item_id": item_id,
            "caption_path": str(caption_path),
            "caption_sha256": common.sha256_text(caption),
            "caption_chars": len(caption),
            "prompt_file": str(args.prompt_file),
            "prompt_sha256": prompt_hash,
            "model": {
                "name": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "request_timeout": args.request_timeout,
                "reasoning_effort": args.reasoning_effort,
            },
            "created_at": common.utc_now(),
            "breakdown": breakdown,
        }
        common.write_json_atomic(out_path, payload)
        return {"item_id": item_id, "status": "done", "usage": usage}
    except Exception as exc:
        return {
            "item_id": item_id,
            "status": "error",
            "caption_path": str(caption_path),
            "error": repr(exc),
        }
    finally:
        lock_path.unlink(missing_ok=True)


async def main_async(args: argparse.Namespace) -> None:
    args.reasoning_effort = common.effective_reasoning_effort(args.reasoning_effort)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.run_id is None:
        args.run_id = f"{common.utc_now().replace(':', '').replace('-', '')}_{socket.gethostname()}_{os.getpid()}"

    template = args.prompt_file.read_text(encoding="utf-8")
    prompt_hash = common.sha256_text(template)
    caption_files = common.discover_caption_files(args.captions_dir)
    selected_files = common.select_by_common_args(
        caption_files,
        item_ids=common.load_item_ids(args.item_ids),
        offset=args.offset,
        limit=args.limit,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    client = None
    if not args.dry_run:
        api_key = common.load_api_key(args.api_key_file)
        client = httpx.AsyncClient(
            base_url=args.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    manifest = {
        "schema_version": 1,
        "status": "running",
        "run_id": args.run_id,
        "created_at": common.utc_now(),
        "finished_at": None,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "captions_dir": str(args.captions_dir),
        "output_dir": str(args.output_dir),
        "output_index": "{output_dir}/{item_id[:4]}/{item_id}.json",
        "prompt_file": str(args.prompt_file),
        "prompt_sha256": prompt_hash,
        "model": {
            "name": args.model,
            "base_url": args.base_url,
            "reasoning_effort": args.reasoning_effort,
        },
        "request_timeout": args.request_timeout,
        "concurrency": args.concurrency,
        "num_caption_files": len(caption_files),
        "num_selected": len(selected_files),
        "limit": args.limit,
        "offset": args.offset,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "dry_run": args.dry_run,
        "counts": {},
        "items": [],
    }
    manifest_path = args.output_dir / "_manifests" / f"{args.run_id}.json"
    common.write_json_atomic(manifest_path, manifest)

    semaphore = asyncio.Semaphore(args.concurrency)
    counts: dict[str, int] = {}

    async def gated(caption_path: Path) -> dict:
        async with semaphore:
            result = await process_caption(
                caption_path=caption_path,
                args=args,
                template=template,
                prompt_hash=prompt_hash,
                client=client,
            )
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            print(
                json.dumps({"item_id": result["item_id"], "status": result["status"]}),
                flush=True,
            )
            return result

    results = await asyncio.gather(*(gated(path) for path in selected_files))
    if client is not None:
        await client.aclose()
    usage_total: dict = {}
    for result in results:
        common.merge_usage(usage_total, result.pop("usage", None))
    manifest["status"] = "finished"
    manifest["finished_at"] = common.utc_now()
    manifest["counts"] = counts
    manifest["api_usage_total"] = usage_total
    manifest["items"] = results
    common.write_json_atomic(manifest_path, manifest)
    print(
        json.dumps(
            {"run_id": args.run_id, "counts": counts, "api_usage_total": usage_total},
            indent=2,
        )
    )


def main() -> None:
    args = argparser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
