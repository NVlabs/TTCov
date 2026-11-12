# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caption video scenes with a vision LLM over an OpenAI-compatible API.

The data layout is abstracted behind scripts/data_interfaces/; each format is
a subcommand that owns its own options (`run_caption.py <format> --help`),
so adding or changing an interface never touches this script:

    python scripts/run_caption.py physical-ai --index-path clips.parquet ...

Captions are written to <output-dir>/captions/<item_id[:4]>/<item_id>.txt.
Runs are resumable: existing captions are skipped and in-flight items are
guarded by lock files under <output-dir>/locks/, so multiple shards or hosts
can share one output directory.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path

from openai import AsyncOpenAI

import common
from data_interfaces import CaptionItem, available_formats, get_interface


def parse_args() -> argparse.Namespace:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Run root; captions land in captions/<id[:4]>/<id>.txt",
    )
    common_parser.add_argument(
        "--prompt-file",
        type=Path,
        default=common.PROMPTS_DIR / "caption.txt",
        help="Prompt sent after the image frames (default: prompts/caption.txt)",
    )
    common_parser.add_argument("--max-tokens", type=int, default=4096)
    common_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omitted from the request when unset)",
    )
    common_parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max simultaneous API requests (default: 16)",
    )
    common_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="API retries per item, with exponential backoff (default: 3)",
    )
    common_parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=6 * 60 * 60,
        help="Reclaim lock files older than this (default: 6h)",
    )
    common.add_llm_args(common_parser)
    common.add_selection_args(common_parser)

    parser = argparse.ArgumentParser(
        description="Caption video scenes with a vision LLM (OpenAI-compatible API)."
    )
    subparsers = parser.add_subparsers(
        dest="data_format", required=True, metavar="<data-format>"
    )
    for name in available_formats():
        interface_cls = get_interface(name)
        sub = subparsers.add_parser(
            name,
            parents=[common_parser],
            help=interface_cls.cli_help,
            description=interface_cls.cli_help,
        )
        interface_cls.add_cli_args(sub)
    return parser.parse_args()


def caption_path(output_dir: Path, item_id: str) -> Path:
    return output_dir / "captions" / item_id[:4] / f"{item_id}.txt"


def lock_path(output_dir: Path, item_id: str) -> Path:
    return output_dir / "locks" / item_id[:4] / f"{item_id}.lock"


def build_image_parts(frames: list) -> list[dict]:
    """Turn Path or raw-JPEG-bytes frames into image_url data-URI content parts."""
    parts = []
    for frame in frames:
        if isinstance(frame, (bytes, bytearray)):
            data, mime = bytes(frame), "image/jpeg"
        else:
            path = Path(frame)
            data = path.read_bytes()
            ext = path.suffix.lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            mime = (
                f"image/{ext}"
                if ext in ("jpeg", "png", "webp", "gif")
                else "image/jpeg"
            )
        b64 = base64.b64encode(data).decode("ascii")
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
        )
    return parts


async def call_model(
    client: AsyncOpenAI, args: argparse.Namespace, content: list[dict]
) -> tuple[str, dict | None]:
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.temperature is not None:
        request["temperature"] = args.temperature
    if args.reasoning_effort is not None:
        request["reasoning_effort"] = args.reasoning_effort

    last_error = None
    for attempt in range(args.max_retries):
        try:
            response = await client.chat.completions.create(**request)
            caption = response.choices[0].message.content.strip()
            usage = response.usage.model_dump() if response.usage is not None else None
            return caption, usage
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"LLM request failed after {args.max_retries} attempts: {last_error}"
    )


async def process_item(
    item: CaptionItem,
    interface,
    args: argparse.Namespace,
    prompt: str,
    client: AsyncOpenAI,
) -> tuple[str, dict | None]:
    out_path = caption_path(args.output_dir, item.item_id)
    if out_path.exists():
        return "skipped_existing", None
    if not common.acquire_lock(
        lock_path(args.output_dir, item.item_id), args.stale_lock_seconds
    ):
        return "skipped_locked", None

    try:
        frames = await asyncio.to_thread(interface.load_frames, item)
        content = await asyncio.to_thread(build_image_parts, frames)
        content.append({"type": "text", "text": prompt})
        caption, usage = await call_model(client, args, content)
        common.write_text_atomic(out_path, caption)
        return "done", usage
    finally:
        lock_path(args.output_dir, item.item_id).unlink(missing_ok=True)


async def main_async(args: argparse.Namespace) -> None:
    args.reasoning_effort = common.effective_reasoning_effort(args.reasoning_effort)
    prompt = args.prompt_file.read_text()
    interface = get_interface(args.data_format).from_args(args)
    items = common.select_by_common_args(
        list(interface.iter_items()),
        common.load_item_ids(args.item_ids),
        args.offset,
        args.limit,
        args.num_shards,
        args.shard_index,
        key=lambda item: item.item_id,
    )
    prepare = getattr(interface, "prepare", None)
    if prepare is not None:
        await asyncio.to_thread(prepare, items)

    client = AsyncOpenAI(
        base_url=args.base_url, api_key=common.load_api_key(args.api_key_file)
    )
    print(
        json.dumps(
            {
                "data_format": args.data_format,
                "model": args.model,
                "num_items": len(items),
                "output_dir": str(args.output_dir),
                "prompt_file": str(args.prompt_file),
                "prompt_sha256": common.sha256_text(prompt),
            },
            indent=2,
        ),
        flush=True,
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    counts: dict[str, int] = {}
    usage_total: dict = {}

    async def gated(item: CaptionItem) -> None:
        async with semaphore:
            line = {"item_id": item.item_id}
            try:
                status, usage = await process_item(
                    item, interface, args, prompt, client
                )
            except Exception as exc:
                status, usage = "error", None
                line["error"] = repr(exc)
            if usage is not None:
                common.merge_usage(usage_total, usage)
            counts[status] = counts.get(status, 0) + 1
            line["status"] = status
            print(json.dumps(line), flush=True)

    await asyncio.gather(*(gated(item) for item in items))
    print(json.dumps({"counts": counts, "api_usage_total": usage_total}, indent=2))


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
