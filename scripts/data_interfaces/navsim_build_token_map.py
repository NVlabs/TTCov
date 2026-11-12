# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the navsim item-id -> {log, tokens} map for training-config generation.

The optimizer (stage 10) emits an opaque list of selected item ids. For navsim,
each id is a caption chunk's first-frame token. To turn a selection into navsim
trainer configs we need, per id, the source log and the chunk's frame tokens.

This script reconstructs that map by driving the SAME ``NavsimInterface`` used
for captioning, so the ids here match exactly the ids the pipeline selects.
Pass the same ``--num-frames`` / ``--interval`` you captioned with, otherwise
the chunking (and thus the ids) will not line up.

Output is a JSON map ``item_id -> {"log": <log_name>, "tokens": [token, ...]}``.
With ``--valid-tokens`` the token list is restricted to a navsim/OpenScenes
valid-token set (e.g. openscene_log_valid_tokens.pkl) so only trainable frames
are kept; without it, every frame of the chunk is retained.

Usage:
  python scripts/data_interfaces/navsim_build_token_map.py \
      --data-root /data/navsim --data-type trainval \
      --num-frames 20 --interval 7 \
      -o output/navsim_configs/token_map.json
"""

import argparse
import json
import sys
from pathlib import Path

# Reuse the captioning interface so ids are guaranteed identical. The script
# lives in the data_interfaces package; add scripts/ to the path so the package
# imports the same way it does under run_caption.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_interfaces.navsim import NavsimInterface  # noqa: E402


def load_valid_tokens(path):
    """Load a valid-token set from JSON or pickle.

    Accepts a flat list of tokens or a dict whose values are token lists
    (e.g. openscene's log_name -> [token, ...]); values are flattened.
    """
    p = Path(path)
    try:
        if p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
        else:
            import dill

            with open(p, "rb") as f:
                data = dill.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: --valid-tokens file not found: {p}")
    except Exception as exc:
        sys.exit(f"Error: could not read --valid-tokens {p}: {exc}")
    if isinstance(data, dict):
        return {token for tokens in data.values() for token in tokens}
    return set(data)


def main(args):
    interface = NavsimInterface(
        Path(args.data_root), args.data_type, args.num_frames, args.interval
    )
    valid_tokens = load_valid_tokens(args.valid_tokens) if args.valid_tokens else None

    token_map = {}
    n_items = n_tokens = n_empty = 0
    for item in interface.iter_items():
        tokens = item.meta["tokens"]
        if valid_tokens is not None:
            tokens = [token for token in tokens if token in valid_tokens]
        token_map[item.item_id] = {"log": item.meta["log"], "tokens": tokens}
        n_items += 1
        n_tokens += len(tokens)
        n_empty += not tokens

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(token_map, f)

    logs = {entry["log"] for entry in token_map.values()}
    print(
        f"Built token map: {len(logs):,} logs, {n_items:,} items, "
        f"{n_tokens:,} tokens -> {output_file}"
    )
    if valid_tokens is not None and n_empty:
        print(
            f"Warning: {n_empty:,} item(s) mapped to zero valid tokens — check "
            f"that --valid-tokens matches this dataset."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the navsim item-id -> {log, tokens} map"
    )
    parser.add_argument(
        "--data-root", type=str, required=True, help="navsim dataset root directory"
    )
    parser.add_argument(
        "--data-type",
        choices=("trainval", "test"),
        required=True,
        help="navsim split to enumerate",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="frames per chunk — MUST match the captioning run (default: 20)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=7,
        help="chunk stride — MUST match the captioning run (default: 7)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Output JSON path for the token map",
    )
    parser.add_argument(
        "--valid-tokens",
        type=str,
        default=None,
        help="Optional JSON/pickle of valid navsim tokens (flat list or "
        "dict-of-lists, e.g. openscene_log_valid_tokens.pkl) to restrict "
        "tokens to trainable frames",
    )
    args = parser.parse_args()
    main(args)
