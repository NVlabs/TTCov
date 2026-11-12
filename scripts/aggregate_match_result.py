# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Aggregate per-item match results into one test-distribution JSON.

Sums the ap_frequency dicts across all match JSONs produced by
match_data_to_k_atlas_faiss.py.

Usage:
  python scripts/aggregate_match_result.py \
      -i output/match_test \
      -o output/atlas/test_frequency.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import common


def main(args):
    input_folder = Path(args.input_folder)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frequency_keys = ("ap_frequency",)
    result = {key: defaultdict(int) for key in frequency_keys}
    json_files = common.discover_json_items(input_folder)
    if not json_files:
        sys.exit(f"Error: no match JSON files found under {input_folder}")
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            sys.exit(f"Error: could not parse {json_file}: {exc}")
        for frequency_key in frequency_keys:
            for key, value in data.get(frequency_key, {}).items():
                result[frequency_key][key] += value
    result = {key: dict(value) for key, value in result.items()}
    common.write_json_atomic(output_file, result)
    counts = ", ".join(f"{len(result[key]):,} {key}" for key in frequency_keys)
    print(f"Aggregated {len(json_files):,} match files -> {output_file} ({counts})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-item match results into one distribution JSON"
    )
    parser.add_argument(
        "-i",
        "--input-folder",
        type=str,
        required=True,
        help="Folder of per-item match JSONs (stage-8 output)",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        required=True,
        help="Aggregated frequency JSON to write",
    )
    args = parser.parse_args()
    main(args)
