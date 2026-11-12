# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate navsim training configs from a data-selection result.

Turns the optimizer's selection (stage 10) into the two YAML files the navsim
training framework expects:

  <output>/scene_filter/<batch>/<exp>.yaml       (a SceneFilter listing the
                                                   selected log_names + tokens)
  <output>/train_test_split/<batch>/<exp>.yaml   (a split that points at the
                                                   scene_filter above)

The optimizer writes an opaque list of selected item ids (JSON). For navsim,
each id is a caption chunk's first-frame token. This script looks each id up in
the navsim token map (navsim_build_token_map.py output:
item_id -> {"log": <log_name>, "tokens": [token, ...]}), groups the selected
clips' frame tokens by source log, and writes the configs. <exp> is taken from
the selection file's parent directory name, matching the layout optimizers.py
writes (exp<NN>_.../greedy_selection.json).

Usage:
  python scripts/data_interfaces/navsim_generate_train_configs.py \
      -i output/selection/exp01_proportional_p1.0_b14756_total_14756/greedy_selection.json \
      -o output/navsim_configs \
      --exp-batch-name my_batch \
      --token-map output/navsim_configs/token_map.json
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import yaml

# navsim SceneFilter header — these values are part of the training recipe and
# are written verbatim ahead of the selected log_names / tokens.
SCENE_FILTER_HEADER = {
    "_target_": "navsim.common.dataclasses.SceneFilter",
    "_convert_": "all",
    "num_history_frames": 4,
    "num_future_frames": 10,
    "frame_interval": 1,
    "has_route": True,
    "max_scenes": None,
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {path}")
    except json.JSONDecodeError as exc:
        sys.exit(f"Error: could not parse JSON {path}: {exc}")


def main(args):
    selection_path = Path(args.input_file)
    output_folder = Path(args.output_folder)
    exp_name = selection_path.parent.name

    selected_ids = load_json(selection_path)
    token_map = load_json(args.token_map)

    # Group each selected id's frame tokens under its source log, preserving
    # first-seen order. Deduplicate tokens (overlapping chunks share frames).
    per_log_tokens = OrderedDict()
    seen_tokens = set()
    missing = 0
    for item_id in selected_ids:
        entry = token_map.get(item_id)
        if entry is None:
            missing += 1
            continue
        bucket = per_log_tokens.setdefault(entry["log"], [])
        for token in entry["tokens"]:
            if token not in seen_tokens:
                seen_tokens.add(token)
                bucket.append(token)
    if missing:
        sys.exit(
            f"Error: {missing} selected id(s) missing from token map "
            f"{args.token_map}; the map may not cover this selection (check "
            f"--num-frames/--interval matched captioning)."
        )

    log_names = list(per_log_tokens.keys())
    tokens = [token for log in log_names for token in per_log_tokens[log]]

    # 1) scene_filter YAML: header, then the selected log_names and tokens.
    scene_filter_dir = output_folder / "scene_filter" / args.exp_batch_name
    scene_filter_dir.mkdir(parents=True, exist_ok=True)
    scene_filter_path = scene_filter_dir / f"{exp_name}.yaml"
    with open(scene_filter_path, "w") as f:
        yaml.safe_dump(
            SCENE_FILTER_HEADER, f, sort_keys=False, default_flow_style=False
        )
        f.write("\nlog_names:\n")
        for log_name in log_names:
            f.write(f"  - '{log_name}'\n")
        f.write("\ntokens:\n")
        for token in tokens:
            f.write(f"  - '{token}'\n")

    # 2) train_test_split YAML: points at the scene_filter written above.
    split_dir = output_folder / "train_test_split" / args.exp_batch_name
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{exp_name}.yaml"
    with open(split_path, "w") as f:
        f.write("defaults:\n")
        f.write(f"  - scene_filter: {args.exp_batch_name}/{exp_name}\n")
        f.write("data_split: trainval\n")

    print(
        f"Wrote {len(log_names):,} log_names / {len(tokens):,} tokens\n"
        f"  scene_filter      -> {scene_filter_path}\n"
        f"  train_test_split  -> {split_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate navsim training configs from a selection id list"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        required=True,
        help="Selection JSON from optimizers.py (e.g. greedy_selection.json)",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        required=True,
        help="Root folder to write scene_filter/ and train_test_split/ under",
    )
    parser.add_argument(
        "--exp-batch-name",
        type=str,
        required=True,
        help="Batch name; used as the sub-folder grouping generated configs",
    )
    parser.add_argument(
        "--token-map",
        type=str,
        required=True,
        help="JSON map item_id -> {log, tokens} from navsim_build_token_map.py",
    )
    args = parser.parse_args()
    main(args)
