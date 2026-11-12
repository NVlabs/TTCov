# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Caption work-item readers for the NAVSIM and NAVHARD dataset layouts.

Both layouts store scene metadata as ``.pkl`` files plus pre-extracted camera
images on disk, so ``load_frames`` simply returns the image paths collected at
enumeration time.

- navsim (``trainval``/``test``): ``<data_root>/navsim_logs/<data_type>/*.pkl``
  metadata and ``<data_root>/sensor_blobs/<data_type>/`` images. Each log is
  split into overlapping chunks of ``num_frames`` front-camera frames taken
  every ``interval`` frames; each chunk is one caption item, identified by the
  token of its first frame.
- navhard stage1: same chunking over
  ``<data_root>/navhard_two_stage/openscene_meta_datas/*.pkl``.
- navhard stage2: one item per synthetic scene in
  ``<data_root>/navhard_two_stage/synthetic_scene_pickles/*.pkl``, with the
  four front-camera frame paths looked up in
  ``synthetic_scenes_attributes.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import dill

from data_interfaces import CaptionItem


def _iter_log_chunks(
    meta_dir: Path,
    image_dir: Path,
    num_frames: int,
    interval: int,
) -> Iterator[CaptionItem]:
    """Yield one item per chunk of ``num_frames`` frames, strided by ``interval``.

    Trailing chunks shorter than ``num_frames`` (down to one frame) are kept,
    so every frame of a log is covered by at least one item.
    """
    if not meta_dir.is_dir():
        raise FileNotFoundError(
            f"Log metadata directory not found: {meta_dir} — check --data-root/--data-type"
        )
    for meta_file in sorted(meta_dir.glob("*.pkl")):
        with open(meta_file, "rb") as f:
            frame_infos = dill.load(f)
        image_list = [
            frame_info["cams"]["CAM_F0"]["data_path"] for frame_info in frame_infos
        ]
        token_list = [frame_info["token"] for frame_info in frame_infos]
        if not token_list:
            print(f"  Warning: skipping empty log metadata file {meta_file}")
            continue

        for chunk_index, start in enumerate(range(0, len(image_list), interval)):
            tokens = token_list[start : start + num_frames]
            frames = [
                image_dir / rel_path
                for rel_path in image_list[start : start + num_frames]
            ]
            yield CaptionItem(
                item_id=tokens[0],
                frames=frames,
                meta={
                    "log": meta_file.stem,
                    "chunk_index": chunk_index,
                    "tokens": tokens,
                },
            )


def _add_log_chunk_cli_args(
    parser, data_format: str, data_type_choices: tuple[str, ...]
) -> None:
    """CLI options shared by the navsim and navhard log-chunking readers."""
    group = parser.add_argument_group(f"{data_format} options")
    group.add_argument(
        "--data-root", type=Path, required=True, help="dataset root directory"
    )
    group.add_argument("--data-type", required=True, choices=data_type_choices)
    group.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="frames per caption item (chunk size; default: 20)",
    )
    group.add_argument(
        "--interval",
        type=int,
        default=7,
        help="frame stride between chunk starts (default: 7)",
    )


class NavsimInterface:
    """NAVSIM trainval/test reader."""

    cli_help = "NAVSIM/OpenScenes-style logs with pre-extracted camera frames"

    def __init__(self, data_root: Path, data_type: str, num_frames: int, interval: int):
        if data_type not in ("trainval", "test"):
            raise ValueError(
                "--data-type must be 'trainval' or 'test' for the navsim format."
            )
        self.meta_dir = data_root / "navsim_logs" / data_type
        self.image_dir = data_root / "sensor_blobs" / data_type
        self.num_frames = num_frames
        self.interval = interval

    @classmethod
    def add_cli_args(cls, parser) -> None:
        _add_log_chunk_cli_args(parser, "navsim", ("trainval", "test"))

    @classmethod
    def from_args(cls, args) -> "NavsimInterface":
        return cls(args.data_root, args.data_type, args.num_frames, args.interval)

    def iter_items(self) -> Iterator[CaptionItem]:
        yield from _iter_log_chunks(
            self.meta_dir, self.image_dir, self.num_frames, self.interval
        )

    def load_frames(self, item: CaptionItem) -> list:
        return item.frames


class NavhardInterface:
    """NAVHARD two-stage reader (stage1: real logs, stage2: synthetic scenes)."""

    cli_help = "NAVHARD two-stage data (stage1 real logs, stage2 synthetic scenes)"

    def __init__(self, data_root: Path, data_type: str, num_frames: int, interval: int):
        if data_type not in ("stage1", "stage2"):
            raise ValueError(
                "--data-type must be 'stage1' or 'stage2' for the navhard format."
            )
        root = data_root / "navhard_two_stage"
        self.data_type = data_type
        self.image_dir = root / "sensor_blobs"
        if data_type == "stage1":
            self.meta_dir = root / "openscene_meta_datas"
            self.attributes_csv = None
        else:
            self.meta_dir = root / "synthetic_scene_pickles"
            self.attributes_csv = root / "synthetic_scenes_attributes.csv"
        self.num_frames = num_frames
        self.interval = interval

    @classmethod
    def add_cli_args(cls, parser) -> None:
        _add_log_chunk_cli_args(parser, "navhard", ("stage1", "stage2"))

    @classmethod
    def from_args(cls, args) -> "NavhardInterface":
        return cls(args.data_root, args.data_type, args.num_frames, args.interval)

    def _load_stage2_frame_map(self) -> dict[str, list[str]]:
        columns = [f"frame_{i}_cam_f0_path" for i in range(4)]
        with open(self.attributes_csv, newline="") as f:
            reader = csv.DictReader(f)
            return {
                row["synthetic_scene_token"]: [row[column] for column in columns]
                for row in reader
            }

    def iter_items(self) -> Iterator[CaptionItem]:
        if self.data_type == "stage1":
            yield from _iter_log_chunks(
                self.meta_dir, self.image_dir, self.num_frames, self.interval
            )
            return

        token_to_images = self._load_stage2_frame_map()
        for pkl_file in sorted(self.meta_dir.glob("*.pkl")):
            token = pkl_file.stem
            if token not in token_to_images:
                raise KeyError(
                    f"Synthetic scene token {token!r} not found in {self.attributes_csv}"
                )
            frames = [self.image_dir / rel_path for rel_path in token_to_images[token]]
            yield CaptionItem(item_id=token, frames=frames, meta={"stage": "stage2"})

    def load_frames(self, item: CaptionItem) -> list:
        return item.frames
