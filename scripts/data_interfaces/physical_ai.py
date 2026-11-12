# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical AI reader backed by NVIDIA's official ``physical_ai_av`` SDK.

The interface keeps feature chunks packed in the Hugging Face cache. It
enumerates clips from a small parquet index (``clip_id`` plus an optional
``split`` column) and materializes JPEG frames lazily from the front-wide
camera after TTCov's skip and lock checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from data_interfaces import CaptionItem


def resize_frame(frame: np.ndarray, max_image_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = max(height, width)
    if side <= max_image_side:
        return frame
    scale = max_image_side / side
    return cv2.resize(
        frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA
    )


def effective_sampling_window(
    start_sec: float,
    duration_sec: float,
    num_frames: int,
    original_duration: float | None,
) -> tuple[float, float, int]:
    """Clamp the requested window to the video length, keeping the sampling rate."""
    requested_end_sec = start_sec + duration_sec
    if original_duration is None or original_duration >= requested_end_sec:
        return start_sec, duration_sec, num_frames

    effective_start_sec = min(start_sec, max(original_duration - duration_sec, 0.0))
    effective_duration_sec = max(original_duration - effective_start_sec, 0.0)
    requested_hz = num_frames / duration_sec if duration_sec > 0 else 1.0
    effective_num_frames = max(1, int(effective_duration_sec * requested_hz))
    return effective_start_sec, effective_duration_sec, effective_num_frames


class PhysicalAIInterface:
    """Parquet-indexed Physical AI reader using cached Hugging Face chunks."""

    cli_help = "NVIDIA Physical AI AV dataset via the official SDK (HF chunk cache)"

    def __init__(
        self,
        index_path: Path,
        split: str | None,
        start_sec: float,
        duration_sec: float,
        num_frames: int,
        max_image_side: int,
        jpeg_quality: int,
        cache_dir: Path | None,
        revision: str | None,
        stream: bool,
        prefetch: bool,
        confirm_download_threshold_gb: float,
        download_workers: int,
    ) -> None:
        try:
            import physical_ai_av
        except ImportError as exc:
            raise RuntimeError(
                "The physical-ai data format requires the optional SDK dependency; "
                "run `uv sync --extra physical-ai`."
            ) from exc

        self.index_path = index_path
        self.split = split
        self.start_sec = start_sec
        self.duration_sec = duration_sec
        self.num_frames = num_frames
        self.max_image_side = max_image_side
        self.jpeg_quality = jpeg_quality
        self.stream = stream
        self.prefetch = prefetch
        self.download_workers = download_workers
        self.dataset = physical_ai_av.PhysicalAIAVDatasetInterface(
            revision=revision,
            cache_dir=cache_dir,
            confirm_download_threshold_gb=confirm_download_threshold_gb,
        )
        self.camera = self.dataset.features.CAMERA.CAMERA_FRONT_WIDE_120FOV

    @classmethod
    def add_cli_args(cls, parser) -> None:
        group = parser.add_argument_group("physical-ai options")
        group.add_argument(
            "--index-path",
            type=Path,
            required=True,
            help="parquet clip index with a 'clip_id' column "
            "(and an optional 'split' column)",
        )
        group.add_argument(
            "--split",
            default=None,
            help="optional filter on the index's 'split' column",
        )
        group.add_argument(
            "--num-frames",
            type=int,
            default=20,
            help="sampled frame count per clip (default: 20)",
        )
        group.add_argument(
            "--start-sec",
            type=float,
            default=5.0,
            help="sampling window start (default: 5.0)",
        )
        group.add_argument(
            "--duration-sec",
            type=float,
            default=10.0,
            help="sampling window length (default: 10.0)",
        )
        group.add_argument(
            "--max-image-side",
            type=int,
            default=768,
            help="resize frames so the longer side is at most "
            "this many pixels (default: 768)",
        )
        group.add_argument(
            "--jpeg-quality",
            type=int,
            default=85,
            help="JPEG encode quality (default: 85)",
        )
        group.add_argument(
            "--hf-cache-dir",
            type=Path,
            default=None,
            help="Hugging Face cache containing packed chunks",
        )
        group.add_argument(
            "--hf-revision", default=None, help="dataset tag or full commit hash"
        )
        group.add_argument(
            "--hf-stream",
            action="store_true",
            help="stream uncached chunks (smoke tests only)",
        )
        group.add_argument(
            "--hf-prefetch",
            action="store_true",
            help="download required chunks before captioning",
        )
        group.add_argument(
            "--hf-confirm-download-threshold-gb",
            type=float,
            default=10.0,
            help="prompt before downloading more than this many "
            "uncached GB in one prefetch (default: 10)",
        )
        group.add_argument(
            "--hf-download-workers",
            type=int,
            default=8,
            help="concurrent chunk download workers (default: 8)",
        )

    @classmethod
    def from_args(cls, args) -> "PhysicalAIInterface":
        return cls(
            index_path=args.index_path,
            split=args.split,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
            num_frames=args.num_frames,
            max_image_side=args.max_image_side,
            jpeg_quality=args.jpeg_quality,
            cache_dir=args.hf_cache_dir,
            revision=args.hf_revision,
            stream=args.hf_stream,
            prefetch=args.hf_prefetch,
            confirm_download_threshold_gb=args.hf_confirm_download_threshold_gb,
            download_workers=args.hf_download_workers,
        )

    def iter_items(self) -> Iterator[CaptionItem]:
        table = pq.read_table(self.index_path)
        if "clip_id" not in table.column_names:
            raise ValueError(
                f"Clip index {self.index_path} is missing column 'clip_id'"
            )
        if self.split is not None:
            if "split" not in table.column_names:
                raise ValueError(
                    f"--split given but {self.index_path} has no 'split' column"
                )
            table = table.filter(pc.equal(table["split"], self.split))

        for row in table.to_pylist():
            yield CaptionItem(
                item_id=row["clip_id"],
                frames=[],
                meta={"split": row.get("split")},
            )

    def prepare(self, items: list[CaptionItem]) -> None:
        """Download the unique front-wide chunks needed by ``items``.

        No-op unless ``--hf-prefetch`` was given (reusing a warm cache).
        """
        if self.prefetch and items:
            self.dataset.download_clip_features(
                [item.item_id for item in items],
                features=[self.camera],
                max_workers=self.download_workers,
            )

    def load_frames(self, item: CaptionItem) -> list[bytes]:
        reader = self.dataset.get_clip_feature(
            item.item_id,
            feature=self.camera,
            maybe_stream=self.stream,
        )
        try:
            timestamps = np.asarray(reader.timestamps, dtype=np.int64)
            if timestamps.size == 0:
                raise RuntimeError(f"Clip {item.item_id} has no frame timestamps")

            first_timestamp = int(timestamps[0])
            original_duration = max(
                0.0,
                (int(timestamps[-1]) - first_timestamp) / 1_000_000,
            )
            start_sec, window_sec, frame_count = effective_sampling_window(
                self.start_sec,
                self.duration_sec,
                self.num_frames,
                original_duration,
            )
            sample_seconds = np.linspace(
                start_sec,
                start_sec + window_sec,
                frame_count,
                endpoint=False,
                dtype=np.float64,
            )
            sample_seconds += window_sec / (2 * frame_count)
            requested_timestamps = first_timestamp + np.rint(
                sample_seconds * 1_000_000
            ).astype(np.int64)
            requested_timestamps = np.clip(
                requested_timestamps,
                timestamps[0],
                timestamps[-1],
            )

            frames_rgb, _ = reader.decode_images_from_timestamps(requested_timestamps)
            images: list[bytes] = []
            for frame_rgb in frames_rgb:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_bgr = resize_frame(frame_bgr, self.max_image_side)
                ok, encoded = cv2.imencode(
                    ".jpg",
                    frame_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError(
                        f"Could not JPEG-encode a frame for {item.item_id}"
                    )
                images.append(encoded.tobytes())
            return images
        finally:
            reader.close()
