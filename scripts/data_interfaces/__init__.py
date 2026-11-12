# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-format interfaces for the unified captioning stage.

Each interface enumerates caption work items for one dataset layout and knows
how to materialize the image frames for an item. Interfaces own their CLI:
each format is a run_caption.py subcommand, so adding or changing an
interface never touches the main script. The contract (see this package's
README.md for the full guide):

- ``cli_help`` (class attribute) is the one-line description shown in
  run_caption.py's subcommand listing.
- ``add_cli_args(parser)`` (classmethod) registers the format's own argparse
  options on its subcommand parser (generic options are already attached).
- ``from_args(args)`` (classmethod) constructs the interface from the parsed
  namespace.
- ``iter_items()`` yields :class:`CaptionItem` objects cheaply (no video
  decoding), so selection/sharding and skip-existing checks stay fast.
  ``item_id`` must be unique and filesystem-safe (it becomes the caption
  output path ``captions/<id[:4]>/<id>.txt``).
- ``load_frames(item)`` returns the final frame list for an item: image file
  ``Path`` objects (navsim/navhard) or raw JPEG ``bytes`` (physical-ai). It
  is called after skip/lock checks, so finished items never pay decode cost.
- ``prepare(items)`` (optional) pre-stages data for the selected items; it is
  called once after item selection when defined.

Register new interfaces under a subcommand name in ``_REGISTRY``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

# subcommand name -> (module, class). Modules are imported lazily so formats
# with heavy or optional dependencies don't slow down or break the others.
_REGISTRY = {
    "navsim": ("data_interfaces.navsim", "NavsimInterface"),
    "navhard": ("data_interfaces.navsim", "NavhardInterface"),
    "physical-ai": ("data_interfaces.physical_ai", "PhysicalAIInterface"),
}


@dataclass
class CaptionItem:
    """One caption request: an id, its image frames, and reader metadata.

    ``frames`` entries are image file paths (``Path``) or raw JPEG ``bytes``.
    Interfaces that decode frames lazily yield items with ``frames=[]`` and
    fill them in ``load_frames``.
    """

    item_id: str
    frames: list
    meta: dict = field(default_factory=dict)


def available_formats() -> list[str]:
    """Return the registered data-format subcommand names."""
    return list(_REGISTRY)


def get_interface(name: str):
    """Return the interface class registered under ``name``."""
    try:
        module_name, class_name = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown data format: {name!r}") from None
    return getattr(importlib.import_module(module_name), class_name)
