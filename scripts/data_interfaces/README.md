# Data interfaces

Captioning (stage 1, `scripts/run_caption.py`) is the only data-format-specific
stage of the TTCov pipeline. Each dataset layout is handled by an interface in
this package, selected as a subcommand (`run_caption.py <format> ...`; run
`run_caption.py <format> --help` for that format's options). Every format
writes the same output layout (`captions/<id[:4]>/<id>.txt`), so everything
downstream is format-independent.

## Provided formats

### navsim

NAVSIM/OpenScenes-style logs: scene-metadata pickles plus pre-extracted camera
frames under `--data-root`. Frames are chunked into caption items by
`--num-frames` (chunk size) and `--interval` (stride); the item id is the
first frame token of each chunk.

```bash
python scripts/run_caption.py navsim --data-root /data/navsim \
    --data-type trainval -o output/caption_train --num-frames 20 --interval 7
```

`--data-type` selects `trainval` or `test`.

### navhard

NAVHARD two-stage data under `--data-root`. Run both stages into the same
output directory: stage1 chunks logs like navsim; stage2 captions the four
synthetic-scene frames per scene token (from the scene-attributes CSV).

```bash
python scripts/run_caption.py navhard --data-root /data/navsim \
    --data-type stage1 -o output/caption_test
python scripts/run_caption.py navhard --data-root /data/navsim \
    --data-type stage2 -o output/caption_test
```

### physical-ai

NVIDIA's Physical AI AV dataset, read through the official
[`physical_ai_av`](https://pypi.org/project/physical-ai-av/) SDK.

Requirements: a Hugging Face login (`hf auth login`) and the dataset's
gated-access agreement accepted. Install the SDK extra:

```bash
uv sync --extra physical-ai
```

The input is a small parquet clip index with a `clip_id` column (and an
optional `split` column, filterable with `--split`). Build one with the SDK:

```python
import physical_ai_av

dataset = physical_ai_av.PhysicalAIAVDatasetInterface(
    revision="<tag-or-full-commit>", cache_dir="/data/hf-cache",
)
camera = dataset.features.CAMERA.CAMERA_FRONT_WIDE_120FOV
eligible = dataset.clip_index["clip_is_valid"] & dataset.feature_presence[camera]
selected = dataset.clip_index.loc[eligible].query("split == 'train'").head(100)
selected[["split"]].reset_index().to_parquet("selected_clip_ids.parquet", index=False)
```

Then caption. Prefetch on the first run; omit `--hf-prefetch` when reusing the
same cache:

```bash
python scripts/run_caption.py physical-ai \
    --index-path /data/selected_clip_ids.parquet -o output/caption_clips \
    --hf-cache-dir /data/hf-cache --hf-revision <tag-or-full-commit> \
    --hf-prefetch --num-frames 40 --start-sec 0 --duration-sec 20 \
    --max-image-side 1280
```

The SDK maps the selected clip ids to their containing feature chunks and
downloads only those (front-wide camera only).

Format-specific frame flags: `--num-frames`, `--start-sec`, `--duration-sec`,
`--max-image-side`, `--jpeg-quality` (the sampling window is clamped to each
clip's length, preserving the sampling rate).

## From a selection to training configs (the id-to-config contract)

The optimizer (main README step 10) emits an **opaque list of selected item
ids** (`<optimizer>_selection.json`) — the same ids the captioning interface
minted. It never interprets an id, so the output is path-free and portable.
Converting those ids into a trainer's configs is each data interface's job,
since only it knows what its ids mean. A future dataset supports this by adding
its own id-to-config step next to its captioning interface; nothing in the
optimizer changes.

### navsim

Two navsim helpers turn a selection into navsim trainer configs (standalone CLI
scripts, navsim-format-specific):

- **`navsim_build_token_map.py`** — builds the map
  `item_id -> {"log": <log_name>, "tokens": [token, ...]}` by driving the same
  `NavsimInterface` used for captioning, so the ids match the pipeline's exactly
  (each id is a chunk's first-frame token). Pass the same `--data-root`,
  `--data-type`, `--num-frames`, and `--interval` you captioned with. Optionally
  add `--valid-tokens` (a JSON/pickle of valid navsim tokens, e.g. OpenScenes'
  `openscene_log_valid_tokens.pkl`) to keep only trainable frames. Output is a
  portable JSON map.

- **`navsim_generate_train_configs.py`** — reads the selection id list and the
  token map, groups the selected clips' tokens by source log (deduplicating
  overlapping frames), and writes `scene_filter/<batch>/<exp>.yaml` and
  `train_test_split/<batch>/<exp>.yaml`. `<exp>` is the selection file's parent
  directory name.

```bash
python scripts/data_interfaces/navsim_build_token_map.py \
    --data-root /data/navsim --data-type trainval --num-frames 20 --interval 7 \
    -o output/navsim_configs/token_map.json

python scripts/data_interfaces/navsim_generate_train_configs.py \
    -i output/selection/<exp-dir>/greedy_selection.json \
    -o output/navsim_configs --exp-batch-name my_batch \
    --token-map output/navsim_configs/token_map.json
```

## Adding a custom data interface

To support a new dataset, add a module in this package with a class
implementing this contract. Each format is a `run_caption.py` subcommand that
owns its CLI options — the main script never needs to change:

```python
class MyDatasetInterface:
    cli_help = "one-line description shown in run_caption.py's format listing"

    @classmethod
    def add_cli_args(cls, parser) -> None:
        # register this format's own options on its subcommand parser (the
        # generic run_caption.py options are already attached); flag names may
        # overlap with other formats' (e.g. --num-frames), never with the
        # generic options
        group = parser.add_argument_group("my-dataset options")
        group.add_argument("--my-index", type=Path, required=True)
        ...

    @classmethod
    def from_args(cls, args) -> "MyDatasetInterface":
        # construct from the parsed namespace (generic + this format's flags)
        ...

    def iter_items(self) -> Iterator[CaptionItem]:
        # yield CaptionItem(item_id, frames, meta) CHEAPLY — no video decoding
        # or image IO here. item_id must be unique and filesystem-safe: it
        # becomes captions/<id[:4]>/<id>.txt. Selection, sharding, and
        # skip-existing checks iterate these before any heavy work.
        ...

    def load_frames(self, item: CaptionItem) -> list:
        # materialize the frames for one item: a list of image file Paths or
        # raw JPEG bytes. Called in a worker thread only after the item has
        # passed skip-existing checks and acquired its lock, so already-done
        # items never pay decode cost. Interfaces that decode lazily yield
        # items with frames=[] from iter_items and fill them here.
        ...
```

Optionally implement `prepare(items)` to pre-stage data for the selected
items; it is called once after item selection whenever it is defined (the
physical-ai interface downloads missing chunks there when `--hf-prefetch` is
given, and no-ops otherwise).

Register the class in `_REGISTRY` in `__init__.py`; it then appears as a
`run_caption.py <name>` subcommand. See `navsim.py` for a frames-on-disk
example (frames as `Path`s) and `physical_ai.py` for a lazy-decoding example
(frames as JPEG `bytes`).
