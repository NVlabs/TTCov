# TTCov

TTCov is a coverage-driven pipeline for **selecting training data** so that
the selected subset matches the concept distribution of a target (test)
domain. Each item is described in natural language and reduced to **atomic
propositions (APs)** — `[subject, predicate, object]` triplets such as
`["ego vehicle", "yields to", "pedestrian"]`. APs observed in the target data
form a deduplicated **proposition atlas**; candidate training items are
matched against it, and an optimizer picks a budget-constrained subset that
minimizes the distribution gap to the target.

The method is domain-agnostic — anything that can be captioned into natural
language can be selected this way. Domain knowledge lives in the data
interfaces and the text prompt files in `prompts/`, plus the LLM-judge
prompts and relation filter inside `clean_atlas.py` (stage 7). The bundled
ones target driving video, the domain of our experiments (NAVSIM, NAVHARD,
and NVIDIA's Physical AI AV dataset); adapting to another domain means
providing a small captioning interface (see
[`scripts/data_interfaces/README.md`](scripts/data_interfaces/README.md)),
your own `prompts/` files (stages 1–4 take them as flags), and a pass over
stage 7's in-code prompts.

## Pipeline overview

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 1 | Video captioning | `scripts/run_caption.py` | `captions/<id[:4]>/<id>.txt` per item |
| 2 | Text distillation | `scripts/text_distillation.py` | atomic sentences per item (JSON) |
| 3 | Proposition conversion | `scripts/generate_propositions.py` | `graph.json` (AP triplets) per item |
| 4 | Nuisance filtering | `scripts/clean_propositions.py` | denoised per-item graphs |
| 5 | Atlas expansion | `scripts/combine_propositions.py` | raw atlas (union of graphs) |
| 6 | AP embedding cache | `scripts/ap_embeddings.py` | embedding pickle |
| 7 | Atlas unification | `scripts/clean_atlas.py` | cleaned atlas |
| 8 | Matching | `scripts/match_data_to_k_atlas_faiss.py` | per-item atlas match frequencies |
| 9 | Target distribution | `scripts/aggregate_match_result.py` | aggregated AP frequency |
| 10 | Data selection | `scripts/optimizers.py` | selected training subset |

Stages 2–4 run on the target (test) data and on the candidate training pool;
stages 5–7 build the atlas from the target data;
stages 8–10 run on both to produce the selection.

## Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- An **OpenAI-compatible chat-completions endpoint** for the LLM stages.
  There is no built-in endpoint or model default: every LLM stage requires
  `--base-url` and `--model` (any OpenAI-compatible endpoint and a model it
  serves), or the equivalent `TTCOV_API_BASE` / `TTCOV_MODEL` environment
  variables. Provide an API key via `--api-key-file <path>` (or the
  `TTCOV_API_KEY` environment variable). Our experiments used Gemini 2.5 Pro.
  All LLM stages default to `--reasoning-effort low`. Thinking models such as
  Gemini 2.5 otherwise reason with an unbounded budget billed at the output
  rate; on a 100-clip end-to-end benchmark that budget was ~81% of all
  generated tokens, and capping it cut total API cost ~69% with output quality
  indistinguishable from an unchanged rerun of the pipeline. Pass
  `--reasoning-effort default` to restore the provider's dynamic budget (e.g.
  for endpoints that reject the OpenAI `reasoning_effort` field), or `none` to
  disable thinking where the model allows it.
- A **GPU** for the embedding stages 6–7 (`Qwen/Qwen3-Embedding-8B`).
- Data behind a captioning interface — `navsim`, `navhard`, and `physical-ai`
  are provided; custom datasets plug in via
  [`scripts/data_interfaces/README.md`](scripts/data_interfaces/README.md).

## Installation

```bash
uv sync
uv sync --extra physical-ai # if using NVIDIA Physical AI AV input
```

On Linux, `torch` resolves from the PyTorch cu126 wheel index (works on
CUDA 12.x and newer drivers; see `[tool.uv.sources]` in `pyproject.toml`).
With plain pip, mirror it:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`.

Run all commands from the repository root with `uv run python ...` (or
activate the venv). Every LLM-backed script requires `--model` and
`--base-url` (plus `--api-key-file` or `TTCOV_API_KEY`). The example commands
below omit these for brevity — set them once so the examples run as written:

```bash
export TTCOV_API_BASE="https://<your-openai-compatible-endpoint>/v1"
export TTCOV_MODEL="<model-served-by-that-endpoint>"
export TTCOV_API_KEY="<your-api-key>"   # or pass --api-key-file <path>
```

## Running the pipeline

### 1. Video captioning

```bash
python scripts/run_caption.py navsim --data-root /data/navsim \
    --data-type trainval -o output/caption_train --num-frames 20 --interval 7
```

For instructions on data formats and using custom data inputs:
[`scripts/data_interfaces/README.md`](scripts/data_interfaces/README.md).
All formats write the same layout, so downstream stages are
format-independent. Runs are resumable (existing outputs skipped, in-flight
items lock-protected) and shardable with `--num-shards N --shard-index i`.

### 2. Text distillation

```bash
python scripts/text_distillation.py -i output/caption_test -o output/distill_test
```

### 3. Proposition conversion

```bash
python scripts/generate_propositions.py -i output/distill_test \
    -o output/propositions_test
```

The extraction prompts embed a domain-context fragment from `prompts/context.txt`
(`--context-file`) — the only domain-specific piece of this stage.

### 4. Nuisance filtering

```bash
python scripts/clean_propositions.py \
    --input-dir output/propositions_test --output-dir output/propositions_test_cleaned \
    --entity-cache output/entity_cleaning_cache.json
```

**Run stages 2–4 twice** — once for the target (test) captions, once for the
training pool — reusing the same `--entity-cache` so entities are
canonicalized identically on both sides.

### 5. Atlas expansion

```bash
python scripts/combine_propositions.py output/propositions_test_cleaned \
    -o output/atlas/raw_atlas.json
```

Inputs are any mix of graph JSON files and folders (searched recursively for
`graph.json`).

### 6. AP embedding cache

```bash
python scripts/ap_embeddings.py \
    output/atlas/raw_atlas.json \
    output/propositions_test_cleaned output/propositions_train_cleaned \
    --output-file output/atlas/ap_embeddings.pkl --batch-size 256 --checkpoint-every 20
```

Pass the atlas **and** the extraction output folders you plan to match
(folders are searched recursively for `graph.json`). Checkpoints as it
goes and resumes from a partial cache. Required before stage 8; stage 7's
default mode computes its own embeddings, but `clean_atlas.py --component`
can reuse this cache via `--embedding-cache`.

### 7. Atlas unification

Whole-AP cleaning — deduplicates entire AP triplets:

```bash
python scripts/clean_atlas.py \
    --input output/atlas/raw_atlas.json --output output/atlas/atlas.json \
    --log-dir output/atlas/clean_logs \
    --embedding-cache output/atlas/atlas_ap_embeddings.pkl \
    --batch-size 256 --checkpoint-every 20
```

Add `--component` for component-level cleaning — deduplicates
entities and predicates:

```bash
python scripts/clean_atlas.py --component \
    --input output/atlas/raw_atlas.json --output output/atlas/atlas.json \
    --log-dir output/atlas/clean_logs \
    --embedding-cache output/atlas/ap_embeddings.pkl
```

Merges are applied only with LLM-judge approval. Tune `--threshold` per
dataset; `--dry-run` previews candidate groups without an API key. Groups
whose verification fails stay unmerged and are retried on re-run (verified
results are cached). `--embed-only` runs just the embedding part (e.g. on a
GPU node) so the LLM part can run elsewhere.

### 8. Match data to the atlas

```bash
# training pool
python scripts/match_data_to_k_atlas_faiss.py output/atlas/atlas.json \
    output/propositions_train_cleaned -o output/match_train \
    --threshold 0.85 --embedding-cache-path output/atlas/ap_embeddings.pkl \
    --ap-match full

# target (test) data
python scripts/match_data_to_k_atlas_faiss.py output/atlas/atlas.json \
    output/propositions_test_cleaned -o output/match_test \
    --threshold 0.85 --embedding-cache-path output/atlas/ap_embeddings.pkl \
    --ap-match full
```

Per-item KGs are discovered as `graph.json` files; use `--item-glob '*.json'`
for other layouts.

### 9. Aggregate the target distribution

```bash
python scripts/aggregate_match_result.py -i output/match_test \
    -o output/atlas/test_frequency.json
```

### 10. Data selection

```bash
python scripts/optimizers.py --atlas output/atlas/test_frequency.json \
    --caption-dir output/match_train --budget 14756 \
    --optimizer greedy --metric kl_divergence --exp-idx 1 -o output/selection
```

`--budget` is the number of items to select; `--exp-idx` tags the output
directory. Optimizers: `greedy`, `lp_round`, `mip`, `lns`; metrics:
`kl_divergence`, `l1_delta`, `dot_product`, `dot_product_norm`. See
`python scripts/optimizers.py --help` for reweighting and refinement options.

The output is `<optimizer>_selection.json` — a portable JSON list of the
selected **item ids** (the ids minted by the captioning interface, e.g. a
navsim frame token or a physical-ai clip id). The optimizer is dataset-agnostic:
it never interprets an id, so selections carry no filesystem paths and stay
portable across machines.

## Generating training configs (per data interface)

Turning a selection into trainer-ready configs is **not part of the core 1–10
pipeline** — it's a downstream step owned by each **data interface**, since only
it knows what its item ids mean. The optimizer (stage 10) just emits item ids;
each dataset converts them into its own trainer's config format. Here we use
**navsim** as the example: two navsim helpers under `scripts/data_interfaces/`
convert the stage-10 id list into the scene-filter and split YAMLs the navsim
trainer consumes:

```bash
# build the id -> {log, tokens} map (same interface/flags used for captioning)
python scripts/data_interfaces/navsim_build_token_map.py \
    --data-root /data/navsim --data-type trainval --num-frames 20 --interval 7 \
    -o output/navsim_configs/token_map.json

# emit scene_filter/<batch>/<exp>.yaml and train_test_split/<batch>/<exp>.yaml
python scripts/data_interfaces/navsim_generate_train_configs.py \
    -i output/selection/exp01_proportional_p1.0_b14756_total_14756/greedy_selection.json \
    -o output/navsim_configs --exp-batch-name my_batch \
    --token-map output/navsim_configs/token_map.json
```

See `scripts/data_interfaces/README.md` for the id-to-config contract and each
dataset's specifics.

## Contributing

This project is currently not accepting contributions. See `CONTRIBUTING.md`.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

No third-party code, model weights, or datasets are bundled with this project;
dependencies are installed from their own upstream distributions. See
`THIRD_PARTY_NOTICES.md` for the dependency list and their licenses.
