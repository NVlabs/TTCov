# Third-Party Notices

TTCov does **not** bundle, vendor, or redistribute any third-party source code,
model weights, or datasets. There are no git submodules and nothing third-party
is copied into this repository. All third-party dependencies listed below are
resolved and installed from their own upstream distributions at install time
(`uv sync` / pip), each under its own upstream license; none of their code is
shipped as part of this project.

This file is provided for attribution and convenience. Each dependency remains
governed by the license linked below, which is delivered with that package by
its own distribution.

## Direct dependencies

Resolved versions are pinned in `uv.lock`; the constraints below are from
`pyproject.toml`. Licenses are the upstream project licenses.

### Core (installed by `uv sync`)

| Package | Constraint | License | Project |
|---------|------------|---------|---------|
| dill | >=0.3.8 | BSD-3-Clause | https://pypi.org/project/dill/ |
| faiss-cpu | >=1.8 | MIT | https://pypi.org/project/faiss-cpu/ |
| highspy | >=1.7 | MIT | https://pypi.org/project/highspy/ |
| httpx | >=0.27 | BSD-3-Clause | https://pypi.org/project/httpx/ |
| numpy | >=1.23 | BSD-3-Clause | https://pypi.org/project/numpy/ |
| openai | >=1.0,<2.3 | Apache-2.0 | https://pypi.org/project/openai/ |
| opencv-python-headless | >=4.8 | Apache-2.0 (OpenCV); MIT packaging | https://pypi.org/project/opencv-python-headless/ |
| pyarrow | >=14.0 | Apache-2.0 | https://pypi.org/project/pyarrow/ |
| scipy | >=1.10 | BSD-3-Clause | https://pypi.org/project/scipy/ |
| sentence-transformers | >=5.1.0,<5.2 | Apache-2.0 | https://pypi.org/project/sentence-transformers/ |
| torch | >=2.7 | BSD-3-Clause | https://pypi.org/project/torch/ |
| tqdm | >=4.66 | MPL-2.0 AND MIT | https://pypi.org/project/tqdm/ |
| transformers | >=4.52 | Apache-2.0 | https://pypi.org/project/transformers/ |

`torch` is resolved on Linux from the PyTorch CUDA 12.6 wheel index
(`https://download.pytorch.org/whl/cu126`); see `pyproject.toml`. Its wheels
bundle components under additional permissive licenses (e.g. BSD-2-Clause, MIT,
BSL-1.0, Apache-2.0-WITH-LLVM-exception), delivered with the wheel.

### Optional extra `physical-ai` (installed by `uv sync --extra physical-ai`)

| Package | Constraint | License | Project |
|---------|------------|---------|---------|
| physical_ai_av | >=0.2.2 | MIT | https://github.com/NVlabs/physical_ai_av |

`physical_ai_av` is NVIDIA's official Physical AI AV dataset SDK and is required
only for the `physical-ai` captioning interface. It is not installed by the
default `uv sync`.
