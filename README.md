# Models & Systems

From a tensor expression to training, GPU execution, distributed training, and production LLM inference.

This repository is a public lab notebook for that path. Each day is a small runnable artifact with a correctness check. The 16-week plan is in [`curriculm.md`](curriculm.md).

## Setup

Python 3.11 or 3.12. Dependencies are locked in `uv.lock`.

```bash
uv sync
uv run jupyter lab
```

Open a notebook under `days/` and select the `.venv` kernel. The virtualenv is not committed.

## Days

- [Day 1: NumPy, shapes, and matrix multiplication](days/day-001-numpy-matmul/numpy_matmul.ipynb). A linear layer is `x @ W + b`. The last two sizes are the matrix. A leading axis stays.
- [Day 2: From NumPy to PyTorch](days/day-002-pytorch-basics/pytorch_basics.ipynb). The same layer, with the spellings that change: `float32`, `dim`, `permute`, `view`, and `.grad`.
