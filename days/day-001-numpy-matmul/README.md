# Day 1: NumPy, Shapes, and Matrix Multiplication

**Question:** How does a neural-network layer turn input features into output
features?

**Notebook:** [`numpy_matmul.ipynb`](numpy_matmul.ipynb)

## What this lab proves

- An array's `shape` assigns meaning to each axis.
- `*` is elementwise multiplication; `@` contracts matching inner dimensions.
- A linear layer is batched matrix multiplication followed by a bias broadcast.
- Leading dimensions are preserved by `@`; this is why the same linear layer
  works for one token, a sequence, or a batch of sequences.

## Run

```bash
jupyter notebook days/day-001-numpy-matmul/numpy_matmul.ipynb
```

## Exit criteria

You can predict every displayed output shape before running a cell and explain
why `(batch, tokens, features) @ (features, outputs)` produces
`(batch, tokens, outputs)`.
