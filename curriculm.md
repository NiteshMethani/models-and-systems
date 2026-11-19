# Full-Stack AI Systems Scientist / Engineer Curriculum

## Mission

Build the ability to reason from a tensor expression through model training,
GPU execution, distributed systems, and production LLM inference. This is an
**accelerated 16-week / 96-workday plan** for someone who already has a deep
learning degree and wants deliberate revision rather than an introductory
course.

The end state is to be able to:

- implement and train a small GPT;
- explain and measure a GPU bottleneck;
- design a parallel training plan for a large dense or MoE model;
- read, modify, and benchmark vLLM;
- understand modern architecture choices (GQA, MLA, MoE, linear/recurrent
  attention); and
- build a compact post-training/RLHF pipeline.

This repository is a **public technical lab notebook**, not a collection of
tutorial solutions. Every day ends with evidence.

## Working Agreement

- Target 2-4 focused hours per workday. Take a catch-up or reading day after
  any difficult block; do not optimize for a streak.
- Every implementation needs a correctness check before a performance claim.
- Every performance claim needs a baseline, measurement environment, and
  chart/table.
- Prefer tiny reproducible models first; scale only after the mechanism is
  understood.
- Write after building: a short `notes.md` explaining the question, result,
  and remaining uncertainty.

## Suggested Repository Layout

```text
.
├── CURRICULUM.md
├── README.md                    # portfolio index and current progress
├── env/                         # reproducible environment notes
├── days/
│   ├── day-001-numpy/           # code, tests, notes, figures
│   └── ...
├── shared/                      # reusable benchmarking and plotting helpers
├── papers/                      # reading notes; do not commit copyrighted PDFs
├── benchmarks/                  # raw configs and summarized results
└── capstones/
```

Each `days/day-NNN-topic/` should include:

```text
README.md        # question, intuition, result, next question
main.py|ipynb    # minimal runnable artifact
test_*.py        # correctness checks where applicable
results/         # small plots/tables; avoid large generated files
```

## Portfolio Standard

At the end of every phase, publish one polished note with a diagram and a
reproducible command. Good artifacts demonstrate judgment:

1. What is the system/model claim?
2. What is the smallest experiment that can falsify it?
3. What did the measurement show?
4. What would change at a larger scale?

## Phase Map

| Weeks | Theme | Flagship proof |
|---:|---|---|
| 1-2 | Fast foundations and training refresh | NumPy/PyTorch MLP with manual gradient checks |
| 3-4 | Attention, Transformers, GPT-2 | Tiny GPT trained and compared with Hugging Face |
| 5-6 | GPU performance and kernels | Profiled attention/MLP plus one Triton kernel |
| 7-9 | Distributed training | Two-GPU training and a parallelism/memory calculator |
| 10-12 | Inference and **vLLM** | PagedAttention simulator, vLLM code map, benchmark report |
| 13-14 | Modern architectures | Dense/MoE/MLA/linear-attention comparison |
| 15-16 | Post-training, RLHF, capstone | Preference optimization and production serving report |

---

# Daily Plan

## Phase 0 - Lab Setup and Fast Foundations (Days 1-8)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 1 | Create the lab skeleton, environment lockfile, formatter, test runner, and plotting helper. | Reproducible setup and a `hello-tensor` test. |
| 2 | NumPy shapes, strides, views/copies, broadcasting, reductions. | Shape exercises plus assertions explaining every output shape. |
| 3 | Matrix multiplication, batched matmul, einsum, and FLOP counting. | Naive vs NumPy matmul benchmark and a shape cheat sheet. |
| 4 | Numerical stability: floating-point formats, overflow, underflow, log-sum-exp. | Stable vs unstable softmax demonstration. |
| 5 | Probability refresher: categorical distribution, likelihood, cross-entropy, KL divergence. | Sampling and cross-entropy from first principles. |
| 6 | Calculus refresh: chain rule, Jacobians, vector-Jacobian products. | Manual derivatives checked with finite differences. |
| 7 | PyTorch tensors, devices, autograd graph, `nn.Module`, optimizer loop. | Linear-regression training loop and gradient inspection. |
| 8 | MLP revision: initialization, activations, normalization, residual paths. | MLP classification baseline with ablations. |

**Phase checkpoint:** explain why a shape error can be mathematically valid yet
semantically wrong; explain why softmax needs stabilization.

## Phase 1 - Training Mechanics and Autoregression (Days 9-16)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 9 | Implement a two-layer network and backprop manually. | Autograd parity test for every parameter gradient. |
| 10 | SGD, momentum, Adam/AdamW, weight decay, learning-rate schedules. | Optimizer comparison on one controlled task. |
| 11 | Batching, data loading, mixed precision, gradient accumulation. | Memory/throughput measurements for batch sizes. |
| 12 | Tokenization: bytes, characters, BPE concept, vocabulary trade-offs. | Tiny BPE trainer or documented minimal implementation. |
| 13 | Autoregressive factorization and next-token cross-entropy. | Character-level language-model data pipeline. |
| 14 | RNN and LSTM: one compact implementation only. | Copy-task comparison showing long-range failure/success. |
| 15 | Encoder-decoder and teacher forcing. | Toy sequence-to-sequence model. |
| 16 | Bahdanau attention as a selective read over encoder states. | Attention heatmap and concise bridge-to-Transformer note. |

**Deliberate scope:** CNNs are optional one-day revision; they are not a
prerequisite for LLM systems work.

## Phase 2 - Transformers and GPT-2 (Days 17-29)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 17 | Scaled dot-product attention, masks, and tensor shapes. | NumPy attention with unit tests. |
| 18 | Causal self-attention and why future tokens are masked. | Causal-mask test that catches leakage. |
| 19 | Multi-head attention: split, attend, concatenate, project. | PyTorch implementation compared to `scaled_dot_product_attention`. |
| 20 | Positional information: learned embeddings, sinusoidal positions, RoPE intuition. | Position-sensitivity experiment. |
| 21 | Transformer block: pre-norm, residual stream, MLP/SwiGLU. | Minimal decoder block with shape/gradient tests. |
| 22 | Build a tiny decoder-only Transformer end to end. | Train it on a deliberately small corpus. |
| 23 | Sampling: greedy, temperature, top-k, top-p, repetition effects. | Side-by-side generations under fixed seed. |
| 24 | GPT-2 architecture: token/position embedding, blocks, final norm, LM head. | Annotated GPT-2 forward-pass trace. |
| 25 | Hugging Face model/config/tokenizer inspection. | Script that prints and validates GPT-2 module/config shapes. |
| 26 | Load GPT-2 weights into your equivalent implementation. | Logit parity test on a fixed prompt. |
| 27 | Training diagnostics: loss curves, overfit-one-batch, validation, checkpoints. | Tiny-GPT training report. |
| 28 | Read the original Transformer and GPT-2 papers selectively. | Two-page architecture comparison note. |
| 29 | Consolidation and cleanup. | Phase write-up: “From tokens to GPT-2 logits.” |

## Phase 3 - KV Cache, Performance, and GPU Fundamentals (Days 30-42)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 30 | Prefill versus decode: what is parallel and what is sequential? | Instrumented generation timeline. |
| 31 | Implement a correct KV cache for one attention layer. | Cached/uncached logits parity and speed comparison. |
| 32 | KV-cache memory formula across layers, heads, dtype, batch, context. | Interactive calculator or table. |
| 33 | Compute-bound versus memory-bound; FLOPs, bytes, arithmetic intensity. | Roofline-style estimate for decode. |
| 34 | GPU overview: SMs, warps, tensor cores, registers, shared memory, L2, HBM. | One-page annotated memory-hierarchy note. |
| 35 | CUDA execution model: grids, blocks, threads, occupancy. | CPU/GPU vector-op benchmark with explanation. |
| 36 | PyTorch profiler and Nsight Systems/Compute basics. | Profile trace annotated with three observations. |
| 37 | GPU memory allocation, fragmentation, and asynchronous execution. | Allocator experiment with measured peak memory. |
| 38 | FlashAttention: online softmax and tiling; distinguish algorithmic complexity from IO cost. | Small online-softmax implementation and explanation. |
| 39 | Implement simplified tiled attention in PyTorch/NumPy. | Correctness and memory comparison to naïve attention. |
| 40 | Triton basics: indexing, program instances, memory coalescing. | Vector-add or softmax kernel with benchmark. |
| 41 | Kernel fusion and launch overhead. | Fused elementwise kernel benchmark. |
| 42 | Consolidation. | Phase write-up: “Why decode becomes memory-bound.” |

## Phase 4 - Distributed Training Foundations (Days 43-56)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 43 | Distributed systems refresher: latency/bandwidth, topology, failures, collectives. | Communication-cost worksheet. |
| 44 | Implement/simulate all-reduce, reduce-scatter, and all-gather. | Tiny collective simulator with tests. |
| 45 | Data parallelism and gradient synchronization. | Two-process DDP run with gradient parity test. |
| 46 | ZeRO stages and FSDP: what state is sharded and what moves. | Per-rank memory calculator. |
| 47 | Activation memory, checkpointing, and recomputation. | Memory/time trade-off plot. |
| 48 | Tensor parallel MLP: column/row sharding and required collectives. | Two-rank TP MLP matched to single-device output. |
| 49 | Tensor-parallel attention; head partitioning. | TP attention sketch with communication annotations. |
| 50 | Sequence parallelism and why it pairs with TP. | Activation-memory accounting note. |
| 51 | Pipeline parallelism, microbatches, 1F1B, pipeline bubbles. | Pipeline-schedule simulator. |
| 52 | Context parallelism and ring attention. | Ring-attention communication simulation. |
| 53 | MoE routing and expert parallelism; all-to-all communication. | Tiny top-k MoE with token-dispatch simulation. |
| 54 | Compose parallelism dimensions: DP × TP × PP × CP × EP. | Layout planner for a hypothetical model/hardware budget. |
| 55 | Read a large-scale training case study (Megatron/DeepSeek). | Design memo for training a 70B dense or MoE model. |
| 56 | Consolidation. | Phase write-up: “What moves between GPUs, and why?” |

Use Xuewen Yang’s worked examples as a recurring reference for the six forms
of parallelism and collective communication: <https://xuewyang.github.io/posts/parallelism/>.

## Phase 5 - Inference Systems and vLLM (Days 57-75)

This is the core systems phase. Read the PagedAttention paper alongside the
implementation in `/Users/niteshmethani/Documents/workspace/my-projects/vllm`.

| Day | Focus | Evidence to commit |
|---:|---|---|
| 57 | Serving metrics: TTFT, TBT, end-to-end latency, throughput, goodput, queueing delay. | Benchmark-harness interface and metric definitions. |
| 58 | Static batching, padding waste, iteration-level/continuous batching. | Discrete-event batch scheduler simulator. |
| 59 | OS memory refresher in Galvin terms: logical/physical address space, paging, fragmentation, sharing, replacement. | Mapping table from OS paging to KV-cache blocks. |
| 60 | Contiguous KV allocation and internal/external fragmentation. | Fragmentation simulator and visualization. |
| 61 | PagedAttention paper: block tables, on-demand blocks, near-zero waste. | Paper figure reproduced as a toy simulation. |
| 62 | Implement a CPU KV block allocator: free list, reference counts, copy-on-write. | Unit-tested allocator. |
| 63 | Implement block tables and non-contiguous logical token access. | Toy paged-attention address translation. |
| 64 | Sharing for parallel sampling and beam search. | Copy-on-write demonstration with reference-count tests. |
| 65 | Scheduling under memory pressure: admission, preemption, recompute/swap concepts. | Scheduler policy comparison. |
| 66 | vLLM repository tour: engine, scheduler, worker/executor, cache manager, attention backend. | Current code map with file links and request lifecycle diagram. |
| 67 | Trace one request through vLLM’s code and logs. | Annotated sequence diagram. |
| 68 | Run a small local vLLM model and validate its OpenAI-compatible API. | Reproducible launch command and smoke test. |
| 69 | Benchmark concurrency, request lengths, and sampling parameters. | TTFT/TBT/throughput plots with methodology. |
| 70 | Inspect vLLM memory utilization, KV blocks, and scheduler behaviour. | Memory-budget report. |
| 71 | Prefix caching and cache reuse; compare with SGLang/RadixAttention conceptually. | Shared-prefix benchmark. |
| 72 | Speculative decoding: draft/verify mechanics and acceptance rate. | Small acceptance-rate simulator. |
| 73 | Tensor-parallel inference, model loading, and serving topology. | Deployment design for a model larger than one GPU. |
| 74 | Read PagedAttention critically: assumptions, 2023 context, and what has changed. | Technical critique, not a summary. |
| 75 | Consolidation. | Phase write-up: “Build a toy vLLM from first principles.” |

## Phase 6 - Modern LLM Architectures (Days 76-84)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 76 | Llama/Qwen design: RMSNorm, RoPE, SwiGLU, GQA/MQA. | Architecture-diff table and KV-cache-size calculation. |
| 77 | Grouped-query attention and multi-query attention. | MHA/GQA/MQA memory and output-shape comparison. |
| 78 | Multi-head latent attention (MLA): compression and inference implications. | Toy latent-cache calculation. |
| 79 | MoE economics: activated parameters, capacity factor, routing load balance. | MoE capacity/load-balance experiment. |
| 80 | DeepSeek-style architecture: select one technical report and reconstruct its design choices. | Evidence-backed architecture note. |
| 81 | Linear/recurrent attention: associative memory, fixed state, interference. | Compare cache growth against a fixed-size recurrent state. |
| 82 | Delta rule, gating, Mamba/Kimi-style trade-offs. | Small recurrent-memory POC; clearly state approximation limits. |
| 83 | Quantization: FP8/INT8/INT4, calibration, error, bandwidth trade-offs. | Quantize/dequantize experiment with error and speed measurement. |
| 84 | Consolidation. | Phase write-up: “What modern LLMs choose to store, retrieve, and route.” |

## Phase 7 - Post-Training and RL (Days 85-92)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 85 | Supervised fine-tuning: data formatting, masking, evaluation leakage. | SFT a small model on a constrained task. |
| 86 | Preference datasets, reward modeling, and preference-pair losses. | Synthetic preference-data pipeline with tests. |
| 87 | RLHF mechanics: policy, reference policy, reward, KL control, PPO vocabulary. | Toy bandit PPO implementation. |
| 88 | Direct Preference Optimization (DPO) and why it can avoid explicit reward-model RL. | DPO-loss derivation and toy optimization. |
| 89 | GRPO/RLOO and group-relative methods at a conceptual + implementation level. | Controlled comparison of two preference objectives. |
| 90 | LLM evaluation: task metrics, judges, contamination, win rates, uncertainty. | Small evaluation harness and failure-analysis template. |
| 91 | Safety and alignment systems: refusal behavior, reward hacking, adversarial evaluation. | Threat model and red-team test set for your task. |
| 92 | Consolidation. | Phase write-up: “From next-token prediction to preference optimization.” |

## Phase 8 - Capstone and Public Proof (Days 93-96)

| Day | Focus | Evidence to commit |
|---:|---|---|
| 93 | Select a narrow thesis: e.g. KV-cache policy, serving scheduler, attention kernel, or distributed-training memory model. | One-page proposal with falsifiable hypothesis. |
| 94 | Implement baseline and improved design. | Reproducible experiment config and correctness tests. |
| 95 | Benchmark, ablate, and document failure cases. | Results table, plots, and limitations section. |
| 96 | Publish. | Polished README/blog post/demo; a short video walkthrough is optional. |

---

# Reading Spine

Read selectively and implement more than you read.

1. Fleuret, *The Little Book of Deep Learning* - concise revision spine.
2. Vaswani et al., *Attention Is All You Need* - attention and Transformer.
3. Radford et al., GPT-2 - decoder-only language modeling.
4. Dao et al., FlashAttention - IO-aware exact attention.
5. Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* - vLLM’s original systems argument.
6. Megatron-LM, ZeRO, GPipe, Ring Attention, and a recent DeepSeek technical report - distributed training.
7. vLLM source tree and its tests - the implementation authority for vLLM internals.

# What “Done” Looks Like

At the end, this repository should let a technically strong reviewer quickly
find:

- a Transformer you built and verified;
- a performance investigation grounded in hardware measurements;
- a distributed-training design whose communication and memory numbers can be
  checked by hand;
- a vLLM/PagedAttention implementation study plus real benchmark;
- an informed comparison of current model architectures; and
- a compact post-training experiment with honest evaluation.

Do not wait for the capstone to share work. Publish the phase write-ups as you
go; thoughtful technical notes, clean measurements, and useful diagrams are
how you enter the right technical conversations.
