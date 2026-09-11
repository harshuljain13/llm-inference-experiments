# Module 1 — Model: Prefill vs Decode

> Course **class 1**. Where an LLM call spends its time.

**One sentence:** An LLM call is two completely different workloads glued together — a compute-bound *prefill* and a memory-bound *decode* — and almost everything in later classes exists to manage that split.

**Everything lives in one file:** `class1.ipynb`. No servers, no GPUs required (it detects CPU / MPS / CUDA and adapts).

---

## The core idea

When you send a prompt to a model, two phases happen:

| | Prefill | Decode |
|---|---|---|
| What it does | Processes the whole prompt at once | Generates one token at a time |
| Parallelism | All prompt tokens in parallel | Strictly sequential — token N needs token N−1 |
| Bottleneck | **Compute** (big matmuls, GPU saturated) | **Memory bandwidth** (tiny matmuls, GPU mostly idle waiting on weights) |
| Scales with | Prompt length | Output length |
| User feels it as | Time to first token (TTFT) | Inter-token latency (ITL) |

This asymmetry is the reason for everything that follows. A GPU running decode is bandwidth-starved, not compute-starved — which is why batching many decodes together is nearly free, and why one huge prefill can stall everyone else.

## The KV cache

During decode, the model must attend to every previous token. Recomputing their key/value projections each step would be O(n²) work. Instead they're cached.

- **Without cache:** each new token reprocesses the entire sequence — quadratic, brutally slow.
- **With cache:** each new token only computes its own K/V and appends — linear.
- **The cost:** memory. KV cache size grows with `sequence_length × layers × heads × head_dim × 2 (K and V) × dtype_bytes`.

That memory is the scarce resource. Module 3 builds a block allocator for it; module 4 sheds load when it runs low.

## Notebook structure

| Section | What happens |
|---|---|
| 0–3 | Install, detect device, load GPT-2 / TinyLlama |
| 4 | **Prefill + decode with KV cache** — the central demo; time each phase separately |
| 5 | PyTorch profiler focused on attention and matmul ops |
| 6–7 | Run experiments across prompt/output lengths, tabulate results |
| 8 | **Cost** — map the two phases onto OpenAI's input vs. output token pricing |
| 12 | **With vs. without KV cache** — speed difference, and GPT-2 vs. TinyLlama memory layout |

## Why section 8 matters

API providers charge *more for output tokens than input tokens* — often 3–5×. That's not arbitrary pricing. Input tokens are prefill: parallel, efficient, high tokens/sec. Output tokens are decode: sequential, bandwidth-bound, low tokens/sec. **The price list is a direct readout of the hardware asymmetry you just measured.**

## What to take away

- TTFT and ITL are different metrics governed by different bottlenecks. Never average them together.
- The KV cache trades memory for time, and memory is finite — that trade is the whole game.
- Long prompts are expensive once (prefill); long outputs are expensive continuously (decode).

## Where this leads

- **Module 2** — put a server in front of this and watch it collapse under concurrency.
- **Module 3** — build the engine that manages KV memory in blocks.
- **Module 4** — build the gateway that decides who gets KV memory at all.
