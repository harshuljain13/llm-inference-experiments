# Class 5 — Overview

**One sentence:** Build a miniature vLLM — paged KV memory, a batching scheduler, preemption — so the engine layer from class 2 stops being a black box.

Class 2 treated the engine as something you download. Class 5 opens it up. It runs on CPU with a fake model by default; a real model is optional.

---

## The engine loop

Everything in `smol-vllm` is this loop:

```
add_request()  →  waiting queue
                      ↓
   ┌──────────  step()  ──────────┐
   │  scheduler.schedule()        │   who runs this step?
   │      ↓                       │
   │  block_manager.allocate()    │   is there KV memory?
   │      ↓                       │
   │  model.prefill() / decode()  │   do the work
   │      ↓                       │
   │  append_token / free         │   grow or release memory
   └──────────────────────────────┘
                      ↓
                RequestOutput
```

`step()` is called repeatedly. Each call is one GPU turn. **Requests are not run to completion — they interleave**, one token at a time, which is what "continuous batching" means.

## File map

### The engine (`smol-vllm/smol_vllm/`)

| File | Lines | Role |
|---|---|---|
| `block_manager.py` | 89 | **Paged KV memory.** The heart of the class. |
| `scheduler.py` | 117 | Who runs next; when to preempt |
| `engine.py` | 198 | `step()` — ties scheduler + blocks + model together |
| `sequence.py` | 50 | `Sequence`, `SequenceGroup`, status enum (WAITING/RUNNING/SWAPPED/FINISHED) |
| `model.py` | 63 | `FakeModel` — no weights, simulated timing |
| `causal_model.py` | 204 | Real HuggingFace model (optional) |
| `metrics.py` | 190 | TTFT, ITL, block utilization, per-step table |
| `demo.py` | 490 | The 5 experiments |
| `exercises.py` | 66 | **Student stubs** — implement these yourself |
| `reference/exercises_ref.py` | — | Solutions |

### Class harness
| File | Role |
|---|---|
| `class5.ipynb` | Part A (manual walkthrough) + Part C (experiments) + Part B (CrewAI) |
| `agent_demo.py` | CrewAI agent driving the engine from the CLI |
| `lib/smol_crew_llm.py` | Adapter: CrewAI `BaseLLM` → smol-vllm engine |
| `lib/engine_state.py`, `lib/tokenizer.py` | Notebook support |
| `scripts/sync_to_lambda.sh`, `lambda_*.sh` | Optional GPU runs |

## Paged attention, concretely

`BlockSpaceManager` is the idea that made vLLM famous. KV cache is not one contiguous slab per sequence — it's **fixed-size blocks** (`block_size=16` tokens) handed out from a free list, exactly like OS virtual memory pages.

```python
_free_blocks: deque       # the free list
_block_tables: {seq_id: [phys_block_ids]}   # the "page table"
_ref_count: {phys_id: n}  # how many sequences share this block
```

Four operations, and why each matters:

- **`allocate(seq_id, num_tokens)`** — `ceil(tokens/16)` blocks off the free list. Non-contiguous is fine; the block table indirects.
- **`append_token(seq_id)`** — decode grew the sequence by one. Only grabs a new block when crossing a 16-token boundary. This is why growth is cheap.
- **`free(seq_id)`** — decrement refcounts; blocks return to the free list only at zero.
- **`copy_on_write(src, dst)`** — two sequences share the same physical blocks, refcount incremented. **This is prefix sharing.** Two agents with the same system prompt occupy one copy of it.

**Why this beats contiguous allocation:** no fragmentation (any free block fits any sequence), no over-reservation (you don't pre-allocate for max possible length), and sharing becomes a refcount increment instead of a memcpy.

`utilization()` — used/total blocks — is the pressure signal. Class 7's gateway scrapes exactly this from real vLLM as `vllm:kv_cache_usage_perc`.

## The scheduler

`schedule()` runs three phases in order:

1. **Promote** waiting → running, FIFO, while under `max_batch_size` *and* blocks are available. Note the extra guard: `free_after < running_after` breaks the loop — it keeps a reserve so running sequences can still append tokens. Admitting a sequence you can't feed is worse than not admitting it.
2. **Preempt** if pressured — `utilization > 0.95`, or free blocks fewer than running sequences. The victim is **evicted, not swapped**: `block_manager.free()`, status → SWAPPED, back to the queue. Its KV is *destroyed* and must be recomputed later.
3. **Swap in** previously preempted groups when room reappears.

### `preempt_guard` — the livelock lesson

`_should_preempt` refuses to preempt when only one sequence is running, or when *every* running sequence was scheduled this same step. Without those guards you get **livelock**: admit a sequence → immediately preempt it → readmit → preempt, forever. Zero forward progress at 100% utilization.

`demo.py:reproduce_preemption_livelock()` exists to reproduce this deliberately. It's the most valuable failure in the class — a system that is fully busy and accomplishing nothing.

## `engine.step()` in detail

Look at the prefill/decode split (`engine.py`):

```python
prefill_groups = [g for g in scheduled if not g.sequences[0].output_tokens]
decode_groups  = [g for g in scheduled if     g.sequences[0].output_tokens]
```

A sequence with no output tokens yet needs prefill; one with output tokens needs decode. They're timed separately (`prefill_ms`, `decode_ms`) because — straight from class 1 — **they're different workloads**. Prefill is compute-bound and chunky; decode is bandwidth-bound and steady.

On finish: mark FINISHED, `record_request_finish`, **`block_manager.free()`**, drop from running. Forgetting the free is a KV leak — the class-5 version of a memory leak.

## The five experiments (`demo.py`)

| # | Focus |
|---|---|
| 1 | Basic scheduling — watch waiting/running/finished move |
| 2 | Batching — throughput vs. batch size |
| 3 | **Prefix sharing** — `copy_on_write`, blocks saved |
| 4 | Cost model — `naive` vs. `roofline` timing |
| 5 | **Preemption / livelock** — the guard's reason to exist |

Plus `measure_kv_bytes_per_token()` — grounds the abstraction in real bytes.

## The exercises

`exercises.py` has three stubs with full docstring specs and no implementations:

- `allocate()` — the free-list pop
- `append_slot()` — the block-boundary check
- `schedule_promotions()` — FIFO promote, **stop at first failure, don't skip ahead**

That last invariant is the subtle one: skipping a blocked head-of-queue request to serve an easier one behind it starves large requests indefinitely. Checkpoints in `demo.py` verify your implementations.

## Part B — CrewAI

`agent_demo.py` + `lib/smol_crew_llm.py` wrap the engine in a CrewAI `BaseLLM`, so agent requests flow into `add_request()`. The point: **an agent framework is just a client.** It generates bursty, prefix-heavy traffic — which is precisely the traffic pattern that makes paging and prefix sharing pay off.

Class 7 uses the identical adapter pattern, but the engine behind it is real vLLM across two replicas.

## What to take away

- Paging solves fragmentation *and* enables sharing. One design, two wins.
- Preemption is not free — you throw away computed KV. Count it.
- A busy system is not a productive system. Livelock is real, and guards are what prevent it.
- Every number the class 7 gateway scrapes (`num_requests_waiting`, `kv_cache_usage_perc`, `num_preemptions_total`) is something you built here by hand.

## Where this leads

- **Class 7** — the engine is now real vLLM. Your job moves up a layer: decide what enters, in what order, and on which replica.
