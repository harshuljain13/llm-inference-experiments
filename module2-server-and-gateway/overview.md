# Module 2 — Inference Server and Gateway

> Course **class 2**. The four layers — model → engine → server → gateway — and which layer owns which problem.

**One sentence:** Wrap a model in an HTTP server the obvious way, break it with concurrency, then compare it against a real engine and a gateway — learning the four-layer stack **model → engine → server → gateway**.

Module 1 looked *inside* one inference call. Module 2 looks at it *from the outside*, as a service.

---

## The four layers

This is the mental model the whole class is built around:

```
model     the weights                      TinyLlama-1.1B
  ↓
engine    batching, KV cache, scheduling   llama.cpp / vLLM
  ↓
server    HTTP, OpenAI-shaped API          FastAPI
  ↓
gateway   routing, keys, fallback, quotas  LiteLLM / RelayServe
```

Most of the class is figuring out **which layer a given problem belongs to.** When the naive server melts, the fix isn't more HTTP workers — it's an engine.

## The four systems under test

| System | What it is | The point |
|---|---|---|
| **A · naive-server** | Your own FastAPI + HuggingFace `generate()` | Deliberately bad. The control group. |
| **B · llama-engine** | llama.cpp server with a GGUF model | A real engine: continuous batching, proper KV management |
| **C · relay-serve** | RelayServe batching proxy in front of B | A gateway that adds micro-batching (`BATCH_SIZE=4`, `BATCH_WAIT_MS=10`) |
| **D · litellm** | LiteLLM proxy in front of A | A gateway that adds keys/routing but *no* batching |

Running the same load against all four is the experiment. **C vs. D is the sharp comparison**: both are "gateways," but only one changes throughput — because batching is an engine concern, not a routing concern.

## File map

### Core
| File | Role |
|---|---|
| `naive_server/server.py` | **The centerpiece.** FastAPI, OpenAI-shaped `/v1/chat/completions`, streaming, `/metrics`, `/healthz` |
| `inference_101.py` | Warm-up: boots llama.cpp in Docker on :8081, sends a request |
| `class2.ipynb` | Main notebook — Parts 0–7 |
| `intro.ipynb` | Warm-up notebook — walks the four layers one at a time |

### Deployment
| File | Role |
|---|---|
| `modal_apps/naive_server_app.py` | Deploys system A to Modal (GPU) |
| `modal_apps/llama_engine.py` | System B — llama.cpp CUDA image, T4, GGUF cached in a Modal volume |
| `modal_apps/relay_serve.py` | System C — RelayServe pointed at B |
| `modal_apps/litellm_app.py` | System D — LiteLLM proxy pointed at A |
| `modal_apps/_common.py` | Shared app names, model IDs, GGUF URL |
| `scripts/modal.sh` | Deploy / env / verify / compare / logs / stop |
| `docker-compose.yml`, `Dockerfile*` | Local (CPU) alternative to Modal |

### Experiment scripts
| Script | Part | What it does |
|---|---|---|
| `part1_anatomy.py` | 1 | Anatomy of one call — tokenize → prefill → decode → detokenize |
| `part3_break_server.py` | 3 | **The critical one.** Concurrent load until the server falls over |
| `part4_compare.py` | 4 | Same load across systems A/B/C/D |
| `part5_observability.py` | 5 | Polls `/metrics` on an interval, prints a live table |
| `part6_litellm_demo.py` | 6 | Gateway features — model aliasing, keys |

## Why the naive server breaks

`server.py` is bad *on purpose*, and the file comments say so. Three specific flaws:

1. **`MODEL_LOCK` serializes every generate.** Two concurrent requests queue behind each other. Throughput is flat no matter how many clients arrive — the lock is deliberately exposed "for teaching."
2. **No batching.** Real engines run many sequences through the GPU in one forward pass. This runs them one at a time, so the GPU sits idle during decode (recall module 1: decode is bandwidth-bound and *wants* company).
3. **No queue, no backpressure, no admission control.** Every request is accepted. Under load they all get slower together rather than some failing fast — latency collapse instead of graceful degradation.

Plus: `workers=1` is intentional (more workers would duplicate model weights in GPU memory), and blocking `generate()` inside an async route blocks the event loop.

**This is the exact problem module 4 solves at the gateway layer and module 3 solves at the engine layer.**

## Notebook parts

| Part | Content |
|---|---|
| 0 | Setup, device check |
| 1 | Anatomy of an LLM call *(skippable)* |
| 2 | Naive inference server — get it running |
| 3 | **Break the server** *(critical)* — concurrency, then "why it breaks" |
| 4 | RelayServe → llama.cpp — a real engine, compared |
| 5 | Observability — what to measure and why |
| 6 | LiteLLM *(skippable)* — gateway features |
| 7 | Reflection |

## What to take away

- **Layer discipline.** Batching belongs to the engine. Routing and keys belong to the gateway. Confusing the two produces C-vs-D: a gateway that looks helpful but moves no throughput.
- **The OpenAI API shape is a contract, not an implementation.** All four systems speak it; their performance differs by orders of magnitude.
- **Concurrency is where naive designs die** — and they die by getting uniformly slow, not by returning errors. Which is why you need admission control.

## Where this leads

- **Module 3** — stop treating the engine as a black box; build one.
- **Module 4** — build the gateway properly: admission, queueing, prefix-aware routing.
