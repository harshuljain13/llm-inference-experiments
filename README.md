<p align="center">
  <img src="assets/banner.svg" alt="LLM Inference Experiments" width="100%">
</p>

<p align="center">
  <strong>Runnable experiments for serving large language models — the empirical companion to the handbook.</strong>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-the-experiments">Experiments</a> •
  <a href="#-metrics-vocabulary">Metrics</a> •
  <a href="#-what-moves-what">Levers</a> •
  <a href="#-adding-an-experiment">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/vLLM-0.8+-orange.svg" alt="vLLM">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.0+-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/CPU--friendly-most%20labs-success.svg" alt="CPU friendly">
</p>

---

## Why This Exists

[**LLM Inference at Scale**](https://github.com/harshuljain13/llm-inference-at-scale) explains how LLM serving works. This repo is where those claims get **measured**.

Reading that paged attention reduces fragmentation is one thing. Watching your block utilization curve bend when you change `block_size` is another. Reading that admission control protects tail latency is one thing. Watching goodput collapse to zero *without* it, at a load you can point to on a graph, is another.

Every experiment here answers a question of the form **"what happens to metric M when I change knob K?"** — and reports the answer as a curve, not an anecdote.

One idea runs through all of it:

> **The GPU is scarce.** Decide carefully what work enters, what runs next, and where it runs. Then measure whether your decisions actually helped.

---

## 🚀 Quick Start

```bash
git clone https://github.com/harshuljain13/llm-inference-experiments.git
cd llm-inference-experiments
```

Most labs are **CPU-friendly and deterministic** — start there, no GPU bill required.

```bash
cd module3-build-your-own-engine && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m smol_vllm.demo          # paged KV, scheduling, preemption
```

New here? Read [`module1-prefill-vs-decode/overview.md`](module1-prefill-vs-decode/overview.md) first. Every lab has an `overview.md` that maps the files and explains the concept before you run anything.

---

## 🧭 The Experiments

Each lab moves one layer up the stack. Read the `overview.md`, then run the code.

| Lab | Layer | The question it answers | GPU | Overview |
|---|---|---|---|---|
| **[module1-prefill-vs-decode](module1-prefill-vs-decode/)** | Model | Why is decode slow and prefill fast? | Optional | [overview](module1-prefill-vs-decode/overview.md) |
| **[module2-server-and-gateway](module2-server-and-gateway/)** | Server + Gateway | Which layer owns which problem? | Optional | [overview](module2-server-and-gateway/overview.md) |
| **[module3-build-your-own-engine](module3-build-your-own-engine/)** | Engine | How is KV memory paged and work scheduled? | No | [overview](module3-build-your-own-engine/overview.md) |
| **[module4-admission-and-routing](module4-admission-and-routing/)** | Gateway | Who gets in, in what order, on which GPU? | **Yes** | [overview](module4-admission-and-routing/overview.md) |
| **[module5-multi-gpu-scaling](module5-multi-gpu-scaling/)** | Multi-GPU | Split the model or replicate it? | **Yes** | [overview](module5-multi-gpu-scaling/overview.md) |
| **[module6-observability-and-cost](module6-observability-and-cost/)** | Production | What does the whole stack cost, and where does time go? | **Yes** | [overview](module6-observability-and-cost/overview.md) |

### The stack, assembled

```
   module1-prefill-vs-decode       prefill vs decode · KV cache · why output tokens cost more
        ↓
   module2-server-and-gateway      model → engine → server → gateway · why naive servers melt
        ↓
   module3-build-your-own-engine   paged KV blocks · continuous batching · preemption livelock
        ↓
   module4-admission-and-routing   admission control · deadline-ordered queue · prefix routing
        ↓
   module5-multi-gpu-scaling       tensor / pipeline / data parallel · when each is worth it
        ↓
   module6-observability-and-cost  the whole stack, instrumented · engine flag A/B · $ per request
```

### Where did each module come from?

<details>
<summary><strong>Provenance — course class / source repo → module</strong></summary>

Modules are numbered sequentially. If you're looking for "class 7" or one of the standalone repos, it's here:

| Was | Now | Why it moved |
|---|---|---|
| `class1` | [`module1-prefill-vs-decode`](module1-prefill-vs-decode/) | renumbered sequentially |
| `class2` | [`module2-server-and-gateway`](module2-server-and-gateway/) | renumbered sequentially |
| `class5` | [`module3-build-your-own-engine`](module3-build-your-own-engine/) | renumbered sequentially |
| `class7` | [`module4-admission-and-routing`](module4-admission-and-routing/) | renumbered sequentially |
| `class3` — `fullstack-inferencing` repo | [`module6-observability-and-cost`](module6-observability-and-cost/) | absorbed via `git subtree` |
| `class6` — `ray_project` repo | [`module5-multi-gpu-scaling`](module5-multi-gpu-scaling/) | absorbed via `git subtree` |

Only course class 4 had no lab code.

**Modules are ordered by stack layer, not by course order.** Classes 3 and 6 shipped as standalone repos and sit late in the module numbering because observability is cross-cutting and multi-GPU builds on the single-GPU modules.

**Original history is intact.** Pre-rename state is on the `upstream-course` remote:

```bash
git fetch upstream-course
git show upstream-course/main:class7/gateway/router.py   # browse a file
git log --follow -- module4-admission-and-routing/gateway/router.py    # history across the rename
```

`git log --follow` tracks a file through the rename, so `git blame` and history are unaffected.

</details>

### What's in each

<details>
<summary><strong>module1-prefill-vs-decode — Prefill vs Decode</strong></summary>

A single notebook. Times the two phases of an LLM call separately and shows they are different workloads: prefill is compute-bound and parallel, decode is memory-bandwidth-bound and sequential. Ends by mapping the asymmetry onto API pricing — output tokens cost more because decode is slower, and the price list is a readout of the hardware.

**Files:** `class1.ipynb`
**Handbook:** Ch. 00 (Transformer at Inference), Ch. 01 (GPU Hardware)
</details>

<details>
<summary><strong>module2-server-and-gateway — Inference Server and Gateway</strong></summary>

Wrap a model in FastAPI the obvious way, then break it with concurrency. Compares four systems under identical load: a naive server, llama.cpp as a real engine, a batching proxy, and a routing gateway. The naive server is bad *on purpose* — a global model lock, no batching, no backpressure.

The sharp comparison is the batching proxy vs. the routing gateway: both are "gateways," but only one moves throughput, because batching is an engine concern.

**Files:** `naive_server/`, `modal_apps/`, `scripts/part*.py`, `class2.ipynb`
**Handbook:** Ch. 05.3 (Continuous Batching), Ch. 06 (Engines), Ch. 08 (Serving)
</details>

<details>
<summary><strong>module3-build-your-own-engine — smol-vllm: Build the Engine</strong></summary>

A miniature vLLM in ~1,500 lines. Paged KV blocks with a free list and refcounts, a scheduler that promotes and preempts, and a metrics layer. Runs on CPU against a fake model, so sweeps are free and seeded runs are reproducible.

Includes `exercises.py` — three stubs (`allocate`, `append_slot`, `schedule_promotions`) with full specs and no implementations — plus a deliberate **preemption livelock** reproduction: a system at 100% utilization making zero forward progress.

**Files:** `smol-vllm/smol_vllm/{block_manager,scheduler,engine}.py`, `demo.py`
**Handbook:** Ch. 04.1 (PagedAttention), Ch. 04.5 (Prefix Caching), Ch. 05.3 (Continuous Batching)
</details>

<details>
<summary><strong>module4-admission-and-routing — Gateway over Two vLLM Replicas</strong></summary>

Real vLLM, two replicas, one GPU, `--max-num-seqs 8` and not raisable. The constraint is the lesson. A gateway makes three decisions: admit or shed (five checks, including refusing work that provably cannot meet its deadline), queue order (EDF with anti-starvation aging), and replica choice (prefix-cache affinity scored against load).

Ships with four presets — `baseline` → `route` → `queue` → `full` — benchmarked under identical load.

**Files:** `gateway/`, `app.py`, `limiter.py`, `bench/report.py`
**Handbook:** Ch. 08.6 (Cache-Aware Routing), Ch. 09.1 (Benchmarking), Ch. 11.5 (Agentic Workload)
</details>

<details>
<summary><strong>module5-multi-gpu-scaling — More Than One GPU</strong></summary>

Every earlier module assumes a single GPU. This is where that breaks. Ray Serve and vLLM on Modal, with tensor / pipeline / data parallelism as the knobs.

The rule under test: use TP/PP only when you *must* (the model doesn't fit), use DP when you *can* (you need throughput). Splitting a model that already fits pays all-reduce cost for nothing.

Carries the metric trap worth internalizing — `run_bench.py` reports **aggregate** and **per-stream** throughput separately, because per-stream stays flat when you scale out and will tell you, wrongly, that adding GPUs did nothing.

**Files:** `serve_app.py`, `modal_app.py`, `load_test/run_bench.py`, `profiling/profile_{tp,dp,tp_pp}.sh`
**Handbook:** Ch. 07.1 (Tensor Parallelism), Ch. 08.1 (Ray Serve)
</details>

<details>
<summary><strong>module6-observability-and-cost — The Whole Stack, Instrumented</strong></summary>

CrewAI → nginx → gateway → vLLM on a Lambda GPU, with Prometheus, Grafana, and per-request cost accounting. Every other module measures with a purpose-built script; this one measures the way production does.

Includes an engine-flag A/B harness — chunked prefill, prefix caching, speculative decoding — each as a separate vLLM profile driven under identical load. The real-vLLM counterpart to the mechanisms `module3-build-your-own-engine` builds from scratch.

Key idea: the gateway exports `upstream_duration` *and* `request_duration`, so `total − upstream` isolates gateway and tunnel overhead. Without that subtraction a slow gateway and a slow engine look identical.

**Files:** `gateway.py`, `monitoring/`, `scripts/vllm_engine/`, `lambda_pricing.py`
**Handbook:** Ch. 09 (Benchmarking & Observability), Ch. 11.5 (Agentic Workload)
</details>

---

## 📊 Metrics Vocabulary

Used identically across every lab. Ambiguity here is the fastest way to draw a wrong conclusion — if a column says TTFT, it must actually be TTFT.

| Metric | Definition | Governed by |
|---|---|---|
| **TTFT** | submit → first token | prefill + queue wait |
| **ITL** | interval between output tokens | decode speed, batch size |
| **Throughput** | completions / sec | batch efficiency |
| **Goodput** | completions / sec **within deadline** | ← the number that matters |
| **KV utilization** | blocks used / blocks total | admission pressure |
| **Preempt rate** | evictions / sec | KV overcommit |
| **Wasted tokens** | decode work thrown away by preemption | preemption cost |
| **Queue wait** | enqueue → dispatch | backlog depth |
| **Prefix hit rate** | requests reusing ≥1 cached block | routing quality |
| **Load spread** | max − min requests per replica | balancing quality |
| **Shed rate** | refusals / sec, **broken out by reason** | admission policy |

> **Throughput and goodput diverge under overload — and that divergence is the whole point.** A system can maximize throughput while goodput falls to zero: everything completes, all of it too late to matter.

---

## 🎛 What Moves What

The causal map these experiments exist to establish. Direction matters more than magnitude.

**Model layer** — `module1-prefill-vs-decode`

| Lever | Primary effect | Should *not* move |
|---|---|---|
| prompt length ↑ | TTFT ↑ (≈linear) | ITL |
| output length ↑ | total latency ↑ | TTFT |
| KV cache off | ITL ↑↑ (quadratic) | TTFT |

**Engine layer** — `module3-build-your-own-engine`

| Lever | Primary effect | Trade / watch for |
|---|---|---|
| `max_batch_size` ↑ | throughput ↑, **then collapses** | ITL ↑; preempts spike at the KV wall |
| `num_gpu_blocks` ↓ | preempts ↑ | wasted tokens ↑, goodput ↓ nonlinearly |
| `block_size` ↑ | internal fragmentation ↑ | block-table overhead ↓ — a sweet spot exists |
| `preempt_guard` off | **livelock** | throughput → 0 at 100% utilization |
| prefix sharing on | effective KV capacity ↑ | more concurrency at the same memory |

**Gateway layer** — `module4-admission-and-routing`

| Lever | Primary effect | Trade | Visible only when |
|---|---|---|---|
| admission on | **goodput ↑** | throughput ↓, admitted ↓ | overloaded |
| `KV_CEILING` ↓ | sheds earlier, p99 ↓ | more false rejects | near the KV wall |
| queue on | interactive p99 ↓ | long-prompt p99 ↑ | queue depth > 1 |
| `AGING_GAIN` ↑ | starvation ↓ | EDF purity ↓ | mixed prompt sizes |
| `W_LOAD` vs `W_PREFIX` | spread ↔ hit rate | direct tradeoff | replicas actually busy |
| `DISPATCH_OVERSHOOT` ↑ | GPU utilization ↑ | queue wait ↑ | saturated |

**Read the last column carefully.** Most gateway levers are no-ops below saturation. An experiment run at 30% utilization will show admission control doing nothing — correctly, and uninformatively.

---

## 📐 Measurement Discipline

Three rules, each learned the hard way.

**1. Sweep load. Never measure at a single point.**
Every metric is a function of offered load. The interesting behavior is at the knee, and single-point measurements almost always land on the flat part of the curve.

```
   goodput
     │      ╭─────╮
     │     ╱       ╲          without admission control:
     │    ╱          ╲           collapses past the knee
     │   ╱             ╲___
     │  ╱      ╭────────────  with admission control:
     │ ╱      ╱                  plateaus instead
     │╱______╱
     └──────────────────────  offered load
             ↑ knee
```

**2. Check what should *not* move.**
If changing prompt length shifts your ITL, the measurement is wrong before the conclusion is. Every experiment declares its invariants.

**3. Report goodput, not throughput.**
Throughput rewards a system for finishing work nobody is waiting for anymore.

---

## ➕ Adding an Experiment

```
<name>/
  overview.md      what it measures, why, and how it connects
  README.md        how to run it
  requirements.txt pinned
  <code>
  results/         committed outputs — CSV, plots (never raw logs or secrets)
```

`overview.md` is the contract. It states:

1. **The question** — "what happens to M when I change K?"
2. **The hypothesis** — expected direction, before running
3. **The invariants** — what should *not* move
4. **The load range** — where on the curve this was measured
5. **The handbook link** — which chapter this grounds

Then add a row to [The Experiments](#-the-experiments) and cross-link the handbook chapter.

**Conventions:** pin dependencies · seed anything random · commit results, gitignore secrets and raw logs · a negative result is a result, and belongs in `overview.md` with the reason.

---

## 🔗 Related

| Repo | What it is |
|---|---|
| [llm-inference-at-scale](https://github.com/harshuljain13/llm-inference-at-scale) | The handbook — 12 chapters, ~55 modules. Theory these experiments test. |
| **This repo** | The lab bench. Runnable code, measured curves. |

---

## 👤 About the Author

**Harshul Jain** is a Senior ML Infrastructure Engineer specializing in real-time ML systems, feature stores, and LLM serving infrastructure. He builds and operates ML platforms serving millions of users, mentors 300+ engineers through an eMentoring program, and is a recurring speaker at ML infrastructure conferences.

- GitHub: [@harshuljain13](https://github.com/harshuljain13)
- Newsletter: [The Engineer's Digest](https://harshuljain.substack.com)

---

<p align="center">
  <em>Labs originate from an LLM inference course; overviews, fixes, and measurement methodology are my own.</em>
</p>
