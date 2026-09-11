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

New here? Start with [module 1](module1-prefill-vs-decode/overview.md).

---

## 🧭 The Experiments

Six modules, bottom-up. Each `overview.md` maps that module's files and explains the concept before you run anything.

| # | Module | Was | What it covers | Question it answers | GPU |
|---|---|---|---|---|---|
| 1 | [**prefill-vs-decode**](module1-prefill-vs-decode/overview.md) | class 1 | prefill vs decode · KV cache · why output tokens cost more | Why is decode slow and prefill fast? | optional |
| 2 | [**server-and-gateway**](module2-server-and-gateway/overview.md) | class 2 | model → engine → server → gateway · why naive servers melt | Which layer owns which problem? | optional |
| 3 | [**build-your-own-engine**](module3-build-your-own-engine/overview.md) | class 5 | paged KV blocks · continuous batching · preemption livelock | How is KV memory paged and work scheduled? | none |
| 4 | [**observability-and-cost**](module4-observability-and-cost/overview.md) | class 3 | Prometheus + Grafana · engine-flag A/B · $ per request | Where does time go, and what does it cost? | yes |
| 5 | [**multi-gpu-scaling**](module5-multi-gpu-scaling/overview.md) | class 6 | tensor / pipeline / data parallel · Ray Serve | Split the model, or replicate it? | yes |
| 6 | [**admission-and-routing**](module6-admission-and-routing/overview.md) | class 7 | admission control · deadline-ordered queue · prefix routing | Who gets in, in what order, on which GPU? | yes |

**Reading order:** measure before you optimize (4), scale out once one GPU isn't enough (5), then govern the fleet you now have (6). Modules 1–3 are CPU-friendly — start there.

Modules 4 and 5 began as the standalone `fullstack-inferencing` and `ray_project` repos, absorbed with `git subtree`. Course class 4 had no lab code. Pre-rename state lives on the `upstream-course` remote, and `git log --follow` tracks any file across the renames:

```bash
git show upstream-course/main:class7/gateway/router.py
git log --follow -- module6-admission-and-routing/gateway/router.py
```

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

**Gateway layer** — `module6-admission-and-routing`

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
