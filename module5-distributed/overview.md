# Module 5 — Distributed: More Than One GPU

> From the standalone `ray_project` repo. Tensor, pipeline, and data parallelism.

**One sentence:** One GPU is not enough — so split the model across GPUs (tensor / pipeline parallel), replicate it across workers (data parallel), and measure what each split actually buys.

Modules 1–4 all assume a single GPU. This is where that assumption breaks.

---

## The three ways to use more than one GPU

| Strategy | What it splits | Use when | Cost |
|---|---|---|---|
| **Tensor parallel (TP)** | each layer's weights across GPUs | model doesn't fit on one GPU | all-reduce on every layer — needs fast interconnect |
| **Pipeline parallel (PP)** | layers across GPUs | model still doesn't fit | bubbles; stages idle waiting |
| **Data parallel (DP)** | whole replicas | model fits, you need throughput | none — replicas are independent |

**The rule of thumb this module tests:** use TP/PP only when you *must* (memory), use DP when you *can* (throughput). Splitting a model that fits is paying communication cost for nothing.

## File map

| File | Role |
|---|---|
| `serve_app.py` | Ray Serve deployment — the distributed serving entrypoint |
| `serve_config.yaml` | Ray Serve config: replicas, resources |
| `modal_app.py` | vLLM on Modal; `MODAL_TENSOR_PARALLEL_SIZE` / `MODAL_PIPELINE_PARALLEL_SIZE` are the knobs |
| `llamacpp_backend.py` | llama.cpp backend (CPU path) |
| `llamaserver_proxy_backend.py` | proxy to an external llama-server |
| `load_test/run_bench.py` | **the measurement tool** — warmup, aggregate vs per-stream throughput |
| `load_test/variable_concurrency.py` | concurrency sweep — the load ladder |
| `load_test/locustfile.py` | Locust scenarios |
| `profiling/profile_{single_gpu,tp,dp,tp_pp}.sh` | per-strategy profiling runs |
| `docker/`, `docker-compose.yml` | CPU and GPU profiles |

## The metric trap this module exists to teach

`run_bench.py` reports **two** throughput numbers, and confusing them invalidates every conclusion:

```
output_tok_per_sec            = completion_tokens / wall_s      ← aggregate
per_stream_output_tok_per_sec = completion_tokens / Σ latencies ← per-stream
```

- **Aggregate** is what must rise when you add GPUs. It's the number that justifies the spend.
- **Per-stream** is roughly `aggregate / concurrency` and stays *flat* when you scale out — because each individual user doesn't get faster, there are just more of them served at once.

Benchmark data-parallel scaling with the per-stream number and you'll conclude, wrongly, that adding GPUs did nothing.

`--warmup` exists for the same reason: cold start, weight load, and CUDA graph capture land on the first requests. Timing them swamps p99 and understates RPS.

## What to measure

| Lever | Expect | Watch for |
|---|---|---|
| `TP=2` vs `TP=1` (model fits both) | aggregate throughput **↓** | all-reduce overhead with no memory benefit |
| `TP=2` (model doesn't fit on 1) | it runs at all | compare against the PP alternative |
| `PP=2` | latency ↑, throughput modest | pipeline bubbles at low concurrency |
| `DP=2` (2 replicas) | aggregate ≈ **2×**, per-stream flat | the metric trap above |
| concurrency sweep | find the knee | same discipline as every other module |

## Where this sits

```
module4-gateway      one gateway, two replicas, one GPU
      ↓
module5-distributed  many GPUs — split the model, or replicate it
```

Module 4 decided *which* replica gets a request. Module 5 asks how those replicas should be constituted in the first place.

**Handbook:** Ch. 07.1 (Tensor Parallelism), Ch. 07.2 (MoE Inference), Ch. 08.1 (Ray Serve)

---

*Absorbed from the standalone `ray_project` repo; full history preserved via `git subtree`.*
