# Module 6 — Overview

**One sentence:** Wire a full production stack — CrewAI → nginx → gateway → vLLM on a real GPU — then instrument it end to end with Prometheus and Grafana, and A/B the engine flags that actually move the numbers.

Every other module measures with a purpose-built script. This one measures the way production does: **continuously, from the outside, through dashboards you didn't write for one experiment.**

---

## The stack

```
CrewAI (laptop)
   ↓
nginx :8780          load balancer
   ↓
FastAPI gateway :8765    metrics, cost accounting
   ↓  SSH tunnel
vLLM :8000-8005      Lambda GPU — one port per engine profile
   ↓
Prometheus → Grafana     scrape, store, visualize
```

The SSH tunnel matters: it means the *gateway runs on your laptop* while the GPU is remote. That split is what makes the `UPSTREAM_DURATION` metric below necessary.

## File map

| File | Role |
|---|---|
| `gateway.py` | **The instrumented gateway** (~1,100 lines). Prometheus metrics, cost accounting, routing. |
| `crew.py` | CrewAI client — the agentic workload generator |
| `lambda_pricing.py` | GPU $/hour → per-request cost attribution |
| `monitoring/docker-compose.yml` | Prometheus + Grafana + Jaeger |
| `monitoring/prometheus.yml` | scrape config |
| `monitoring/nginx-gateway-lb.conf` | load balancer in front of the gateway |
| `monitoring/grafana_dashboards/` | 5 dashboards — overview, gateway-proxy, technique-cost, tinyllama-ops, fullstack |
| `scripts/vllm_engine/*.sh` | **engine profiles** — one per flag combination |
| `scripts/run_server_ab.sh`, `ab_arms.sh` | A/B harness that drives those profiles |

## The engine A/B experiments

`scripts/vllm_engine/` is a set of launchers, each starting vLLM with one flag combination. `ab_arms.sh` drives them as arms of an A/B test.

| Profile | Flag under test | Expected effect |
|---|---|---|
| `baseline.sh` | — | control |
| `baseline_strict.sh` | features explicitly disabled | hard control |
| `chunked_prefill.sh` | chunked prefill | long prompts stop blocking decode → interactive p99 ↓ |
| `prefix_caching.sh` | prefix caching | shared prefixes → TTFT ↓ |
| `chunked_prefill_and_prefix_caching.sh` | both | do they compose, or interfere? |
| `speculative_decoding.sh` | draft model | ITL ↓ if acceptance rate is high; ↑ if not |
| `run_engine_fleet.sh` | all at once, ports 8000–8005 | side-by-side under one load generator |

> These stay in this module rather than moving to `module3-engine` because the A/B harness here *is* what drives them. Module 3 builds these mechanisms from scratch on CPU; module 6 measures the real vLLM implementations of the same ideas.

## Metrics worth understanding

`gateway.py` exports a Prometheus surface designed around three questions:

| Metric | Answers |
|---|---|
| `llm_gateway_inflight_requests` | how much work is *resident* — the middle of the concurrency funnel |
| `llm_gateway_upstream_duration_seconds` | time awaiting the engine |
| `llm_gateway_request_duration_seconds` | total end-to-end |
| `llm_gateway_estimated_gpu_cost_usd_total` | $ attributed per request |
| `llm_gateway_completion_tokens_total` | output volume |

**`total − upstream` isolates gateway + tunnel overhead.** Without that subtraction, a slow gateway and a slow engine look identical on a latency graph — the single most common misdiagnosis in a tunneled setup.

The inflight gauge matters for the same reason: arrivals and completions alone can't tell you whether the system is saturated or just idle between bursts.

## Cost

`lambda_pricing.py` turns GPU $/hour into per-request dollars, and `technique-cost.json` graphs it. This is the only module that closes the loop from **latency → throughput → dollars** — which is ultimately the argument any of this has to win.

## What to measure

| Lever | Primary | Watch for |
|---|---|---|
| chunked prefill on | interactive p99 ↓ | only under mixed prompt sizes |
| prefix caching on | TTFT ↓ | needs genuinely shared prefixes |
| both on | compose? | may contend for the same budget |
| speculative decoding | ITL ↓ | **↑** if draft acceptance is low — measure, don't assume |
| nginx replicas ↑ | throughput ↑ | gateway becomes the bottleneck |

## Where this sits

```
modules 1-5    build and measure each layer
      ↓
module6        run the whole stack and watch it in production terms
```

**Handbook:** Ch. 05.3 (Continuous Batching), Ch. 09 (Benchmarking & Observability), Ch. 11.5 (Agentic Workload)

---

*Absorbed from the standalone `fullstack-inferencing` repo; full history preserved via `git subtree`.*
