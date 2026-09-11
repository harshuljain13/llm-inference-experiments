# Distributed LLM Playground — Performance Report

**Date:**  
**Cluster:** (e.g. Anyscale workspace, instance types, GPU count)  
**Ray / vLLM / CUDA:** (versions from `ray --version`, cluster image)  
**Model:** (`PLAYGROUND_MODEL_KEY`, Hugging Face id)  
**Preset:** (`PLAYGROUND_PRESET`)

## 1. Configuration matrix

| Run ID | Preset | Replicas (DP) | TP | PP | EP | Quantization | Notes |
|--------|--------|---------------|----|----|----|--------------|-------|
| A | tiny_single | 1 | 1 | 1 | off | none | Baseline |
| B | tiny_dp | 2+ | 1 | 1 | off | none | Ray horizontal |
| C | tiny_tp | 1 | 2–4 | 1 | off | none | All-reduce comms |
| D | tiny_tp_pp | 1 | 2 | 2 | off | none | Pipeline bubbles |
| E | llm_70b_tp_awq | 1 | 4–8 | 1 | off | AWQ | Memory / throughput |

## 2. Methodology

- **Warmup:** (e.g. 10 requests discarded)  
- **Load generator:** Locust / `load_test/run_bench.py` (concurrency, total requests)  
- **Metrics collected:** p50/p99 latency, aggregate RPS, completion tokens/sec (from API `usage`), GPU util (DCGM / `nvidia-smi dmon` / Nsight)  
- **Nsight report paths:** (`.nsys-rep` files per scenario)

## 3. Results

### 3.1 Latency (seconds)

| Run | p50 | p99 | Notes |
|-----|-----|-----|-------|
| A | | | |
| B | | | |

### 3.2 Throughput

| Run | Aggregate RPS | Output tok/s (approx) | GPU util (avg) |
|-----|-----------------|------------------------|----------------|
| A | | | |

## 4. Nsight / GPU observations

- **Single GPU:** Kernel timeline, CUDA graphs, batching gaps  
- **DP:** Multiple independent GPU timelines; Ray scheduling overhead  
- **TP:** NCCL / all-reduce segments vs compute  
- **TP+PP:** Stage idle time (pipeline bubbles), chunking / prefill  
- **KV cache / memory:** pressure points at 70B scale  

## 5. Bottlenecks (checklist)

- [ ] Ray Serve queueing / `max_ongoing_requests`  
- [ ] Ingress router saturation (`PLAYGROUND_NUM_INGRESS_REPLICAS`)  
- [ ] TP communication (dominates for small models — “overkill” demo)  
- [ ] PP bubbles  
- [ ] KV cache growth / `max_model_len`  
- [ ] Disk / HF download / checkpoint I/O (first load)

## 6. Stretch goals status

- [ ] Disaggregated prefill/decode (vLLM / Ray Serve LLM version capabilities)  
- [ ] Replica kill / recovery demo  
- [ ] Variable concurrency traffic simulator  
- [ ] MoE + EP (`moe_ep` + Mixtral)

## 7. Conclusions

(3–5 bullets: what scaled, what did not, next experiment.)
