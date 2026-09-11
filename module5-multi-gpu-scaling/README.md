# Distributed LLM playground — runbook

```bash
cd /path/to/ray_project
```

---

## 1. Modal (cloud GPU, OpenAI API)

```bash
pip install -r requirements-modal.txt
modal setup
modal deploy modal_app.py
```

The Modal app name is `ray-project-vllm` (see `modal_app.py`). Use the URL printed after `modal deploy` (ends in `.modal.run`) as the API base; HTTP routes are under `/v1/...`. In the examples below, set `MODAL_BASE_URL` to that URL (no trailing slash). Default OpenAI `model` name: `tinyllama-chat`.

Optional before `modal deploy`:

```bash
export MODAL_GPU_TYPE=A10G
export MODAL_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export MODAL_SERVED_MODEL_NAME=tinyllama-chat
```

Gated Hugging Face models:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token
export MODAL_USE_HF_SECRET=1
modal deploy modal_app.py
```

Smoke test:

```bash
modal run modal_app.py
```

---

## 2. Docker — CPU (`llama-server` + Ray on :8000)

```bash
mkdir -p models
cp /your/real/file.gguf models/model.gguf
docker compose --profile cpu up --build
```

- API via Ray: `http://127.0.0.1:8000/v1/...`
- Direct llama: `http://127.0.0.1:8080`

---

## 3. Docker — GPU (Ray + vLLM, needs NVIDIA Container Toolkit)

```bash
docker compose --profile gpu up --build
```

API: `http://127.0.0.1:8000/v1/...`

Optional:

```bash
export PLAYGROUND_PRESET=tiny_tp
export PLAYGROUND_MODEL_KEY=tinyllama
docker compose --profile gpu up --build
```

---

## 4. macOS — prebuilt `llama-server` + Ray (two terminals)

Terminal A:

```bash
chmod +x scripts/download_llamaserver_macos.sh
./scripts/download_llamaserver_macos.sh
LLAMA="$(find "$(pwd)/third_party/llama.cpp-bin" -type f -name llama-server | head -1)"
"$LLAMA" -m /path/to/your.gguf --host 127.0.0.1 --port 8080
```

Terminal B:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PLAYGROUND_BACKEND=llamaserver
export LLAMASERVER_BASE_URL=http://127.0.0.1:8080
serve run serve_config.yaml
```

---

## 5. Linux — Ray + vLLM (local GPU)

Python **3.10–3.12** recommended.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PLAYGROUND_BACKEND=vllm
export PLAYGROUND_PRESET=tiny_single
export PLAYGROUND_MODEL_KEY=tinyllama
serve run serve_config.yaml
```

---

## Exploring parallelism (where it actually lives)

| Path | What you can change | Notes |
|------|---------------------|--------|
| **Modal** (`modal_app.py`) | **Tensor parallel (TP)** and **pipeline parallel (PP)** across `MODAL_GPU_COUNT` GPUs | One vLLM process. Require `MODAL_TENSOR_PARALLEL_SIZE * MODAL_PIPELINE_PARALLEL_SIZE == MODAL_GPU_COUNT`. Default is all GPUs in TP, PP=1. **`@modal.concurrent`** + vLLM batching handle many in-flight requests; that is not the same as Ray **data parallel** replicas. |
| **Ray + vLLM** (Linux GPU, `serve run serve_config.yaml`) | **DP** (`num_replicas`), **TP/PP/EP** via `config.py` presets and `PLAYGROUND_PRESET` | This is the main playground for comparing `tiny_dp`, `tiny_tp`, `tiny_pp`, `tiny_tp_pp`, MoE **EP**, etc. Use **Docker `--profile gpu`** on a Linux host with NVIDIA, or a cloud GPU box. |
| **CPU / `llama-server`** | Mostly single-process | Good for API and load-shape experiments, not multi-GPU vLLM-style TP/PP. |

Example (Modal, 4 GPUs split 2×2):

```bash
export MODAL_GPU_COUNT=4
export MODAL_TENSOR_PARALLEL_SIZE=2
export MODAL_PIPELINE_PARALLEL_SIZE=2
modal deploy modal_app.py
```

Then re-run **`run_bench.py`** / Locust and compare latency/throughput to your 1-GPU deploy.

---

## Check that it responds

Local:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fake-key" \
  -d '{"model":"tinyllama-chat","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

Modal (deployed playground):

```bash
MODAL_BASE_URL='https://YOUR-WORKSPACE--YOUR-APP.modal.run'
curl -s "${MODAL_BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer fake-key" \
  -d '{"model":"tinyllama-chat","messages":[{"role":"user","content":"Hi"}],"max_tokens":32}'
```

---

## Load test (optional)

```bash
pip install -r requirements.txt
export PLAYGROUND_LOADTEST_MODEL=tinyllama-chat
locust -f load_test/locustfile.py --host=http://127.0.0.1:8000
```

```bash
export PLAYGROUND_BENCH_MODEL=tinyllama-chat
python load_test/run_bench.py --url http://127.0.0.1:8000 --concurrency 16 --requests 200
```

Against the deployed Modal app:

```bash
export MODAL_BASE_URL='https://YOUR-WORKSPACE--YOUR-APP.modal.run'
export PLAYGROUND_LOADTEST_MODEL=tinyllama-chat
locust -f load_test/locustfile.py --host="${MODAL_BASE_URL}"

export PLAYGROUND_BENCH_MODEL=tinyllama-chat
python load_test/run_bench.py --url "${MODAL_BASE_URL}" --concurrency 8 --requests 50
```

---

## Next steps (pick in order)

1. **Modal:** `pip install -r requirements-modal.txt` → `modal deploy modal_app.py` (after any `modal_app.py` change).
2. **Smoke:** run the **Modal** `curl` in [Check that it responds](#check-that-it-responds) and confirm JSON with `choices`.
3. **Load:** run **Locust** or **`run_bench.py`** against your Modal URL (same section).
4. **Ray + parallelism (Linux/GPU or cloud):** `pip install -r requirements.txt`, set `PLAYGROUND_PRESET` / `PLAYGROUND_MODEL_KEY`, `serve run serve_config.yaml` — see `config.py` presets (`tiny_tp`, `tiny_dp`, …).
5. **Profiling (Linux + GPU):** `./profiling/profile_single_gpu.sh` (with `serve` running); fill `report_template.md` with numbers.

**Mac without Docker running:** use **Modal** or **section 4** (llama-server + Ray), not `docker compose --profile gpu` (no NVIDIA GPU in Docker on Apple Silicon).

---

## Where to change settings

| What | File / env |
|------|------------|
| Ray bind host/port | `serve_config.yaml` → `http_options` |
| vLLM parallelism (TP/DP/PP presets) | `config.py` presets + `PLAYGROUND_*` env vars |
| Modal model / GPU / TP–PP | `modal_app.py` or `MODAL_*` (`MODAL_GPU_COUNT`, `MODAL_TENSOR_PARALLEL_SIZE`, `MODAL_PIPELINE_PARALLEL_SIZE`) |
| Nsight scripts | `profiling/*.sh` |
| Report template | `report_template.md` |

Full API details: [Ray Serve LLM](https://docs.ray.io/en/latest/serve/llm/quick-start.html), [vLLM](https://docs.vllm.ai/), [Modal vLLM example](https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/llm-serving).
