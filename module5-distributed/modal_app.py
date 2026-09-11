from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import modal

MODEL_NAME = os.environ.get("MODAL_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SERVED_MODEL_NAME = os.environ.get("MODAL_SERVED_MODEL_NAME", "tinyllama-chat")
MODAL_GPU_TYPE = os.environ.get("MODAL_GPU_TYPE", "T4")
MODAL_GPU_COUNT = int(os.environ.get("MODAL_GPU_COUNT", "1"))
MODAL_TENSOR_PARALLEL_SIZE = int(
    os.environ.get("MODAL_TENSOR_PARALLEL_SIZE", str(MODAL_GPU_COUNT))
)
MODAL_PIPELINE_PARALLEL_SIZE = int(os.environ.get("MODAL_PIPELINE_PARALLEL_SIZE", "1"))
if MODAL_TENSOR_PARALLEL_SIZE * MODAL_PIPELINE_PARALLEL_SIZE != MODAL_GPU_COUNT:
    raise ValueError(
        "Modal vLLM requires MODAL_TENSOR_PARALLEL_SIZE * MODAL_PIPELINE_PARALLEL_SIZE "
        f"== MODAL_GPU_COUNT (got {MODAL_TENSOR_PARALLEL_SIZE} * "
        f"{MODAL_PIPELINE_PARALLEL_SIZE} != {MODAL_GPU_COUNT}). "
        "Example: 4 GPUs → TP=2 and PP=2, or TP=4 and PP=1."
    )
FAST_BOOT = os.environ.get("MODAL_FAST_BOOT", "1").lower() in ("1", "true", "yes")
MAX_MODEL_LEN = int(os.environ.get("MODAL_MAX_MODEL_LEN", "2048"))
VLLM_PORT = 8000
MINUTES = 60

# MODAL_* vars are read on the laptop AND again when the container re-imports this
# module. They do not propagate on their own, so bake the ones we set into the image:
# otherwise TP/PP silently fall back to defaults in the container, and the conditional
# secret below makes the local/remote dependency counts disagree.
_PLAYGROUND_ENV_KEYS = (
    "MODAL_MODEL_NAME",
    "MODAL_SERVED_MODEL_NAME",
    "MODAL_GPU_TYPE",
    "MODAL_GPU_COUNT",
    "MODAL_TENSOR_PARALLEL_SIZE",
    "MODAL_PIPELINE_PARALLEL_SIZE",
    "MODAL_FAST_BOOT",
    "MODAL_MAX_MODEL_LEN",
    "MODAL_ALLOW_LONG_MAX_MODEL_LEN",
    "MODAL_USE_HF_SECRET",
)
_baked_env = {k: os.environ[k] for k in _PLAYGROUND_ENV_KEYS if os.environ.get(k, "").strip()}

vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", **_baked_env})
)

hf_cache_vol = modal.Volume.from_name("playground-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("playground-vllm-cache", create_if_missing=True)

app = modal.App("ray-project-vllm")

_USE_HF_SECRET = os.environ.get("MODAL_USE_HF_SECRET", "0").lower() in ("1", "true", "yes")
_MODAL_SECRETS = (
    [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]
    if _USE_HF_SECRET
    else []
)


def _gpu_spec() -> str:
    return f"{MODAL_GPU_TYPE}:{MODAL_GPU_COUNT}"


@app.function(
    image=vllm_image,
    gpu=_gpu_spec(),
    timeout=15 * MINUTES,
    scaledown_window=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=_MODAL_SECRETS,
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=VLLM_PORT, startup_timeout=15 * MINUTES)
def serve():
    if os.environ.get("MODAL_ALLOW_LONG_MAX_MODEL_LEN", "").lower() in ("1", "true", "yes"):
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    cmd: list[str] = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(MODAL_TENSOR_PARALLEL_SIZE),
        "--pipeline-parallel-size",
        str(MODAL_PIPELINE_PARALLEL_SIZE),
        "--uvicorn-log-level",
        "info",
    ]
    if FAST_BOOT:
        cmd.append("--enforce-eager")
    else:
        cmd.append("--no-enforce-eager")

    print("Starting:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test():
    import aiohttp

    url = await serve.get_web_url.aio()
    print("URL:", url)
    async with aiohttp.ClientSession(base_url=url) as session:
        async with session.get("/health", timeout=aiohttp.ClientTimeout(total=600)) as resp:
            print("GET /health", resp.status)
            resp.raise_for_status()
        payload: dict[str, Any] = {
            "model": SERVED_MODEL_NAME,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 32,
        }
        async with session.post("/v1/chat/completions", json=payload, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            print("POST /v1/chat/completions", resp.status)
            body = await resp.text()
            print(body[:800])
