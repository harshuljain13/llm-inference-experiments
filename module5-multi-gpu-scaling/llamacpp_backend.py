from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


def llamacpp_settings() -> Dict[str, Any]:
    path = os.environ.get("LLAMACPP_MODEL_PATH") or os.environ.get("PLAYGROUND_LLAMACPP_GGUF", "")
    return {
        "model_path": path.strip(),
        "n_ctx": int(os.environ.get("LLAMACPP_N_CTX", "4096")),
        "n_gpu_layers": int(os.environ.get("LLAMACPP_N_GPU_LAYERS", "-1")),
        "n_threads": int(os.environ.get("LLAMACPP_N_THREADS", str(os.cpu_count() or 4))),
        "model_id": _env_model_id(),
        "verbose": os.environ.get("LLAMACPP_VERBOSE", "").lower() in ("1", "true", "yes"),
    }


def _env_model_id() -> str:
    return (
        os.environ.get("PLAYGROUND_MODEL_ID")
        or os.environ.get("PLAYGROUND_BENCH_MODEL")
        or os.environ.get("PLAYGROUND_LOADTEST_MODEL")
        or "tinyllama-chat"
    )


def validate_llamacpp_env() -> None:
    s = llamacpp_settings()
    p = s["model_path"]
    if not p:
        raise RuntimeError(
            "llama.cpp backend: set LLAMACPP_MODEL_PATH to a .gguf file "
            "(e.g. TinyLlama-1.1B-Chat Q4_K_M from Hugging Face / TheBloke)."
        )
    if not os.path.isfile(p):
        raise RuntimeError(f"llama.cpp backend: GGUF not found: {p}")


def build_llamacpp_application():
    try:
        __import__("llama_cpp")
    except ImportError as e:
        raise ImportError(
            "llama.cpp backend requires `llama-cpp-python`. Install a prebuilt wheel: "
            "`pip install llama-cpp-python` (see PyPI release files for your OS/arch)."
        ) from e

    from ray import serve

    validate_llamacpp_env()
    settings = llamacpp_settings()

    fastapi_app = FastAPI(title="ray-project LLM playground (llama.cpp)", version="0.1")
    n_replicas = max(1, int(os.environ.get("LLAMACPP_NUM_REPLICAS", "1")))

    @serve.deployment(name="llamacpp_openai", num_replicas=n_replicas)
    @serve.ingress(fastapi_app)
    class LlamaCppOpenAI:
        def __init__(self):
            from llama_cpp import Llama

            mp = settings["model_path"]
            print(
                "[playground] llama.cpp backend: loading GGUF=%s n_gpu_layers=%s n_ctx=%s replicas=%s"
                % (mp, settings["n_gpu_layers"], settings["n_ctx"], n_replicas),
                flush=True,
            )
            self._model_id = settings["model_id"]
            self._llm = Llama(
                model_path=mp,
                n_ctx=settings["n_ctx"],
                n_gpu_layers=settings["n_gpu_layers"],
                n_threads=settings["n_threads"],
                verbose=settings["verbose"],
            )

        @fastapi_app.post("/v1/chat/completions")
        async def chat_completions(self, body: ChatCompletionRequest):
            if body.stream:
                raise HTTPException(
                    status_code=501,
                    detail="Streaming is not implemented for the llama.cpp fallback.",
                )

            def _run():
                return self._llm.create_chat_completion(
                    messages=[m.model_dump() for m in body.messages],
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                )

            out = await asyncio.to_thread(_run)
            if isinstance(out, dict):
                out = dict(out)
                out["model"] = body.model or self._model_id
            return JSONResponse(out)

        @fastapi_app.post("/v1/completions")
        async def completions(self, body: CompletionRequest):
            if body.stream:
                raise HTTPException(status_code=501, detail="Streaming not implemented.")

            def _run():
                return self._llm.create_completion(
                    prompt=body.prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                )

            out = await asyncio.to_thread(_run)
            if isinstance(out, dict):
                out = dict(out)
                out["model"] = body.model or self._model_id
            return JSONResponse(out)

        @fastapi_app.get("/v1/models")
        async def list_models(self):
            return {
                "object": "list",
                "data": [
                    {
                        "id": self._model_id,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "llama-cpp",
                    }
                ],
            }

        @fastapi_app.get("/health")
        async def health(self):
            return {"status": "ok", "backend": "llama.cpp", "model_id": self._model_id}

    return LlamaCppOpenAI.bind()
