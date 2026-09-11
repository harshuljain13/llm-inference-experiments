from __future__ import annotations

import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("playground.serve")


def _resolve_backend() -> str:
    raw = os.environ.get("PLAYGROUND_BACKEND", "auto").strip().lower()
    if raw in ("llamaserver", "llama_server", "llama-server"):
        return "llamaserver"
    if raw in ("llamacpp", "llama.cpp", "gguf", "llama_cpp"):
        return "llamacpp"
    if raw == "vllm":
        return "vllm"
    if raw not in ("auto", ""):
        logger.warning("Unknown PLAYGROUND_BACKEND=%r; using auto", raw)
    if os.environ.get("PLAYGROUND_FORCE_LLAMACPP", "").lower() in ("1", "true", "yes", "on"):
        return "llamacpp"
    if os.environ.get("PLAYGROUND_FORCE_LLAMASERVER", "").lower() in ("1", "true", "yes", "on"):
        return "llamaserver"
    try:
        __import__("vllm")
    except ImportError:
        if os.environ.get("LLAMASERVER_BASE_URL", "").strip():
            logger.info(
                "vLLM not importable; using llamaserver HTTP proxy (LLAMASERVER_BASE_URL is set)."
            )
            return "llamaserver"
        logger.info(
            "vLLM not importable; using llama-cpp-python backend "
            "(install requirements-optional-llamacpp-pypi.txt if you have a wheel, "
            "or set LLAMASERVER_BASE_URL + PLAYGROUND_BACKEND=llamaserver)."
        )
        return "llamacpp"
    return "vllm"


def build_application():
    backend = _resolve_backend()
    if backend == "vllm":
        from ray.serve.llm import build_openai_app

        from config import build_llm_config

        llm_config = build_llm_config()
        return build_openai_app({"llm_configs": [llm_config]})

    if backend == "llamaserver":
        from llamaserver_proxy_backend import build_llamaserver_proxy_application

        return build_llamaserver_proxy_application()

    from llamacpp_backend import build_llamacpp_application

    return build_llamacpp_application()


app = build_application()


def main():
    import ray
    from ray import serve

    host = os.environ.get("PLAYGROUND_SERVE_HOST", "0.0.0.0")
    port = int(os.environ.get("PLAYGROUND_SERVE_PORT", "8000"))
    ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
    logger.info("Starting OpenAI-compatible gateway on %s:%s", host, port)
    serve.run(build_application(), blocking=True, host=host, port=port)


if __name__ == "__main__":
    main()
