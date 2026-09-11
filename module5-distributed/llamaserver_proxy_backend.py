from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)


def _upstream_base() -> str:
    base = os.environ.get("LLAMASERVER_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError(
            "Set LLAMASERVER_BASE_URL to the llama-server root, e.g. http://127.0.0.1:8080"
        )
    return base


def build_llamaserver_proxy_application():
    from ray import serve

    upstream = _upstream_base()
    print("[playground] llamaserver HTTP proxy → %s" % upstream, flush=True)

    fastapi_app = FastAPI(title="playground → llama-server proxy", version="0.1")
    timeout = float(os.environ.get("LLAMASERVER_HTTP_TIMEOUT", "600"))

    @serve.deployment(name="llamaserver_proxy", num_replicas=1)
    @serve.ingress(fastapi_app)
    class LlamaServerProxy:
        @fastapi_app.get("/health")
        async def health(self):
            return {"status": "ok", "backend": "llamaserver-proxy", "upstream": upstream}

        @fastapi_app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
        async def forward_v1(self, request: Request, path: str):
            import httpx

            url = f"{upstream}/v1/{path}"
            if request.url.query:
                url = f"{url}?{request.url.query}"
            hop = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "content-length", "connection")
            }
            content: bytes | None = None
            if request.method in ("POST", "PUT", "PATCH"):
                content = await request.body()

            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.request(request.method, url, headers=hop, content=content)

            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type"),
            )

    return LlamaServerProxy.bind()
