from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

_ROOT = Path(__file__).resolve().parent
_ENV_FILE = _ROOT / ".env"


def _reload_dotenv() -> None:
    load_dotenv(_ENV_FILE, override=True)


_reload_dotenv()

_log = logging.getLogger("gateway")

_metrics_http_lock = threading.Lock()
_metrics_http_started = False


def _ensure_prometheus_http_server() -> None:
    global _metrics_http_started
    with _metrics_http_lock:
        if _metrics_http_started:
            return
        from prometheus_client import start_http_server

        start_http_server(GATEWAY_METRICS_PORT)
        _metrics_http_started = True
        _log.info(
            "Prometheus scrape URL (for Docker Prometheus): http://127.0.0.1:%s/metrics",
            GATEWAY_METRICS_PORT,
        )


def _vllm_base_url() -> str:
    _reload_dotenv()
    return os.environ.get("VLLM_BASE_URL", "").strip().rstrip("/")
VLLM_SERVER_PROFILE = (os.environ.get("VLLM_SERVER_PROFILE", "") or "unset").strip().lower()
_ENGINE_TECHNIQUE_PORT_OFFSETS: dict[str, int] = {
    "baseline": 0,
    "beam_search": 0,
    "chunked_prefill": 1,
    "prefix_caching": 2,
    "chunked_prefill_and_prefix_caching": 3,
    "baseline_strict": 4,
    "speculative_decoding": 5,
}
GPU_HOURLY_COST_USD = float(os.environ.get("GPU_HOURLY_COST_USD", "2.5"))
GATEWAY_METRICS_PORT = int(os.environ.get("GATEWAY_METRICS_PORT", "9101"))
OTEL_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.environ.get(
    "OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4317"
)
SERVICE_NAME = os.environ.get("OTEL_SERVICE_NAME", "gateway")


def _otlp_traces_enabled() -> bool:
    v = os.environ.get("OTEL_TRACES_EXPORTER", "none").strip().lower()
    return v in ("otlp", "grpc", "1", "true", "yes")


def _otlp_grpc_hostport(url: str) -> str:
    u = url.strip()
    for p in ("http://", "https://", "grpc://"):
        if u.startswith(p):
            u = u[len(p) :]
    return u.split("/")[0].split("?")[0]


resource = Resource.create({"service.name": SERVICE_NAME})
provider = TracerProvider(resource=resource)
if _otlp_traces_enabled():
    try:
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=_otlp_grpc_hostport(OTEL_ENDPOINT), insecure=True)
            )
        )
    except Exception:
        pass
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)
HTTPXClientInstrumentor().instrument()

_MLABELS = ["technique", "server_profile"]

REQUEST_DURATION = Histogram(
    "llm_gateway_request_duration_seconds",
    "End-to-end gateway→vLLM time (non-streaming or first byte not split)",
    _MLABELS,
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60),
)
REQUEST_COST_USD = Counter(
    "llm_gateway_estimated_gpu_cost_usd_total",
    "Estimated GPU $ = (duration_s/3600) * hourly_rate (env GPU_HOURLY_COST_USD or Lambda API if configured)",
    _MLABELS,
)
PROMPT_TOKENS = Counter(
    "llm_gateway_prompt_tokens_total",
    "Prompt tokens reported by vLLM usage",
    _MLABELS,
)
COMPLETION_TOKENS = Counter(
    "llm_gateway_completion_tokens_total",
    "Completion tokens reported by vLLM usage",
    _MLABELS,
)
INFLIGHT_REQUESTS = Gauge(
    "llm_gateway_inflight_requests",
    "Requests currently in flight through the gateway (middle of the concurrency funnel)",
    _MLABELS,
)
UPSTREAM_DURATION = Histogram(
    "llm_gateway_upstream_duration_seconds",
    "Time spent awaiting the upstream engine; total minus this = gateway + tunnel overhead",
    _MLABELS,
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 60),
)
WASTED_COMPLETION_TOKENS = Counter(
    "llm_gateway_wasted_completion_tokens_total",
    "Completion tokens generated for streams the client abandoned — GPU work paid for and discarded",
    _MLABELS,
)
REQUEST_OUTCOMES = Counter(
    "llm_gateway_request_outcomes_total",
    "Requests by outcome class — the error-rate signal (ok|client_error|server_error|upstream_unreachable)",
    _MLABELS + ["outcome", "status_code"],
)
REQUESTS_TOTAL = Counter(
    "llm_gateway_requests_total",
    "Completed OpenAI proxy requests (non-streaming path increments once)",
    _MLABELS,
)
TTFT_SECONDS = Histogram(
    "llm_gateway_time_to_first_token_seconds",
    "Time from proxy send to first non-empty upstream byte (streaming); non-stream ≈ full response time",
    _MLABELS,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1, 2, 5, 15),
)
STREAM_INTER_CHUNK_SECONDS = Histogram(
    "llm_gateway_stream_inter_chunk_delay_seconds",
    "Wall time between successive non-empty upstream byte chunks (streaming only)",
    _MLABELS,
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
)
TPOT_SECONDS = Histogram(
    "llm_gateway_time_per_output_token_seconds",
    "Streaming: mean inter-chunk delay; non-stream: e2e/max(completion_tokens,1)",
    _MLABELS,
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1, 5),
)
COMPLETION_TOKENS_PER_SECOND = Histogram(
    "llm_gateway_completion_tokens_per_second",
    "completion_tokens / e2e_s when both known (non-stream or parsed SSE usage)",
    _MLABELS,
    buckets=(1, 5, 10, 20, 50, 100, 200, 500, 1000),
)
GATEWAY_METRICS_INFO = Info(
    "llm_gateway_info",
    "Gateway build metadata; extended_timing=true means TTFT / inter-chunk / TPOT / tok/s histograms are registered",
)
GATEWAY_METRICS_INFO.info({"extended_timing": "true", "schema": "v2"})


def _drop_builtin_python_export_collectors() -> None:
    for coll in list(getattr(REGISTRY, "_collector_to_names", {}).keys()):
        if coll.__class__.__name__ in ("GCCollector", "PlatformCollector"):
            try:
                REGISTRY.unregister(coll)
            except KeyError:
                pass


_drop_builtin_python_export_collectors()


def _metrics_summary_payload() -> dict[str, Any]:
    gateway_build: dict[str, str] = {}
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for mf in REGISTRY.collect():
        name = mf.name
        if name == "llm_gateway_info" and mf.type == "info":
            for s in mf.samples:
                gateway_build.update(dict(s.labels))
            continue
        if not name.startswith("llm_gateway_") or name.endswith("_created"):
            continue

        if mf.type == "counter":
            for s in mf.samples:
                labels = dict(s.labels) if s.labels else {}
                if "technique" not in labels:
                    continue
                key = (labels["technique"], labels.get("server_profile", "?"))
                g = groups.setdefault(
                    key,
                    {"technique": key[0], "server_profile": key[1]},
                )
                short = name.removeprefix("llm_gateway_")
                g[short] = float(s.value)
        elif any(s.name.endswith("_bucket") for s in mf.samples):
            for s in mf.samples:
                if not s.name.endswith("_count") and not s.name.endswith("_sum"):
                    continue
                labels = dict(s.labels) if s.labels else {}
                if "technique" not in labels:
                    continue
                key = (labels["technique"], labels.get("server_profile", "?"))
                g = groups.setdefault(
                    key,
                    {"technique": key[0], "server_profile": key[1]},
                )
                base = name.removeprefix("llm_gateway_")
                if s.name.endswith("_count"):
                    g[f"{base}_count"] = int(s.value)
                elif s.name.endswith("_sum"):
                    g[f"{base}_sum"] = round(float(s.value), 6)

    return {
        "about": {
            "prometheus_text_format": (
                "The /metrics page follows the Prometheus exposition spec: every metric family has "
                "# HELP and # TYPE lines, and histograms expose one line per bucket — that repetition is normal."
            ),
            "vllm_engine_metrics": (
                "vLLM’s own metrics (prefill/decode, KV cache, etc.) are served by the vLLM process, "
                "not this gateway. With the README SSH tunnel they are at http://127.0.0.1:8000/metrics "
                "(Prometheus job vllm_tunnel). Console log lines from vLLM are separate from Prometheus."
            ),
        },
        "gateway_build": gateway_build,
        "per_technique_profile": sorted(
            groups.values(),
            key=lambda r: (r.get("server_profile", ""), r.get("technique", "")),
        ),
    }


def _metrics_summary_as_text(data: dict[str, Any]) -> str:
    lines = [
        "LLM gateway — readable metrics summary",
        "(Raw Prometheus: GET /metrics on port " + str(GATEWAY_METRICS_PORT) + " and on the gateway API port)",
        "",
    ]
    lines.append("Gateway build: " + json.dumps(data.get("gateway_build") or {}, sort_keys=True))
    lines.append("")
    for row in data.get("per_technique_profile") or []:
        lines.append(
            f"--- technique={row.get('technique')}  server_profile={row.get('server_profile')} ---"
        )
        for k in sorted(x for x in row if x not in ("technique", "server_profile")):
            lines.append(f"  {k}: {row[k]}")
        lines.append("")
    lines.append(data["about"]["vllm_engine_metrics"])
    return "\n".join(lines).rstrip() + "\n"


def _metrics_summary_as_html(data: dict[str, Any]) -> str:
    rows_html = ""
    for row in data.get("per_technique_profile") or []:
        cells = "".join(
            f"<tr><td><code>{k}</code></td><td>{row[k]}</td></tr>"
            for k in sorted(x for x in row if x not in ("technique", "server_profile"))
        )
        rows_html += (
            f"<h3>technique=<code>{row.get('technique')}</code> "
            f"server_profile=<code>{row.get('server_profile')}</code></h3>"
            f"<table border='1' cellpadding='6' cellspacing='0'>{cells}</table>"
        )
    about = data.get("about") or {}
    ptxt = about.get("prometheus_text_format", "")
    vtxt = about.get("vllm_engine_metrics", "")
    gb = json.dumps(data.get("gateway_build") or {}, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Gateway metrics summary</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:1rem auto;}}
code{{background:#f4f4f4;padding:0 4px;}}</style></head><body>
<h1>LLM gateway metrics (readable)</h1>
<p>{ptxt}</p>
<p><strong>vLLM engine metrics:</strong> {vtxt}</p>
<p>Gateway build: <code>{gb}</code></p>
{rows_html}
<p><a href="?format=json">JSON</a> · <a href="?format=text">plain text</a> · <a href="/metrics">raw Prometheus</a></p>
</body></html>"""


def _estimate_cost_usd(duration_s: float, app: FastAPI) -> float:
    rate = float(getattr(app.state, "gpu_hourly_usd", GPU_HOURLY_COST_USD))
    return (duration_s / 3600.0) * rate


def _resolve_technique(request: Request, body: dict[str, Any] | None) -> str:
    h = request.headers.get("x-technique") or request.headers.get("X-Technique")
    if h:
        return h.strip().lower()
    if body and isinstance(body.get("metadata"), dict):
        m = body["metadata"].get("technique")
        if isinstance(m, str):
            return m.strip().lower()
    return "baseline"


def _parse_backend_map(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().rstrip("/") for k, v in data.items() if v}


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "on")


def _url_with_port(base_url: str, new_port: int) -> str:
    p = urlparse(base_url.strip().rstrip("/"))
    scheme = p.scheme or "http"
    host = p.hostname
    if not host:
        return ""
    host_fmt = f"[{host}]" if ":" in host else host
    if p.username is not None:
        auth = p.username
        if p.password:
            auth += f":{p.password}"
        netloc = f"{auth}@{host_fmt}:{new_port}"
    else:
        netloc = f"{host_fmt}:{new_port}"
    path = p.path or ""
    out = urlunparse((scheme, netloc, path, "", "", "")).rstrip("/")
    return out


def _default_base_port(base_url: str) -> int | None:
    p = urlparse(base_url.strip())
    if not p.hostname:
        return None
    if p.port is not None:
        return int(p.port)
    scheme = p.scheme or "http"
    return 443 if scheme == "https" else 8000


def _auto_engine_routes_from_base(base_url: str) -> dict[str, str]:
    if not base_url:
        return {}
    bp = _default_base_port(base_url)
    if bp is None:
        return {}
    out: dict[str, str] = {}
    for tech, delta in _ENGINE_TECHNIQUE_PORT_OFFSETS.items():
        u = _url_with_port(base_url, bp + delta)
        if u:
            out[tech] = u
    return out


def _effective_backend_map() -> dict[str, str]:
    _reload_dotenv()
    raw = os.environ.get("VLLM_BACKEND_MAP_JSON", "").strip()
    explicit = _parse_backend_map(raw)
    if not _env_truthy("VLLM_AUTO_ENGINE_ROUTING"):
        return explicit
    base = _vllm_base_url()
    auto = _auto_engine_routes_from_base(base)
    merged = dict(auto)
    merged.update(explicit)
    return merged


def _upstream_base(technique: str) -> str:
    m = _effective_backend_map()
    t = technique.lower()
    if t in m and m[t]:
        return m[t].rstrip("/")
    return _vllm_base_url()


def _metric_lp(technique: str) -> dict[str, str]:
    return {"technique": technique, "server_profile": VLLM_SERVER_PROFILE}


def _record_outcome(lp: dict[str, str], status_code: int) -> None:
    """Classify a completed request so error rate is computable.

    status_code 0 is reserved for 'never reached upstream' (connect/transport failure),
    which is otherwise invisible: those paths return early without touching any counter.
    """
    if status_code == 0:
        outcome = "upstream_unreachable"
    elif status_code >= 500:
        outcome = "server_error"
    elif status_code >= 400:
        outcome = "client_error"
    else:
        outcome = "ok"
    REQUEST_OUTCOMES.labels(**lp, outcome=outcome, status_code=str(status_code)).inc()


def _parse_openai_sse_usage(buffer: bytes) -> tuple[int, int]:
    if not buffer:
        return 0, 0
    text = buffer.decode(errors="replace")
    pt, ct = 0, 0
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            s = line.strip()
            if not s.startswith("data:"):
                continue
            data = s[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            u = obj.get("usage")
            if not isinstance(u, dict):
                continue
            try:
                p = int(u.get("prompt_tokens") or 0)
                c = int(u.get("completion_tokens") or 0)
            except (TypeError, ValueError):
                continue
            if p or c:
                pt, ct = p, c
    return pt, ct


def _smol_style_row(
    *,
    ttft_s: float,
    e2e_s: float,
    tpot_avg_s: float,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, float | int]:
    return {
        "prompt_tokens_total": prompt_tokens,
        "generation_tokens_total": completion_tokens,
        "time_to_first_token_avg_ms": ttft_s * 1000.0,
        "tpot_avg_ms": tpot_avg_s * 1000.0,
        "e2e_request_latency_avg_ms": e2e_s * 1000.0,
        "prompt_len_avg": float(prompt_tokens),
        "prefill_latency_avg_ms": ttft_s * 1000.0,
        "decode_latency_avg_ms": max(0.0, e2e_s - ttft_s) * 1000.0,
    }


def _build_metrics_log_row(
    *,
    technique: str,
    upstream_path: str,
    streaming: bool,
    status_code: int,
    trace_id: str,
    e2e_s: float,
    ttft_s: float,
    tpot_avg_s: float,
    inter_chunk_delays_s: list[float],
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    inter_ms = [d * 1000.0 for d in inter_chunk_delays_s]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "technique": technique,
        "server_profile": VLLM_SERVER_PROFILE,
        "upstream_path": upstream_path,
        "streaming": streaming,
        "status_code": status_code,
        "trace_id": trace_id,
        "e2e_latency_s": e2e_s,
        "time_to_first_token_s": ttft_s,
        "tpot_avg_s": tpot_avg_s,
        "inter_chunk_count": len(inter_chunk_delays_s),
        "inter_chunk_delay_avg_ms": (
            sum(inter_ms) / len(inter_ms) if inter_ms else 0.0
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "smol_style": _smol_style_row(
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            tpot_avg_s=tpot_avg_s,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    }


async def _append_gateway_metrics_log(app: FastAPI, row: dict[str, Any]) -> None:
    d = getattr(app.state, "metrics_log_dir", None)
    if not d:
        return
    lock = getattr(app.state, "metrics_log_lock", None)
    if lock is None:
        return
    async with lock:
        path = Path(d)
        path.mkdir(parents=True, exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fpath = path / f"gateway_metrics_{day}.jsonl"

        def _write() -> None:
            with open(fpath, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)


def _observe_tpot_and_tok_s(
    lp: dict[str, str],
    *,
    tpot_s: float,
    e2e_s: float,
    completion_tokens: int,
) -> None:
    if tpot_s > 0:
        TPOT_SECONDS.labels(**lp).observe(tpot_s)
    if e2e_s > 0 and completion_tokens > 0:
        COMPLETION_TOKENS_PER_SECOND.labels(**lp).observe(completion_tokens / e2e_s)


def _tls_verify() -> bool:
    return os.environ.get("VLLM_TLS_VERIFY", "true").lower() in ("1", "true", "yes")


def _is_vllm_placeholder_url(url: str) -> bool:
    if not url:
        return False
    try:
        host = (urlparse(url).hostname or url).lower()
    except Exception:
        host = url.lower()
    return "your_workspace" in host or "your-workspace" in host


def _is_web_console_not_api_url(url: str) -> bool:
    if not url:
        return False
    try:
        p = urlparse(url)
        h = (p.hostname or "").lower()
        if h in ("modal.com", "www.modal.com"):
            return True
    except Exception:
        pass
    return False


def _plaintext_upstream_error_response(
    status_code: int, content: bytes, trace_id: str
) -> JSONResponse | None:
    raw = (content or b"")[:4096].decode(errors="replace").strip()
    if not raw.lower().startswith("modal-http:"):
        return None
    hint = (
        "Upstream returned a plain-text busy/cold-start line instead of JSON. "
        "Wait 30–60s and retry. On Lambda: confirm `vllm serve` is running, the SSH tunnel is up, "
        "and VLLM_BASE_URL is http://127.0.0.1:8000 (or your tunneled port)."
    )
    return JSONResponse(
        status_code=status_code,
        content={"error": "upstream_plaintext", "detail": raw, "hint": hint},
        headers={"X-Trace-Id": trace_id},
    )


def _connect_error_response(exc: Exception) -> JSONResponse:
    detail = str(exc)
    hint = (
        "Set VLLM_BASE_URL to your vLLM OpenAI base URL (README: Lambda + tunnel → http://127.0.0.1:8000). "
        "SSL hostname mismatch usually means the URL is still a placeholder from .env.example."
    )
    if "CERTIFICATE_VERIFY_FAILED" in detail or "hostname" in detail.lower():
        pass
    return JSONResponse(
        {"error": "upstream_unreachable", "detail": detail, "hint": hint},
        status_code=502,
    )


def _inject_beam_if_needed(technique: str, body: dict[str, Any]) -> None:
    if technique != "beam_search":
        return
    body.setdefault("use_beam_search", True)
    body.setdefault("best_of", 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gpu_hourly_usd = float(os.environ.get("GPU_HOURLY_COST_USD", "2.5"))
    app.state.gpu_cost_source = "env"
    api_key = os.environ.get("LAMBDA_CLOUD_API_KEY", "").strip()
    use_api = os.environ.get("LAMBDA_COST_USE_API", "true").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if api_key and use_api:
        try:
            from lambda_pricing import resolve_gpu_hourly_usd_from_lambda

            resolved = resolve_gpu_hourly_usd_from_lambda(api_key)
            if resolved is not None:
                app.state.gpu_hourly_usd = resolved
                app.state.gpu_cost_source = "lambda_api"
                _log.info("GPU hourly rate from Lambda API: $%.4f/hr", resolved)
        except Exception as e:
            _log.warning(
                "Lambda pricing lookup failed (%s); using GPU_HOURLY_COST_USD=$%.4f",
                e,
                app.state.gpu_hourly_usd,
            )
    app.state.client = httpx.AsyncClient(
        timeout=httpx.Timeout(600.0, connect=30.0),
        verify=_tls_verify(),
    )
    app.state.metrics_log_lock = asyncio.Lock()
    mdir = os.environ.get("GATEWAY_METRICS_LOG_DIR", "").strip()
    if mdir.lower() in ("-", "none", "false", "0"):
        app.state.metrics_log_dir = None
        _log.info("JSONL request metrics logging disabled (GATEWAY_METRICS_LOG_DIR=%r).", mdir)
    elif not mdir:
        app.state.metrics_log_dir = (_ROOT / "logs" / "gateway").resolve()
        _log.info(
            "JSONL metrics log dir (default): %s — set GATEWAY_METRICS_LOG_DIR to override or '-' to disable.",
            app.state.metrics_log_dir,
        )
    else:
        p = Path(mdir)
        app.state.metrics_log_dir = (
            p.resolve() if p.is_absolute() else (_ROOT / p).resolve()
        )
        _log.info("JSONL metrics log dir: %s", app.state.metrics_log_dir)
    _ensure_prometheus_http_server()
    if _env_truthy("VLLM_AUTO_ENGINE_ROUTING"):
        _log.info(
            "VLLM_AUTO_ENGINE_ROUTING on — backend_map keys: %s",
            sorted(_effective_backend_map().keys()),
        )
    yield
    await app.state.client.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
FastAPIInstrumentor.instrument_app(app)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "llm-gateway",
        "endpoints": {
            "health": "/health",
            "openai_proxy": "/v1/",
            "prometheus": "/metrics",
            "metrics_summary_json": "/metrics/summary",
            "metrics_summary_text": "/metrics/summary?format=text",
            "metrics_summary_html": "/metrics/summary?format=html",
            "prometheus_scrape_port": GATEWAY_METRICS_PORT,
        },
    }


@app.get("/health")
async def health(request: Request) -> dict[str, Any]:
    v = _vllm_base_url()
    bad_ph = _is_vllm_placeholder_url(v)
    bad_dash = _is_web_console_not_api_url(v)
    bad = bad_ph or bad_dash
    if bad_dash:
        warn = (
            "VLLM_BASE_URL hostname looks like a cloud *web console*, not the vLLM API. "
            "Use the OpenAI-compatible base URL (Lambda: http://127.0.0.1:8000 through your SSH tunnel)."
        )
    elif bad_ph:
        warn = (
            "VLLM_BASE_URL still looks like .env.example — set it to your real vLLM base "
            "(Lambda: http://127.0.0.1:8000 with the tunnel from the README)."
        )
    else:
        warn = None
    return {
        "status": "ok" if not bad else "misconfigured",
        "default_upstream": v or "(unset)",
        "server_profile": VLLM_SERVER_PROFILE,
        "vllm_auto_engine_routing": _env_truthy("VLLM_AUTO_ENGINE_ROUTING"),
        "backend_map_keys": sorted(_effective_backend_map().keys()),
        "vllm_url_placeholder": bad_ph,
        "web_console_not_api_url": bad_dash,
        "warning": warn,
        "gpu_hourly_cost_usd": getattr(request.app.state, "gpu_hourly_usd", None),
        "gpu_cost_source": getattr(request.app.state, "gpu_cost_source", "env"),
    }


@app.get("/metrics")
async def metrics() -> Response:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/metrics/summary")
async def metrics_summary(format: str = "json") -> Response:
    data = _metrics_summary_payload()
    f = (format or "json").strip().lower()
    if f in ("text", "txt", "plain"):
        return PlainTextResponse(_metrics_summary_as_text(data))
    if f in ("html", "htm"):
        return HTMLResponse(_metrics_summary_as_html(data))
    return JSONResponse(data)


@app.get("/v1")
@app.get("/v1/")
async def openai_v1_index() -> dict[str, Any]:
    return {
        "service": "llm-gateway",
        "note": "Not proxied upstream. Use GET /v1/models or POST /v1/chat/completions to hit vLLM.",
        "gateway_health": "/health",
    }


async def _proxy(request: Request, upstream_path: str) -> Response:
    body_bytes = await request.body()
    body: dict[str, Any] | None = None
    if body_bytes:
        try:
            body = json.loads(body_bytes)
        except json.JSONDecodeError:
            body = None

    technique = _resolve_technique(request, body)
    if body is not None:
        _inject_beam_if_needed(technique, body)
        body_bytes = json.dumps(body).encode()

    base = _upstream_base(technique)
    if not base:
        return JSONResponse(
            {"error": "No upstream URL: set VLLM_BASE_URL or VLLM_BACKEND_MAP_JSON for this technique"},
            status_code=500,
        )
    if _is_vllm_placeholder_url(base) or _is_web_console_not_api_url(base):
        hint = (
            "Use the vLLM OpenAI API base URL, not a cloud provider browser UI hostname. "
            "Lambda + README tunnel: http://127.0.0.1:8000"
            if _is_web_console_not_api_url(base)
            else "Replace template hostnames in VLLM_BASE_URL with your real vLLM base (e.g. http://127.0.0.1:8000)."
        )
        return JSONResponse(
            {"error": "upstream_url_misconfigured", "detail": base, "hint": hint},
            status_code=502,
        )

    parent_ctx = extract(dict(request.headers))

    _accept = request.headers.get("accept", "")
    stream = "text/event-stream" in _accept.lower() or (
        body is not None and body.get("stream") is True
    )

    t0 = time.perf_counter()
    client: httpx.AsyncClient = request.app.state.client

    lp = _metric_lp(technique)
    INFLIGHT_REQUESTS.labels(**lp).inc()
    full_url = f"{base}{upstream_path}"
    if request.url.query:
        full_url = f"{full_url}?{request.url.query}"

    if stream:
        span = tracer.start_span(
            "gateway.vllm.proxy",
            context=parent_ctx,
            attributes={
                "llm.technique": technique,
                "llm.server_profile": VLLM_SERVER_PROFILE,
                "http.route": upstream_path,
                "llm.streaming": True,
            },
        )
        out_ctx = trace.set_span_in_context(span)
        up_accept = request.headers.get("accept", "*/*")
        if "text/event-stream" not in up_accept.lower():
            up_accept = "text/event-stream"
        headers: dict[str, str] = {
            "content-type": request.headers.get("content-type", "application/json"),
            "accept": up_accept,
            "x-technique": technique,
        }
        inject(headers, context=out_ctx)
        req = client.build_request(
            request.method,
            full_url,
            headers=headers,
            content=body_bytes if body_bytes else None,
        )
        try:
            resp = await client.send(req, stream=True)
        except httpx.ConnectError as e:
            span.end()
            _record_outcome(lp, 0)
            INFLIGHT_REQUESTS.labels(**lp).dec()
            return _connect_error_response(e)

        tid = format(span.get_span_context().trace_id, "032x")
        gw_app = request.app

        async def gen():
            buf = bytearray()
            first_ttft_s: float | None = None
            inter: list[float] = []
            last_nonempty: float | None = None
            # The finally block below also runs when the client disconnects mid-stream.
            # Without this flag an abandoned stream would be recorded as a 200/ok.
            completed = False
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        buf.extend(chunk)
                        now = time.perf_counter()
                        if first_ttft_s is None:
                            first_ttft_s = now - t0
                        if last_nonempty is not None:
                            inter.append(now - last_nonempty)
                        last_nonempty = now
                    yield chunk
                completed = True
            finally:
                await resp.aclose()
                dt = time.perf_counter() - t0
                pt, ct = _parse_openai_sse_usage(bytes(buf))
                ttft_s = first_ttft_s if first_ttft_s is not None else dt
                for d in inter:
                    STREAM_INTER_CHUNK_SECONDS.labels(**lp).observe(d)
                TTFT_SECONDS.labels(**lp).observe(ttft_s)
                if inter:
                    tpot_avg_s = sum(inter) / len(inter)
                elif ct > 0:
                    tpot_avg_s = dt / ct
                else:
                    tpot_avg_s = 0.0
                _observe_tpot_and_tok_s(lp, tpot_s=tpot_avg_s, e2e_s=dt, completion_tokens=ct)
                if pt:
                    PROMPT_TOKENS.labels(**lp).inc(pt)
                if ct:
                    COMPLETION_TOKENS.labels(**lp).inc(ct)
                span.set_attribute("gateway.duration_s", dt)
                span.set_attribute("llm.prompt_tokens", pt)
                span.set_attribute("llm.completion_tokens", ct)
                REQUEST_DURATION.labels(**lp).observe(dt)
                REQUEST_COST_USD.labels(**lp).inc(_estimate_cost_usd(dt, gw_app))
                REQUESTS_TOTAL.labels(**lp).inc()
                if completed:
                    _record_outcome(lp, resp.status_code)
                else:
                    # Client hung up mid-generation. Those tokens were paid for and
                    # thrown away — "wasted-token rate" in the concurrency taxonomy.
                    REQUEST_OUTCOMES.labels(
                        **lp, outcome="cancelled", status_code="499"
                    ).inc()
                    WASTED_COMPLETION_TOKENS.labels(**lp).inc(ct)
                INFLIGHT_REQUESTS.labels(**lp).dec()
                await _append_gateway_metrics_log(
                    gw_app,
                    _build_metrics_log_row(
                        technique=technique,
                        upstream_path=upstream_path,
                        streaming=True,
                        status_code=resp.status_code,
                        trace_id=tid,
                        e2e_s=dt,
                        ttft_s=ttft_s,
                        tpot_avg_s=tpot_avg_s,
                        inter_chunk_delays_s=inter,
                        prompt_tokens=pt,
                        completion_tokens=ct,
                    ),
                )
                span.end()

        h = {**dict(resp.headers), "X-Trace-Id": tid}
        return StreamingResponse(gen(), status_code=resp.status_code, headers=h)

    with tracer.start_as_current_span(
        "gateway.vllm.proxy",
        context=parent_ctx,
        attributes={
            "llm.technique": technique,
            "llm.server_profile": VLLM_SERVER_PROFILE,
            "http.route": upstream_path,
            "llm.streaming": False,
        },
    ) as span:
        headers = {
            "content-type": request.headers.get("content-type", "application/json"),
            "accept": request.headers.get("accept", "*/*"),
            "x-technique": technique,
        }
        inject(headers)
        req = client.build_request(
            request.method,
            full_url,
            headers=headers,
            content=body_bytes if body_bytes else None,
        )
        try:
            resp = await client.send(req, stream=False)
        except httpx.ConnectError as e:
            _record_outcome(lp, 0)
            INFLIGHT_REQUESTS.labels(**lp).dec()
            return _connect_error_response(e)
        UPSTREAM_DURATION.labels(**lp).observe(time.perf_counter() - t0)
        content = await resp.aread()
        await resp.aclose()
        if resp.status_code >= 400:
            snippet = content[:2048].decode(errors="replace") if content else ""
            _log.warning("upstream HTTP %s %s — %s", resp.status_code, full_url, snippet)
            span.set_attribute("upstream.status_code", resp.status_code)
        dt = time.perf_counter() - t0
        span.set_attribute("gateway.duration_s", dt)
        REQUEST_DURATION.labels(**lp).observe(dt)
        REQUEST_COST_USD.labels(**lp).inc(_estimate_cost_usd(dt, request.app))
        REQUESTS_TOTAL.labels(**lp).inc()
        _record_outcome(lp, resp.status_code)
        INFLIGHT_REQUESTS.labels(**lp).dec()

        pt, ct = 0, 0
        try:
            payload = json.loads(content)
            usage = payload.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            PROMPT_TOKENS.labels(**lp).inc(pt)
            COMPLETION_TOKENS.labels(**lp).inc(ct)
            span.set_attribute("llm.prompt_tokens", pt)
            span.set_attribute("llm.completion_tokens", ct)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        ttft_s = dt
        tpot_avg_s = (dt / ct) if ct > 0 else 0.0
        TTFT_SECONDS.labels(**lp).observe(ttft_s)
        _observe_tpot_and_tok_s(lp, tpot_s=tpot_avg_s, e2e_s=dt, completion_tokens=ct)

        tid = format(span.get_span_context().trace_id, "032x")
        await _append_gateway_metrics_log(
            request.app,
            _build_metrics_log_row(
                technique=technique,
                upstream_path=upstream_path,
                streaming=False,
                status_code=resp.status_code,
                trace_id=tid,
                e2e_s=dt,
                ttft_s=ttft_s,
                tpot_avg_s=tpot_avg_s,
                inter_chunk_delays_s=[],
                prompt_tokens=pt,
                completion_tokens=ct,
            ),
        )
        if resp.status_code >= 400:
            wrapped = _plaintext_upstream_error_response(resp.status_code, content, tid)
            if wrapped is not None:
                return wrapped
        out_headers = {
            k: v for k, v in resp.headers.items() if k.lower() != "content-length"
        }
        out_headers["X-Trace-Id"] = tid
        return Response(
            content=content,
            status_code=resp.status_code,
            headers=out_headers,
            media_type=resp.headers.get("content-type"),
        )


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def openai_proxy(path: str, request: Request) -> Response:
    return await _proxy(request, f"/v1/{path}")


def run() -> None:
    from uvicorn import Config, Server

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if _otlp_traces_enabled():
        _log.info(
            "OTLP trace export is on (→ %s). If you see UNAVAILABLE on 4317, set OTEL_TRACES_EXPORTER=none or start Jaeger.",
            OTEL_ENDPOINT,
        )
    host = os.environ.get("GATEWAY_HOST", "127.0.0.1")
    port = int(os.environ.get("GATEWAY_PORT", "8765"))
    server = Server(Config(app, host=host, port=port, log_level="info"))
    server.run()


if __name__ == "__main__":
    run()
