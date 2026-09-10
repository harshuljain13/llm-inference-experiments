from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway import config as cfg
from gateway.admission import should_shed
from gateway.internal.bucket import Buckets
from gateway.internal.pending_queue import Expired, PendingQueue, QueueFull
from gateway.internal.stats import Ring
from gateway.internal.tokenize import messages_to_text, tokenize
from gateway.router import Router
from gateway.scrape import scrape_loop
from gateway.state import FleetState, PendingRequest, QueueItem, ReplicaState


def _deadline_ms(body: dict, tenant: str) -> int:
    if body.get("deadline_ms") is not None:
        return int(body["deadline_ms"])
    tier = cfg.TENANT_TIER.get(tenant, "interactive")
    return cfg.DEADLINE_BY_TIER_MS[tier]


def _estimate_ms(fleet: FleetState, req: PendingRequest) -> int:
    prefill = req.n_in / max(fleet.prefill_tokens_per_s, 1e-3)
    decode = req.n_out * fleet.inter_token_latency_s
    return int(1000 * (prefill + decode + fleet.queue_wait_s))


def refuse(reason: str, req: PendingRequest | None, fleet: FleetState, stats: Ring) -> JSONResponse:
    status = cfg.QUOTA_STATUS if reason == "quota" else cfg.SHED_STATUS
    retry = cfg.RETRY_AFTER_S.get(reason, 1)
    est = _estimate_ms(fleet, req) if req is not None else 0
    deadline_ms = req.deadline_ms if req is not None else 0
    stats.emit(
        "refused",
        reason=reason,
        tenant=getattr(req, "tenant", "default"),
        n_in=getattr(req, "n_in", 0),
        n_out=getattr(req, "n_out", 0),
        estimated_total_ms=est,
        deadline_ms=deadline_ms,
        outcome="refused",
    )
    return JSONResponse(
        status_code=status,
        headers={"Retry-After": str(retry)},
        content={
            "error": {
                "type": "request_refused",
                "reason": reason,
                "estimated_total_ms": est,
                "deadline_ms": deadline_ms,
            }
        },
    )


def create_app() -> FastAPI:
    fleet = FleetState()
    stats = Ring()
    buckets = Buckets()
    router = Router()
    queue = PendingQueue(stats, fleet)
    stop = asyncio.Event()
    client = httpx.AsyncClient(timeout=cfg.HTTP_TIMEOUT_S)
    worker_task: asyncio.Task | None = None
    scrape_task: asyncio.Task | None = None

    async def wait_slot() -> None:
        while True:
            cap = sum(r.max_num_seqs for r in fleet.replicas) + cfg.DISPATCH_OVERSHOOT
            if sum(r.in_flight for r in fleet.replicas) < max(cap, 1):
                return
            await asyncio.sleep(0.005)

    async def assign(req: PendingRequest) -> tuple[ReplicaState, int]:
        await wait_slot()
        replica, match = router.pick(fleet.replicas, req)
        replica.in_flight += 1
        return replica, match

    async def worker() -> None:
        while not stop.is_set():
            try:
                item = await asyncio.wait_for(queue.get(), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            try:
                replica, match = await assign(item.req)
            except Exception as exc:
                if not item.ready.done():
                    item.ready.set_exception(exc)
                continue
            stats.emit("dequeued", tenant=item.tenant, n_in=item.n_in, n_out=item.n_out)
            if not item.ready.done():
                item.ready.set_result((replica, match))

    async def proxy(request: Request, req: PendingRequest, replica: ReplicaState, match: int):
        t0 = time.monotonic()
        ttft_ms: float | None = None
        body = dict(req.body)
        body.pop("deadline_ms", None)
        if req.priority is not None:
            body["priority"] = req.priority
        url = f"{replica.url}/v1/chat/completions"
        outcome = "ok"
        reason = None
        try:
            if req.stream:

                async def stream_out():
                    nonlocal ttft_ms, outcome, reason
                    async with client.stream("POST", url, json=body) as up:
                        async for raw in up.aiter_bytes():
                            if await request.is_disconnected():
                                outcome = "cancelled"
                                reason = "client_disconnect"
                                await up.aclose()
                                return
                            if ttft_ms is None:
                                ttft_ms = (time.monotonic() - t0) * 1000
                            yield raw
                        if up.status_code >= 400:
                            outcome = "error"
                            reason = f"upstream_{up.status_code}"

                return StreamingResponse(
                    _tee(stream_out(), finally_=lambda: _finish(replica, req, match, t0, ttft_ms, outcome, reason)),
                    media_type="text/event-stream",
                    headers=_hdr(replica, match, "ok", None),
                )

            up = await client.post(url, json=body)
            ttft_ms = (time.monotonic() - t0) * 1000
            if up.status_code >= 400:
                outcome = "error"
                reason = f"upstream_{up.status_code}"
            _finish(replica, req, match, t0, ttft_ms, outcome, reason)
            headers = _hdr(replica, match, outcome, reason)
            try:
                payload = up.json()
            except Exception:
                payload = {"error": {"type": "upstream", "reason": up.text[:200]}}
            return JSONResponse(status_code=up.status_code, content=payload, headers=headers)
        except httpx.HTTPError as exc:
            outcome = "error"
            reason = "upstream_error"
            _finish(replica, req, match, t0, ttft_ms, outcome, reason)
            return JSONResponse(
                status_code=502,
                content={"error": {"type": "upstream", "reason": str(exc)}},
                headers=_hdr(replica, match, outcome, reason),
            )

    def _finish(
        replica: ReplicaState,
        req: PendingRequest,
        match: int,
        t0: float,
        ttft_ms: float | None,
        outcome: str,
        reason: str | None,
    ) -> None:
        replica.in_flight = max(0, replica.in_flight - 1)
        replica.completed += 1
        stats.emit(
            "completed",
            reason=reason,
            tenant=req.tenant,
            replica=replica.id,
            n_in=req.n_in,
            n_out=req.n_out,
            ttft_ms=ttft_ms,
            total_ms=(time.monotonic() - t0) * 1000,
            prefix_match_tokens=match,
            outcome=outcome,
        )

    def _hdr(replica: ReplicaState, match: int, outcome: str, reason: str | None) -> dict[str, str]:
        h = {
            "X-Replica": replica.id,
            "X-Prefix-Match-Tokens": str(match),
            "X-Outcome": outcome,
        }
        if reason:
            h["X-Reason"] = reason
        return h

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal worker_task, scrape_task
        urls = list(cfg.REPLICA_URLS)
        fleet.replicas = [
            ReplicaState(url=u.rstrip("/"), id=f"r{i}", max_num_seqs=cfg.DEFAULT_MAX_NUM_SEQS)
            for i, u in enumerate(urls)
        ]
        fleet.scrape_ok_at = 0.0
        scrape_task = asyncio.create_task(scrape_loop(fleet, stop))
        worker_task = asyncio.create_task(worker())
        for _ in range(20):
            if fleet.scrape_ok_at > 0:
                break
            await asyncio.sleep(0.05)
        yield
        stop.set()
        if scrape_task:
            scrape_task.cancel()
        if worker_task:
            worker_task.cancel()
        await client.aclose()

    app = FastAPI(title="llm-gateway-lab", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "replicas": [r.url for r in fleet.replicas],
            "stale_for": fleet.stale_for,
            "queue_depth": len(queue),
            "admission": cfg.ADMISSION_ENABLED,
            "queue": cfg.QUEUE_ENABLED,
            "prefix": cfg.USE_PREFIX_ROUTING,
        }

    @app.get("/_stats")
    async def get_stats() -> dict:
        return {
            "events": stats.snapshot(),
            "queue_depth": len(queue),
            "fleet": {
                "stale_for": fleet.stale_for,
                "kv_usage_max": fleet.kv_usage_max,
                "waiting_total": fleet.waiting_total,
                "running_total": fleet.running_total,
                "headroom_tokens": fleet.headroom_tokens,
                "queue_wait_s": fleet.queue_wait_s,
                "replicas": [
                    {
                        "id": r.id,
                        "url": r.url,
                        "waiting": r.waiting,
                        "running": r.running,
                        "kv_usage": r.kv_usage,
                        "in_flight": r.in_flight,
                        "completed": r.completed,
                    }
                    for r in fleet.replicas
                ],
            },
        }

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": cfg.SERVED_MODEL_NAME, "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        messages = body.get("messages") or []
        text = messages_to_text(messages)
        token_ids = tokenize(text)
        n_in = len(token_ids)
        n_out = int(body.get("max_tokens") or body.get("max_completion_tokens") or cfg.DEFAULT_MAX_TOKENS)
        tenant = request.headers.get("X-Tenant-Id") or "default"
        deadline_ms = _deadline_ms(body, tenant)
        now = time.monotonic()
        prio_raw = request.headers.get("X-Priority")
        priority = int(prio_raw) if prio_raw is not None and prio_raw != "" else None
        req = PendingRequest(
            n_in=n_in,
            n_out=n_out,
            deadline_s=deadline_ms / 1000.0,
            tenant=tenant,
            token_ids=token_ids,
            messages=messages,
            stream=bool(body.get("stream")),
            body=body,
            deadline_at=now + deadline_ms / 1000.0,
            priority=priority,
            deadline_ms=deadline_ms,
        )
        stats.emit("recv", tenant=tenant, n_in=n_in, n_out=n_out, deadline_ms=deadline_ms)

        if not buckets.allow(tenant, n_in + n_out):
            return refuse("quota", req, fleet, stats)

        if cfg.ADMISSION_ENABLED:
            reason = should_shed(fleet, req)
            if reason:
                return refuse(reason, req, fleet, stats)

        stats.emit("admitted", tenant=tenant, n_in=n_in, n_out=n_out)

        if not cfg.QUEUE_ENABLED:
            replica, match = await assign(req)
            stats.emit("dispatched", replica=replica.id, prefix_match_tokens=match, tenant=tenant)
            return await proxy(request, req, replica, match)

        loop = asyncio.get_event_loop()
        item = QueueItem(
            enqueued_at=now,
            deadline_at=req.deadline_at,
            n_in=n_in,
            n_out=n_out,
            tenant=tenant,
            passed_over=0,
            req=req,
            ready=loop.create_future(),
        )
        try:
            await queue.put(item)
        except QueueFull:
            return refuse("queue_full", req, fleet, stats)
        stats.emit("enqueued", tenant=tenant, n_in=n_in, n_out=n_out, queue_depth=len(queue))
        try:
            replica, match = await item.ready
        except Expired:
            return refuse("expired_in_queue", req, fleet, stats)
        stats.emit("dispatched", replica=replica.id, prefix_match_tokens=match, tenant=tenant)
        return await proxy(request, req, replica, match)

    app.state.fleet = fleet
    app.state.stats = stats
    app.state.queue = queue
    app.state.router = router
    return app


async def _tee(agen, finally_):
    try:
        async for item in agen:
            yield item
    finally:
        finally_()
