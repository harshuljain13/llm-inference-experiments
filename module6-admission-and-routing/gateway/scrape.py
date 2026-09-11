from __future__ import annotations

import asyncio
import re
import time

import httpx

from gateway import config as cfg
from gateway.state import FleetState, ReplicaState

_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*$"
)


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        out[m.group("name")] = float(m.group("value"))
    return out


def _gauge(metrics: dict[str, float], name: str, default: float = 0.0) -> float:
    return metrics.get(name, default)


def apply_metrics(r: ReplicaState, metrics: dict[str, float]) -> None:
    r.running = int(_gauge(metrics, "vllm:num_requests_running"))
    r.waiting = int(_gauge(metrics, "vllm:num_requests_waiting"))
    kv = _gauge(metrics, "vllm:kv_cache_usage_perc")
    r.kv_usage = kv / 100.0 if kv > 1.0 else kv
    r.prefix_queries = int(_gauge(metrics, "vllm:prefix_cache_queries_total"))
    r.prefix_hits = int(_gauge(metrics, "vllm:prefix_cache_hits_total"))
    r.preemptions = int(_gauge(metrics, "vllm:num_preemptions_total"))
    r.queue_time_sum = _gauge(metrics, "vllm:request_queue_time_seconds_sum")
    r.queue_time_count = _gauge(metrics, "vllm:request_queue_time_seconds_count")
    r.ttft_sum = _gauge(metrics, "vllm:time_to_first_token_seconds_sum")
    r.ttft_count = _gauge(metrics, "vllm:time_to_first_token_seconds_count")
    r.itl_sum = _gauge(metrics, "vllm:inter_token_latency_seconds_sum")
    r.itl_count = _gauge(metrics, "vllm:inter_token_latency_seconds_count")


def refresh_fleet_rates(fleet: FleetState) -> None:
    ttft_s = sum(r.ttft_sum for r in fleet.replicas)
    ttft_n = sum(r.ttft_count for r in fleet.replicas)
    itl_s = sum(r.itl_sum for r in fleet.replicas)
    itl_n = sum(r.itl_count for r in fleet.replicas)
    if ttft_n > 0 and ttft_s > 0:
        fleet.prefill_tokens_per_s = max(200.0 / (ttft_s / ttft_n), 1.0)
    if itl_n > 0:
        fleet.inter_token_latency_s = itl_s / itl_n


async def scrape_once(client: httpx.AsyncClient, fleet: FleetState) -> bool:
    ok = True
    for r in fleet.replicas:
        try:
            resp = await client.get(f"{r.url}/metrics", timeout=2.0)
            resp.raise_for_status()
            apply_metrics(r, parse_prometheus(resp.text))
        except Exception:
            ok = False
    if ok:
        fleet.scrape_ok_at = time.monotonic()
        refresh_fleet_rates(fleet)
    return ok


async def scrape_loop(fleet: FleetState, stop: asyncio.Event) -> None:
    async with httpx.AsyncClient() as client:
        while not stop.is_set():
            await scrape_once(client, fleet)
            try:
                await asyncio.wait_for(stop.wait(), timeout=cfg.SCRAPE_INTERVAL_S)
            except asyncio.TimeoutError:
                pass
