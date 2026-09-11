from __future__ import annotations

from gateway import config as cfg
from gateway.state import FleetState, PendingRequest


def should_shed(fleet: FleetState, req: PendingRequest) -> str | None:
    if fleet.stale_for > cfg.STALE_CEILING_S:
        return "no_signal"
    if fleet.kv_usage_max > cfg.KV_CEILING:
        return "kv_pressure"
    n = max(len(fleet.replicas), 1)
    if fleet.waiting_total > cfg.WAITING_CEILING_PER_REPLICA * n:
        return "queue_depth"
    if fleet.headroom_tokens < req.n_in + req.n_out:
        return "no_headroom"
    prefill = req.n_in / max(fleet.prefill_tokens_per_s, 1.0)
    decode = req.n_out * max(fleet.inter_token_latency_s, 0.0)
    if prefill + decode + fleet.queue_wait_s > req.deadline_s:
        return "deadline_unmeetable"
    return None
