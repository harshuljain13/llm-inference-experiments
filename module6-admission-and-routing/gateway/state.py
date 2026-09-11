from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from gateway import config as cfg


@dataclass
class ReplicaState:
    url: str
    id: str
    waiting: int = 0
    running: int = 0
    kv_usage: float = 0.0
    max_num_seqs: int = cfg.DEFAULT_MAX_NUM_SEQS
    kv_capacity_tokens: int = cfg.KV_CAPACITY_TOKENS
    prefix_queries: int = 0
    prefix_hits: int = 0
    preemptions: int = 0
    ttft_sum: float = 0.0
    ttft_count: float = 0.0
    itl_sum: float = 0.0
    itl_count: float = 0.0
    queue_time_sum: float = 0.0
    queue_time_count: float = 0.0
    in_flight: int = 0
    completed: int = 0


@dataclass
class FleetState:
    replicas: list[ReplicaState] = field(default_factory=list)
    scrape_ok_at: float = 0.0
    prefill_tokens_per_s: float = cfg.INIT_PREFILL_TOKENS_PER_S
    inter_token_latency_s: float = cfg.INIT_INTER_TOKEN_LATENCY_S
    queue_wait_s: float = 0.0

    @property
    def stale_for(self) -> float:
        if self.scrape_ok_at <= 0:
            return 1e9
        return time.monotonic() - self.scrape_ok_at

    @property
    def kv_usage_max(self) -> float:
        return max((r.kv_usage for r in self.replicas), default=0.0)

    @property
    def waiting_total(self) -> int:
        return sum(r.waiting for r in self.replicas)

    @property
    def running_total(self) -> int:
        return sum(r.running for r in self.replicas)

    @property
    def headroom_tokens(self) -> int:
        return int(sum((1.0 - r.kv_usage) * r.kv_capacity_tokens for r in self.replicas))

    def update_queue_wait(self, depth: int) -> None:
        slots = max(sum(r.max_num_seqs for r in self.replicas), 1)
        svc = 400.0 / max(self.prefill_tokens_per_s, 1.0) + 200.0 * self.inter_token_latency_s
        self.queue_wait_s = depth * svc / slots


@dataclass
class PendingRequest:
    n_in: int
    n_out: int
    deadline_s: float
    tenant: str
    token_ids: list[int]
    messages: list
    stream: bool
    body: dict
    deadline_at: float
    priority: int | None = None
    deadline_ms: int = 2000


@dataclass
class QueueItem:
    enqueued_at: float
    deadline_at: float
    n_in: int
    n_out: int
    tenant: str
    passed_over: int
    req: PendingRequest
    ready: Any = field(repr=False)
