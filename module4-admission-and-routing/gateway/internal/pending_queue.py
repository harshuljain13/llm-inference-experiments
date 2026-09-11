from __future__ import annotations

import asyncio
import time

from gateway import config as cfg
from gateway.internal.stats import Ring
from gateway.queue import queue_key
from gateway.state import FleetState, QueueItem


class QueueFull(Exception):
    pass


class PendingQueue:
    def __init__(self, stats: Ring, fleet: FleetState, maxsize: int | None = None) -> None:
        self._items: list[QueueItem] = []
        self._cv = asyncio.Condition()
        self._stats = stats
        self._fleet = fleet
        self.maxsize = cfg.QUEUE_MAXSIZE if maxsize is None else maxsize

    def __len__(self) -> int:
        return len(self._items)

    async def put(self, item: QueueItem) -> None:
        async with self._cv:
            if len(self._items) >= self.maxsize:
                raise QueueFull
            self._items.append(item)
            self._fleet.update_queue_wait(len(self._items))
            self._cv.notify()

    def _expire(self) -> None:
        now = time.monotonic()
        keep: list[QueueItem] = []
        for it in self._items:
            if it.deadline_at <= now:
                if not it.ready.done():
                    it.ready.set_exception(Expired())
                self._stats.emit(
                    "expired",
                    reason="expired_in_queue",
                    tenant=it.tenant,
                    n_in=it.n_in,
                    n_out=it.n_out,
                    outcome="expired",
                )
            else:
                keep.append(it)
        self._items = keep
        self._fleet.update_queue_wait(len(self._items))

    async def get(self) -> QueueItem:
        async with self._cv:
            while True:
                self._expire()
                if self._items:
                    now = time.monotonic()
                    self._items.sort(key=lambda it: queue_key(it, now))
                    item = self._items.pop(0)
                    for other in self._items:
                        other.passed_over += 1
                    self._fleet.update_queue_wait(len(self._items))
                    return item
                try:
                    await asyncio.wait_for(self._cv.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue


class Expired(Exception):
    reason = "expired_in_queue"
