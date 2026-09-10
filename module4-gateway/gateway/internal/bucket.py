from __future__ import annotations

import time

from gateway import config as cfg


class TokenBucket:
    def __init__(self, rate: float | None = None, burst: float | None = None) -> None:
        self.rate = cfg.BUCKET_RATE_TOKENS_PER_S if rate is None else rate
        self.burst = cfg.BUCKET_BURST_TOKENS if burst is None else burst
        self._tokens = self.burst
        self._t = time.monotonic()

    def allow(self, n: float) -> bool:
        now = time.monotonic()
        self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rate)
        self._t = now
        if self._tokens < n:
            return False
        self._tokens -= n
        return True


class Buckets:
    def __init__(self) -> None:
        self._by: dict[str, TokenBucket] = {}

    def allow(self, tenant: str, n: float) -> bool:
        b = self._by.get(tenant)
        if b is None:
            b = TokenBucket()
            self._by[tenant] = b
        return b.allow(n)
