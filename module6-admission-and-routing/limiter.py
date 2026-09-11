from __future__ import annotations

import threading
import time

from gateway import config as cfg


class RateLimited(Exception):
    pass


class RateLimiter:
    def __init__(
        self,
        rps: float | None = None,
        burst: float | None = None,
    ) -> None:
        self.rps = cfg.LIMITER_RPS if rps is None else rps
        self.burst = cfg.LIMITER_BURST if burst is None else burst
        self._tokens = float(self.burst)
        self._t = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.burst, self._tokens + (now - self._t) * self.rps)
            self._t = now
            if self._tokens < n:
                return False
            self._tokens -= n
            return True

    def acquire(self, n: float = 1.0) -> float:
        waited = 0.0
        while not self.try_acquire(n):
            time.sleep(0.02)
            waited += 0.02
        return waited
