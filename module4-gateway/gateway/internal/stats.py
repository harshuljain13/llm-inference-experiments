from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

from gateway import config as cfg


class Ring:
    def __init__(self, maxlen: int = cfg.STATS_RING) -> None:
        self._buf: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields: Any) -> None:
        rec = {"ts": time.monotonic(), "event": event, **fields}
        with self._lock:
            self._buf.append(rec)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
