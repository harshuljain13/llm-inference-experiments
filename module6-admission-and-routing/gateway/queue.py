from __future__ import annotations

from gateway import config as cfg
from gateway.state import QueueItem


def queue_key(item: QueueItem, now: float) -> tuple:
    starved = item.passed_over >= cfg.MAX_OVERTAKES
    long = 0 if starved or item.n_in < cfg.LONG_PROMPT_TOKENS else 1
    slack = (item.deadline_at - now) - cfg.AGING_GAIN * item.passed_over
    return (long, slack, item.enqueued_at)
