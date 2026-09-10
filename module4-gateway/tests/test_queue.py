from __future__ import annotations

from gateway.queue import queue_key
from gateway.state import PendingRequest, QueueItem


def _item(**kw) -> QueueItem:
    req = PendingRequest(
        n_in=kw.get("n_in", 100),
        n_out=50,
        deadline_s=2.0,
        tenant="chat",
        token_ids=[],
        messages=[],
        stream=False,
        body={},
        deadline_at=kw.get("deadline_at", 10.0),
    )
    return QueueItem(
        enqueued_at=kw.get("enqueued_at", 0.0),
        deadline_at=kw.get("deadline_at", 10.0),
        n_in=kw.get("n_in", 100),
        n_out=50,
        tenant="chat",
        passed_over=kw.get("passed_over", 0),
        req=req,
        ready=object(),
    )


def test_queue_key_deadline_beats_fcfs():
    slack = _item(enqueued_at=0.0, deadline_at=100.0, n_in=100)
    urgent = _item(enqueued_at=10.0, deadline_at=20.0, n_in=100)
    now = 15.0
    assert queue_key(urgent, now) < queue_key(slack, now)


def test_queue_key_long_prompt_sorts_later():
    short = _item(enqueued_at=5.0, deadline_at=50.0, n_in=100, passed_over=0)
    long = _item(enqueued_at=0.0, deadline_at=50.0, n_in=8_000, passed_over=0)
    now = 1.0
    assert queue_key(short, now) < queue_key(long, now)


def test_queue_key_aging_waives_long_penalty():
    long_starved = _item(enqueued_at=10.0, deadline_at=50.0, n_in=8_000, passed_over=99)
    short = _item(enqueued_at=0.0, deadline_at=50.0, n_in=100, passed_over=0)
    now = 1.0
    assert queue_key(long_starved, now) < queue_key(short, now)
