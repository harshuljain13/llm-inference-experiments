from __future__ import annotations

from gateway.admission import should_shed
from gateway.state import FleetState, PendingRequest, ReplicaState


def _req(**kw) -> PendingRequest:
    defaults = dict(
        n_in=200,
        n_out=100,
        deadline_s=2.0,
        tenant="chat",
        token_ids=[1] * 200,
        messages=[],
        stream=False,
        body={},
        deadline_at=10.0,
        deadline_ms=2000,
    )
    defaults.update(kw)
    return PendingRequest(**defaults)


def _fleet(**kw) -> FleetState:
    r = ReplicaState(url="http://x", id="r0", waiting=0, running=1, kv_usage=0.2)
    f = FleetState(
        replicas=[r, ReplicaState(url="http://y", id="r1", waiting=0, running=1, kv_usage=0.1)],
        scrape_ok_at=1e18,
        prefill_tokens_per_s=4_000.0,
        inter_token_latency_s=0.004,
        queue_wait_s=0.0,
    )
    for k, v in kw.items():
        setattr(f, k, v)
    return f


def test_should_shed_no_signal_when_stale():
    fleet = _fleet()
    fleet.scrape_ok_at = 0.0
    assert should_shed(fleet, _req()) == "no_signal"


def test_should_shed_kv_pressure():
    fleet = _fleet()
    fleet.replicas[0].kv_usage = 0.95
    assert should_shed(fleet, _req()) == "kv_pressure"


def test_should_shed_queue_depth():
    fleet = _fleet()
    fleet.replicas[0].waiting = 9
    fleet.replicas[1].waiting = 9
    assert should_shed(fleet, _req()) == "queue_depth"


def test_should_shed_no_headroom():
    fleet = _fleet()
    for r in fleet.replicas:
        r.kv_usage = 0.80
        r.kv_capacity_tokens = 200
    assert should_shed(fleet, _req(n_in=500, n_out=500)) == "no_headroom"


def test_should_shed_deadline_unmeetable():
    fleet = _fleet()
    fleet.inter_token_latency_s = 0.05
    fleet.queue_wait_s = 4.0
    assert should_shed(fleet, _req(n_in=100, n_out=200, deadline_s=0.5)) == "deadline_unmeetable"


def test_should_shed_admits_when_healthy():
    assert should_shed(_fleet(), _req()) is None
