from __future__ import annotations

import time

from gateway.scrape import parse_prometheus
from limiter import RateLimiter


def test_scrape_parses_vllm_028_names():
    text = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="lab"} 3.0
vllm:num_requests_waiting{model_name="lab"} 2.0
vllm:kv_cache_usage_perc{model_name="lab"} 0.4
vllm:prefix_cache_queries_total{model_name="lab"} 100
vllm:prefix_cache_hits_total{model_name="lab"} 40
vllm:request_queue_time_seconds_sum 1.5
vllm:request_queue_time_seconds_count 3
vllm:time_to_first_token_seconds_sum 0.6
vllm:time_to_first_token_seconds_count 3
vllm:inter_token_latency_seconds_sum 0.03
vllm:inter_token_latency_seconds_count 10
vllm:num_preemptions_total 0
"""
    m = parse_prometheus(text)
    assert m["vllm:kv_cache_usage_perc"] == 0.4
    assert m["vllm:num_requests_waiting"] == 2.0
    assert "vllm:gpu_cache_usage_perc" not in m


def test_rate_limiter_burst_then_block():
    lim = RateLimiter(rps=0.0, burst=2)
    assert lim.try_acquire()
    assert lim.try_acquire()
    assert not lim.try_acquire()


def test_rate_limiter_refills():
    lim = RateLimiter(rps=100.0, burst=1)
    assert lim.try_acquire()
    assert not lim.try_acquire()
    time.sleep(0.03)
    assert lim.try_acquire()
