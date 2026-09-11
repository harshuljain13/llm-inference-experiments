#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional

import httpx


def _usage_tokens(resp_json: Dict[str, Any]) -> Optional[int]:
    u = resp_json.get("usage") or {}
    return int(u.get("completion_tokens") or 0) or None


async def one_chat(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    max_tokens: int,
    temperature: float,
    prompt: str,
):
    t0 = time.perf_counter()
    r = await client.post(
        f"{base}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'fake-key')}"},
        timeout=600.0,
    )
    r.raise_for_status()
    data = r.json()
    dt = time.perf_counter() - t0
    tok = _usage_tokens(data)
    return dt, tok or 0


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[k]


async def run(args: argparse.Namespace) -> None:
    base = args.url.rstrip("/")
    model = args.model or os.environ.get("PLAYGROUND_BENCH_MODEL", "tinyllama-chat")
    sem = asyncio.Semaphore(args.concurrency)

    prompts = [
        "Give a two-sentence overview of Ray Serve LLM.",
        "What is vLLM continuous batching?",
        "List three causes of pipeline bubbles in PP inference.",
    ]

    latencies: List[float] = []
    tokens_out: List[int] = []
    errors = 0

    async with httpx.AsyncClient() as client:

        async def wrapped(i: int):
            nonlocal errors
            async with sem:
                prompt = prompts[i % len(prompts)]
                try:
                    dt, toks = await one_chat(
                        client,
                        base,
                        model,
                        args.max_tokens,
                        args.temperature,
                        prompt,
                    )
                    latencies.append(dt)
                    tokens_out.append(toks)
                except Exception as e:
                    errors += 1
                    print("request error:", e)

        if args.warmup:
            # The first requests pay container cold start, weight load and CUDA
            # graph capture. Timing them swamps p99 and understates RPS, so run
            # them untimed and throw the samples away.
            print("warmup: %d request(s)..." % args.warmup, flush=True)
            await asyncio.gather(*(wrapped(i) for i in range(args.warmup)))
            latencies.clear()
            tokens_out.clear()
            errors = 0

        t_wall0 = time.perf_counter()
        await asyncio.gather(*(wrapped(i) for i in range(args.requests)))
        wall_s = time.perf_counter() - t_wall0

    total_toks = sum(tokens_out)
    print(json.dumps({"model": model, "errors": errors, "ok": len(latencies)}, indent=2))
    if not latencies:
        return
    print(
        "latency_s  p50=%.3f p90=%.3f p99=%.3f max=%.3f"
        % (pct(latencies, 50), pct(latencies, 90), pct(latencies, 99), max(latencies))
    )
    mean_lat = statistics.mean(latencies)
    rps = len(latencies) / wall_s if wall_s > 0 else float("nan")
    print("mean_latency_s=%.4f wall_s=%.3f aggregate_rps=%.2f" % (mean_lat, wall_s, rps))
    if total_toks > 0 and wall_s > 0:
        # The headline throughput number: what the whole system delivered per
        # second of wall clock. This is what has to move when you add GPUs.
        print("output_tok_per_sec=%.2f (aggregate: completion_tokens / wall_s)" % (total_toks / wall_s))
    if total_toks > 0 and sum(latencies) > 0:
        # Per-stream rate: roughly aggregate / concurrency, so it stays flat when
        # you scale out. Good for single-user speed, useless for comparing DP.
        print(
            "per_stream_output_tok_per_sec=%.2f (completion_tokens summed / sum(latencies))"
            % (total_toks / sum(latencies))
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5, help="Untimed requests to discard first")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
