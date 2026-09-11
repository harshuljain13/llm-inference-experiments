#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import httpx


async def worker(
    client: httpx.AsyncClient,
    base: str,
    model: str,
    max_tokens: int,
    stop_at: float,
    latencies: list,
):
    prompt = "Reply with a single short sentence about Ray Serve."
    while time.perf_counter() < stop_at:
        t0 = time.perf_counter()
        r = await client.post(
            f"{base}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'fake-key')}"},
            timeout=600.0,
        )
        r.raise_for_status()
        latencies.append(time.perf_counter() - t0)


async def phase(base: str, model: str, concurrency: int, duration_s: float, max_tokens: int):
    stop_at = time.perf_counter() + duration_s
    latencies: list = []
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *[
                worker(client, base, model, max_tokens, stop_at, latencies)
                for _ in range(concurrency)
            ]
        )
    return latencies


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default=os.environ.get("PLAYGROUND_BENCH_MODEL", "tinyllama-chat"))
    p.add_argument("--steps", default="2,4,8,16", help="Comma-separated concurrency levels")
    p.add_argument("--duration", type=float, default=20.0, help="Seconds per step")
    p.add_argument("--max-tokens", type=int, default=48)
    args = p.parse_args()
    base = args.url.rstrip("/")
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    for c in steps:
        t0 = time.perf_counter()
        lats = asyncio.run(phase(base, args.model, c, args.duration, args.max_tokens))
        wall = time.perf_counter() - t0
        print(
            json.dumps(
                {
                    "concurrency": c,
                    "duration_s": args.duration,
                    "requests": len(lats),
                    "wall_s": round(wall, 3),
                    "mean_latency_s": round(sum(lats) / len(lats), 4) if lats else None,
                }
            )
        )


if __name__ == "__main__":
    main()
