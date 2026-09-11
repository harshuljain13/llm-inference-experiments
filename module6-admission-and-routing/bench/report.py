from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from app import DEFAULT_QUERIES, run_one
from limiter import RateLimiter

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Record:
    stream: str
    submit_t: float
    ttft_ms: float | None
    total_ms: float | None
    status: int
    reason: str | None
    replica: str | None
    made_deadline: bool
    prefix_hit: bool
    n_in: int
    n_out: int
    deadline_ms: int


PRESETS = [
    ("1 · baseline", "baseline"),
    ("2 · route", "route"),
    ("3 · queue", "queue"),
    ("4 · full", "full"),
]


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(xs, q))


def summarize(name: str, recs: list[Record], stats: dict) -> dict:
    admitted = [r for r in recs if r.status < 400]
    refused = [r for r in recs if r.status >= 400]
    reasons = Counter(r.reason or "unknown" for r in refused)
    ttfts = [r.ttft_ms for r in admitted if r.ttft_ms is not None]
    totals = [r.total_ms for r in admitted if r.total_ms is not None]
    tokens = sum(r.n_in + r.n_out for r in admitted)
    elapsed = 0.0
    if recs:
        elapsed = max(r.submit_t for r in recs) - min(r.submit_t for r in recs)
        elapsed = max(elapsed, 1e-3)
    by_rep: dict[str, int] = Counter(r.replica or "?" for r in admitted)
    loads = list(by_rep.values()) or [0]
    hits = sum(1 for r in admitted if r.prefix_hit)
    events = stats.get("events") or []
    q_depths = [e.get("queue_depth") for e in events if e.get("event") == "enqueued"]
    q_depths = [d for d in q_depths if isinstance(d, (int, float))]
    return {
        "config": name,
        "n": len(recs),
        "admitted": len(admitted),
        "refused": len(refused),
        "refused_by_reason": dict(reasons),
        "ttft_p50": _pct(ttfts, 50),
        "ttft_p99": _pct(ttfts, 99),
        "total_p50": _pct(totals, 50),
        "total_p99": _pct(totals, 99),
        "deadline_pct": 100.0 * (sum(1 for r in recs if r.made_deadline) / max(len(recs), 1)),
        "prefix_hit_rate": 100.0 * hits / max(len(admitted), 1),
        "load_spread": max(loads) - min(loads),
        "tokens_per_s": tokens / elapsed,
        "queue_depth_max": max(q_depths) if q_depths else 0,
        "records": [
            {
                "stream": r.stream,
                "submit_t": r.submit_t,
                "ttft_ms": r.ttft_ms,
                "total_ms": r.total_ms,
                "status": r.status,
                "reason": r.reason,
                "replica": r.replica,
                "made_deadline": r.made_deadline,
                "prefix_hit": r.prefix_hit,
            }
            for r in recs
        ],
        "events": events,
    }


def _fmt(x: float) -> str:
    if x != x:
        return "  n/a"
    if abs(x) >= 100:
        return f"{x:6.0f}"
    return f"{x:6.1f}"


def print_table(rows: list[dict]) -> None:
    print()
    print(
        f"{'Config':<16} {'adm':>5} {'ref':>5} {'reasons':<28} "
        f"{'p50':>6} {'p99':>6} {'%dl':>6} {'hit%':>6} {'spread':>6} {'tok/s':>6}"
    )
    print("-" * 100)
    for r in rows:
        reasons = ",".join(f"{k}:{v}" for k, v in sorted(r["refused_by_reason"].items())[:4]) or "-"
        print(
            f"{r['config']:<16} {r['admitted']:5d} {r['refused']:5d} {reasons:<28} "
            f"{_fmt(r['ttft_p50'])} {_fmt(r['ttft_p99'])} {_fmt(r['deadline_pct'])} "
            f"{_fmt(r['prefix_hit_rate'])} {r['load_spread']:6d} {_fmt(r['tokens_per_s'])}"
        )
    print()


def write_html(rows: list[dict], path: Path) -> None:
    slim = []
    for r in rows:
        slim.append(
            {
                "config": r["config"],
                "admitted": r["admitted"],
                "refused_by_reason": r["refused_by_reason"],
                "ttft_p99": r["ttft_p99"],
                "deadline_pct": r["deadline_pct"],
                "prefix_hit_rate": r["prefix_hit_rate"],
                "load_spread": r["load_spread"],
                "records": [
                    {
                        "submit_t": rec["submit_t"],
                        "ttft_ms": rec["ttft_ms"],
                        "status": rec["status"],
                        "reason": rec["reason"],
                        "made_deadline": rec["made_deadline"],
                        "prefix_hit": rec["prefix_hit"],
                    }
                    for rec in r["records"]
                ],
                "events": [
                    {
                        "ts": e.get("ts"),
                        "event": e.get("event"),
                        "reason": e.get("reason"),
                        "queue_depth": e.get("queue_depth"),
                    }
                    for e in r["events"]
                    if e.get("event") in ("enqueued", "refused", "admitted", "completed")
                ],
            }
        )
    path.write_text(HTML.replace("__DATA__", json.dumps(slim)), encoding="utf-8")


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>llm-gateway-lab</title>
<style>
body{font:14px/1.4 -apple-system,BlinkMacSystemFont,sans-serif;background:#111;color:#eee;margin:24px}
h1{font-size:18px;font-weight:600}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.box{background:#1c1c1c;border:1px solid #333;padding:12px}
canvas{width:100%;height:220px;background:#111}
.legend{color:#aaa;font-size:12px;margin-top:6px}
</style></head><body>
<h1>four configs, one load</h1>
<div class="grid">
  <div class="box"><div>1 · admission outcomes over time</div><canvas id="c1"></canvas><div class="legend" id="l1"></div></div>
  <div class="box"><div>2 · queue depth + time-in-queue p99</div><canvas id="c2"></canvas><div class="legend" id="l2"></div></div>
  <div class="box"><div>3 · prefix hit rate + load spread</div><canvas id="c3"></canvas><div class="legend" id="l3"></div></div>
  <div class="box"><div>4 · TTFT p99 + deadline attainment</div><canvas id="c4"></canvas><div class="legend" id="l4"></div></div>
</div>
<script>
const DATA = __DATA__;
const C = ["#6ea8fe","#75b798","#ffc107","#e35d6a"];
function ax(ctx,w,h,pad){ctx.strokeStyle="#444";ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke();}
function bar(id, values, labels, legend){
  const c=document.getElementById(id), ctx=c.getContext("2d");
  const dpr=window.devicePixelRatio||1; c.width=c.clientWidth*dpr; c.height=c.clientHeight*dpr; ctx.scale(dpr,dpr);
  const w=c.clientWidth,h=c.clientHeight,pad=28, n=values.length, max=Math.max(...values.map(v=>Math.max(...v.flat?v: [v])), 1);
  ax(ctx,w,h,pad);
  const bw=(w-2*pad)/n*0.6;
  values.forEach((v,i)=>{
    const arr = Array.isArray(v)?v:[v];
    arr.forEach((x,j)=>{
      ctx.fillStyle=C[j%C.length];
      const bh=(h-2*pad)*(x/max);
      const x0=pad+(i+0.2)*(w-2*pad)/n + j*bw/arr.length;
      ctx.fillRect(x0, h-pad-bh, bw/arr.length-2, bh);
    });
    ctx.fillStyle="#aaa"; ctx.font="11px sans-serif";
    ctx.fillText(labels[i], pad+(i+0.25)*(w-2*pad)/n, h-8);
  });
  document.getElementById(legend).textContent = labels.map((l,i)=>l+": "+JSON.stringify(values[i])).join("   ");
}
const labels=DATA.map(d=>d.config.split("·")[0].trim());
bar("c1", DATA.map(d=>[d.admitted, d.records.filter(r=>r.status>=400).length]), labels, "l1");
bar("c2", DATA.map(d=>{
  const ev=d.events.filter(e=>e.event==="enqueued" && e.queue_depth!=null);
  const depths=ev.map(e=>e.queue_depth);
  const p99 = depths.length? depths.sort((a,b)=>a-b)[Math.floor(0.99*(depths.length-1))]:0;
  return [Math.max(...depths,0), p99];
}), labels, "l2");
bar("c3", DATA.map(d=>[d.prefix_hit_rate, d.load_spread]), labels, "l3");
bar("c4", DATA.map(d=>[isNaN(d.ttft_p99)?0:d.ttft_p99, d.deadline_pct]), labels, "l4");
</script></body></html>
"""


def _require_replicas(replicas: str) -> None:
    urls = [u.strip() for u in replicas.split(",") if u.strip()]
    if len(urls) < 2:
        raise SystemExit("need two replica URLs, e.g. http://127.0.0.1:8001,http://127.0.0.1:8002")
    bad: list[str] = []
    for u in urls:
        try:
            r = httpx.get(u.rstrip("/") + "/v1/models", timeout=3.0)
            if r.status_code != 200:
                bad.append(u)
        except Exception:
            bad.append(u)
    if bad:
        raise SystemExit(
            "replicas not up: "
            + ", ".join(bad)
            + "\nOn Lambda:  bash setup/launch_replicas.sh && make smoke"
        )


def _wait_health(base: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            r = httpx.get(base + "/health", timeout=1.0)
            if r.status_code == 200:
                return
            last = r.text
        except Exception as exc:
            last = str(exc)
        time.sleep(0.1)
    raise RuntimeError(f"gateway not healthy at {base}: {last}")


def _start_gateway(port: int, preset: str, replicas: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "gateway.main",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--preset",
        preset,
        "--replicas",
        replicas,
        "--tokenizer",
        os.environ.get("MODEL", "Qwen/Qwen3-0.6B"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.Popen(cmd, cwd=str(ROOT), env=env)


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _crew_records(gateway: str, n: int, workers: int) -> list:
    limiter = RateLimiter()
    queries = (DEFAULT_QUERIES * ((n // len(DEFAULT_QUERIES)) + 1))[:n]
    out: list[Record] = []

    def one(q: str) -> Record:
        rec = run_one(q, gateway=gateway, limiter=limiter, dual_agent=True, max_tokens=64)
        meta = rec.get("last") or {}
        status = int(meta.get("status") or (429 if not rec["ok"] else 200))
        reason = meta.get("reason")
        if rec["error"] and not reason:
            reason = "rate_limited" if "limiter" in rec["error"] else "error"
        return Record(
            stream="C",
            submit_t=time.perf_counter(),
            ttft_ms=meta.get("total_ms"),
            total_ms=rec["total_ms"],
            status=status,
            reason=reason,
            replica=meta.get("replica"),
            made_deadline=rec["ok"] and rec["total_ms"] <= 5000,
            prefix_hit=int(meta.get("prefix_match") or 0) >= 16,
            n_in=0,
            n_out=64,
            deadline_ms=5000,
        )

    with ThreadPoolExecutor(max_workers=max(workers, 1)) as pool:
        futs = [pool.submit(one, q) for q in queries]
        for fut in as_completed(futs):
            out.append(fut.result())
    return out


def run_experiment(n: int, workers: int, replicas: str, port: int) -> list[dict]:
    rows = []
    for label, preset in PRESETS:
        print(f"→ {label}  (CrewAI × {n} queries, {workers} workers)", flush=True)
        proc = _start_gateway(port, preset, replicas)
        try:
            _wait_health(f"http://127.0.0.1:{port}")
            recs = _crew_records(f"http://127.0.0.1:{port}", n, workers)
            stats = httpx.get(f"http://127.0.0.1:{port}/_stats", timeout=5.0).json()
            rows.append(summarize(label, recs, stats))
        finally:
            _stop(proc)
            time.sleep(0.3)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--queries", type=int, default=12)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument(
        "--replicas",
        default=os.environ.get("GATEWAY_REPLICAS", "http://127.0.0.1:8001,http://127.0.0.1:8002"),
    )
    p.add_argument("--out", default=str(ROOT))
    args = p.parse_args()
    _require_replicas(args.replicas)
    rows = run_experiment(args.queries, args.workers, args.replicas, args.port)
    print_table(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [
            {k: v for k, v in r.items() if k not in ("records", "events")} | {"n_events": len(r["events"])}
            for r in rows
        ]
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_html(rows, out / "results.html")
    print(f"wrote {out / 'results.json'} and {out / 'results.html'}")


if __name__ == "__main__":
    main()
