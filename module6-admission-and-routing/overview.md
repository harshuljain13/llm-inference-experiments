# Module 6 — Gateway: Admission, Queueing, Routing

> Course **class 7**. Decide what enters, in what order, and on which replica.

**One sentence:** Put a gateway in front of two real vLLM replicas and make three decisions well — *should this request be accepted, in what order should it run, and which replica should get it.*

Module 3 built the engine. Module 6 accepts it as given (real vLLM, `--max-num-seqs 8`, not raisable) and moves up a layer. **The constraint is the lesson:** 16 total slots on one GPU, and you may not add more.

---

## The stack

```
CrewAI agents            app.py           the workload
   ↓
RateLimiter              limiter.py       client-side, fail fast
   ↓
Gateway :8080            gateway/
   ├─ Buckets            per-tenant token quota      → 429
   ├─ should_shed        load-aware admission        → 503
   ├─ PendingQueue       bounded, deadline-ordered
   ├─ Router + Trie      which replica
   └─ proxy              forward, stream, account
        ↓
   vLLM :8001   vLLM :8002      max-num-seqs 8 each
        ↑
   scrape_loop reads /metrics every 250ms
```

Each layer exists because the layer below has a small, hard capacity.

## File map

| File | Role |
|---|---|
| `app.py` | CrewAI client — `GatewayLLM(BaseLLM)` posts OpenAI-shaped JSON to the gateway |
| `limiter.py` | Client-side token bucket. **`try_acquire`, not wait** — refuses locally, no round trip |
| `gateway/config.py` | Every knob, plus the four presets |
| `gateway/main.py` | CLI → config → uvicorn |
| `gateway/internal/app.py` | **The centerpiece.** Handler, worker, dispatch, proxy, accounting |
| `gateway/scrape.py` | Polls vLLM `/metrics`, parses Prometheus, learns fleet speed |
| `gateway/state.py` | `ReplicaState`, `FleetState`, `PendingRequest`, `QueueItem` |
| `gateway/admission.py` | `should_shed` — 14 lines, returns a reason or `None` |
| `gateway/queue.py` | `queue_key` — the priority tuple |
| `gateway/internal/pending_queue.py` | Bounded queue, expiry, anti-starvation counters |
| `gateway/router.py` | `score()` and `pick()` — cache affinity vs. load |
| `gateway/trie.py` | Block-hash prefix trie mirroring vLLM's cache |
| `gateway/internal/bucket.py` | Per-tenant token buckets |
| `gateway/internal/stats.py` | Ring buffer of events, served at `/_stats` |
| `bench/report.py` | Runs all four presets, writes `results.json` / `results.html` |
| `setup/*.sh` | Mac→Lambda sync, SSH, replica launch, smoke test |

## Request lifecycle

Trace one request through `internal/app.py`:

1. **Tokenize** — `messages_to_text` → `tokenize`. Needed for prefix matching and size estimates.
2. **Deadline** — `deadline_ms` from the body, else from tenant tier (`agent → agentic → 5000ms`, `default → interactive → 2000ms`).
3. **Quota** — `buckets.allow(tenant, n_in + n_out)`. Fail → **429**. The only 429 path.
4. **Admission** — `should_shed(fleet, req)` if enabled. Fail → **503** with `Retry-After`.
5. **Queue** — if enabled, enqueue and await a future; a worker task resolves it. Else dispatch directly.
6. **Dispatch** — `wait_slot()` then `router.pick()`.
7. **Proxy** — forward to the replica, stream or not, then account.

## The five admission checks

`admission.py`, in order:

| Check | Reason | Why |
|---|---|---|
| `stale_for > 2s` | `no_signal` | Blind. Refuse rather than guess. |
| `kv_usage_max > 0.85` | `kv_pressure` | Near the memory wall |
| `waiting_total > 4 × replicas` | `queue_depth` | Backlog already deep |
| `headroom < n_in + n_out` | `no_headroom` | Literally doesn't fit |
| `prefill + decode + wait > deadline` | `deadline_unmeetable` | Can't finish in time |

**The last one is the interesting one.** The gateway predicts completion from scraped rates and refuses work it can't finish. A refusal at t=0 beats a timeout at t=5s, because the refused request never consumed KV.

**`no_signal` is the subtle one.** `scrape_ok_at` is set *only* when every replica answers, and `stale_for` returns `1e9` before the first success. That makes "I have no information" a distinct state — otherwise blindness looks identical to zero load, and you'd cheerfully dispatch into a fire.

## The queue's priority tuple

`queue.py` — three policies in one sort key:

```python
starved = item.passed_over >= MAX_OVERTAKES        # 8
long    = 0 if starved or n_in < LONG_PROMPT_TOKENS else 1
slack   = (deadline_at - now) - AGING_GAIN * passed_over
return (long, slack, enqueued_at)
```

- **Long prompts last** — a 1024+ token prefill blocks the batch (module 1: prefill is compute-bound and chunky).
- **EDF by slack** — least time-to-deadline first.
- **Anti-starvation** — each overtake increments `passed_over`, which ages the item's slack downward and, after 8, promotes it out of the "long" penalty class entirely.

Without the aging term, a long prompt under steady short-request load never runs. `_expire()` fails items whose deadline passed while queued — no point dispatching work that's already useless.

## The router

```python
load  = min((waiting + running) / max_num_seqs, LOAD_CEILING)   # 2.0
score = W_PREFIX * match_tokens - W_LOAD * load                  # 1.0, 64.0
```

Cache affinity vs. load balance, as one number. `W_LOAD = 64` means one full unit of load costs 64 tokens of prefix match — a single 16-token block won't drag you onto a busy replica, but a few hundred matched tokens will.

**Known weakness, observed in the bench:** at low load the `load` term is ≈0, so prefix match wins unopposed and everything piles onto whichever replica won first. The balancing term only bites when replicas are genuinely busy.

`USE_P2C` samples 2 random candidates first — power-of-two-choices, O(1) as the fleet grows, and avoids everyone stampeding to the same "best" replica.

### The trie

`trie.py` mirrors vLLM's own block cache: tokens → 16-token blocks → blake2s hashes → a path under each replica's root. `match()` walks until a miss and returns matched *tokens*. Nodes carry a 90s TTL so the gateway's belief decays like the real cache does.

Critically, `pick()` **inserts on dispatch** — the routing decision updates the model. This is what makes the researcher→writer crew hit: same prefix, remembered replica.

## Dispatch control

`wait_slot()` blocks until in-flight < `Σ max_num_seqs + DISPATCH_OVERSHOOT` (8+8+4 = 20). The overshoot is deliberate: keep vLLM's own queue slightly non-empty so the GPU never idles between batches, without letting the backlog balloon.

On the proxy path: `deadline_ms` is stripped (ours, not vLLM's) and `priority` passes through — replicas run `--scheduling-policy priority`, so priority reaches vLLM's scheduler. Streams are teed so `_finish()` runs even on client disconnect; mid-stream `is_disconnected()` aborts upstream and marks `cancelled`. **Cancellation propagation matters** — an abandoned stream that keeps decoding burns slots on output nobody reads.

## The four presets

`config.apply_preset` flips three booleans; `bench/report.py` runs all four under identical load:

| preset | admission | queue | prefix routing |
|---|---|---|---|
| `baseline` | ✗ | ✗ | ✗ — round-robin, unbounded |
| `route` | ✗ | ✗ | ✓ |
| `queue` | ✗ | ✓ | ✓ |
| `full` | ✓ | ✓ | ✓ |

**The intended story:** `route` buys TTFT via prefix hits; `queue` buys deadline attainment by reordering; `full` trades throughput for a better *made-deadline* rate, because the requests it drops were going to miss anyway.

**Caveat from actual runs:** with 12–60 queries the fleet was never saturated, so admission never fired and the queue never held more than a few items. The presets only diverge under genuine overload. If `reasons` shows only `rate_limited`, the client limiter is the bottleneck and the gateway isn't being tested.

## Reading the bench output

| col | means |
|---|---|
| `adm` / `ref` | completed vs. rejected |
| `reasons` | **who** rejected and why — `rate_limited` = client, anything else = gateway |
| `p50`/`p99` | *not* TTFT despite the name — last gateway call's wall time |
| `%dl` | share of all crews under the deadline; rejections count as misses |
| `hit%` | calls with ≥1 cached prefix block |
| `spread` | replica imbalance; 0 = even |
| `tok/s` | effectively admitted-crews/sec (token counts are hardcoded) |

Two known accounting quirks: `queue_depth_max` is 0 for presets 1–2 because the queue is *disabled*, not because it was empty; and `_crew_records` falls back to a stale `last_meta`, so a crew rate-limited on its *second* call is miscounted as admitted.

## What to take away

- **Refuse early or fail late — pick one.** Admission control is how a system degrades gracefully instead of collapsing uniformly (the class-2 failure mode).
- **Unknown ≠ idle.** Stale telemetry must be its own state.
- **Cache affinity and load balance are in tension**, and the weights that resolve it only work in the load regime you tuned them for.
- **Aging is mandatory** anywhere you reorder a queue, or you've built starvation.
- **Measure under real overload**, or your policies are untested no-ops that look like they work.

## The thread through the whole repo

| Module | Layer | Question |
|---|---|---|
| 1 | Model | Why is decode slow and prefill fast? |
| 2 | Server + Gateway | Which layer owns which problem? |
| 3 | Engine | How is KV memory paged and work scheduled? |
| 4 | Production | What does the stack cost, and where does time go? |
| 5 | Multi-GPU | Split the model, or replicate it? |
| 6 | Gateway | Who gets in, in what order, and on which GPU? |

One idea throughout: **the GPU is scarce.** Decide carefully what enters, what runs next, where it runs — then measure whether the decisions helped.
