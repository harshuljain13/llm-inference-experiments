# Mini LLM Serving System

> **Capstone project.** Build admission control, a scheduler, and a router against a simulated GPU — then connect all three and prove the whole thing holds under overload.
>
> Companion to [Class 7](../class7/overview.md), which makes the same three decisions against real vLLM replicas.

**No model. No GPU.** Everything is simulated, which is exactly the point: you can sweep load, inject failures, and rewind the clock for free.

---

## The One Idea

> **The GPU is scarce.** Decide carefully what work enters, what runs next, and where it runs. Then measure whether your decisions actually helped.

Imagine 100 users hitting a server with limited capacity. Your system answers three questions:

| Question | Component | File |
|---|---|---|
| Should I accept this request? | **Admission control** | `admit.py` |
| Which request runs next? | **Scheduler** | `sched.py` |
| Which worker gets it? | **Router** | `router.py` |
| *All three, wired together* | **Integration** | `serve.py` |

---

## The Simulated World

**Every request carries:**

| Field | Meaning |
|---|---|
| `id` | request identifier |
| `arrival_t` | when it showed up |
| `priority` | `0` = interactive, `20` = batch |
| `prompt_tokens` | input size |
| `max_new_tokens` | maximum output size |
| `prefix_hash` | identifies shared input, or `None` |
| `timeout_s` | how long the user will wait |
| `tenant` | customer / user group |

**Your server has three scarce resources:**

| Resource | Meaning |
|---|---|
| `decode_slots` | how many requests can generate tokens at once |
| `KV blocks` | memory available to running requests |
| `prefill tokens` | how much input can be processed per step |

Every rule you write below exists to protect one of these three.

---

## Part 1 — Admission Control

**File:** `admit.py` · **Question:** *is the server too busy to safely accept this?*

```python
should_shed(req, snap) -> (shed: bool, code: int, retry_after_s: int)
```

### Write these tests first

| Situation | Expected result |
|---|---|
| Tenant used 96% of **token** allowance | reject `429` |
| Tenant used 96% of **request** allowance | reject `429` |
| Request would likely wait > half its timeout | reject `503`/`529` |
| Only 5% KV remains, **new** prefix | reject `503`/`529` |
| Only 5% KV remains, prefix **already cached** | **accept** |
| Very bad tail latency, **interactive** request | **accept** |
| Very bad tail latency, **batch** request | reject |

The two "accept" rows are the interesting ones. They're what stops this from being a blunt load shedder.

### Then implement these rules

**1 · Protect tenants.** Check the tenant's *token* limit before its *request-count* limit.
> Ten requests are not necessarily ten times the work. Counting requests lets one tenant smuggle in enormous jobs.

**2 · Don't accept doomed requests.**
```text
expected_queue_wait = queue_length × p50_TTFT
```
If that exceeds **half** the request's timeout, reject it now. A refusal at t=0 beats a timeout at t=5s — the refused request never consumed KV.

**3 · Don't run out of KV memory.** If less than **8%** of KV remains, reject requests whose prefix is *not* cached. A request reusing an existing prefix costs far less memory, so let it in.

**4 · Protect interactive users.** If `p99 > 4 × p50` **and** the queue is growing:
- keep `priority < 10`
- reject `priority >= 10`

> **Scope:** admission control only says accept or reject. It does **not** retry or reroute.

### Status codes

| Code | Means |
|---|---|
| `429` | tenant limit — *you* asked for too much |
| `503` / `529` | server capacity — *we* have no room |

Getting this wrong is a real production bug: a `429` tells the client to slow down, a `503` tells it to retry elsewhere.

### Short answer (≈ ½ page)

Explain which of your rules embodies the thinking behind:
- DALL·E's 5-minute cancellation
- Anthropic's late-capacity behavior
- Cloudflare overload protection

---

## Part 2 — Scheduler

**File:** `sched.py` · **Question:** *which accepted request gets the GPU next?*

```python
step(waiting, running, budget)
```

Each `step()` call is one chance for the GPU to do work.

### Policies

Support three: `fcfs`, `priority`, `drr`.

For `priority`: `0` before `20`; ties broken by arrival time.

### Rules

**1 · Chunked prefill.** A 32,000-token prompt against a 2,048-token budget does **not** get processed at once. Take at most 2,048 this step and continue later.

**2 · Don't let one huge prompt block decoding.** After at most *one* prefill chunk, spend the remaining budget on requests already decoding.
```text
one prefill chunk  →  remaining capacity goes to decode
```

**3 · Preempt, don't swap.** When KV runs out, stop the lowest-priority running request, **throw away its KV**, and return it to the waiting queue. It recomputes its prompt when it runs again. Count `preempts` and `wasted_decode_tokens`.

**4 · Honor aborts.** When `req.aborted == True`, remove it immediately, free its KV, generate nothing more. Count `aborted_freed`.

### The workload

Build `traces/mixed.jsonl`:

| Share | Class | Priority | Prompt | Output |
|---|---|---|---|---|
| 70% | interactive | 0 | 200–800 | 64–256 |
| 20% | batch | 20 | 2,000–8,000 | 512–2,048 |
| 10% | shared-prefix agents | — | 4,000 (3,500 shared) | — |

Run **60 simulated seconds** under FCFS, Priority, and DRR.

### Measure

`completed/s` · `interactive p99 TTFT` · `batch p99 TTFT` · `preempts/s` · `wasted decode tokens` · `requests rejected`

**Answer:** which scheduler wastes the most decode work, and why? (one paragraph)

---

## Part 3 — Router

**File:** `router.py` · **Question:** *which worker should receive this?*

```python
pick(req, workers)
```

Two fake workers, A and B. Each reports: free KV, running count, waiting count, cached prefixes, p99 latency, health.

**Strategies:** `random` · `least_loaded` · `p2c` · `prefix_then_load`

### Safety rules

**H6 · Unknown ≠ idle.** If a worker is unhealthy or its load is unknown, do **not** treat it as empty. Use unknown workers only when *all* workers are unknown.
> Missing telemetry looks identical to zero load. Systems that confuse the two route everything at the one worker that stopped reporting — because it crashed.

**H4 · Don't bounce an overloaded request.** Before choosing, check whether each worker would reject via your Part 1 logic. If both reject:
```python
return Shed(503, retry_after=2)
```
Never loop A → B → A → B. Failover under global overload multiplies the load instead of relieving it.

### Experiments

| Trace | Setup | Question |
|---|---|---|
| **T1** | no shared prefixes | Does P2C balance better than random? |
| **T2** | 40% share one prefix | Does prefix-aware routing save KV? |
| **T3** | worker B's telemetry is 15s stale, claims "empty" while busy | Does `least_loaded` overload B — and does H6 prevent it? |

**T2 target:**
```text
KV(prefix_then_load)  <  0.4 × KV(least_loaded)
```

**Report for all four strategies:** p99 TTFT · KV allocated · shed % · traffic sent to stale B

---

## Part 4 — Integration

**File:** `serve.py`

```text
client
   ↓
admit()     should we accept it?
   ↓
pick()      which worker?
   ↓
worker queue
   ↓
step()      GPU does work
```

One process · two fake workers · one simulated clock.

---

## Deliverables

```text
admit.py
sched.py
router.py
serve.py
traces/
plots/soak.png
tests/            (advanced track only)
REPORT.pdf        ≤ 2 pages — tables with short explanations
```

---

## Self-Check

You should be able to point at a line of code for each:

- [ ] Where do I prevent accepting work that will time out?
- [ ] Where do I protect KV memory?
- [ ] Where do I prioritize interactive traffic?
- [ ] Where do I stop one tenant monopolizing the GPU?
- [ ] Where do I preempt a request?
- [ ] Where do I exploit shared prefixes?
- [ ] Where do I handle missing worker telemetry?
- [ ] Where do I stop failover from making overload worse?

---

## Advanced Track

> These reproduce real postmortems from OpenAI, Anthropic, and Cloudflare. The question is whether you can fix them the way those engineering teams did.

### 1 · DALL·E soak

Set `timeout = 5s`, `p50 job time = 2s`. Keep offering load until estimated queue wait passes 2.5s.

**Requirement:** shedding must begin *before* completions fall to zero.

Produce `plots/soak.png` showing **admitted**, **completed**, and **shed** over time.

### 2 · Worker failure and recovery

Kill worker B → admission should drop. Bring B back → ramp admission 10% → 100% over 30s.

Continue ramping **only while** `p99 TTFT < 2 × baseline`. If it exceeds that, stop ramping and shed.
> This is why recovery is gradual. Restoring full traffic instantly re-kills the worker that just came back.

### 3 · Correct error code

| Condition | Must return | Must NOT return |
|---|---|---|
| Tenant over token limit, fleet empty | `429` | `503` |
| Fleet out of KV, tenant under limit | `503`/`529` | `429` |

### 4 · Prefix stickiness

On T2: `KV(prefix_then_load) < 0.4 × KV(least_loaded)`

### 5 · Abort

Abort a request mid-decode. The next `step()` must free its KV, increment `aborted_freed`, and generate **zero** further tokens.
