# Module 7 — Mini Serving System: Admission, Scheduler, Router

> Course **class 7, homework 2**. Build the three decisions yourself, against a simulated GPU.

**Status: not started.** This module is a spec and a set of stubs. Nothing is implemented yet.

---

## The ask, in one line

> **The GPU is scarce.** Decide what work enters, what runs next, and where it runs — then measure whether your decisions helped.

No model. No GPU. Everything simulated, which is the point: you can sweep load, inject failures, and rewind the clock for free.

## Why simulated

[Module 6](../module6-admission-and-routing/overview.md) makes these same three decisions against real vLLM. So why build a fake one?

| | Real (module 6) | Simulated (here) |
|---|---|---|
| Determinism | run-to-run variance swamps small effects | same seed, same answer |
| Time | 60 seconds takes 60 seconds | 60 simulated seconds is instant |
| Failures | staging a 15s-stale worker is an ordeal | one line |
| Cost | GPU-hours per sweep | free |

Policy is easier to get right when engine noise is removed. **Module 6 proves it works on real hardware; module 7 proves you understand why it works.**

## The three decisions

```
request arrives
      ↓
  admit.py     should_shed()   accept, or reject with a reason
      ↓
  router.py    pick()          which of two workers
      ↓
  sched.py     step()          which request gets the GPU this tick
      ↓
  serve.py                     all three, one clock
```

| File | Decision | The hard part |
|---|---|---|
| `admit.py` | accept or reject | telling `429` (your quota) from `503` (our capacity) |
| `sched.py` | what runs next | chunked prefill, and preemption that throws away work |
| `router.py` | which worker | unknown ≠ idle; don't bounce requests between workers |
| `serve.py` | integration | one process, two workers, one simulated clock |

## Three scarce resources

Everything you write protects one of these:

| Resource | What runs out |
|---|---|
| `decode_slots` | how many sequences can generate at once |
| `KV blocks` | memory held by running requests |
| `prefill tokens` | input processed per step |

Module 3 built the mechanisms that manage these. Here you write the **policy** that decides who gets them.

## Design decisions the spec leaves open

The assignment does not answer these. They are the assignment.

1. **Where does admission run** — before routing, inside it, or both? Part 4's diagram and H4 in Part 3 disagree. Resolve it and write down why.
2. **What is `snap`?** Rule 4 needs *"the queue is growing"* — a derivative. You cannot compute that from a point-in-time snapshot. So what does `snap` carry, and over what window?
3. **What is `budget`?** Prefill is measured in tokens, decode in slots. The spec passes one `budget`. Scalar, tuple, or something else?
4. **Rule evaluation order.** The spec orders the tenant checks but not rules 1–4. Advanced #3 constrains you: *tenant over limit + empty fleet → 429, not 503.* Derive the order that satisfies it.
5. **DRR across what?** Tenants, or priority classes? The spec never says. What's your quantum?

## How will you know the simulator is right?

There's no ground truth. **A bug and a finding look identical** — you'll get a confident table either way.

Pick invariants you can assert every step:

- KV blocks allocated + free == total, always
- no sequence RUNNING without a block table
- tokens produced + tokens wasted == tokens attempted
- a preempted request's KV returns to the free list before the next step

Without these, every number in the report is unfalsifiable.

## Layout

```
module7-mini-serving-system/
  ASSIGNMENT.md   the brief, restructured
  overview.md     this file
  admit.py        stub — Part 1
  sched.py        stub — Part 2
  router.py       stub — Part 3
  serve.py        stub — Part 4
  traces/         mixed.jsonl and the router traces
  tests/          the seven admission cases, then the rest
  plots/          soak.png
```

## Deliverables

`admit.py` · `sched.py` · `router.py` · `serve.py` · `traces/` · `plots/soak.png` · `REPORT.pdf` (≤ 2 pages)

Advanced track reproduces real postmortems — DALL·E soak, worker failure and gradual recovery, correct error codes, prefix stickiness, abort handling.

## Where this sits

```
module3-build-your-own-engine   the mechanisms: paging, batching, preemption
      ↓
module6-admission-and-routing   the policy, on real vLLM
      ↓
module7-mini-serving-system     the policy, built from scratch and stress-tested
```

**Handbook:** Ch. 05.3 (Continuous Batching), Ch. 08.6 (Cache-Aware Routing), Ch. 09.1 (Benchmarking)
