"""Part 2 — Scheduler.

The request is already accepted. The GPU can do a limited amount of work per
step. Decide what runs.

Each call to step() means: "the GPU gets one more chance to do work." Requests
are NOT run to completion — they interleave, which is what continuous batching
means.

Three policies: fcfs, priority, drr.
    priority 0 before priority 20; ties broken by arrival time.
"""

from __future__ import annotations


def step(waiting, running, budget):
    """Advance the simulated GPU by one step.

    Four rules:

    1. Chunked prefill.
       A 32,000-token prompt against a 2,048-token budget does NOT get
       processed at once. Take at most 2,048 this step and continue later.

    2. One prefill chunk, then decode.
       After at most ONE prefill chunk, spend what remains on requests already
       decoding. A long prefill must not starve sequences mid-generation.

    3. Preempt, don't swap.
       When KV runs out, stop the lowest-priority running request, THROW AWAY
       its KV, and return it to the waiting queue. It recomputes its prompt
       when it runs again.
           count: preempts, wasted_decode_tokens

    4. Honour aborts.
       If req.aborted, remove it immediately, free its KV, generate nothing
       more.
           count: aborted_freed

    OPEN QUESTION — what is `budget`?
        Prefill is measured in TOKENS (chunk <= 2048). Decode is measured in
        SLOTS (N sequences each advance one token). The spec passes one
        `budget`. "Use whatever remains for decoding" — remaining WHAT? Tokens
        and slots are not the same currency. Scalar, tuple, or something that
        prices a decode pass properly?

    OPEN QUESTION — which victim?
        "Lowest-priority running request." Priority 20 is batch, so is lowest
        priority the highest number? And among ties: newest (least work lost)
        or oldest (most progress, so most waste)? These give opposite
        wasted_decode_tokens columns.

    OPEN QUESTION — is the metric named right?
        Preemption discards KV, so the victim must recompute its PROMPT. That
        is wasted prefill. The metric is called wasted_decode_tokens. Are you
        counting the whole cost, or the part the spec happened to name?

    OPEN QUESTION — DRR across what?
        Tenants? Priority classes? The spec never says. What is your quantum,
        and what makes the result fair?

    OPEN QUESTION — starvation.
        One prefill chunk per step. With eight 32k-token prompts queued, when
        does the eighth start? Does anything here prevent that?
    """
    raise NotImplementedError("Part 2 — see docstring.")
