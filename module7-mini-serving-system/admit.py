"""Part 1 — Admission control.

One decision: let this request in, or turn it away. Nothing else. No queueing,
no routing, no retrying. The spec is explicit that admission "does not retry
requests or move them to another worker."

Two kinds of "no", and the difference is the point of this file:

    429       the tenant used its quota      -> the client's doing
    503/529   the fleet is out of capacity   -> ours

A client receiving 429 should slow down. One receiving 503 should retry, maybe
elsewhere. Returning the wrong code sends the wrong instruction.

Write the seven tests in ASSIGNMENT.md before implementing anything here.
"""

from __future__ import annotations


def should_shed(req, snap) -> tuple[bool, int | None, int | None]:
    """Decide whether to reject `req`, given server state `snap`.

    Returns (shed, code, retry_after_seconds). On accept: (False, None, None).

    Four rules. Evaluation ORDER is yours to derive — the spec does not give it,
    but Advanced #3 constrains you: a tenant over its token limit against an
    empty fleet must get 429, not 503. Work out what that forces.

    1. Protect tenants.
       Check the tenant's TOKEN limit before its REQUEST-COUNT limit. Ten
       requests are not ten times the same work.  -> 429

    2. Don't accept the doomed.
           expected_queue_wait = queue_length * p50_TTFT
       If that exceeds half of req.timeout_s, reject. Running work the user has
       already abandoned spends KV on output nobody reads.  -> 503/529

    3. Protect KV memory.
       If less than 8% of KV remains, reject a request whose prefix is NOT
       cached. One reusing a cached prefix costs far less, so let it in.
                                                                -> 503/529

    4. Protect interactive users.
       If p99 > 4 * p50 AND the queue is growing:
           keep   priority <  10
           reject priority >= 10
                                                                -> 503/529

    OPEN QUESTION — what is `snap`?
        Rule 4 needs "the queue is growing", which is a derivative. You cannot
        compute it from a point-in-time snapshot. So either snap carries
        history, or something upstream maintains the trend and hands it to you.
        Decide which, and over what window. Too short and you shed on noise;
        too long and you shed after the collapse.

    OPEN QUESTION — whose queue, whose KV?
        Rules 2 and 3 say "queue length" and "8% of KV" without saying whose.
        Your answer depends on where admission runs relative to routing.
        See overview.md.

    OPEN QUESTION — retry_after.
        The spec never gives values. Should a quota rejection and a KV-pressure
        rejection tell the client to come back at the same time? Why not?
    """
    raise NotImplementedError("Part 1 — see docstring, and write the tests first.")
