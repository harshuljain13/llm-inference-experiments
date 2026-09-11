"""Part 4 — Integration.

One process. Two fake workers. One simulated clock.

    client
       ↓
    admit()      should we accept it?
       ↓
    pick()       which worker?
       ↓
    worker queue
       ↓
    step()       GPU does work

OPEN QUESTION — where does admission actually run?
    This diagram puts admit() before pick(). But H4 in Part 3 says routing
    checks admission PER WORKER. Both cannot be literally true. Resolve it and
    write down why, because the deliverable question "where do I stop failover
    from making overload worse?" points straight at this seam.

OPEN QUESTION — what kind of clock?
    Fixed-tick is simpler, but the tick size becomes a hidden parameter that
    changes your results. Event-driven is accurate and harder. If fixed: what
    tick, and how would you notice it was too coarse?

INVARIANTS — assert these every step.
    There is no ground truth here. A bug and a finding look identical; you get
    a confident table either way. Without invariants the report is
    unfalsifiable.

        kv_allocated + kv_free == kv_total
        no request RUNNING without a block table
        tokens_produced + tokens_wasted == tokens_attempted
        a preempted request's KV is back on the free list before the next step
        every request ends in exactly one terminal state
            (completed | rejected | aborted | expired)
"""

from __future__ import annotations


def main():
    raise NotImplementedError("Part 4 — wire admit, pick and step to one clock.")


if __name__ == "__main__":
    main()
