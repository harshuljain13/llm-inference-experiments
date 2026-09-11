"""Part 3 — Router.

Two fake workers, A and B. Each reports: free KV, running count, waiting count,
cached prefixes, p99 latency, health.

Four strategies: random, least_loaded, p2c, prefix_then_load.
"""

from __future__ import annotations


def pick(req, workers):
    """Choose which worker receives `req`.

    Two safety rules, both named after real failure modes:

    H6 — unknown is NOT idle.
        If a worker is unhealthy or its load is unknown, do not treat it as
        empty. Use unknown workers only when ALL workers are unknown.

        Missing telemetry looks exactly like zero load. A router that confuses
        the two sends everything to the one worker that stopped reporting —
        because it died.

    H4 — don't bounce an overloaded request.
        Before choosing, check whether each worker would reject `req` using
        your Part 1 logic. If both reject:
            return Shed(503, retry_after=2)
        Never loop A -> B -> A -> B. Failover under global overload multiplies
        load rather than relieving it.

    OPEN QUESTION — define "load".
        least_loaded and p2c both need one number per worker. Free KV? Running
        count? Queue depth? p99? A composite? T1 only shows something if this
        is defined well.

    OPEN QUESTION — how much prefix is worth how much load?
        prefix_then_load must trade cache affinity against balance. A hard
        threshold, or a weighted score? If weighted, what is the exchange rate
        between "matched tokens" and "one unit of load"? Module 6's router
        answers this with W_PREFIX and W_LOAD — worth reading, then deciding
        for yourself.

    OPEN QUESTION — how do you DETECT staleness?
        T3 makes worker B's telemetry 15 seconds old while it claims to be
        empty. Detecting that needs a timestamp. Does the worker stamp its own
        report? What if its clock is skewed?

    OPEN QUESTION — what load does an unknown worker get?
        Infinity, or excluded from candidates entirely? Those differ when every
        worker is unknown.
    """
    raise NotImplementedError("Part 3 — see docstring.")
