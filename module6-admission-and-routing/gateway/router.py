from __future__ import annotations

import random

from gateway import config as cfg
from gateway.state import PendingRequest, ReplicaState
from gateway.trie import PrefixTrie


def score(r: ReplicaState, req: PendingRequest, match_tokens: int) -> float:
    load = (r.waiting + r.running) / max(r.max_num_seqs, 1)
    load = min(load, cfg.LOAD_CEILING)
    return cfg.W_PREFIX * match_tokens - cfg.W_LOAD * load


class Router:
    def __init__(self) -> None:
        self.trie = PrefixTrie()
        self._rr = 0

    def pick(self, replicas: list[ReplicaState], req: PendingRequest) -> tuple[ReplicaState, int]:
        if not replicas:
            raise RuntimeError("no replicas")
        if cfg.USE_P2C and len(replicas) >= 2:
            candidates = random.sample(replicas, 2)
        else:
            candidates = list(replicas)
        scored: list[tuple[float, int, ReplicaState]] = []
        for r in candidates:
            match = self.trie.match(r.id, req.token_ids)
            s = score(r, req, match) if cfg.USE_PREFIX_ROUTING else 0.0
            scored.append((s, match, r))
        best = max(s for s, _, _ in scored)
        tied = [(m, r) for s, m, r in scored if s == best]
        chosen_match, chosen = tied[self._rr % len(tied)]
        self._rr += 1
        self.trie.insert(chosen.id, req.token_ids)
        return chosen, chosen_match
