from __future__ import annotations

from gateway.router import Router, score
from gateway.state import PendingRequest, ReplicaState
from gateway.trie import PrefixTrie, block_hashes


def _req(n: int = 64) -> PendingRequest:
    return PendingRequest(
        n_in=n,
        n_out=16,
        deadline_s=2.0,
        tenant="chat",
        token_ids=list(range(n)),
        messages=[],
        stream=False,
        body={},
        deadline_at=10.0,
    )


def _r(i: str, waiting: int = 0, running: int = 0) -> ReplicaState:
    return ReplicaState(url="http://x", id=i, waiting=waiting, running=running, max_num_seqs=8)


def test_score_prefers_prefix():
    r = _r("r0")
    req = _req()
    assert score(r, req, 128) > score(r, req, 0)


def test_score_prefers_lighter_replica():
    req = _req()
    light = _r("a", waiting=0, running=1)
    heavy = _r("b", waiting=6, running=8)
    assert score(light, req, 0) > score(heavy, req, 0)


def test_trie_match_is_tokens_not_blocks():
    t = PrefixTrie()
    ids = list(range(64))
    t.insert("r0", ids)
    assert t.match("r0", ids) == 64
    assert t.match("r1", ids) == 0
    assert len(block_hashes(ids)) == 4


def test_pick_inserts_at_dispatch_not_later():
    router = Router()
    req = _req(32)
    a, b = _r("r0"), _r("r1")
    chosen, match = router.pick([a, b], req)
    assert match == 0
    assert router.trie.match(chosen.id, req.token_ids) == 32
