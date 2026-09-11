from __future__ import annotations

import hashlib
import struct
import time
from dataclasses import dataclass, field

from gateway import config as cfg


@dataclass
class _Node:
    children: dict[int, _Node] = field(default_factory=dict)
    expires_at: float = 0.0


def block_hashes(token_ids: list[int], block_size: int = cfg.BLOCK_SIZE) -> list[int]:
    out: list[int] = []
    n = len(token_ids) - (len(token_ids) % block_size)
    for i in range(0, n, block_size):
        raw = struct.pack(f"{block_size}I", *[int(t) & 0xFFFFFFFF for t in token_ids[i : i + block_size]])
        digest = hashlib.blake2s(raw, digest_size=8).digest()
        out.append(int.from_bytes(digest, "little"))
    return out


class PrefixTrie:
    def __init__(self, ttl_s: float = cfg.TRIE_TTL_S) -> None:
        self._ttl = ttl_s
        self._roots: dict[str, _Node] = {}

    def insert(self, replica: str, token_ids: list[int]) -> None:
        now = time.monotonic()
        node = self._roots.setdefault(replica, _Node())
        node.expires_at = now + self._ttl
        for h in block_hashes(token_ids):
            node = node.children.setdefault(h, _Node())
            node.expires_at = now + self._ttl

    def match(self, replica: str, token_ids: list[int]) -> int:
        node = self._roots.get(replica)
        if node is None:
            return 0
        now = time.monotonic()
        matched = 0
        for h in block_hashes(token_ids):
            nxt = node.children.get(h)
            if nxt is None or nxt.expires_at < now:
                break
            matched += cfg.BLOCK_SIZE
            node = nxt
        return matched
