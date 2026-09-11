"""Reference solutions for build exercises (import if stuck)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from ..block_manager import BlockSpaceManager
    from ..sequence import SequenceGroup, SequenceStatus


def allocate(bm: "BlockSpaceManager", seq_id: int, num_tokens: int) -> None:
    n = math.ceil(num_tokens / bm.block_size)
    if len(bm._free_blocks) < n:
        raise ValueError(f"need {n} blocks, have {len(bm._free_blocks)} free")
    blocks = []
    for _ in range(n):
        phys = bm._free_blocks.popleft()
        bm._ref_count[phys] = 1
        blocks.append(phys)
    bm._block_tables[seq_id] = blocks
    bm._tokens_allocated[seq_id] = num_tokens


def append_slot(bm: "BlockSpaceManager", seq_id: int) -> None:
    if seq_id not in bm._block_tables:
        raise ValueError(f"seq {seq_id} not allocated")
    bm._tokens_allocated[seq_id] += 1
    needed = math.ceil(bm._tokens_allocated[seq_id] / bm.block_size)
    current = len(bm._block_tables[seq_id])
    if needed > current:
        if not bm._free_blocks:
            raise ValueError("no free blocks")
        phys = bm._free_blocks.popleft()
        bm._ref_count[phys] = 1
        bm._block_tables[seq_id].append(phys)


def schedule_promotions(
    waiting: List["SequenceGroup"],
    running: List["SequenceGroup"],
    *,
    max_batch_size: int,
    can_allocate: Callable[[int], bool],
    allocate_fn: Callable[[int, int], None],
) -> tuple[List["SequenceGroup"], List["SequenceGroup"]]:
    from ..sequence import SequenceStatus

    w = list(waiting)
    r = list(running)
    while w and len(r) < max_batch_size:
        group = w[0]
        ntok = group.sequences[0].num_tokens
        if not can_allocate(ntok):
            break
        w.pop(0)
        allocate_fn(group.group_id, ntok)
        group.sequences[0].status = SequenceStatus.RUNNING
        r.append(group)
    return w, r
