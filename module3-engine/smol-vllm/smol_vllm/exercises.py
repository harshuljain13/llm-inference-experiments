"""Student build exercises — implement stubs, then run checkpoints in the notebook."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .block_manager import BlockSpaceManager
    from .sequence import SequenceGroup


def allocate(bm: "BlockSpaceManager", seq_id: int, num_tokens: int) -> None:
    """Allocate KV blocks for a sequence.

    Algorithm:
      blocks_needed = ceil(num_tokens / block_size)
      Pop that many physical block IDs from the free list (FIFO order).
      Store them as this sequence's block table; set ref_count=1 on each.

    Invariant: after allocate, len(block_table) == blocks_needed and
    num_free_blocks decreased by blocks_needed.
    """
    raise NotImplementedError("Implement allocate() — see docstring for the spec.")


def append_slot(bm: "BlockSpaceManager", seq_id: int) -> None:
    """Grow KV storage when a new generated token crosses a block boundary.

    Algorithm:
      Increment the sequence's token count by 1 (caller may have done this —
      here you only grow blocks when needed).
      If num_tokens % block_size == 1 after a new block is required
      (i.e. tokens filled the previous block), pop one free block and append
      to the block table. Otherwise no-op.

    Invariant: blocks allocated == ceil(num_tokens / block_size).
    """
    raise NotImplementedError("Implement append_slot() — see docstring for the spec.")


def schedule_promotions(
    waiting: List["SequenceGroup"],
    running: List["SequenceGroup"],
    *,
    max_batch_size: int,
    can_allocate,
    allocate_fn,
) -> tuple[List["SequenceGroup"], List["SequenceGroup"]]:
    """Promote sequences from waiting → running.

    Algorithm:
      While waiting is non-empty and len(running) < max_batch_size:
        Peek head of waiting. If can_allocate(prompt_len) succeeds, pop it,
        call allocate_fn, mark RUNNING, append to running.
        On first failure, STOP — do not skip ahead in the queue.

    Invariant: len(running) <= max_batch_size always; FIFO order preserved.
    """
    raise NotImplementedError(
        "Implement schedule_promotions() — see docstring for the spec."
    )


def blocks_needed(num_tokens: int, block_size: int) -> int:
    return math.ceil(num_tokens / block_size)
