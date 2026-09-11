import math
from collections import deque
from typing import Dict, List


class BlockSpaceManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._ref_count: Dict[int, int] = {}
        self._block_tables: Dict[int, List[int]] = {}
        self._tokens_allocated: Dict[int, int] = {}
        self._free_blocks: deque = deque(range(num_blocks))

    def can_allocate(self, num_tokens_needed: int) -> bool:
        num_blocks_needed = math.ceil(num_tokens_needed / self.block_size)
        return len(self._free_blocks) >= num_blocks_needed

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        num_blocks_needed = math.ceil(num_tokens / self.block_size)
        if len(self._free_blocks) < num_blocks_needed:
            raise ValueError(
                f"Cannot allocate {num_blocks_needed} blocks for seq {seq_id}: "
                f"only {len(self._free_blocks)} free"
            )

        blocks = []
        for _ in range(num_blocks_needed):
            phys_id = self._free_blocks.popleft()
            self._ref_count[phys_id] = 1
            blocks.append(phys_id)

        self._block_tables[seq_id] = blocks
        self._tokens_allocated[seq_id] = num_tokens

    def append_token(self, seq_id: int) -> None:
        if seq_id not in self._block_tables:
            raise ValueError(f"Sequence {seq_id} has no blocks (call allocate first)")

        self._tokens_allocated[seq_id] += 1
        blocks_needed = math.ceil(self._tokens_allocated[seq_id] / self.block_size)
        blocks_current = len(self._block_tables[seq_id])

        if blocks_needed > blocks_current:
            if not self._free_blocks:
                raise ValueError(f"No free blocks to append token for seq {seq_id}")
            phys_id = self._free_blocks.popleft()
            self._ref_count[phys_id] = 1
            self._block_tables[seq_id].append(phys_id)

    def free(self, seq_id: int) -> None:
        if seq_id not in self._block_tables:
            return

        for phys_id in self._block_tables[seq_id]:
            self._ref_count[phys_id] -= 1
            if self._ref_count[phys_id] == 0:
                self._free_blocks.append(phys_id)
                del self._ref_count[phys_id]

        del self._block_tables[seq_id]
        del self._tokens_allocated[seq_id]

    def get_block_table(self, seq_id: int) -> List[int]:
        return self._block_tables.get(seq_id, []).copy()

    def copy_on_write(self, src_seq_id: int, dst_seq_id: int) -> None:
        if src_seq_id not in self._block_tables:
            raise ValueError(f"Source sequence {src_seq_id} has no blocks")

        src_blocks = self._block_tables[src_seq_id]
        self._block_tables[dst_seq_id] = src_blocks.copy()
        self._tokens_allocated[dst_seq_id] = self._tokens_allocated[src_seq_id]

        for phys_id in src_blocks:
            self._ref_count[phys_id] = self._ref_count.get(phys_id, 0) + 1

    def utilization(self) -> float:
        used = self.num_blocks - len(self._free_blocks)
        return used / self.num_blocks if self.num_blocks > 0 else 0.0

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def ref_counts(self) -> Dict[int, int]:
        return dict(self._ref_count)

    def all_block_tables(self) -> Dict[int, List[int]]:
        return {sid: blocks.copy() for sid, blocks in self._block_tables.items()}
