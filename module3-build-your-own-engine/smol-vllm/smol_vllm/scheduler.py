import math
from collections import deque
from typing import Deque, List

from .block_manager import BlockSpaceManager
from .sequence import SchedulerOutputs, SequenceGroup, SequenceStatus


class Scheduler:
    def __init__(
        self,
        block_manager: BlockSpaceManager,
        max_batch_size: int,
        *,
        preempt_guard: bool = True,
    ):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.preempt_guard = preempt_guard
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

    def schedule(self) -> SchedulerOutputs:
        blocks_to_swap_in: List = []
        blocks_to_swap_out: List = []
        newly_scheduled: set[int] = set()

        while self.waiting and len(self.running) < self.max_batch_size:
            group = self.waiting[0]
            seq = group.sequences[0]
            num_tokens = seq.num_tokens
            num_blocks = math.ceil(num_tokens / self.block_manager.block_size)
            free_after = self.block_manager.num_free_blocks() - num_blocks
            running_after = len(self.running) + 1

            if not self.block_manager.can_allocate(num_tokens):
                break
            if free_after < running_after:
                break

            self.block_manager.allocate(group.group_id, num_tokens)
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(group)
            newly_scheduled.add(group.group_id)

        while self.running and self._should_preempt(newly_scheduled):
            group = self._pick_preempt_candidate(newly_scheduled)
            if group is None:
                break
            self.running.remove(group)
            seq = group.sequences[0]
            seq.status = SequenceStatus.SWAPPED
            self.block_manager.free(group.group_id)
            self.swapped.append(group)
            blocks_to_swap_out.append(group.group_id)
            print(f"  [preempt] group {group.group_id} -> swapped (blocks freed)")

        i = 0
        while i < len(self.swapped) and len(self.running) < self.max_batch_size:
            group = self.swapped[i]
            seq = group.sequences[0]
            num_tokens = seq.num_tokens
            num_blocks_needed = math.ceil(
                num_tokens / self.block_manager.block_size
            )
            free_after_swap = (
                self.block_manager.num_free_blocks()
                - num_blocks_needed
            )
            running_after_swap = len(self.running) + 1

            if (
                self.block_manager.can_allocate(num_tokens)
                and free_after_swap >= running_after_swap
            ):
                self.block_manager.allocate(group.group_id, num_tokens)
                seq.status = SequenceStatus.RUNNING
                self.running.append(group)
                self.swapped.pop(i)
                blocks_to_swap_in.append(group.group_id)
                print(f"  [swap-in] group {group.group_id} <- swapped")
            else:
                i += 1

        return SchedulerOutputs(
            scheduled_groups=self.running.copy(),
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
        )

    def _should_preempt(self, newly_scheduled: set[int]) -> bool:
        if not self.running:
            return False
        over_batch = len(self.running) > self.max_batch_size
        high_util = self.block_manager.utilization() > 0.95
        low_free = self.block_manager.num_free_blocks() < len(self.running)
        if not (over_batch or high_util or low_free):
            return False
        if self.preempt_guard and len(self.running) == 1:
            return False
        if self.preempt_guard and all(g.group_id in newly_scheduled for g in self.running):
            return False
        return True

    def _pick_preempt_candidate(
        self, newly_scheduled: set[int]
    ) -> SequenceGroup | None:
        for i in range(len(self.running) - 1, -1, -1):
            group = self.running[i]
            if self.preempt_guard and group.group_id in newly_scheduled:
                continue
            return group
        if self.preempt_guard:
            return None
        return self.running[-1] if self.running else None
