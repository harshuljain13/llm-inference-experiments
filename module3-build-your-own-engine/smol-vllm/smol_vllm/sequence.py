import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


@dataclass
class Sequence:
    seq_id: int
    prompt_tokens: List[int]
    output_tokens: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.time)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED


@dataclass
class SequenceGroup:
    group_id: int
    sequences: List[Sequence]
    sampling_params: Dict


@dataclass
class RequestOutput:
    group_id: int
    seq_id: int
    output_tokens: List[int]
    finished: bool


@dataclass
class SchedulerOutputs:
    scheduled_groups: List[SequenceGroup]
    blocks_to_swap_in: List
    blocks_to_swap_out: List
