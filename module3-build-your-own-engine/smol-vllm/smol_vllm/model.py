import math
import random
import time
from typing import Literal

from .sequence import SequenceGroup


class FakeModel:
    """Simulated model for teaching the control plane without GPU matmuls."""

    TEXT_CORPUS = (
        "Paged attention maps logical kv blocks to physical gpu memory "
        "for efficient continuous batching in large language model inference"
    )

    # Roofline timing (seconds): fixed_step_cost + per_token_cost * batch_size
    fixed_step_cost: float = 0.020
    per_token_cost: float = 0.0005
    naive_per_group_cost: float = 0.005

    def __init__(
        self,
        *,
        seed: int | None = None,
        output_mode: Literal["ids", "text"] = "ids",
        timing: Literal["naive", "roofline"] = "naive",
    ):
        self.seed = seed
        self.output_mode = output_mode
        self.timing = timing
        self._rng = random.Random(seed)

    def prefill(self, groups: list[SequenceGroup]) -> list[int]:
        total_prompt = sum(g.sequences[0].num_tokens for g in groups)
        time.sleep(0.01 * total_prompt / 100)
        return [self._fake_next_token(g) for g in groups]

    def decode(
        self, groups: list[SequenceGroup], block_tables: list[list[int]]
    ) -> list[int]:
        _ = block_tables
        batch = len(groups)
        if self.timing == "roofline":
            time.sleep(self.fixed_step_cost + self.per_token_cost * batch)
        else:
            time.sleep(self.naive_per_group_cost * batch)
        return [self._fake_next_token(g) for g in groups]

    def _fake_next_token(self, group: SequenceGroup) -> int:
        seq = group.sequences[0]
        stop_ids = group.sampling_params.get("stop_token_ids", [0])

        if self.output_mode == "text":
            idx = (seq.num_tokens + group.group_id * 3) % len(self.TEXT_CORPUS)
            ch = self.TEXT_CORPUS[idx]
            token = ord(ch)
            if token in stop_ids:
                token = ord("a") + (idx % 26)
            return token

        # ids mode — deterministic arithmetic tokens (no random stop)
        return (seq.num_tokens * 7 + group.group_id * 13 + 17) % 1000 + 1
