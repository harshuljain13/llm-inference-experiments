from __future__ import annotations

from smol_vllm import LLMEngine


def show_engine_state(engine: LLMEngine, step: int | None = None) -> None:
    """Print scheduler queues + block-manager stats (matches workflow diagram)."""
    prefix = f"step {step:2d} | " if step is not None else ""
    waiting = [g.group_id for g in engine.scheduler.waiting]
    running = [g.group_id for g in engine.scheduler.running]
    swapped = [g.group_id for g in engine.scheduler.swapped]
    free = engine.block_manager.num_free_blocks()
    total = engine.block_manager.num_blocks
    util = engine.block_manager.utilization() * 100
    print(
        f"{prefix}waiting={waiting}  running={running}  swapped={swapped}  "
        f"| free={free}/{total} blocks  util={util:.1f}%"
    )
