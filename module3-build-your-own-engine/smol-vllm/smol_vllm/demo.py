import random
import time
from datetime import datetime
from pathlib import Path

from . import BlockSpaceManager, LLMEngine
from .seed_utils import seed_all


def _rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def run_block_manager_checkpoint(*, verbose: bool = True) -> None:
    """Assert BlockSpaceManager allocate / append / free invariants."""
    if verbose:
        print("\n" + "=" * 60)
        print("Checkpoint: BlockSpaceManager")
        print("=" * 60)

    bm = BlockSpaceManager(num_blocks=20, block_size=16)

    for seq_id in range(3):
        bm.allocate(seq_id, 48)
    util_after_alloc = bm.utilization()
    expected_util = 9 / 20  # 3 seqs × 3 blocks
    assert abs(util_after_alloc - expected_util) < 0.01, (
        f"FAIL allocate: expected util≈{expected_util:.2f}, got {util_after_alloc:.2f}"
    )
    if verbose:
        print(f"  Allocated 3 seqs × 3 blocks: util={util_after_alloc*100:.0f}%")

    for seq_id in range(3):
        for _ in range(16):
            bm.append_token(seq_id)
    util_after_append = bm.utilization()
    expected_append = 12 / 20  # 3 seqs × 4 blocks
    assert abs(util_after_append - expected_append) < 0.01, (
        f"FAIL append_token: expected util≈{expected_append:.2f}, got {util_after_append:.2f}"
    )
    if verbose:
        print(f"  After appends (16 each): util={util_after_append*100:.0f}%")

    bm.free(1)
    util_after_free = bm.utilization()
    expected_free = 8 / 20
    assert abs(util_after_free - expected_free) < 0.01, (
        f"FAIL free: expected util≈{expected_free:.2f}, got {util_after_free:.2f}"
    )
    free_count = bm.num_free_blocks()
    assert free_count == 12, f"FAIL free: expected 12 free blocks, got {free_count}"
    if verbose:
        print(f"  After free(seq 1): util={util_after_free*100:.0f}%")
        print(f"  Free blocks: {free_count} (4 blocks returned from seq 1)")
        print("  PASS: BlockSpaceManager checkpoint\n")


def run_scheduler_checkpoint(*, seed: int = 0, verbose: bool = True) -> None:
    """Assert continuous batching caps running count."""
    if verbose:
        print("\n" + "=" * 60)
        print("Checkpoint: Scheduler (continuous batching)")
        print("=" * 60)

    engine = LLMEngine(
        num_gpu_blocks=64,
        block_size=16,
        max_batch_size=4,
        seed=seed,
    )

    for i in range(10):
        engine.add_request(list(range(10)), max_tokens=5)

    if verbose:
        print("  Submitting 10 requests, max_batch_size=4")
    for step in range(8):
        engine.step()
        running = len(engine.scheduler.running)
        waiting = len(engine.scheduler.waiting)
        assert running <= 4, f"FAIL scheduler: expected ≤4 running, got {running} at step {step+1}"
        if verbose:
            print(f"  step {step+1}: running={running}, waiting={waiting}")

    if verbose:
        print("  PASS: Scheduler checkpoint\n")


def run_build_allocate_checkpoint(allocate_fn, *, verbose: bool = True) -> None:
    """Run student allocate() against the block-manager spec."""
    bm = BlockSpaceManager(num_blocks=8, block_size=16)
    allocate_fn(bm, seq_id=0, num_tokens=48)
    table = bm.get_block_table(0)
    assert len(table) == 3, f"FAIL allocate: expected 3 blocks, got {table}"
    assert bm.num_free_blocks() == 5, f"FAIL allocate: expected 5 free, got {bm.num_free_blocks()}"
    if verbose:
        print(f"  PASS: allocate() → block_table={table}")


def run_build_append_checkpoint(append_fn, *, verbose: bool = True) -> None:
    bm = BlockSpaceManager(num_blocks=8, block_size=16)
    bm.allocate(0, 16)
    append_fn(bm, 0)
    assert len(bm.get_block_table(0)) == 2, (
        f"FAIL append_slot: expected 2 blocks after 17 tokens, got {bm.get_block_table(0)}"
    )
    if verbose:
        print(f"  PASS: append_slot() → block_table={bm.get_block_table(0)}")


def _run_exp1(seed: int = 0):
    print("\n" + "=" * 60)
    print("Experiment 1: Continuous Batching")
    print("=" * 60)
    print("Submit 20 requests of different lengths")
    print("Columns: step | running | waiting | swapped | blocks_used (%)\n")

    rng = _rng(seed)
    engine = LLMEngine(
        num_gpu_blocks=64, block_size=16, max_batch_size=4, seed=seed
    )

    for i in range(20):
        length = rng.randint(10, 80)
        prompt = list(range(i * 100, i * 100 + length))
        engine.add_request(prompt, max_tokens=20)

    step = 0
    total_finished = 0
    while total_finished < 20:
        outputs = engine.step()
        total_finished += sum(1 for o in outputs if o.finished)
        step += 1

        running = len(engine.scheduler.running)
        waiting = len(engine.scheduler.waiting)
        swapped = len(engine.scheduler.swapped)
        util = engine.block_manager.utilization() * 100

        print(
            f"  step {step:3d} | running={running:2d} | waiting={waiting:2d} | "
            f"swapped={swapped:2d} | blocks_used={util:5.1f}%"
        )

    print(f"\n  Done in {step} steps. All 20 requests finished.")
    engine.metrics.print_summary()
    return engine


def _run_exp2(seed: int = 0):
    print("\n" + "=" * 60)
    print("Experiment 2: Memory Pressure & Preemption")
    print("=" * 60)
    print("num_blocks=16, 10 long sequences (64 tokens each) -> watch preemption\n")

    engine = LLMEngine(
        num_gpu_blocks=16, block_size=16, max_batch_size=10, seed=seed
    )

    for i in range(10):
        prompt = list(range(1000, 1064))
        engine.add_request(prompt, max_tokens=10)

    step = 0
    total_finished = 0
    while total_finished < 10:
        outputs = engine.step()
        total_finished += sum(1 for o in outputs if o.finished)
        step += 1
        util = engine.block_manager.utilization() * 100
        print(
            f"  step {step:3d} | running={len(engine.scheduler.running):2d} | "
            f"swapped={len(engine.scheduler.swapped):2d} | util={util:.0f}%"
        )

    print(f"\n  Done in {step} steps.")
    engine.metrics.print_summary()
    return engine


def run_exp3_prefix_sharing():
    print("\n" + "=" * 60)
    print("Experiment 3: Prefix Sharing (copy_on_write)")
    print("=" * 60)
    print("5 sequences with same 32-token prefix -> compare utilization\n")

    bm = BlockSpaceManager(num_blocks=64, block_size=16)

    tokens_without = 32 + 16
    for i in range(5):
        bm.allocate(i, tokens_without)

    util_without = bm.utilization()
    tables_without = bm.all_block_tables()
    print("  Without sharing — block tables:")
    for sid, tbl in sorted(tables_without.items()):
        print(f"    seq {sid}: {tbl}")
    print(f"  ref_counts: {bm.ref_counts()}")

    for i in range(5):
        bm.free(i)

    bm.allocate(0, 48)
    for i in range(1, 5):
        bm.copy_on_write(0, i)
        for _ in range(16):
            bm.append_token(i)

    util_with_sharing = bm.utilization()
    tables_with = bm.all_block_tables()
    print(
        f"\n  Without prefix sharing: 5 seqs × 3 blocks = 15 blocks -> util={util_without*100:.0f}%"
    )
    print(
        f"  With copy_on_write: shared prefix -> util={util_with_sharing*100:.0f}%"
    )
    print("\n  With sharing — block tables (note shared block IDs):")
    for sid, tbl in sorted(tables_with.items()):
        print(f"    seq {sid}: {tbl}")
    shared = set(tables_with[0][:2])
    for sid in range(1, 5):
        assert shared.issubset(set(tables_with[sid])), "prefix blocks should be shared"
    print(f"  ref_counts: {bm.ref_counts()}")
    print("  Shared prefix block IDs:", sorted(shared))
    print()


def _print_exp4_cost_model(timing: str) -> None:
    fm = LLMEngine(num_gpu_blocks=64, block_size=16, max_batch_size=1, timing=timing).model
    if timing == "naive":
        print(
            f"  [naive] cost ∝ batch_size only — "
            f"sleep({fm.naive_per_group_cost} × batch_size) per decode step (no fixed overhead)"
        )
    else:
        print(
            f"  [roofline] fixed + marginal — "
            f"sleep({fm.fixed_step_cost} + {fm.per_token_cost} × batch_size) per decode step"
        )


def _run_exp4(timing: str = "naive", seed: int = 0):
    print("\n" + "=" * 60)
    print(f"Experiment 4: Throughput Scaling (timing={timing})")
    print("=" * 60)
    _print_exp4_cost_model(timing)
    print("Measure tok/s at batch_size=1, 8, 16 -> ASCII bar chart")
    print("(step metrics suppressed here so the bar chart stays visible)\n")

    batch_sizes = [1, 8, 16]
    tok_per_secs = []
    engines = []

    for batch_size in batch_sizes:
        engine = LLMEngine(
            num_gpu_blocks=64,
            block_size=16,
            max_batch_size=batch_size,
            seed=seed,
            timing=timing,
            enable_metrics=False,
        )
        engines.append(engine)

        for i in range(batch_size):
            engine.add_request(list(range(50)), max_tokens=20)

        start = time.perf_counter()
        total_tokens = 0
        while True:
            outputs = engine.step()
            total_tokens += len(outputs)
            if all(o.finished for o in outputs):
                break
        elapsed = time.perf_counter() - start
        tps = total_tokens / elapsed if elapsed > 0 else 0
        tok_per_secs.append(tps)
        print(f"  batch_size={batch_size:2d}: {tps:.1f} tok/s")

    print("\n  Throughput (tok/s) bar chart:")
    max_tps = max(tok_per_secs) if tok_per_secs else 1
    for bs, tps in zip(batch_sizes, tok_per_secs):
        bar_len = int(40 * tps / max_tps)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"    batch={bs:2d} |{bar}| {tps:.1f}")
    if engines:
        engines[-1].metrics.print_summary()

    print()
    return engines


def reproduce_preemption_livelock(*, seed: int = 0) -> LLMEngine:
    """Reproduce preempt/swap-in thrash with preempt_guard=False (debug exercise)."""
    print("\n" + "=" * 60)
    print("Debug: preemption livelock (preempt_guard=False)")
    print("=" * 60)

    engine = LLMEngine(
        num_gpu_blocks=24,
        block_size=16,
        max_batch_size=4,
        seed=seed,
        preempt_guard=False,
    )
    long_prompt = list(range(350))
    engine.add_request(long_prompt, max_tokens=12)

    for step in range(20):
        engine.step()
        if not engine.scheduler.running and not engine.scheduler.waiting:
            break
    return engine


def _run_exp5(seed: int = 0):
    print("\n" + "=" * 60)
    print("Experiment 5: Fake vs Real Model (Educational)")
    print("=" * 60)

    print("\n=== Fake model (simulated timing, zero deps) ===")
    engine_fake = LLMEngine(
        num_gpu_blocks=128, block_size=16, max_batch_size=4, seed=seed
    )
    for i in range(4):
        engine_fake.add_request(list(range(10 + i * 5, 30 + i * 5)), max_tokens=10)

    start = time.perf_counter()
    total = 0
    while True:
        outputs = engine_fake.step()
        total += len(outputs)
        if all(o.finished for o in outputs):
            break
    elapsed = time.perf_counter() - start
    print(
        f"  4 prompts, ~20 tok each, max 10 output → {elapsed:.2f}s, "
        f"{total/elapsed:.0f} tok/s (simulated)"
    )

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError:
        print("\n=== CausalLM (skipped: pip install torch transformers accelerate) ===")
        print("  Install extras to compare: pip install torch transformers accelerate")
        return None

    _ = AutoTokenizer
    print("\n=== CausalLM (actual compute + memory) ===")
    engine_real = LLMEngine(
        num_gpu_blocks=128,
        block_size=16,
        max_batch_size=2,
        use_real_model=True,
        seed=seed,
    )
    tokenizer = engine_real.model.tokenizer
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
    ]
    for p in prompts:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        engine_real.add_request(tokens, max_tokens=8)

    start = time.perf_counter()
    total = 0
    while True:
        outputs = engine_real.step()
        total += len(outputs)
        if all(o.finished for o in outputs):
            break
    elapsed = time.perf_counter() - start
    print(f"  2 prompts → {elapsed:.2f}s, {total/elapsed:.1f} tok/s (real)")
    if torch.cuda.is_available():
        mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  VRAM peaked at {mb:.0f} MB")
    engine_real.metrics.print_summary()
    return engine_real


def measure_kv_bytes_per_token(
    *,
    seed: int = 0,
    max_tokens: int = 32,
    prompt: str = "Measure KV cache growth one token at a time for TinyLlama.",
) -> float | None:
    """Return measured KV bytes per generated token (GPU + real model only)."""
    try:
        import torch
    except ImportError:
        print("  KV/token probe skipped (torch not installed)")
        return None
    if not torch.cuda.is_available():
        print("  KV/token probe skipped (no CUDA)")
        return None

    engine = LLMEngine(
        num_gpu_blocks=128,
        block_size=16,
        max_batch_size=1,
        use_real_model=True,
        seed=seed,
        enable_metrics=True,
    )
    tokenizer = engine.model.tokenizer
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    engine.add_request(ids, max_tokens=max_tokens)
    while True:
        outputs = engine.step()
        if outputs and all(o.finished for o in outputs):
            break
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    gen = engine.metrics.generation_tokens_total
    if gen <= 0:
        print("  KV/token probe: no tokens generated")
        return None
    bytes_per_token = (mem_after - mem_before) / gen
    kb = bytes_per_token / 1024
    # TinyLlama: 2 × 22 layers × 4 kv_heads × 64 dim × 2 bytes ≈ 22 KB/token
    expected_kb = 2 * 22 * 4 * 64 * 2 / 1024
    print(f"  KV measured ≈ {kb:.1f} KB/token ({gen} tokens)")
    print(f"  Class 3 slide 41 expects ≈ {expected_kb:.0f} KB/token (2×L×H×D×bytes)")
    return bytes_per_token


def _save_metrics_log(all_metrics: list):
    if not all_metrics:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"smol_vllm_{ts}.csv"
    import csv

    all_rows = []
    for name, m in all_metrics:
        for r in m.to_csv_rows():
            r["experiment"] = name
            r["timestamp"] = datetime.now().isoformat()
            all_rows.append(r)
    if all_rows:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  [metrics] saved to {path}")


def main():
    seed_all(0)
    print("\n" + "#" * 60)
    print("#  smol-vLLM Demo - Paged Attention Inference Engine")
    print("#" * 60)

    all_metrics: list[tuple[str, object]] = []

    run_block_manager_checkpoint()
    run_scheduler_checkpoint(seed=0)

    engine1 = run_exp1_continuous_batching(seed=0)
    if engine1:
        all_metrics.append(("exp1_continuous_batching", engine1.metrics))

    engine2 = run_exp2_memory_pressure(seed=0)
    if engine2:
        all_metrics.append(("exp2_memory_pressure", engine2.metrics))

    run_exp3_prefix_sharing()

    engines4 = run_exp4_throughput_scaling(timing="naive", seed=0)
    if engines4:
        all_metrics.append(("exp4_throughput", engines4[-1].metrics))

    engine5 = run_exp5_fake_vs_real(seed=0)
    if engine5:
        all_metrics.append(("exp5_educational", engine5.metrics))

    _save_metrics_log(all_metrics)
    print("All experiments complete.")


run_exp1_continuous_batching = _run_exp1
run_exp2_memory_pressure = _run_exp2
run_exp4_throughput_scaling = _run_exp4
run_exp5_fake_vs_real = _run_exp5
