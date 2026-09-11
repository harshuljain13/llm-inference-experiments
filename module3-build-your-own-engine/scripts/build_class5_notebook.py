#!/usr/bin/env python3
"""Generate class5.ipynb with sections 0–9 (observe → build → break → debug → prove)."""
from __future__ import annotations

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md("""# How an Inference Engine Works — smol_vllm

![Engine workflow diagram](assets/engine_workflow.png)

**Arc:** observe → build → break → debug → prove

| # | Section |
|---|---------|
| 0 | Setup + readiness |
| 1 | Walk the diagram — one request by hand |
| 2 | Build it — stubs → green checkpoints |
| 3 | Experiments 1–3 (predict first) |
| 4 | Break it — parameter sweeps |
| 5 | Debug it — preemption livelock |
| 6 | Throughput & simulator fidelity (Exp 4) |
| 7 | Prove Class 3 on GPU |
| 8 | Real traffic — CrewAI |
| 9 | Diff to production vLLM |

All engines use `seed=0` for reproducible demos.
"""),
    md("## 0 — Setup"),
    code("""import inspect
import subprocess
import sys
from pathlib import Path

SEED = 0

_shadow = Path.cwd() / "smol_vllm"
if _shadow.is_dir():
    sys.path = [p for p in sys.path if Path(p).resolve() != _shadow.resolve()]
if str(Path.cwd()) not in sys.path:
    sys.path.insert(0, str(Path.cwd()))

# After sync: reinstall editable package into *this* kernel's Python
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", "./smol-vllm"],
    check=True,
)

# Drop cached imports from a pre-sync kernel session (smol_vllm + lib adapters)
for name in list(sys.modules):
    if (
        name == "smol_vllm"
        or name.startswith("smol_vllm.")
        or name.startswith("lib.")
        or name == "agent_demo"
    ):
        del sys.modules[name]

from smol_vllm import LLMEngine, SequenceStatus
from smol_vllm.demo import (
    run_block_manager_checkpoint,
    run_build_allocate_checkpoint,
    run_build_append_checkpoint,
    run_exp1_continuous_batching,
    run_exp2_memory_pressure,
    run_exp3_prefix_sharing,
    run_exp4_throughput_scaling,
    run_exp5_fake_vs_real,
    run_scheduler_checkpoint,
    reproduce_preemption_livelock,
    measure_kv_bytes_per_token,
)
from smol_vllm import exercises
from lib.engine_state import show_engine_state

_params = inspect.signature(LLMEngine.__init__).parameters
assert "seed" in _params, (
    "Old smol_vllm still loaded — use Kernel → Restart, then re-run this cell"
)
print("smol_vllm imported OK (seed, preempt_guard, output_mode, timing available)")
"""),
    code("""from lib.gpu_check import load_dotenv, print_gpu_report

load_dotenv()
GPU_REPORT = print_gpu_report()
GPU_OK = GPU_REPORT["real_model_ready"]
"""),
    md("""---
## 1 — Walk the diagram (one request by hand)

Follow the workflow diagram: request → waiting → allocate → prefill → decode → free.
"""),
    code("""engine = LLMEngine(
    num_gpu_blocks=24,
    block_size=16,
    max_batch_size=4,
    enable_metrics=True,
    seed=SEED,
)
print(f"Engine: {engine.block_manager.num_blocks} blocks × {engine.block_manager.block_size} tokens/block")
show_engine_state(engine)
"""),
    code("""# Warm-up: allocate + free so later tables get scattered physical IDs (not [0],[1],[2]...)
for sid, ntok in [(90, 32), (91, 48)]:
    engine.add_request(list(range(ntok)), max_tokens=1)
while engine.scheduler.waiting or engine.scheduler.running:
    engine.step()
# Re-create clean engine for the walkthrough
engine = LLMEngine(num_gpu_blocks=24, block_size=16, max_batch_size=4, enable_metrics=True, seed=SEED)

engine.add_request(list(range(10, 25)), max_tokens=12)
engine.add_request(list(range(100, 130)), max_tokens=8)
engine.add_request(list(range(200, 212)), max_tokens=20)
show_engine_state(engine)
"""),
    code("""# Diagram: can_allocate? → allocate → block_table
for gid in [0, 1, 2]:
    ntok = engine.groups[gid].sequences[0].num_tokens
    ok = engine.block_manager.can_allocate(ntok)
    if ok:
        engine.block_manager.allocate(gid, ntok)
    print(f"group {gid}: can_allocate={ok}  block_table={engine.block_manager.get_block_table(gid)}")
show_engine_state(engine)
"""),
    code("""# Prefill + decode steps (full logs — watch queues move)
for step in range(1, 8):
    outputs = engine.step()
    show_engine_state(engine, step=step)
    if outputs and all(o.finished for o in outputs):
        break
"""),
    code("""# generate() convenience API
clean = LLMEngine(num_gpu_blocks=24, block_size=16, max_batch_size=4, enable_metrics=True, seed=SEED)
for tok in clean.generate(list(range(5, 25)), max_tokens=8):
    pass
show_engine_state(clean)
clean.metrics.print_summary()
"""),
    md("""---
## 2 — Build it

You've watched the engine allocate blocks. Now write the allocator. **The checkpoint is the spec.**

Open `smol_vllm/exercises.py` — run each cell below. **First run = RED (expected).** Implement, re-run until green.
"""),
    code("""# 2.1 allocate() — run first. FAIL until you implement exercises.allocate.
try:
    run_build_allocate_checkpoint(exercises.allocate)
except NotImplementedError:
    print("RED: implement allocate() in smol_vllm/exercises.py, then re-run this cell")

# Escape hatch (instructor pre-class / stuck):
# from smol_vllm.reference.exercises_ref import allocate as ref_allocate
# run_build_allocate_checkpoint(ref_allocate)
"""),
    code("""# 2.2 append_slot() — same pattern
try:
    run_build_append_checkpoint(exercises.append_slot)
except NotImplementedError:
    print("RED: implement append_slot() in smol_vllm/exercises.py, then re-run this cell")

# Escape hatch:
# from smol_vllm.reference.exercises_ref import append_slot as ref_append_slot
# run_build_append_checkpoint(ref_append_slot)
"""),
    code("""# 2.3 Built-in checkpoints (always run after build section)
run_block_manager_checkpoint()
run_scheduler_checkpoint(seed=SEED)
"""),
    md("""---
## 3 — Experiments (predict → run)

Scan the logs for `[preempt]`, `swapped=`, and `running=` — there will be many lines; that's intentional.
"""),
    md("""### Predict — Exp 1 (continuous batching)

20 requests, `max_batch_size=4`. **Predict:** peak `running`? Does `waiting` drain monotonically?
"""),
    code("exp1_engine = run_exp1_continuous_batching(seed=SEED)"),
    md("""### Predict — Exp 2 (memory pressure)

`num_blocks=16`, `block_size=16` → **256** tokens capacity. 10 seqs × 64 tokens → **640** needed (4 blocks each). **Predict:** how many run concurrently? Any preemption?
"""),
    code("exp2_engine = run_exp2_memory_pressure(seed=SEED)"),
    md("""### Predict — Exp 3 (prefix sharing)

5 seqs, 32-token shared prefix, `block_size=16` → prefix = 2 blocks. **Predict:** blocks with vs without sharing?
"""),
    code("run_exp3_prefix_sharing()"),
    md("""---
## 4 — Break it

Each cell: run, then explain which limit bound you.
"""),
    code("""# 4.1 Block pool: 8 blocks, 64-token prompts need 4 blocks each → 2 fit, 3rd waits
from smol_vllm import BlockSpaceManager

bm = BlockSpaceManager(num_blocks=8, block_size=16)
for sid in [0, 1]:
    bm.allocate(sid, 64)
    print(f"seq {sid}: table={bm.get_block_table(sid)}")
print(f"After 2 seqs: free={bm.num_free_blocks()}/8  (3rd seq would need 4 — queue or reject)")

# Same budget via scheduler: admission keeps headroom → often running=1 (see logs)
print("\\nScheduler admission (free_after >= running rule):")
break1 = LLMEngine(num_gpu_blocks=8, block_size=16, max_batch_size=8, seed=SEED)
for i in range(3):
    break1.add_request(list(range(1000 + i, 1064 + i)), max_tokens=5)
for step in range(8):
    break1.step()
    show_engine_state(break1, step=step)
"""),
    code("""# 4.2 block_size trade-off — hold ~128-token capacity, same 64-token prompt
import math

PROMPT_TOKENS = 64
CAPACITY_TOKENS = 128

for bs in [1, 16, 64]:
    num_blocks = math.ceil(CAPACITY_TOKENS / bs)
    e = LLMEngine(num_gpu_blocks=num_blocks, block_size=bs, max_batch_size=1, seed=SEED)
    e.add_request(list(range(PROMPT_TOKENS)), max_tokens=4)
    e.step()
    table = e.block_manager.get_block_table(0)
    blocks_used = len(table)
    slots = blocks_used * bs
    wasted = slots - PROMPT_TOKENS
    print(
        f"block_size={bs:2d}  blocks={num_blocks:3d}  table_len={blocks_used:2d}  "
        f"wasted_slots={wasted:2d}  table={table}"
    )
"""),
    code("""# 4.3 max_batch_size=64 but num_gpu_blocks=24 — which limit binds?
break3 = LLMEngine(num_gpu_blocks=24, block_size=16, max_batch_size=64, seed=SEED)
for i in range(12):
    break3.add_request(list(range(50)), max_tokens=10)
for step in range(6):
    break3.step()
    print(f"step {step+1}: running={len(break3.scheduler.running)} waiting={len(break3.scheduler.waiting)}")
"""),
    md("""---
## 5 — Debug it — preemption livelock

1. Run with `preempt_guard=False` — watch thrash  
2. Ask: preempting to make room for *whom*?  
3. Open `scheduler.py` — find `num_free_blocks() < len(running)`  
4. Re-run with `preempt_guard=True` (default)
"""),
    code("reproduce_preemption_livelock(seed=SEED)"),
    code("""# Fixed — default preempt_guard=True
fixed = LLMEngine(num_gpu_blocks=24, block_size=16, max_batch_size=4, seed=SEED, preempt_guard=True)
fixed.add_request(list(range(350)), max_tokens=12)
for step in range(20):
    fixed.step()
show_engine_state(fixed)
"""),
    md("""---
## 6 — Throughput & simulator fidelity (Exp 4)

Run twice: `timing="naive"` then `timing="roofline"`. Which cost model can teach continuous batching?
"""),
    code("""naive_engines = run_exp4_throughput_scaling(timing="naive", seed=SEED)
roof_engines = run_exp4_throughput_scaling(timing="roofline", seed=SEED)
"""),
    md("""---
## 7 — Prove Class 3 on GPU (Lambda / CUDA)

Skip if `GPU_OK` is false. Derive slide numbers from your own `[metrics]` lines.
"""),
    code("""if GPU_OK:
    real_engine = run_exp5_fake_vs_real(seed=SEED)
else:
    print("GPU track skipped — see LAMBDA.md")
"""),
    code("""# Prefill vs decode asymmetry + KV bytes/token (when GPU_OK)
if GPU_OK:
    m = real_engine.metrics
    if m.prefill_latencies and m.decode_latencies:
        p = sum(m.prefill_latencies)/len(m.prefill_latencies)*1000
        d = sum(m.decode_latencies)/len(m.decode_latencies)*1000
        print(f"prefill_avg={p:.0f}ms  decode_avg={d:.0f}ms  ratio={p/max(d,0.001):.1f}x")
    try:
        import torch
        if torch.cuda.is_available():
            mb = torch.cuda.max_memory_allocated()/1024**2
            print(f"VRAM peak={mb:.0f} MB  (TinyLlama ~1.1B × 2B ≈ 2.2 GB weights)")
    except ImportError:
        pass
    measure_kv_bytes_per_token(seed=SEED)
"""),
    md("""---
## 8 — Real traffic — CrewAI through your engine

**What this section is for:** Class 2 showed inference *from the outside* (HTTP, gateways).  
Everything until now was *inside the engine* (scheduler, blocks, prefill/decode).  
Here we connect both: a **real agent framework** calls **your** engine — same `add_request` → `step` loop, but the prompts come from CrewAI agents instead of you typing token lists.

**Watch the logs, not the final prose.** CrewAI wraps the engine; every agent LLM call becomes an `ENGINE REQUEST #N` with the same queues you traced in §1.

### Two layers — same stack as production

| Layer | Job | In this notebook | What you see in logs |
|-------|-----|------------------|----------------------|
| **Orchestration** (CrewAI) | Break a user task into steps; route work between agents; pass context task → task | Research Analyst → Technical Writer | CrewAI task/agent banners (green/purple UI) |
| **Inference** (smol_vllm) | Schedule requests; allocate KV blocks; run prefill/decode; return tokens | **One shared** `LLMEngine` for the whole crew | `ENGINE REQUEST #1`, `#2`, `waiting` / `running` / `swapped`, `[metrics]` |
| **Model** | Turn tokens into next tokens | FakeModel `output_mode="text"` on CPU (optional TinyLlama on GPU) | Readable fragment vs real answer |

### §1 manual vs §8 CrewAI — same engine, different caller

| | **§1 — You drive** | **§8 — CrewAI drives** |
|---|---------------------|-------------------------|
| Trigger | `engine.add_request(...)` in a cell | Agent completes a task → `SmolVLLMCrewLLM.call()` |
| Calls per user question | 1 (or 3 in the walkthrough) | **2** — research facts, then write answer |
| Prompt content | `list(range(...))` token IDs | Real system/user text from agent roles |
| Your job in class | Step through diagram by hand | Point at `#1` and `#2` — *same scheduler, new traffic pattern* |

Uses `output_mode="text"` so FakeModel output looks like words (not `<450>`). For a real answer, run the optional GPU cell at the end of this section.
"""),
    code("""from agent_demo import build_crew, kickoff_crew

agent_engine = LLMEngine(
    num_gpu_blocks=24,
    block_size=16,
    max_batch_size=4,
    enable_metrics=True,
    seed=SEED,
    output_mode="text",
)
crew = build_crew(agent_engine, verbose_engine=True, dual_agent=True)
user_query = "What is paged attention in LLM inference?"
print(f"USER: {user_query}")
print("\\nExpect: ENGINE REQUEST #1 (Research Analyst) → #2 (Technical Writer)")
"""),
    code("""result = await kickoff_crew(crew, {"user_query": user_query})
print(result.raw)
agent_engine.metrics.print_summary()
"""),
    code("""# Comparison recap — print after the crew run
print("=" * 72)
print("CREWAI vs MANUAL ENGINE — what just happened")
print("=" * 72)
print(f"{'':30} {'§1 manual':^18} {'§8 CrewAI':^18}")
print("-" * 72)
rows = [
    ("Who called add_request()", "You", "SmolVLLMCrewLLM adapter"),
    ("Engine instances", "New engine per demo", "One engine, whole crew"),
    ("LLM calls this run", "1 per generate()", str(agent_engine.request_counter)),
    ("Scheduler logs", "show_engine_state()", "ENGINE REQUEST #N + state"),
    ("Orchestration visible?", "No", "Yes (CrewAI tasks/agents)"),
    ("Model on CPU", "FakeModel ids/text", "FakeModel text mode"),
]
for label, manual, crew in rows:
    print(f"{label:30} {manual:^18} {crew:^18}")
print("-" * 72)
print("Takeaway: CrewAI is the control plane *above* the engine — not a replacement for it.")
print("Production stack: Agent framework → inference server (vLLM) → GPU matmuls.")
print("=" * 72)
"""),
    code("""# Optional: CrewAI on GPU (when Lambda/CUDA ready)
if GPU_OK:
    gpu_crew_engine = LLMEngine(
        num_gpu_blocks=128,
        block_size=16,
        max_batch_size=2,
        use_real_model=True,
        enable_metrics=True,
        seed=SEED,
    )
    gpu_crew = build_crew(gpu_crew_engine, verbose_engine=True, dual_agent=True)
    gpu_result = await kickoff_crew(gpu_crew, {"user_query": user_query})
    print(gpu_result.raw)
    gpu_crew_engine.metrics.print_summary()
"""),
    md("""---
## 9 — Diff to production vLLM

Open side by side:

| smol_vllm | production |
|-----------|------------|
| `scheduler.py` → `schedule()` | `vllm/v1/core/sched/scheduler.py` |
| waiting / running / swapped | same skeleton |
| allocate + preempt | + chunked prefill, prefix cache, speculative decode, grammar bitmasks |

The toy wasn't a toy — it's the control plane with the extras removed.
"""),
    md("""---
## Export PDF (optional)

Run after all cells execute.
"""),
    code("""from lib.export_notebook import save_pdf
from IPython.display import FileLink

pdf = save_pdf("class5.ipynb")
FileLink(pdf.name)
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent.parent / "class5.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {out} ({len(cells)} cells)")
