# smol-vllm

[![PyPI version](https://img.shields.io/pypi/v/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/smol-vllm.svg)](https://pypi.org/project/smol-vllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/GitHub-goabiaryan%2Fsmol__vllm-blue)](https://github.com/goabiaryan/smol_vllm)

Paged-attention inference engine: KV cache, continuous batching, preemption. Educational, not production.

## Install

```bash
pip install smol-vllm
```

Real models (TinyLlama, Qwen2, etc.):

```bash
pip install smol-vllm[tinyllama-1.1b]
# or
pip install smol-vllm[qwen2-0.5b]
```

## Quick Start

**FakeModel** (no extras):

```python
from smol_vllm import LLMEngine

engine = LLMEngine()
for token in engine.generate([1, 2, 3, 4, 5], max_tokens=20):
    print(token, end=" ")
```

**CausalLM** (needs `[tinyllama-1.1b]` or `[qwen2-0.5b]`):

```python
engine = LLMEngine(use_real_model=True)
tokenizer = engine.model.tokenizer
tokens = tokenizer.encode("Hello!", add_special_tokens=False)
for token in engine.generate(tokens, max_tokens=20):
    print(tokenizer.decode([token]), end="")
```

## Models

| Model | `model_name` |
|-------|---------------|
| TinyLlama 1.1B | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default) |
| Qwen2 0.5B | `Qwen/Qwen2-0.5B-Instruct` |
| Phi-2 | `microsoft/phi-2` |
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct` |
| Gemma 2 | `google/gemma-2-2b-it` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |

Gated models (Llama, Gemma, etc.) need a HuggingFace token. Options:

**1. Env var** (recommended):
```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

**2. In code**:
```python
LLMEngine(use_real_model=True, model_name="meta-llama/Llama-3.2-1B-Instruct", hf_token="hf_xxxx")
```

Get a token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Accept the model's license on its HF page first.

## Demo

```bash
smol-vllm-demo
```

## What It Teaches

- **PagedAttention** — block-based KV cache, ref counting
- **Continuous batching** — short jobs fill slots immediately
- **Preemption & swapping** — when memory runs low
- **Prefill vs decode** — compute-bound → memory-bound

Workflow: run with FakeModel first (zero deps), then switch to CausalLM to compare.

## Metrics

Step-level: prefill/decode latency, tok/s, KV util. Summary and CSV logs in `logs/`.

## License

MIT
