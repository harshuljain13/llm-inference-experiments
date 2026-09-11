import time
from typing import List

from .sequence import SequenceGroup

# Transformers 4.46+ returns Cache objects for past_key_values (not legacy tuples)
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def _as_legacy_cache(past):
    """Normalize HF Cache objects → tuple of per-layer (key, value) pairs."""
    if past is None:
        return None
    if hasattr(past, "to_legacy_cache") and not isinstance(past, (tuple, list)):
        return past.to_legacy_cache()
    if DynamicCache is not None and isinstance(past, DynamicCache):
        if hasattr(past, "to_legacy_cache"):
            return past.to_legacy_cache()
    return past


def _iter_layer_kv(past):
    """Yield (key, value) tensors for each layer across HF cache formats."""
    if past is None:
        return
    if DynamicCache is not None and isinstance(past, DynamicCache):
        if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
            for k, v in zip(past.key_cache, past.value_cache, strict=False):
                yield k, v
            return
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()
    for layer in past:
        if isinstance(layer, (tuple, list)):
            yield layer[0], layer[1]
        elif DynamicCache is not None and isinstance(layer, DynamicCache):
            yield layer.key_cache[0], layer.value_cache[0]
        else:
            raise TypeError(f"Unknown cache layer type: {type(layer)!r}")


def _split_legacy_cache_per_group(past, batch_index: int):
    """Extract one sequence's KV cache from a batched legacy past."""
    layer_caches = []
    for k, v in _iter_layer_kv(past):
        layer_caches.append(
            (k[batch_index : batch_index + 1].clone(), v[batch_index : batch_index + 1].clone())
        )
    return tuple(layer_caches)


def _to_model_cache(past):
    """Convert stored legacy cache → DynamicCache for model.forward (all HF versions)."""
    if past is None:
        return None
    if DynamicCache is not None and isinstance(past, DynamicCache):
        return past
    if DynamicCache is None:
        return past

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past)

    # transformers ≥4.58: from_legacy_cache removed — use constructor
    try:
        return DynamicCache(past)
    except TypeError:
        return DynamicCache(ddp_cache_data=past)


class CausalLM:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str | None = None,
        token: str | None = None,
    ):
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[smol-vllm] Loading {model_name} on {device} ...")
        t0 = time.perf_counter()
        import os
        load_token = token or os.environ.get("HF_TOKEN")
        load_kw = {"token": load_token} if load_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **load_kw)
        load_dtype = torch.float16 if device == "cuda" else torch.float32
        model_kw = dict(
            device_map={"": device} if device == "cuda" else None,
            low_cpu_mem_usage=True,
            **load_kw,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=load_dtype, **model_kw
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=load_dtype, **model_kw
            )
        if device == "cpu":
            self.model = self.model.to(device)
        self.model.eval()
        self.kv_caches = {}
        print(f"[smol-vllm] Loaded in {time.perf_counter() - t0:.1f}s")

    def prefill(self, groups: List[SequenceGroup]) -> List[int]:
        if not groups:
            return []

        import torch

        prompts = [g.sequences[0].prompt_tokens for g in groups]
        max_len = max(len(p) for p in prompts)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded = []
        for p in prompts:
            padded.append([pad_id] * (max_len - len(p)) + p)

        input_ids = torch.tensor(padded, dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        attention_mask[input_ids == pad_id] = 0

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        elapsed = time.perf_counter() - t0

        logits = outputs.logits[:, -1, :]
        next_tokens = torch.argmax(logits, dim=-1).cpu().tolist()

        past = outputs.past_key_values
        for i, group in enumerate(groups):
            if (
                DynamicCache is not None
                and isinstance(past, DynamicCache)
                and hasattr(past, "batch_split")
            ):
                splits = past.batch_split(full_batch_size=len(groups), split_size=1)
                self.kv_caches[group.group_id] = _as_legacy_cache(splits[i])
            else:
                self.kv_caches[group.group_id] = _split_legacy_cache_per_group(past, i)

        avg_len = sum(len(p) for p in prompts) / len(prompts)
        print(
            f"[edu] Prefill batch={len(groups)} compute-bound "
            f"prompt_tokens≈{avg_len:.0f} {elapsed*1000:.0f}ms → {next_tokens}"
        )
        return next_tokens

    def decode(self, groups: List[SequenceGroup], block_tables: List[List[int]]) -> List[int]:
        _ = block_tables
        if not groups:
            return []

        import torch

        next_tokens = []
        t0 = time.perf_counter()
        for group in groups:
            last_token = group.sequences[0].output_tokens[-1]
            input_ids = torch.tensor([[last_token]], dtype=torch.long).to(self.device)
            past = _to_model_cache(self.kv_caches.get(group.group_id))

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1).item()
            next_tokens.append(next_tok)

            # Store legacy tuple per group (decode converts back to DynamicCache)
            self.kv_caches[group.group_id] = _as_legacy_cache(outputs.past_key_values)

        elapsed = time.perf_counter() - t0
        print(
            f"[edu] Decode batch={len(groups)} memory-bound "
            f"KV cache reads {elapsed*1000:.0f}ms → {next_tokens}"
        )
        return next_tokens

    def clear_cache(self, group_id: int):
        if group_id in self.kv_caches:
            del self.kv_caches[group_id]
