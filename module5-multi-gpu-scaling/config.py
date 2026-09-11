from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from ray.serve.llm import LLMConfig
except ImportError:
    LLMConfig = Any


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or raw.strip() == "" else raw.strip()


@dataclass
class ParallelismFlags:
    num_replicas: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    enable_expert_parallel: bool
    data_parallel_size_vllm: int

    def summary(self) -> str:
        return (
            f"replicas={self.num_replicas} TP={self.tensor_parallel_size} "
            f"PP={self.pipeline_parallel_size} EP={self.enable_expert_parallel} "
            f"vllm_data_parallel_size={self.data_parallel_size_vllm}"
        )


_MODEL_DEFAULTS = {
    "tinyllama": ("tinyllama-chat", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    "llama-70b-instruct": (
        "llama-3.1-70b-instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ),
    "llama-70b-awq": (
        "llama-3.1-70b-awq",
        "casperhansen/llama-3.1-70b-instruct-awq",
    ),
    "mixtral-8x7b": (
        "mixtral-8x7b",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ),
}


def _model_spec(model_key: str) -> Dict[str, str]:
    if model_key not in _MODEL_DEFAULTS:
        raise ValueError(f"Unknown model key {model_key!r}. Valid: {list(_MODEL_DEFAULTS)}")
    default_id, default_src = _MODEL_DEFAULTS[model_key]
    return {
        "model_id": _env_str("PLAYGROUND_MODEL_ID", default_id),
        "model_source": _env_str("PLAYGROUND_MODEL_SOURCE", default_src),
    }


def _base_engine_kwargs() -> Dict[str, Any]:
    return {
        "trust_remote_code": _env_bool("PLAYGROUND_TRUST_REMOTE_CODE", True),
        "max_model_len": _env_int("PLAYGROUND_MAX_MODEL_LEN", 4096),
        "gpu_memory_utilization": float(
            os.environ.get("PLAYGROUND_GPU_MEMORY_UTILIZATION", "0.92")
        ),
        "enable_chunked_prefill": _env_bool("PLAYGROUND_ENABLE_CHUNKED_PREFILL", True),
        "max_num_batched_tokens": _env_int("PLAYGROUND_MAX_NUM_BATCHED_TOKENS", 8192),
    }


def preset_engine_and_deployment(preset: str) -> Dict[str, Any]:
    eng = _base_engine_kwargs()
    dep: Dict[str, Any] = {
        "num_replicas": 1,
        "max_ongoing_requests": _env_int("PLAYGROUND_MAX_ONGOING_REQUESTS", 64),
    }
    accel = _env_str("PLAYGROUND_ACCELERATOR_TYPE", "A10G")
    pg: Optional[Dict[str, Any]] = None

    preset = preset.strip().lower()

    if preset == "tiny_single":
        eng.update({"tensor_parallel_size": 1, "pipeline_parallel_size": 1})
    elif preset == "tiny_dp":
        eng.update({"tensor_parallel_size": 1, "pipeline_parallel_size": 1})
        dep["num_replicas"] = _env_int("PLAYGROUND_NUM_REPLICAS", 2)
    elif preset == "tiny_tp":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 2),
                "pipeline_parallel_size": 1,
            }
        )
    elif preset == "tiny_pp":
        eng.update(
            {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": _env_int("PLAYGROUND_PIPELINE_PARALLEL_SIZE", 2),
            }
        )
    elif preset == "tiny_tp_pp":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 2),
                "pipeline_parallel_size": _env_int("PLAYGROUND_PIPELINE_PARALLEL_SIZE", 2),
            }
        )
    elif preset == "tiny_dp_tp":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 2),
                "pipeline_parallel_size": 1,
            }
        )
        dep["num_replicas"] = _env_int("PLAYGROUND_NUM_REPLICAS", 2)
    elif preset == "moe_ep":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 2),
                "pipeline_parallel_size": 1,
                "enable_expert_parallel": True,
            }
        )
    elif preset == "llm_70b_tp_awq":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 4),
                "pipeline_parallel_size": _env_int("PLAYGROUND_PIPELINE_PARALLEL_SIZE", 1),
                "quantization": _env_str("PLAYGROUND_QUANTIZATION", "awq"),
                "dtype": _env_str("PLAYGROUND_DTYPE", "half"),
            }
        )
        accel = _env_str("PLAYGROUND_ACCELERATOR_TYPE", "A100-80G")
    elif preset == "llm_70b_tp_pp_awq":
        eng.update(
            {
                "tensor_parallel_size": _env_int("PLAYGROUND_TENSOR_PARALLEL_SIZE", 2),
                "pipeline_parallel_size": _env_int("PLAYGROUND_PIPELINE_PARALLEL_SIZE", 2),
                "quantization": _env_str("PLAYGROUND_QUANTIZATION", "awq"),
                "dtype": _env_str("PLAYGROUND_DTYPE", "half"),
            }
        )
        accel = _env_str("PLAYGROUND_ACCELERATOR_TYPE", "A100-80G")
    else:
        raise ValueError(
            f"Unknown PLAYGROUND_PRESET={preset!r}. Valid presets are defined in config.preset_engine_and_deployment."
        )

    vllm_dp = _env_int("PLAYGROUND_VLLM_DATA_PARALLEL_SIZE", 0)
    if vllm_dp > 0:
        eng["data_parallel_size"] = vllm_dp

    if _env_bool("PLAYGROUND_ENABLE_EXPERT_PARALLEL", False):
        eng["enable_expert_parallel"] = True

    pg_raw = os.environ.get("PLAYGROUND_PLACEMENT_GROUP_JSON")
    if pg_raw:
        pg = json.loads(pg_raw)

    return {
        "deployment_config": dep,
        "engine_kwargs": eng,
        "placement_group_config": pg,
        "accelerator_type": accel,
    }


def resolve_parallelism_flags(engine_kwargs: Dict[str, Any], deployment_config: Dict[str, Any]):
    return ParallelismFlags(
        num_replicas=int(deployment_config.get("num_replicas", 1)),
        tensor_parallel_size=int(engine_kwargs.get("tensor_parallel_size", 1)),
        pipeline_parallel_size=int(engine_kwargs.get("pipeline_parallel_size", 1)),
        enable_expert_parallel=bool(engine_kwargs.get("enable_expert_parallel", False)),
        data_parallel_size_vllm=int(engine_kwargs.get("data_parallel_size", 0) or 0),
    )


def build_llm_config() -> "LLMConfig":
    if LLMConfig is Any:
        raise ImportError("ray[serve] is required to build LLMConfig")

    model_key = _env_str("PLAYGROUND_MODEL_KEY", "tinyllama")
    model = _model_spec(model_key)

    preset = _env_str("PLAYGROUND_PRESET", "tiny_single")
    parts = preset_engine_and_deployment(preset)
    flags = resolve_parallelism_flags(parts["engine_kwargs"], parts["deployment_config"])

    logger.info(
        "Playground LLMConfig: model_key=%s preset=%s parallelism=[%s] accelerator=%s",
        model_key,
        preset,
        flags.summary(),
        parts["accelerator_type"],
    )
    print(
        "[playground] model=%s (%s) preset=%s :: %s"
        % (
            model["model_id"],
            model["model_source"],
            preset,
            flags.summary(),
        ),
        flush=True,
    )

    experimental = {}
    n_ingress = _env_int("PLAYGROUND_NUM_INGRESS_REPLICAS", 2)
    if n_ingress > 0:
        experimental["num_ingress_replicas"] = n_ingress
    stream_ms = os.environ.get("PLAYGROUND_STREAM_BATCHING_INTERVAL_MS")
    if stream_ms:
        experimental["stream_batching_interval_ms"] = float(stream_ms)

    return LLMConfig(
        model_loading_config={
            "model_id": model["model_id"],
            "model_source": model["model_source"],
        },
        accelerator_type=parts["accelerator_type"],
        deployment_config=parts["deployment_config"],
        engine_kwargs=parts["engine_kwargs"],
        placement_group_config=parts["placement_group_config"],
        llm_engine="vLLM",
        log_engine_metrics=_env_bool("PLAYGROUND_LOG_ENGINE_METRICS", True),
        experimental_configs=experimental or None,
        runtime_env=_runtime_env_from_env(),
    )


def _runtime_env_from_env() -> Optional[Dict[str, Any]]:
    raw = os.environ.get("PLAYGROUND_RUNTIME_ENV_JSON")
    if not raw:
        return None
    return json.loads(raw)
