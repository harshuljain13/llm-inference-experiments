from __future__ import annotations

MODEL = "Qwen/Qwen3-0.6B"
REPLICA_URLS: list[str] = ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
DEFAULT_MAX_NUM_SEQS = 8
DEFAULT_MAX_MODEL_LEN = 16384
BLOCK_SIZE = 16
KV_CAPACITY_TOKENS = DEFAULT_MAX_MODEL_LEN
TOKENIZER_ID: str | None = None

DEFAULT_MAX_TOKENS = 256
SERVED_MODEL_NAME = "lab"
TENANT_TIER = {"default": "interactive", "chat": "interactive", "agent": "agentic"}
DEADLINE_BY_TIER_MS = {"interactive": 2000, "agentic": 5000, "batch": 8000}

BUCKET_RATE_TOKENS_PER_S = 50_000.0
BUCKET_BURST_TOKENS = 200_000.0
LIMITER_RPS = 4.0
LIMITER_BURST = 8.0

ADMISSION_ENABLED = False
STALE_CEILING_S = 2.0
KV_CEILING = 0.85
WAITING_CEILING_PER_REPLICA = 4
INIT_PREFILL_TOKENS_PER_S = 4_000.0
INIT_INTER_TOKEN_LATENCY_S = 0.008

QUEUE_ENABLED = False
QUEUE_MAXSIZE = 64
AGING_GAIN = 0.15
MAX_OVERTAKES = 8
LONG_PROMPT_TOKENS = 1_024
DISPATCH_OVERSHOOT = 4

USE_PREFIX_ROUTING = False
USE_P2C = False
W_PREFIX = 1.0
W_LOAD = 64.0
LOAD_CEILING = 2.0

SCRAPE_INTERVAL_S = 0.25
TRIE_TTL_S = 90.0
STATS_RING = 8_192
HTTP_TIMEOUT_S = 60.0

RETRY_AFTER_S = {
    "quota": 1,
    "no_signal": 1,
    "kv_pressure": 2,
    "queue_depth": 1,
    "no_headroom": 2,
    "deadline_unmeetable": 1,
    "queue_full": 1,
    "expired_in_queue": 1,
}
QUOTA_STATUS = 429
SHED_STATUS = 503

HOST = "127.0.0.1"
PORT = 8080


def apply_preset(name: str) -> None:
    global ADMISSION_ENABLED, QUEUE_ENABLED, USE_PREFIX_ROUTING
    presets = {
        "baseline": (False, False, False),
        "route": (False, False, True),
        "queue": (False, True, True),
        "full": (True, True, True),
    }
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}; expected {sorted(presets)}")
    ADMISSION_ENABLED, QUEUE_ENABLED, USE_PREFIX_ROUTING = presets[name]


def reset_defaults() -> None:
    global ADMISSION_ENABLED, QUEUE_ENABLED, USE_PREFIX_ROUTING, USE_P2C
    global TOKENIZER_ID, QUEUE_MAXSIZE, BUCKET_BURST_TOKENS
    global BUCKET_RATE_TOKENS_PER_S, REPLICA_URLS, PORT, LIMITER_RPS, LIMITER_BURST
    ADMISSION_ENABLED = False
    QUEUE_ENABLED = False
    USE_PREFIX_ROUTING = False
    USE_P2C = False
    TOKENIZER_ID = None
    QUEUE_MAXSIZE = 64
    BUCKET_BURST_TOKENS = 200_000.0
    BUCKET_RATE_TOKENS_PER_S = 50_000.0
    LIMITER_RPS = 4.0
    LIMITER_BURST = 8.0
    REPLICA_URLS = ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
    PORT = 8080
