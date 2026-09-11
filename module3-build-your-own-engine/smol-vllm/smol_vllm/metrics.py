import time
from datetime import datetime
from pathlib import Path


G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
W = "\033[0m"

_MAX_TRACKED = 50000


class Metrics:
    def __init__(self, *, show_gpu_stats: bool = True):
        self.show_gpu_stats = show_gpu_stats
        self.step_count = 0
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.prefill_latencies: list[float] = []
        self.decode_latencies: list[float] = []
        self.e2e_latencies: list[float] = []
        self.ttft_latencies: list[float] = []
        self.inter_token_deltas: list[float] = []
        self.prompt_lengths: list[int] = []
        self._request_start: dict[int, float] = {}
        self._request_first_token: dict[int, float] = {}
        self._last_token_time: dict[int, float] = {}

    def record_request_start(self, group_id: int, prompt_len: int = 0):
        self._request_start[group_id] = time.perf_counter()
        if prompt_len > 0:
            self.prompt_lengths.append(prompt_len)
        self._maybe_cleanup()

    def record_first_token(self, group_id: int):
        if group_id in self._request_start:
            self.ttft_latencies.append(time.perf_counter() - self._request_start[group_id])
            self._request_first_token[group_id] = time.perf_counter()

    def record_request_finish(self, group_id: int):
        if group_id in self._request_start:
            self.e2e_latencies.append(time.perf_counter() - self._request_start[group_id])
            del self._request_start[group_id]
        if group_id in self._request_first_token:
            del self._request_first_token[group_id]
        if group_id in self._last_token_time:
            del self._last_token_time[group_id]

    def record_inter_token(self, group_id: int) -> float | None:
        now = time.perf_counter()
        delta = None
        if group_id in self._last_token_time:
            delta = now - self._last_token_time[group_id]
            self.inter_token_deltas.append(delta)
        self._last_token_time[group_id] = now
        return delta

    def _maybe_cleanup(self):
        n = len(self._request_start) + len(self._last_token_time)
        if n > _MAX_TRACKED:
            self.clear()

    def clear(self):
        self._request_start.clear()
        self._request_first_token.clear()
        self._last_token_time.clear()

    def _tok_s_color(self, tok_s: float) -> str:
        if tok_s >= 100:
            return f"{G}{tok_s:.0f}{W}"
        if tok_s >= 30:
            return f"{Y}{tok_s:.0f}{W}"
        return f"{R}{tok_s:.0f}{W}"

    def _kv_bar(self, util: float, width: int = 10) -> str:
        filled = int(util * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _gpu_stats(self) -> str:
        try:
            import torch

            if not torch.cuda.is_available():
                return ""
            device = torch.cuda.current_device()
            mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
            # device_map="auto" / first step may show 0 — use peak or driver-reported used
            if mem_alloc <= 0:
                mem_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
            if mem_alloc <= 0 and mem_reserved <= 0:
                free, total = torch.cuda.mem_get_info(device)
                mem_used = (total - free) / 1024**2
                return f" gpu_mem={mem_used:.0f}/{total / 1024**2:.0f}MB"
            return f" gpu_mem={mem_alloc:.0f}/{mem_reserved:.0f}MB"
        except Exception:
            pass
        return ""

    def _cpu_stats(self) -> str:
        try:
            import psutil
            return f" cpu={psutil.cpu_percent():.0f}%"
        except Exception:
            pass
        return ""

    def print_step(
        self,
        step: int,
        prefill_ms: float,
        decode_ms: float,
        prefill_tokens: int,
        gen_tokens: int,
        running: int,
        waiting: int,
        swapped: int,
        block_util: float,
    ):
        self.step_count = step
        self.prompt_tokens_total += prefill_tokens
        self.generation_tokens_total += gen_tokens
        elapsed = (prefill_ms + decode_ms) / 1000
        tok_s = gen_tokens / elapsed if elapsed > 0 else 0

        tok_s_str = self._tok_s_color(tok_s)
        kv_bar = self._kv_bar(block_util)

        line = (
            f"  [metrics] step={step} "
            f"prefill={prefill_ms:.0f}ms decode={decode_ms:.0f}ms "
            f"gen_tokens={gen_tokens} tok/s={tok_s_str} "
            f"running={running} waiting={waiting} swapped={swapped} "
            f"kv={kv_bar} {block_util*100:.0f}%"
        )
        line += self._gpu_stats() if self.show_gpu_stats else ""
        line += self._cpu_stats()
        print(line)

    def print_summary(self):
        avg_ttft = sum(self.ttft_latencies) / len(self.ttft_latencies) * 1000 if self.ttft_latencies else 0
        avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) * 1000 if self.e2e_latencies else 0
        avg_tpot = sum(self.inter_token_deltas) / len(self.inter_token_deltas) * 1000 if self.inter_token_deltas else 0
        avg_prompt = sum(self.prompt_lengths) / len(self.prompt_lengths) if self.prompt_lengths else 0

        print("\n  [metrics] summary:")
        print(f"    prompt_tokens_total={self.prompt_tokens_total} generation_tokens_total={self.generation_tokens_total}")
        print(f"    time_to_first_token_avg_ms={avg_ttft:.0f} tpot_avg_ms={avg_tpot:.0f} e2e_request_latency_avg_ms={avg_e2e:.0f}")
        print(f"    prompt_len_avg={avg_prompt:.0f} (prefix tokens at start)")
        if self.prefill_latencies:
            print(f"    prefill_latency_avg_ms={sum(self.prefill_latencies)/len(self.prefill_latencies)*1000:.0f}")
        if self.decode_latencies:
            print(f"    decode_latency_avg_ms={sum(self.decode_latencies)/len(self.decode_latencies)*1000:.0f}")
        print()

    def to_csv_rows(self) -> list[dict]:
        avg_ttft = sum(self.ttft_latencies) / len(self.ttft_latencies) * 1000 if self.ttft_latencies else 0
        avg_e2e = sum(self.e2e_latencies) / len(self.e2e_latencies) * 1000 if self.e2e_latencies else 0
        avg_tpot = sum(self.inter_token_deltas) / len(self.inter_token_deltas) * 1000 if self.inter_token_deltas else 0
        avg_prompt = sum(self.prompt_lengths) / len(self.prompt_lengths) if self.prompt_lengths else 0
        avg_prefill = sum(self.prefill_latencies) / len(self.prefill_latencies) * 1000 if self.prefill_latencies else 0
        avg_decode = sum(self.decode_latencies) / len(self.decode_latencies) * 1000 if self.decode_latencies else 0

        return [{
            "prompt_tokens_total": self.prompt_tokens_total,
            "generation_tokens_total": self.generation_tokens_total,
            "time_to_first_token_avg_ms": avg_ttft,
            "tpot_avg_ms": avg_tpot,
            "e2e_request_latency_avg_ms": avg_e2e,
            "prompt_len_avg": avg_prompt,
            "prefill_latency_avg_ms": avg_prefill,
            "decode_latency_avg_ms": avg_decode,
        }]

    def save_csv(self, path: str | Path, experiment: str = ""):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.to_csv_rows()
        if not rows:
            return
        import csv
        for r in rows:
            r["experiment"] = experiment
            r["timestamp"] = datetime.now().isoformat()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
