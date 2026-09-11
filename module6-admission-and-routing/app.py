#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from typing import Any

import httpx
from crewai import Agent, Crew, Process, Task
from crewai.llm import BaseLLM

from gateway import config as cfg
from limiter import RateLimited, RateLimiter

GATEWAY = os.environ.get("GATEWAY_URL", f"http://{cfg.HOST}:{cfg.PORT}")


def _as_messages(messages: str | list[dict[str, str]]) -> list[dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out: list[dict[str, str]] = []
    for m in messages:
        out.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})
    return out


class GatewayLLM(BaseLLM):
    llm_type: str = "gateway"
    gateway: str = GATEWAY
    limiter: Any = None
    max_tokens: int = 128
    tenant: str = "agent"
    last_meta: dict[str, Any] = {}

    model_config = {"arbitrary_types_allowed": True}

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        _ = (tools, callbacks, available_functions, from_task, from_agent, response_model)
        if self.limiter is not None and not self.limiter.try_acquire(1.0):
            raise RateLimited(f"app limiter: {self.limiter.rps} rps / burst {self.limiter.burst}")
        body = {
            "model": cfg.SERVED_MODEL_NAME,
            "messages": _as_messages(messages),
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        t0 = time.perf_counter()
        r = httpx.post(
            self.gateway.rstrip("/") + "/v1/chat/completions",
            json=body,
            headers={"X-Tenant-Id": self.tenant, "Content-Type": "application/json"},
            timeout=cfg.HTTP_TIMEOUT_S,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        reason = None
        text = ""
        if r.status_code >= 400:
            try:
                reason = r.json().get("error", {}).get("reason")
            except Exception:
                reason = f"http_{r.status_code}"
        else:
            text = r.json()["choices"][0]["message"]["content"]
        self.last_meta = {
            "status": r.status_code,
            "reason": reason,
            "replica": r.headers.get("X-Replica"),
            "prefix_match": int(r.headers.get("X-Prefix-Match-Tokens") or 0),
            "total_ms": total_ms,
            "text": text,
        }
        if r.status_code >= 400:
            raise RuntimeError(f"gateway {r.status_code}: {reason}")
        return text


def build_crew(llm: GatewayLLM, *, dual_agent: bool = True) -> Crew:
    if not dual_agent:
        agent = Agent(
            role="Inference Assistant",
            goal="Answer the user's question through the local serving gateway.",
            backstory="You call the lab gateway. Every token goes through the rate limiter.",
            llm=llm,
            verbose=True,
            max_iter=1,
        )
        task = Task(
            description="The user asked: {user_query}\n\nGive a short, helpful answer.",
            expected_output="A concise answer in 1-3 sentences.",
            agent=agent,
        )
        return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)

    researcher = Agent(
        role="Research Analyst",
        goal="Extract key facts about the user's topic via the gateway.",
        backstory="You gather bullet-point facts. Each call is rate-limited, then queued.",
        llm=llm,
        verbose=True,
        max_iter=1,
    )
    writer = Agent(
        role="Technical Writer",
        goal="Turn research notes into a clear answer via the gateway.",
        backstory="You polish facts. The second call should prefix-hit if routing works.",
        llm=llm,
        verbose=True,
        max_iter=1,
    )
    research_task = Task(
        description="The user asked: {user_query}\n\nList 3 key facts about this topic.",
        expected_output="Three bullet points of key facts.",
        agent=researcher,
    )
    write_task = Task(
        description=(
            "The user asked: {user_query}\n\n"
            "Using the research notes, write a clear 2-sentence answer."
        ),
        expected_output="A polished 2-sentence answer.",
        agent=writer,
        context=[research_task],
    )
    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True,
    )


DEFAULT_QUERIES = [
    "What is paged attention in LLM inference?",
    "Why does TTFT jump when the KV cache fills?",
    "What is prefix caching and when does it miss?",
    "How does a gateway decide to shed load?",
    "What is power of two choices in replica routing?",
]


def run_one(query: str, *, gateway: str, limiter: RateLimiter, dual_agent: bool, max_tokens: int) -> dict:
    llm = GatewayLLM(
        model="lab",
        gateway=gateway,
        limiter=limiter,
        max_tokens=max_tokens,
        tenant="agent",
    )
    crew = build_crew(llm, dual_agent=dual_agent)
    t0 = time.perf_counter()
    err = None
    raw = ""
    try:
        result = crew.kickoff(inputs={"user_query": query})
        raw = getattr(result, "raw", str(result))
    except (RateLimited, RuntimeError) as exc:
        err = str(exc)
    return {
        "query": query,
        "ok": err is None,
        "error": err,
        "result": raw,
        "total_ms": (time.perf_counter() - t0) * 1000,
        "last": llm.last_meta,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?", default=DEFAULT_QUERIES[0])
    p.add_argument("--gateway", default=GATEWAY)
    p.add_argument("--single", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    limiter = RateLimiter()
    print(f"app → limiter ({limiter.rps} rps, burst {limiter.burst}) → {args.gateway} → vLLM\n")
    queries = [args.query] if args.repeat == 1 else DEFAULT_QUERIES * ((args.repeat // len(DEFAULT_QUERIES)) + 1)
    queries = queries[: args.repeat]
    for i, q in enumerate(queries, 1):
        print(f"USER [{i}/{len(queries)}]: {q}")
        rec = run_one(
            q,
            gateway=args.gateway,
            limiter=limiter,
            dual_agent=not args.single,
            max_tokens=args.max_tokens,
        )
        if rec["ok"]:
            print(rec["result"])
        else:
            print("REFUSED:", rec["error"])
        print()


if __name__ == "__main__":
    main()
