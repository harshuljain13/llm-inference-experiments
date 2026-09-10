#!/usr/bin/env python3
"""User query → CrewAI agent(s) → smol_vllm engine (with visible scheduler steps)."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Drop stale vendor shadow if present (see Readme.md)
_shadow = Path.cwd() / "smol_vllm"
if _shadow.is_dir():
    sys.path = [p for p in sys.path if Path(p).resolve() != _shadow.resolve()]

from crewai import Agent, Crew, Process, Task
from smol_vllm import LLMEngine

from lib.smol_crew_llm import SmolVLLMCrewLLM


def make_llm(engine: LLMEngine, *, verbose_engine: bool = True) -> SmolVLLMCrewLLM:
    return SmolVLLMCrewLLM(
        model="smol-vllm-fake",
        engine=engine,
        verbose_engine=verbose_engine,
        max_output_tokens=20,
        temperature=0.7,
    )


def build_crew(
    engine: LLMEngine,
    *,
    verbose_engine: bool = True,
    dual_agent: bool = True,
) -> Crew:
    """Build a CrewAI crew wired to smol_vllm.

    dual_agent=True (default): Research Analyst → Technical Writer (2 engine calls).
    dual_agent=False: single Inference Assistant (1 engine call).
    """
    llm = make_llm(engine, verbose_engine=verbose_engine)

    if not dual_agent:
        agent = Agent(
            role="Inference Assistant",
            goal="Answer the user's question using the local smol_vllm inference engine.",
            backstory="You route answers through smol_vllm so students can watch the scheduler.",
            llm=llm,
            verbose=True,
            max_iter=1,
        )
        task = Task(
            description=(
                "The user asked: {user_query}\n\n"
                "Use the inference engine to produce a short, helpful answer."
            ),
            expected_output="A concise answer in 1-3 sentences.",
            agent=agent,
        )
        return Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )

    researcher = Agent(
        role="Research Analyst",
        goal="Extract key facts about the user's topic via the inference engine.",
        backstory="You gather bullet-point facts. Every LLM call goes through smol_vllm.",
        llm=llm,
        verbose=True,
        max_iter=1,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Turn research notes into a clear answer via the inference engine.",
        backstory="You polish facts into plain English. Every LLM call goes through smol_vllm.",
        llm=llm,
        verbose=True,
        max_iter=1,
    )

    research_task = Task(
        description=(
            "The user asked: {user_query}\n\n"
            "Use the inference engine to list 3 key facts about this topic."
        ),
        expected_output="Three bullet points of key facts.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "The user asked: {user_query}\n\n"
            "Using the research notes from the previous task, write a clear 2-sentence answer."
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


async def kickoff_crew(crew: Crew, inputs: dict[str, Any]):
    """Run crew. In Jupyter: `result = await kickoff_crew(crew, inputs)`."""
    return await crew.akickoff(inputs=inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI → smol_vllm demo")
    parser.add_argument(
        "query",
        nargs="?",
        default="What is paged attention in LLM inference?",
        help="User question to send through the agent",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Use one agent instead of researcher + writer",
    )
    parser.add_argument("--quiet-engine", action="store_true", help="Hide engine step logs")
    args = parser.parse_args()

    engine = LLMEngine(
        num_gpu_blocks=24,
        block_size=16,
        max_batch_size=4,
        enable_metrics=True,
        seed=0,
        output_mode="text",
    )

    mode = "1 agent" if args.single else "2 agents (researcher → writer)"
    print(f"Workflow: user query → CrewAI ({mode}) → smol_vllm engine → response\n")
    print(f"USER: {args.query}\n")

    crew = build_crew(engine, verbose_engine=not args.quiet_engine, dual_agent=not args.single)
    result = asyncio.run(kickoff_crew(crew, {"user_query": args.query}))

    print("\n" + "=" * 60)
    print("CREW RESULT")
    print("=" * 60)
    print(result.raw)
    print(f"\nTotal engine requests this run: {engine.request_counter}")
    engine.metrics.print_summary()


if __name__ == "__main__":
    main()
