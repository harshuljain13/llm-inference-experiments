from __future__ import annotations

from typing import Any, TYPE_CHECKING

from crewai.llm import BaseLLM

if TYPE_CHECKING:
    from smol_vllm import LLMEngine

from .engine_state import show_engine_state
from .tokenizer import decode_token_ids, encode_text, max_prompt_tokens, messages_to_text


class SmolVLLMCrewLLM(BaseLLM):
    """CrewAI LLM adapter that routes every call through smol_vllm's LLMEngine."""

    llm_type: str = "smol_vllm"
    # Any — not LLMEngine: setup cell reloads smol_vllm; pydantic isinstance would fail
    engine: Any
    verbose_engine: bool = True
    max_output_tokens: int = 32

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

        prompt_text = messages_to_text(messages)
        limit = max_prompt_tokens(self.engine, reserve_output=self.max_output_tokens)
        prompt_tokens = encode_text(prompt_text, max_tokens=limit, engine=self.engine)

        stop_ids = [0]
        if hasattr(self.engine.model, "tokenizer"):
            eos = self.engine.model.tokenizer.eos_token_id
            if eos is not None:
                stop_ids = [eos]

        self.engine.add_request(
            prompt_tokens,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature or 1.0,
            stop_token_ids=stop_ids,
        )
        group_id = self.engine.request_counter - 1

        if self.verbose_engine:
            agent_label = getattr(from_agent, "role", None) if from_agent else "agent"
            req_num = group_id + 1
            print("\n" + "=" * 60)
            print(f"ENGINE REQUEST #{req_num} — CrewAI agent: {agent_label}")
            print("=" * 60)
            if len(prompt_text) > limit:
                print(f"[engine] prompt truncated to {limit} tokens (KV block budget)")
            preview = prompt_text[:200] + ("..." if len(prompt_text) > 200 else "")
            print(f"prompt ({len(prompt_tokens)} tokens): {preview!r}")

        if self.verbose_engine:
            print("\n[engine] request queued — scheduler state:")
            show_engine_state(self.engine)

        output_tokens: list[int] = []
        step = 0
        stalled = 0
        while True:
            step += 1
            outputs = self.engine.step()
            if self.verbose_engine:
                show_engine_state(self.engine, step=step)

            if not outputs:
                stalled += 1
                group = self.engine.groups.get(group_id)
                if group and group in self.engine.scheduler.waiting and stalled >= 2:
                    need = len(group.sequences[0].prompt_tokens)
                    blocks = (need + self.engine.block_manager.block_size - 1) // (
                        self.engine.block_manager.block_size
                    )
                    raise RuntimeError(
                        f"Engine could not allocate {blocks} KV blocks for prompt "
                        f"({need} tokens). Increase num_gpu_blocks or shorten the prompt."
                    )
            else:
                stalled = 0

            for out in outputs:
                if out.group_id == group_id:
                    output_tokens = out.output_tokens.copy()
                    if out.finished:
                        break

            group = self.engine.groups.get(group_id)
            if group and group.sequences[0].is_finished:
                break
            if step > 500:
                raise RuntimeError("smol_vllm engine exceeded step limit")

        text = decode_token_ids(output_tokens, engine=self.engine)
        if self.verbose_engine:
            print(f"\n[engine] finished — output tokens: {output_tokens}")
            print(f"[engine] decoded text: {text!r}\n")

        if not text.strip():
            text = f"[FakeModel output: {output_tokens}]"
        return text
