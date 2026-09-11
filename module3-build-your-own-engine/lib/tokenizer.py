from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from smol_vllm import LLMEngine


def messages_to_text(messages: str | list[dict[str, Any]]) -> str:
    if isinstance(messages, str):
        return messages.strip()
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(part) for part in content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def encode_text(
    text: str,
    *,
    max_tokens: int = 256,
    engine: "LLMEngine | None" = None,
) -> list[int]:
    """FakeModel: char-level IDs. Real model: HF tokenizer on the engine."""
    if engine is not None and hasattr(engine.model, "tokenizer"):
        ids = engine.model.tokenizer.encode(text, add_special_tokens=False)
        return ids[:max_tokens]
    clipped = text[:max_tokens]
    return [ord(c) for c in clipped]


def max_prompt_tokens(engine: "LLMEngine", *, reserve_output: int = 32) -> int:
    """Leave room for KV output tokens when sizing the prompt."""
    capacity = engine.block_manager.num_blocks * engine.block_manager.block_size
    return max(32, capacity - reserve_output - engine.block_manager.block_size)


def decode_token_ids(
    token_ids: list[int],
    *,
    engine: "LLMEngine | None" = None,
) -> str:
    """FakeModel: ASCII decode. Real model: HF tokenizer."""
    if engine is not None and hasattr(engine.model, "tokenizer"):
        return engine.model.tokenizer.decode(token_ids, skip_special_tokens=True)
    parts: list[str] = []
    for tok in token_ids:
        if tok == 0:
            break
        if 32 <= tok <= 126:
            parts.append(chr(tok))
        else:
            parts.append(f"<{tok}>")
    return "".join(parts)
