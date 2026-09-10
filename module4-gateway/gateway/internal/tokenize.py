from __future__ import annotations

from gateway import config as cfg

_hf = None


def messages_to_text(messages: list) -> str:
    parts = []
    for m in messages:
        parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
    return "\n".join(parts)


def tokenize(text: str) -> list[int]:
    global _hf
    if cfg.TOKENIZER_ID:
        if _hf is None:
            from transformers import AutoTokenizer

            _hf = AutoTokenizer.from_pretrained(cfg.TOKENIZER_ID, trust_remote_code=True)
        ids = _hf.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]
    if not text:
        return [0]
    ids: list[int] = []
    for i in range(0, len(text), 4):
        chunk = text[i : i + 4].encode()
        ids.append(int.from_bytes((chunk + b"\x00\x00\x00\x00")[:4], "little") & 0x7FFFFFFF)
    return ids


def reset_tokenizer() -> None:
    global _hf
    _hf = None
