from __future__ import annotations

import pytest

from gateway import config as cfg
from gateway.internal.tokenize import reset_tokenizer


@pytest.fixture(autouse=True)
def _reset_cfg():
    cfg.reset_defaults()
    cfg.TOKENIZER_ID = None
    reset_tokenizer()
    yield
    cfg.reset_defaults()
    reset_tokenizer()
