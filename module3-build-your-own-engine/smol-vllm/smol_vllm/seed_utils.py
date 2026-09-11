from __future__ import annotations


def seed_all(seed: int | None) -> None:
    if seed is None:
        return
    import random

    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
