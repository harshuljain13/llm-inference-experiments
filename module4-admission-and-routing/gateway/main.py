from __future__ import annotations

import argparse

import uvicorn

from gateway import config as cfg


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=cfg.HOST)
    p.add_argument("--port", type=int, default=cfg.PORT)
    p.add_argument("--replicas", default=",".join(cfg.REPLICA_URLS))
    p.add_argument("--tokenizer", default="")
    p.add_argument("--preset", choices=["baseline", "route", "queue", "full"], default="")
    p.add_argument("--admission", action="store_true")
    p.add_argument("--queue-bounded", action="store_true")
    p.add_argument("--prefix-routing", action="store_true")
    p.add_argument("--p2c", action="store_true")
    p.add_argument("--queue-maxsize", type=int, default=0)
    return p.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    cfg.HOST = args.host
    cfg.PORT = args.port
    cfg.REPLICA_URLS = [u.strip() for u in args.replicas.split(",") if u.strip()]
    cfg.TOKENIZER_ID = args.tokenizer or None
    if args.preset:
        cfg.apply_preset(args.preset)
    else:
        cfg.ADMISSION_ENABLED = bool(args.admission)
        cfg.QUEUE_ENABLED = bool(args.queue_bounded)
        cfg.USE_PREFIX_ROUTING = bool(args.prefix_routing)
    cfg.USE_P2C = bool(args.p2c)
    if args.queue_maxsize:
        cfg.QUEUE_MAXSIZE = args.queue_maxsize


def main(argv: list[str] | None = None) -> None:
    apply_args(_parse(argv))
    from gateway.internal.app import create_app

    uvicorn.run(create_app(), host=cfg.HOST, port=cfg.PORT, log_level="info")


if __name__ == "__main__":
    main()
