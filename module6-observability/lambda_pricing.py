# lambda_pricing.py — Resolve GPU $/hour from Lambda Cloud public API (optional; gateway uses if LAMBDA_CLOUD_API_KEY is set).
# API reference: https://docs.lambda.ai/api/cloud#listInstanceTypes

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

_log = logging.getLogger("gateway.lambda_pricing")

LAMBDA_API_BASE = os.environ.get("LAMBDA_CLOUD_API_BASE", "https://cloud.lambdalabs.com/api/v1").rstrip("/")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key.strip()}"}


def _cents_to_usd_hour(cents: Any) -> float | None:
    try:
        v = float(cents)
    except (TypeError, ValueError):
        return None
    if v < 0:
        return None
    return v / 100.0


def _merge_price_entry(prices: dict[str, float], name: str | None, cents: Any) -> None:
    if not name:
        return
    usd = _cents_to_usd_hour(cents)
    if usd is None:
        return
    key = str(name).strip()
    if not key:
        return
    prev = prices.get(key)
    if prev is not None and abs(prev - usd) > 1e-6:
        _log.debug("Lambda price for %s: keeping %.4f (also saw %.4f)", key, prev, usd)
        return
    prices[key] = usd


def _extract_prices_from_obj(obj: Any, prices: dict[str, float]) -> None:
    """Walk nested JSON from GET /instance-types (shape varies by API version)."""
    if isinstance(obj, dict):
        it = obj.get("instance_type")
        if isinstance(it, dict):
            name = it.get("name")
            cents = (
                obj.get("price_cents_per_hour")
                or it.get("price_cents_per_hour")
                or obj.get("price_cents")
            )
            _merge_price_entry(prices, name, cents)
        if "name" in obj and "price_cents_per_hour" in obj:
            _merge_price_entry(prices, obj.get("name"), obj.get("price_cents_per_hour"))
        for v in obj.values():
            _extract_prices_from_obj(v, prices)
    elif isinstance(obj, list):
        for item in obj:
            _extract_prices_from_obj(item, prices)


def fetch_instance_type_prices_usd_per_hour(api_key: str, client: httpx.Client | None = None) -> dict[str, float]:
    """Return map instance type name -> USD/hour for the whole instance (Lambda's posted rate)."""
    own = client is None
    if own:
        client = httpx.Client(timeout=30.0)
    try:
        url = f"{LAMBDA_API_BASE}/instance-types"
        r = client.get(url, headers=_auth_headers(api_key))
        r.raise_for_status()
        body = r.json()
    finally:
        if own:
            client.close()

    prices: dict[str, float] = {}
    data = body.get("data", body)
    _extract_prices_from_obj(data, prices)
    if not prices:
        _extract_prices_from_obj(body, prices)
    return prices


def _running_instance_type_name(api_key: str, client: httpx.Client | None = None) -> str | None:
    own = client is None
    if own:
        client = httpx.Client(timeout=30.0)
    try:
        url = f"{LAMBDA_API_BASE}/instances"
        r = client.get(url, headers=_auth_headers(api_key))
        r.raise_for_status()
        body = r.json()
    finally:
        if own:
            client.close()

    rows = body.get("data")
    if not isinstance(rows, list):
        return None
    active_like = frozenset({"active", "running", "booting", "pending"})
    for inst in rows:
        if not isinstance(inst, dict):
            continue
        status = str(inst.get("status", "")).lower()
        if status and status not in active_like:
            continue
        it = inst.get("instance_type")
        if isinstance(it, dict) and it.get("name"):
            return str(it["name"])
        if inst.get("instance_type_name"):
            return str(inst["instance_type_name"])
    return None


def resolve_gpu_hourly_usd_from_lambda(
    api_key: str,
    *,
    instance_type: str | None = None,
) -> float | None:
    """
    Look up current Lambda list price for the instance type.

    - If ``instance_type`` is set (or env LAMBDA_INSTANCE_TYPE), use that name.
    - Else use the first active-like row from GET /instances.
    """
    explicit = (instance_type or os.environ.get("LAMBDA_INSTANCE_TYPE", "")).strip() or None
    with httpx.Client(timeout=30.0) as client:
        prices = fetch_instance_type_prices_usd_per_hour(api_key, client)
        if not prices:
            _log.warning("Lambda instance-types: no price map parsed (API shape may have changed).")
            return None
        name = explicit or _running_instance_type_name(api_key, client)
        if not name:
            _log.warning(
                "Set LAMBDA_INSTANCE_TYPE (e.g. gpu_1x_a10) or run at least one active instance to infer type."
            )
            return None
        if name not in prices:
            _log.warning("Unknown Lambda instance type %r; known: %s", name, sorted(prices.keys())[:12])
            return None
        return prices[name]
