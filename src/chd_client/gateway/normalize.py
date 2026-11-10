"""The reply the client gets: OpenAI chat.completion fields filled in, cost labelled, and x_gateway attached.

Cost is never invented. usage.cost_source says where usage.cost comes from:
- reported: the server reported it (usage.cost);
- zero_api_charge: the profile says the server charges nothing per request (GPU time is not counted);
- estimated: tokens x the profile's prices (an estimate, not a bill);
- unknown: usage.cost is null, and a client with a budget should stop.
"""

from __future__ import annotations

import time
import uuid


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def cost_of(profile, usage: dict) -> tuple:
    """(cost, source) for a reply's usage."""
    c = profile.cost
    if _num(usage.get("cost")):
        if usage["cost"] == 0 and c.source == "zero_api_charge":
            return 0.0, "zero_api_charge"
        return float(usage["cost"]), "reported"
    if c.source == "zero_api_charge":
        return 0.0, "zero_api_charge"
    if c.source == "estimated" and _num(usage.get("prompt_tokens")) and _num(usage.get("completion_tokens")):
        return (usage["prompt_tokens"] * c.input_per_mtok + usage["completion_tokens"] * c.output_per_mtok) / 1e6, \
            "estimated"
    return None, "unknown"


def finalize(resp: dict, profile, meta: dict) -> dict:
    resp.setdefault("id", "chatcmpl-" + uuid.uuid4().hex[:24])
    resp.setdefault("object", "chat.completion")
    resp.setdefault("created", int(time.time()))
    resp["model"] = profile.name
    usage = resp.get("usage")
    if not isinstance(usage, dict):
        usage = resp["usage"] = {}
    reported = _num(usage.get("prompt_tokens")) and _num(usage.get("completion_tokens"))
    if reported and not _num(usage.get("total_tokens")):
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    cost, source = cost_of(profile, usage)
    usage["cost"] = cost
    usage["cost_source"] = source
    usage["cost_currency"] = profile.cost.currency
    meta["usage_reported"] = reported
    meta["cost_source"] = source
    if source == "estimated":
        meta["pricing_revision"] = profile.cost.pricing_revision
    resp["x_gateway"] = meta
    return resp


def error(status: int, message: str, code: str, meta: dict | None = None) -> dict:
    kind = "invalid_request_error" if 400 <= status < 500 else "server_error"
    out = {"error": {"message": message, "type": kind, "code": code}}
    if meta is not None:
        out["x_gateway"] = meta
    return out
