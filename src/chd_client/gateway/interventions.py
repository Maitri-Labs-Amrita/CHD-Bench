"""Interventions: changes the gateway makes to a model's reply.

All are off by default and turned on per profile. Each one that acts is recorded in x_gateway.interventions with its
version and the original text, and every token it generates is counted in the reply's usage, so assisted results can
be reported apart from baselines.

- tool_call_repair (v1): the reply has no tool call but its text holds one in a damaged form; the lenient parser of
  the profile's tool format turns it into a call.
- forced_tool_call (v1): the reply has no tool call although tools were offered; the gateway asks the same worker to
  continue the reply once after the format's call prefix (continue_final_message), stopping at the call suffix, and
  keeps the first call it parses. The engine must support prefill.
"""

from __future__ import annotations

import copy

from chd_client.gateway.toolcalls import FORMATS

REPAIR_VERSION = 1
FORCED_VERSION = 1


def offered_tools(body: dict) -> set:
    out = set()
    for t in body.get("tools") or []:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and isinstance(f.get("name"), str):
            out.add(f["name"])
    return out


def _choice(resp: dict) -> dict | None:
    try:
        ch = resp["choices"][0]
        return ch if isinstance(ch.get("message"), dict) else None
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def add_usage(a, b):
    """The usage of two generations together; a count missing from either is left out (unknown)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {}
    out = {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost"):
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            out[k] = x + y
    return out


def repair(profile, body: dict, resp: dict) -> dict | None:
    ch = _choice(resp)
    if ch is None or ch["message"].get("tool_calls"):
        return None
    text = ch["message"].get("content") or ""
    content, calls = FORMATS[profile.tool_format].lenient(text, offered_tools(body))
    if not calls:
        return None
    ch["message"]["content"] = content
    ch["message"]["tool_calls"] = calls
    ch["finish_reason"] = "tool_calls"
    return {"kind": "tool_call_repair", "version": REPAIR_VERSION, "original_content": text, "n_calls": len(calls)}


async def force_call(profile, body: dict, resp: dict, send) -> dict | None:
    """send(body) -> (status, json): one more request to the same worker."""
    ch = _choice(resp)
    if ch is None or ch["message"].get("tool_calls"):
        return None
    fmt = FORMATS[profile.tool_format]
    prefix = profile.interventions.call_prefix or fmt.prefix
    suffix = profile.interventions.call_suffix or fmt.suffix
    text = ch["message"].get("content") or ""
    body2 = copy.copy(body)
    body2["messages"] = list(body["messages"]) + [
        {"role": "assistant", "content": text + ("\n" if text and not text.endswith("\n") else "") + prefix}]
    body2.update(continue_final_message=True, add_generation_prompt=False, stop=[suffix], n=1)
    rec = {"kind": "forced_tool_call", "version": FORCED_VERSION, "original_content": text, "prefix": prefix}
    status, resp2 = await send(body2)
    if status != 200 or _choice(resp2) is None:
        rec["result"] = f"error {status}"
        return rec
    cont = _choice(resp2)["message"].get("content") or ""
    rec["continuation"] = cont
    rec["usage"] = resp2.get("usage")
    resp["usage"] = add_usage(resp.get("usage"), resp2.get("usage"))
    seq = prefix + cont + suffix
    if profile.interventions.tool_call_repair:
        _, calls = fmt.lenient(seq, offered_tools(body))
    else:
        _, calls = fmt.strict(seq)
    if not calls:
        rec["result"] = "no_call"
        return rec
    ch["message"]["content"] = text
    ch["message"]["tool_calls"] = calls[:1]
    ch["finish_reason"] = "tool_calls"
    rec["result"] = "call"
    return rec


async def apply(profile, body: dict, resp: dict, send) -> list:
    """Run the profile's interventions on a 200 reply, in place. Returns their records."""
    iv = profile.interventions
    if not (iv.tool_call_repair or iv.forced_tool_call) or not offered_tools(body):
        return []
    if body.get("tool_choice") == "none":
        return []
    records = []
    if iv.tool_call_repair:
        r = repair(profile, body, resp)
        if r:
            records.append(r)
    if iv.forced_tool_call:
        r = await force_call(profile, body, resp, send)
        if r:
            records.append(r)
    return records
