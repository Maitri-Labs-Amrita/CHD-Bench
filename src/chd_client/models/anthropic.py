"""Anthropic Messages API adapter (default model claude-opus-5-5), beta streaming endpoint.

Request: system = [client system prompt] (cache breakpoint on the last block), the environment's
tools verbatim, adaptive thinking with summarized display, effort in output_config, tool_choice auto (forced tool
choice is rejected by this model). The assistant content comes back verbatim (SDK blocks, thinking untouched).
By default every image is kept and the history is append-only, so earlier thinking stays valid. Only with an image
window or an image cap (which edit earlier turns), or in a resumed episode (restore: recorded replies replayed), does
the request set block_binding.prefix_mismatch_behavior = drop_block. No fallback model is ever configured: a refusal
ends the episode.
"""

import base64
import importlib.metadata
import sys
import time

from chd_client.models.base import ModelAdapter, ModelTurn, ToolCall, Usage, placeholder, DEFAULT_MAX_TOKENS
from chd_client.models.openrouter import RETRY_WAITS_S

PRICES = {"claude-opus-5-5": {"in": 4.0, "out": 20.0, "cache_write": 5.0, "cache_read": 0.20},
          "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cache_write": 2.5, "cache_read": 0.10}}
PRICES_SOURCE = {"source": "https://platform.claude.com/docs/en/about-claude/pricing", "confirmed_on": "2026-11-01"}
BETAS = ["thinking-binding-controls-2026-08-01"]
EPHEMERAL = {"type": "ephemeral"}
# A transient failure resends the same request after each of these waits (s), as the OpenRouter adapter does (about
# 20 min in all) before the turn fails. The history is append-only and cached, so a resend costs mostly cache reads;
# the output a dropped stream had produced is billed but not counted in the episode's cost.
# Retried: connection errors and timeouts (also a raw transport error out of the stream), 429, 5xx and 529, and an
# `error` event in the stream of type overloaded_error / api_error / rate_limit_error. Any other 4xx is the request's
# own fault and ends the turn at once.
TRANSIENT_STREAM_ERRORS = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}


def _transient(e) -> bool:
    """Whether a failed attempt is worth sending again (RETRY_WAITS_S)."""
    import anthropic
    if isinstance(e, (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError,
                      anthropic.OverloadedError)):
        return True
    if isinstance(e, anthropic.APIStatusError):
        if e.status_code == 429 or e.status_code >= 500:
            return True
        # an `error` event inside a 200 stream: the SDK raises a plain APIStatusError with the event as its body
        err = e.body.get("error") if isinstance(e.body, dict) else None
        return isinstance(err, dict) and err.get("type") in TRANSIENT_STREAM_ERRORS
    # the stream breaks while it is read: the transport's own error (httpx / httpx2), not wrapped by the SDK
    return any(c.__name__ == "TransportError" and c.__module__.split(".")[0] in ("httpx", "httpx2")
               for c in type(e).__mro__)


def _causes(e) -> list[str]:
    """The chain under an exception ('Connection error.' alone says nothing): type and message of each cause."""
    out, c, seen = [], e.__cause__ or e.__context__, set()
    while c is not None and id(c) not in seen and len(out) < 5:
        seen.add(id(c))
        out.append(f"{type(c).__module__}.{type(c).__name__}: {str(c)[:300]}")
        c = c.__cause__ or c.__context__
    return out


def image_block(png: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}}


def _versions(*pkgs) -> dict:
    out = {}
    for p in pkgs:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    return out


def price_record(model: str) -> dict:
    """The provenance record of the built-in price for `model`: its PRICES rates and PRICES_SOURCE."""
    return {"table": "chd_client.models.anthropic.PRICES", "rates": dict(PRICES[model]), **PRICES_SOURCE}


def cost_usd(model: str, uncached: int, out: int, cache_write: int, cache_read: int, prices=None) -> float | None:
    """USD from the usage tokens; prices: {model: {in, out, cache_write, cache_read}} (default PRICES)."""
    p = (PRICES if prices is None else prices).get(model)
    if p is None:
        return None
    return (uncached * p["in"] + out * p["out"] + cache_write * p["cache_write"] + cache_read * p["cache_read"]) / 1e6


def split_calls(content) -> tuple[list, str]:
    """ToolCalls with their reasoning (non-empty thinking and text blocks since the previous tool_use), and all text."""
    calls, buf, texts = [], [], []
    for b in content:
        if b.type == "thinking" and (b.thinking or "").strip():
            buf.append(b.thinking.strip())
        elif b.type == "text" and b.text.strip():
            buf.append(b.text.strip())
            texts.append(b.text)
        elif b.type == "tool_use":
            calls.append(ToolCall(b.id, b.name, dict(b.input or {}), "\n\n".join(buf) or None))
            buf = []
    return calls, "".join(texts)


class AnthropicModel(ModelAdapter):
    provider = "anthropic"  # the request log's provider: the registry entry's name (models.registry)
    prices = None  # PRICES; a registry entry with cost.prices sets its own (models.registry)
    resumed_binding = False  # set by restore(): the requests of a resumed episode set block_binding drop_block

    def __init__(self, model="claude-opus-5-5", effort="max", max_tokens=DEFAULT_MAX_TOKENS, image_window=None, client=None,
                 sleep=time.sleep):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model_id = self.model_version = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.image_window = image_window
        self.last_request = None
        self.sleep = sleep
        self.retries = []  # (attempt, error type) of each retried request

    def request_config(self) -> dict:
        r = self.render_params()
        return {**r, "image_window": self.image_window, "retry_waits_s": list(RETRY_WAITS_S),
                **self.image_cap_config()}

    def render_params(self) -> dict:
        """Everything of a request except system, tools and messages (provenance)."""
        thinking, extra = {"type": "adaptive", "display": "summarized"}, {}
        # stripping images (a window or a cap) edits earlier turns, and a resumed episode replays recorded replies
        # (restore): let the API drop stale thinking rather than refuse the request
        if self.image_window is not None or self.max_images is not None or self.resumed_binding:
            thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            extra["betas"] = list(BETAS)
        if self.effort:
            extra["output_config"] = {"effort": self.effort}
        return dict(model=self.model_id, max_tokens=self.max_tokens, thinking=thinking,
                    tool_choice={"type": "auto"}, **extra)

    def sdk_versions(self) -> dict:
        return _versions("anthropic")

    # ------------------------------------------------------------------ rendering
    def _obs_blocks(self, turn, obs):
        kept = self.kept_images(turn, obs)
        blocks = [{"type": "text", "text": obs.text}]
        for view, png, keep in zip(obs.frames, obs.images, kept):
            blocks.append(image_block(png) if keep else {"type": "text", "text": placeholder(obs, view)})
        return blocks, not all(kept)

    def render(self) -> dict:
        system = [{"type": "text", "text": b} for b in self.system_blocks]
        system[-1]["cache_control"] = EPHEMERAL
        msgs, stable = [], None
        for e, turn in zip(self.history, self.turn_index()):
            stripped = False
            if e[0] == "obs0":
                blocks, stripped = self._obs_blocks(turn, e[2])
                msgs.append({"role": "user", "content": [{"type": "text", "text": e[1]}] + blocks})
            elif e[0] == "assistant":
                msgs.append({"role": "assistant", "content": e[1].content})
            elif e[0] == "results":
                content = []
                for call, obs in e[1]:
                    blocks, s = self._obs_blocks(turn, obs)
                    stripped |= s
                    content.append({"type": "tool_result", "tool_use_id": call.id, "content": blocks,
                                    "is_error": obs.is_error})
                msgs.append({"role": "user", "content": content})
            elif e[0] == "text":
                block = {"type": "text", "text": e[1]}
                if msgs and msgs[-1]["role"] == "user":
                    msgs[-1]["content"].append(block)
                else:
                    msgs.append({"role": "user", "content": [block]})
            if stripped:
                stable = len(msgs) - 1
        if stable is not None and stable != len(msgs) - 1:
            msgs[stable]["content"][-1]["cache_control"] = EPHEMERAL
        msgs[-1]["content"][-1]["cache_control"] = EPHEMERAL
        p = self.render_params()
        return dict(
            model=p.pop("model"), max_tokens=p.pop("max_tokens"), system=system,
            tools=[{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in self.tools],
            messages=msgs, **p)

    # ------------------------------------------------------------------ one request
    def step(self) -> ModelTurn:
        req = self.render()
        self.last_request = req
        self.record_request(provider=self.provider, system=req["system"], tools=req["tools"], messages=req["messages"],
                            params={k: v for k, v in req.items() if k not in ("system", "tools", "messages")})
        for attempt, wait in enumerate((*RETRY_WAITS_S, None)):
            t0 = time.monotonic()
            self._events = 0
            try:
                with self.client.beta.messages.stream(**req) as s:
                    for _ in s:  # counted, so a failed attempt says how far the stream got
                        self._events += 1
                    resp = s.get_final_message()
                break
            except Exception as e:  # noqa: BLE001 - only transient errors are retried; the rest re-raise
                self._log_failure(attempt + 1, e, t0)
                if wait is None or not _transient(e):
                    raise
                self.retries.append((attempt + 1, type(e).__name__))
                self.sleep(wait)
        self.record_response(resp)
        return self.absorb(resp)

    def _log_failure(self, attempt, e, t0):
        """A failed attempt goes to responses.jsonl and stderr (the episode log): its error, the causes under it (the
        underlying network error: reset, closed by the remote end, ...), how long it ran and how many stream events
        arrived first. Resume skips these records (it keeps only replies)."""
        info = {"attempt": attempt, "type": type(e).__name__, "message": str(e)[:500], "causes": _causes(e),
                "elapsed_s": round(time.monotonic() - t0, 1), "events": self._events,
                "transient": _transient(e), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        status = getattr(e, "status_code", None)
        if status is not None:
            info["status"] = status
        self.record_response({"failed_attempt": info})
        cause = f" <- {info['causes'][0][:200]}" if info["causes"] else ""
        print(f"model request attempt {attempt} failed after {info['elapsed_s']} s ({info['events']} stream events): "
              f"{info['type']}: {info['message'][:200]}{cause}", file=sys.stderr, flush=True)

    def restore(self, raw: dict) -> ModelTurn:
        """Resume (chd_client.resume): a recorded reply (responses.jsonl "raw") goes into the history as step() put it
        there when it arrived; nothing is sent. Its thinking blocks keep their text and their `signature` (the
        encrypted reasoning), so the next request carries the earlier reasoning as the uninterrupted run would have.

        The recording is a full model_dump, so it holds every optional field, null when the reply did not set it; the
        SDK sends only the fields a block has set. Null fields are dropped here so a restored block goes out as the
        live one did (a block arrives from the stream with only the fields the API sent). From then on the resumed
        episode's requests set block_binding drop_block (resumed_binding): should a restored block still differ from
        the original, the API drops that block's reasoning (listed in input_transformations, which every model_turn
        records) instead of refusing the request."""
        from anthropic.types.beta import BetaMessage
        if not isinstance(raw, dict) or not isinstance(raw.get("content"), list):
            from chd_client.resume import ResumeError
            raise ResumeError("resume: a kept turn has no recorded Anthropic reply in responses.jsonl")
        raw = {**raw, "content": [{k: v for k, v in b.items() if v is not None} if isinstance(b, dict) else b
                                  for b in raw["content"]]}
        self.resumed_binding = True
        return self.absorb(BetaMessage.model_validate(raw))

    def absorb(self, resp) -> ModelTurn:
        """One reply into the history and a ModelTurn (step() after a request; restore() on resume)."""
        self.history.append(("assistant", resp))
        calls, text = split_calls(resp.content)
        u = resp.usage
        cw, cr = u.cache_creation_input_tokens or 0, u.cache_read_input_tokens or 0
        usage = Usage(u.input_tokens + cw + cr, u.output_tokens, cr, cw,
                      cost_usd(self.model_id, u.input_tokens, u.output_tokens, cw, cr, self.prices))
        dropped = [t.model_dump(mode="json") if hasattr(t, "model_dump") else t
                   for t in (getattr(resp, "input_transformations", None) or [])]
        extra = {"id": resp.id, "model": resp.model, "input_transformations": dropped}
        if resp.stop_details is not None:
            extra["stop_details"] = resp.stop_details.model_dump(mode="json")
        return ModelTurn(calls, text, resp.stop_reason or "?", usage, resp, extra)
