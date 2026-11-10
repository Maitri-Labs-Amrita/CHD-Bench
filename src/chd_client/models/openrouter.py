"""OpenRouter adapter (OpenAI-compatible chat completions) for non-Anthropic models: --model openrouter:<slug>.

Tools go out as function tools with the environment's schema verbatim. Each tool result is a `tool` message with
the JSON text; the images of one batch of results follow in a single user message, each labelled with its tool,
step and view (tool messages cannot carry images). The same image window applies. Cost comes from OpenRouter's
usage accounting (`usage.cost`); when it is missing, not a number, not finite or below 0, cost_usd is None and the
runner stops (cost_unknown).

Claude models (`anthropic/...`) are cached only with explicit breakpoints (OpenAI and Gemini cache on their own):
`cache_control` goes on the system prompt, on the last message whose images the window stripped, and on the last
message, as the Anthropic adapter does. Their requests are pinned to one provider without fallbacks, because a
cache lives at one provider: Amazon Bedrock, not OpenRouter's "anthropic" provider, whose extra
moderation rejects CT volume renderings as "violence/graphic".
"""

import base64
import json
import math
import os
import sys
import time

from chd_client.models.base import ModelAdapter, ModelTurn, ToolCall, Usage, placeholder, DEFAULT_MAX_TOKENS

BASE_URL = "https://openrouter.ai/api/v1"
EPHEMERAL = {"type": "ephemeral"}
CLAUDE_PROVIDER = {"order": ["amazon-bedrock", "google-vertex"], "allow_fallbacks": False}
STOPS = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}
RETRY_WAITS_S = (5, 10, 20, 40, 80, 160, 300, 300, 300)
REQUEST_TIMEOUT_S = 1800.0
STREAM = True
IMAGE_FORMATS = ("png", "webp_lossless")


def encode_image(png: bytes, image_format: str = "png") -> str:
    """The data URL of a rendered PNG in `image_format` (IMAGE_FORMATS)."""
    if image_format == "png":
        return "data:image/png;base64," + base64.b64encode(png).decode()
    if image_format == "webp_lossless":
        import io
        from PIL import Image
        out = io.BytesIO()
        Image.open(io.BytesIO(png)).save(out, "WEBP", lossless=True, exact=True, method=4)
        return "data:image/webp;base64," + base64.b64encode(out.getvalue()).decode()
    raise ValueError(f"image_format {image_format!r}: one of {', '.join(IMAGE_FORMATS)}")


class ErrorReply(Exception):
    """A reply that carries an `error` object and no choices (OpenRouter puts upstream failures in the body)."""

    def __init__(self, error):
        self.error = error if isinstance(error, dict) else {"message": str(error)}
        self.code = self.error.get("code")
        super().__init__(f"{self.code}: {self.error.get('message')}")

    @property
    def transient(self) -> bool:
        code = self.code if isinstance(self.code, int) else int(self.code) if str(self.code).isdigit() else None
        kind = (self.error.get("metadata") or {}).get("error_type")
        return code is None or code == 429 or code >= 500 or kind == "timeout"


def _transient(e) -> bool:
    if isinstance(e, ErrorReply):
        return e.transient
    import openai
    if type(e) is openai.APIError:
        return ErrorReply(e.body if isinstance(e.body, dict) else {"message": str(e)}).transient
    try:
        import openai
    except ImportError:  # pragma: no cover - the adapter needs openai anyway
        return False
    return isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError,
                          openai.RateLimitError))


MAX_RETRY_AFTER_S = 600


class RateLimited(RuntimeError):
    """The provider still refused with a rate limit after every retry."""


def _is_rate_limit(e) -> bool:
    if isinstance(e, ErrorReply):
        return str(e.code) == "429"
    import openai
    return isinstance(e, openai.RateLimitError)


def _retry_after(e) -> float | None:
    """The Retry-After header of a failed request, in seconds (capped), or None."""
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) or {}
    v = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        return min(float(v), MAX_RETRY_AFTER_S) if v is not None else None
    except (TypeError, ValueError):
        return None


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    v = getattr(obj, key, None)
    if v is None:
        v = (getattr(obj, "model_extra", None) or {}).get(key)
    return default if v is None else v


def _reported_cost(v) -> float | None:
    """usage.cost as USD, or None (cost_unknown) when it is missing, not a number, not finite or below 0: json.loads
    takes NaN and Infinity, and a NaN cost would turn off every cost-cap comparison."""
    if v is None or isinstance(v, bool):
        return None
    try:
        c = float(v)
    except (TypeError, ValueError):
        return None
    return c if math.isfinite(c) and c >= 0 else None


def _as_dict(obj) -> dict:
    if isinstance(obj, dict):
        return obj
    return obj.to_dict() if hasattr(obj, "to_dict") else obj.model_dump()


def _merge_details(acc: list, parts: list):
    """reasoning_details deltas: one entry per `index` (else the last entry of the same type); text fields join."""
    for p in parts or []:
        p = dict(p)
        same = [d for d in acc if "index" in p and d.get("index") == p["index"]] or \
               [d for d in acc[-1:] if "index" not in p and d.get("type") == p.get("type")]
        if not same:
            acc.append(p)
            continue
        d = same[0]
        for k, v in p.items():
            if v is None:
                continue
            d[k] = d.get(k, "") + v if k in ("text", "summary", "data") and isinstance(v, str) else v


def _merge_extra(dst: dict, src: dict) -> dict:
    """Stream deltas of fields the adapter does not know: an object merges key by key, any other value that is not
    None replaces the earlier one."""
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_extra(dst[k], v)
        elif v is not None:
            dst[k] = v
    return dst


def assemble(chunks) -> dict:
    """One chat.completion dict from a stream of chat.completion.chunk objects. A chunk with an `error` (OpenRouter
    sends one when the upstream fails mid-stream) gives a reply with that error and no choices."""
    out = {"id": "", "object": "chat.completion", "created": 0, "model": "", "usage": None}
    content, reasoning, details, calls, finish, error = [], [], [], {}, None, None
    extra_content = None
    for ch in chunks:
        ch = _as_dict(ch)
        for k in ("id", "created", "model"):
            out[k] = ch.get(k) or out[k]
        if ch.get("x_gateway"):
            out["x_gateway"] = ch["x_gateway"]
        if ch.get("usage"):
            out["usage"] = ch["usage"]
        if ch.get("error"):
            error = ch["error"]
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            content.append(d.get("content") or "")
            reasoning.append(d.get("reasoning") or "")
            _merge_details(details, d.get("reasoning_details"))
            for tc in d.get("tool_calls") or []:
                t = calls.setdefault(tc.get("index", len(calls)), {"id": None, "type": "function",
                                                                    "function": {"name": "", "arguments": ""}})
                t["id"] = tc.get("id") or t["id"]
                f = tc.get("function") or {}
                t["function"]["name"] += f.get("name") or ""
                t["function"]["arguments"] += f.get("arguments") or ""
                _merge_extra(t, {k: v for k, v in tc.items() if k not in ("index", "id", "function")})
                _merge_extra(t["function"], {k: v for k, v in f.items() if k not in ("name", "arguments")})
            if d.get("extra_content") is not None:
                extra_content = _merge_extra(extra_content or {}, d["extra_content"])
            finish = c.get("finish_reason") or finish
            if c.get("error"):
                error = c["error"]
    if error is None and finish == "error":
        error = {"message": "stream ended with finish_reason 'error'"}
    if error is not None:
        return {**out, "choices": None, "error": error}
    msg = {"role": "assistant", "content": "".join(content),
           "tool_calls": [calls[k] for k in sorted(calls)] or None}
    if "".join(reasoning):
        msg["reasoning"] = "".join(reasoning)
    if details:
        msg["reasoning_details"] = details
    if extra_content is not None:
        msg["extra_content"] = extra_content
    return {**out, "choices": [{"index": 0, "finish_reason": finish, "message": msg}]}


def openai_client(base_url, api_key, secret_headers=None):
    """The openai SDK client of every OpenAI-compatible adapter: REQUEST_TIMEOUT_S, no silent library retries.
    secret_headers (a registry entry's headers_env values) go in the client's default headers, so they are
    sent with every request but never reach the request log (params.extra_headers) or provenance."""
    import openai
    return openai.OpenAI(base_url=base_url, api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=0,
                         **({"default_headers": dict(secret_headers)} if secret_headers else {}))


class OpenRouterModel(ModelAdapter):
    PROVIDER = "openrouter"
    stream = STREAM
    headers = {}
    routing = None
    image_format = "png"

    def __init__(self, model_slug, image_window=None, max_tokens=DEFAULT_MAX_TOKENS, client=None, sleep=time.sleep, effort=None):
        if client is None:
            client = openai_client(BASE_URL, os.environ["OPENROUTER_API_KEY"])
        self.client = client
        self.model_id = self.model_version = model_slug
        self.image_window = image_window
        self.max_tokens = max_tokens
        self.effort = effort
        self.last_request = None
        self.cached = model_slug.startswith("anthropic/")
        self.sleep = sleep
        self.retries = []

    def _image_parts(self, turn, obs, label):
        """(parts, stripped): stripped when the window or the cap replaced an image of this observation with a
        placeholder."""
        kept = self.kept_images(turn, obs)
        parts = []
        for view, png, keep in zip(obs.frames, obs.images, kept):
            if keep:
                parts.append({"type": "text", "text": f"[{label}, step {obs.step}, {view}]"})
                parts.append({"type": "image_url", "image_url": {"url": encode_image(png, self.image_format)}})
            else:
                parts.append({"type": "text", "text": placeholder(obs, view)})
        return parts, not all(kept)

    @staticmethod
    def _mark(msg):
        """cache_control on the message's last content part (string content becomes one text part)."""
        if isinstance(msg["content"], str):
            msg["content"] = [{"type": "text", "text": msg["content"]}]
        msg["content"][-1]["cache_control"] = EPHEMERAL

    def render(self) -> dict:
        msgs = [{"role": "system", "content": "\n\n".join(self.system_blocks)}]
        stable = None
        for e, turn in zip(self.history, self.turn_index()):
            stripped = False
            if e[0] == "obs0":
                obs = e[2]
                parts, stripped = self._image_parts(turn, obs, "initial observation")
                msgs.append({"role": "user", "content": [{"type": "text", "text": e[1]}, {"type": "text", "text": obs.text}]
                             + parts})
            elif e[0] == "assistant":
                msgs.append(e[1])
            elif e[0] == "results":
                parts = []
                for call, obs in e[1]:
                    msgs.append({"role": "tool", "tool_call_id": call.id, "content": obs.text})
                    p, s = self._image_parts(turn, obs, f"{call.name} result")
                    parts += p
                    stripped |= s
                if parts:
                    msgs.append({"role": "user", "content": parts})
            elif e[0] == "text":
                if msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                    msgs[-1]["content"].append({"type": "text", "text": e[1]})
                else:
                    msgs.append({"role": "user", "content": e[1]})
            if stripped:
                stable = len(msgs) - 1
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                   "parameters": t.input_schema}} for t in self.tools]
        extra_body = {"usage": {"include": True}}
        if self.effort:
            extra_body["reasoning"] = {"effort": self.effort}
        if self.cached:
            self._mark(msgs[0])
            if stable is not None and stable != len(msgs) - 1:
                self._mark(msgs[stable])
            self._mark(msgs[-1])
        if self._provider():
            extra_body["provider"] = self._provider()
        return dict(model=self.model_id, messages=msgs, tools=tools, tool_choice="auto", max_tokens=self.max_tokens,
                    extra_body=extra_body)

    def request_config(self) -> dict:
        """Everything of a request except the messages and tools (provenance)."""
        extra_body = {"usage": {"include": True}, **({"reasoning": {"effort": self.effort}} if self.effort else {}),
                      **({"provider": self._provider()} if self._provider() else {})}
        return {"model": self.model_id, "tool_choice": "auto", "max_tokens": self.max_tokens,
                "extra_body": extra_body, "image_window": self.image_window, "timeout_s": REQUEST_TIMEOUT_S,
                "retry_waits_s": list(RETRY_WAITS_S), "stream": self.stream, **self.image_cap_config(),
                **({"image_format": self.image_format} if self.image_format != "png" else {})}

    def _provider(self) -> dict | None:
        """OpenRouter's `provider` object: Claude's fixed order (CLAUDE_PROVIDER), then the registry's routing rule
        for this model on top (its fields win)."""
        p = {**(CLAUDE_PROVIDER if self.cached else {}), **(self.routing or {})}
        return p or None

    def sdk_versions(self) -> dict:
        from chd_client.models.anthropic import _versions
        return _versions("openai")

    def _extra_headers(self) -> dict:
        """HTTP headers of this turn's request: the entry's static headers (none for OpenRouter); OpenAICompatibleModel
        adds the episode and request ids."""
        return dict(self.headers)

    def _request(self, req, headers=None):
        """One attempt: the reply (streamed when self.stream), recorded; an error reply raises ErrorReply."""
        from openai.types.chat import ChatCompletion
        self._chunks = 0
        headers = headers or {}
        if self.stream:
            r = self.client.chat.completions.create(**req, extra_headers=headers, stream=True,
                                                    stream_options={"include_usage": True})
            raw = _as_dict(r) if hasattr(r, "choices") else assemble(self._count(r))
        else:
            raw = _as_dict(self.client.chat.completions.create(**req, extra_headers=headers))
        self.record_response(raw)
        if raw.get("error") or not raw.get("choices"):
            raise ErrorReply(raw.get("error") or {"message": "reply without choices"})
        return ChatCompletion.model_validate(raw)

    def _count(self, chunks):
        for c in chunks:
            self._chunks += 1
            yield c

    def _log_failure(self, attempt, e, t0):
        """A failed attempt goes to responses.jsonl and stderr (the episode log): its error, how long it ran and how
        many stream chunks arrived first."""
        info = {"attempt": attempt, "type": type(e).__name__, "message": str(e)[:500],
                "elapsed_s": round(time.monotonic() - t0, 1), "chunks": self._chunks,
                **({"code": e.code} if isinstance(e, ErrorReply) else {})}
        if not isinstance(e, ErrorReply):
            self.record_response({"failed_attempt": info})
        print(f"model request attempt {attempt} failed after {info['elapsed_s']} s ({info['chunks']} chunks): "
              f"{info['type']}: {info['message'][:200]}", file=sys.stderr, flush=True)

    def _retryable(self, e) -> bool:
        """Whether a failed attempt is tried again (RETRY_WAITS_S): transient errors only (models.local.LocalModel
        also refuses the gateway's failed worker starts)."""
        return _transient(e)

    def _stored_call(self, tc, raw) -> dict:
        """The tool call as the history keeps it and sends it back: id, type and function only (OpenRouter;
        models.local.OpenAICompatibleModel keeps the vendor's extra fields too)."""
        return {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": raw}}

    def step(self) -> ModelTurn:
        req = self.render()
        self.last_request = req
        headers = self._extra_headers()
        self.record_request(provider=self.PROVIDER, system=req["messages"][0], tools=req["tools"],
                            messages=req["messages"][1:],
                            params={**{k: v for k, v in req.items() if k not in ("messages", "tools")},
                                    "extra_headers": headers})
        for attempt, wait in enumerate((*RETRY_WAITS_S, None)):
            t0 = time.monotonic()
            try:
                resp = self._request(req, headers)
                break
            except Exception as e:  # noqa: BLE001
                self._log_failure(attempt + 1, e, t0)
                if wait is None and _is_rate_limit(e):
                    raise RateLimited(f"rate limited after {attempt + 1} attempts: {e}") from e
                if wait is None or not self._retryable(e):
                    raise
                self.retries.append((attempt + 1, type(e).__name__ if not isinstance(e, ErrorReply)
                                     else f"ErrorReply {e.code}"))
                self.sleep(max(wait, _retry_after(e) or 0))
        return self.absorb(resp)

    def restore(self, raw: dict) -> ModelTurn:
        """Resume (chd_client.resume): a recorded reply (responses.jsonl "raw") goes into the history exactly as step()
        put it there when it arrived; nothing is sent."""
        from openai.types.chat import ChatCompletion
        if not isinstance(raw, dict):
            from chd_client.resume import ResumeError
            raise ResumeError("resume: a kept turn has no recorded reply in responses.jsonl")
        return self.absorb(ChatCompletion.model_validate(raw))

    def absorb(self, resp) -> ModelTurn:
        """One reply into the history and a ModelTurn (step() after a successful request; restore() on resume)."""
        choice = resp.choices[0]
        m = choice.message
        content = m.content or ""
        reasoning = _get(m, "reasoning")
        text = "\n\n".join(x for x in (reasoning, content) if x and x.strip())
        calls, stored = [], []
        for tc in m.tool_calls or []:
            raw = tc.function.arguments or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
            calls.append(ToolCall(tc.id, tc.function.name, args if isinstance(args, dict) else {"_raw": args}, None))
            stored.append(self._stored_call(tc, raw))
        for i, c in enumerate(calls):
            c.reasoning = (text or None) if i == 0 else None
        msg = {"role": "assistant", "content": content}
        if stored:
            msg["tool_calls"] = stored
        details = _get(m, "reasoning_details")
        if details:
            msg["reasoning_details"] = details
        self.history.append(("assistant", msg))
        u = resp.usage
        ptd = _get(u, "prompt_tokens_details")
        cached = _get(ptd, "cached_tokens", 0) or 0
        written = _get(ptd, "cache_write_tokens", 0) or 0
        cost = _get(u, "cost")
        usage = Usage(_get(u, "prompt_tokens", 0), _get(u, "completion_tokens", 0), cached, written,
                      _reported_cost(cost))
        stop = STOPS.get(choice.finish_reason, choice.finish_reason or "?")
        return ModelTurn(calls, content, stop, usage, resp, {"id": resp.id, "model": resp.model})
