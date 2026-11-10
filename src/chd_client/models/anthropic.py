"""Anthropic Messages API adapter (default model claude-opus-5-5), beta streaming endpoint.

Request: system = [client system prompt] (cache breakpoint on the last block), the environment's
tools verbatim, adaptive thinking with summarized display, effort in output_config, tool_choice auto (forced tool
choice is rejected by this model). The assistant content comes back verbatim (SDK blocks, thinking untouched).
By default every image is kept and the history is append-only, so earlier thinking stays valid. Only with an image
window or an image cap (which edit earlier turns) does the request set
block_binding.prefix_mismatch_behavior = drop_block. No fallback model is ever configured: a refusal
ends the episode.
"""

import base64
import importlib.metadata

from chd_client.models.base import ModelAdapter, ModelTurn, ToolCall, Usage, placeholder, DEFAULT_MAX_TOKENS

PRICES = {"claude-opus-5-5": {"in": 4.0, "out": 20.0, "cache_write": 5.0, "cache_read": 0.20},
          "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cache_write": 2.5, "cache_read": 0.10}}
PRICES_SOURCE = {"source": "https://platform.claude.com/docs/en/about-claude/pricing", "confirmed_on": "2026-11-01"}
BETAS = ["thinking-binding-controls-2026-08-01"]
EPHEMERAL = {"type": "ephemeral"}


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
    provider = "anthropic"
    prices = None

    def __init__(self, model="claude-opus-5-5", effort="medium", max_tokens=DEFAULT_MAX_TOKENS, image_window=None, client=None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client = client
        self.model_id = self.model_version = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.image_window = image_window
        self.last_request = None

    def request_config(self) -> dict:
        r = self.render_params()
        return {**r, "image_window": self.image_window, **self.image_cap_config()}

    def render_params(self) -> dict:
        """Everything of a request except system, tools and messages (provenance)."""
        thinking, extra = {"type": "adaptive", "display": "summarized"}, {}
        if self.image_window is not None or self.max_images is not None:
            thinking["block_binding"] = {"prefix_mismatch_behavior": "drop_block"}
            extra["betas"] = list(BETAS)
        return dict(model=self.model_id, max_tokens=self.max_tokens, thinking=thinking,
                    output_config={"effort": self.effort}, tool_choice={"type": "auto"}, **extra)

    def sdk_versions(self) -> dict:
        return _versions("anthropic")

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

    def step(self) -> ModelTurn:
        req = self.render()
        self.last_request = req
        self.record_request(provider=self.provider, system=req["system"], tools=req["tools"], messages=req["messages"],
                            params={k: v for k, v in req.items() if k not in ("system", "tools", "messages")})
        with self.client.beta.messages.stream(**req) as s:
            resp = s.get_final_message()
        self.record_response(resp)
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
