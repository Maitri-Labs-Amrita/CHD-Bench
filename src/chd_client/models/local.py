"""OpenAI-compatible adapters: any OpenAI-compatible URL (a registry entry of kind openai_compatible, models.registry)
and the local model gateway (--model local:<name>, LocalModel).

OpenAICompatibleModel sends the same requests as OpenRouterModel without OpenRouter's fields (usage accounting,
reasoning effort, provider routing, cache marks): extra_body is the entry's own (default empty), and --effort goes out
as the entry's effort parameter (`reasoning_effort`, OpenAI's top-level one) or not at all. Each request carries the
entry's static headers (a headers_env header is in the client's default headers instead, so its value is never
recorded; request_config names its variable) and, with episode_headers, three headers for a gateway's scheduler and
log: X-Episode-Id (the runner sets episode_id after reset; omitted before), X-Request-Id (one per turn, the same across
that turn's retries) and X-Turn-Index. The output cap goes out as max_tokens, or as the entry's max_tokens_param
(max_completion_tokens: OpenAI's reasoning models refuse max_tokens). Cost: "reported" takes usage.cost from the reply
(a reply without it stops the run, cost_unknown); "estimated" computes it from the usage tokens and the model's prices
(cached input at its own price when given, else at the input price) and records cost_source "estimated" in the turn's
transcript line; no price, or a reply without the prompt and completion token counts, gives cost_usd None (never 0.0:
the cost cap would never trip).
The URL (without credentials or query) is in request_config, so provenance names the endpoint, not only the model.
A tool call keeps every field the server sent, and an assistant message its extra_content, and both go back
unchanged in the next request: Gemini's OpenAI-compatible endpoint puts a thought signature in
tool_calls[].extra_content.google.thought_signature and refuses (HTTP 400) a later request that does not return it
(https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures). OpenRouterModel keeps only id, type and
function.

LocalModel is the local model gateway (chd-gateway, chd_client.gateway) at $CHD_LOCAL_BASE_URL: episode headers, no
effort, not streamed. The gateway reports usage.cost (0.0 for a local model) and usage.cost_source
("zero_api_charge"). Its x_gateway (model revision, engine, interventions applied) and cost_source go into the turn's
transcript line; a reply without a model revision is marked there (gateway_revision_missing) and warned about once.
A gateway reply with error code worker_start_failed (its model worker cannot start, e.g. a package is missing) is not
retried: LocalModel raises GatewayWorkerStartError with the gateway's message (it names the worker log), and the
runner ends the episode with stop reason runner_exception. Retrying would start the same broken worker again.

Before the episode (resolve_limits, one GET /v1/models, no model call) it reads the profile's capabilities from the
gateway (x_gateway.capabilities of the model's entry):
- max_images: it caps the images of each request to it (models.base) when --max-images was not given; a --max-images
  above it is refused.
- request_timeout_s (x_gateway, not a capability): how long the gateway waits for one turn. Above the client's own
  timeout (REQUEST_TIMEOUT_S), the client waits that + TIMEOUT_MARGIN_S (request_config timeout_s): a long thinking
  turn or a worker restart after a spot preemption is not cut by the client and resent.
- max_output_tokens (the model creator's maximum output tokens per turn): it is the request's max_tokens when --max-tokens was not given (request_config then records
  max_tokens_source "gateway"); a --max-tokens above it is refused. A --max-tokens at or below it is sent as given.
A gateway that reports no cap, or cannot be asked, leaves the requests unchanged (byte-identical).
vLLM lowers max_tokens itself when the prompt plus max_tokens is longer than its --max-model-len (vLLM 0.32.0
entrypoints/serve/utils/api_utils.py get_max_tokens: min(max-model-len minus the prompt, max_tokens)). The reply then
says nothing about it; a turn that ends with finish_reason length (stop "max_tokens") after fewer output tokens than
max_tokens gets max_tokens_lowered {requested, output_tokens} in its transcript line.
"""

import os
import sys
import time
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from chd_client.models.base import ModelTurn, DEFAULT_MAX_TOKENS
from chd_client.models.openrouter import REQUEST_TIMEOUT_S, OpenRouterModel, _as_dict, _get, openai_client

NO_KEY = "local"

TIMEOUT_MARGIN_S = 300.0
START_FAILED = "worker_start_failed"

BASE_URL_ENV = "CHD_LOCAL_BASE_URL"
API_KEY_ENV = "CHD_LOCAL_API_KEY"
MODELS_TIMEOUT_S = 30.0


def public_url(url) -> str:
    """The URL without credentials, query or fragment (for provenance)."""
    u = urlsplit(str(url))
    host = u.hostname or ""
    return urlunsplit((u.scheme, host + (f":{u.port}" if u.port else ""), u.path, "", ""))


class OpenAICompatibleModel(OpenRouterModel):
    PROVIDER = "openai_compatible"
    stream = False

    def __init__(self, name, image_window=None, max_tokens=DEFAULT_MAX_TOKENS, client=None, base_url=None, sleep=time.sleep, *,
                 provider=None, api_key=None, effort=None, effort_param=None, extra_body=None, headers=None,
                 headers_env=None, episode_headers=False, x_gateway=False, cost=None, stream=None,
                 max_tokens_param="max_tokens"):
        self.headers_env = dict(headers_env or {})
        if client is None:
            if not base_url:
                raise RuntimeError(f"--model {provider or self.PROVIDER}:{name} needs a base URL")
            missing = [v for v in self.headers_env.values() if not os.environ.get(v)]
            if missing:
                raise RuntimeError(f"--model {provider or self.PROVIDER}:{name} needs ${', $'.join(missing)}")
            client = openai_client(base_url, NO_KEY if api_key is None else api_key,
                                   {h: os.environ[v] for h, v in self.headers_env.items()})
        super().__init__(name, image_window=image_window, max_tokens=max_tokens, client=client, sleep=sleep,
                         effort=effort if effort_param else None)
        if provider is not None:
            self.PROVIDER = provider
        if stream is not None:
            self.stream = stream
        self.effort_param = effort_param
        self.max_tokens_param = max_tokens_param
        self.extra_body = dict(extra_body or {})
        self.headers = dict(headers or {})
        self.episode_headers = episode_headers
        self.x_gateway = x_gateway
        self.cost = dict(cost or {"source": "reported"})
        self.price = (self.cost.get("prices") or {}).get(name) if self.cost["source"] == "estimated" else None
        self.base_url = public_url(base_url or getattr(client, "base_url", "") or "")
        self.cached = False
        self.turn = 0
        self._warned = False

    def start(self, *a, **k):
        self.turn = 0
        return super().start(*a, **k)

    def render(self) -> dict:
        req = super().render()
        req["extra_body"] = dict(self.extra_body)
        if self.max_tokens_param != "max_tokens":
            req[self.max_tokens_param] = req.pop("max_tokens")
        if self.effort_param and self.effort:
            req[self.effort_param] = self.effort
        return req

    def request_config(self) -> dict:
        out = {**super().request_config(), "extra_body": dict(self.extra_body), "base_url": self.base_url}
        if self.effort_param:
            out[self.effort_param] = self.effort
        if self.max_tokens_param != "max_tokens":
            out["max_tokens_param"] = self.max_tokens_param
        if self.headers:
            out["headers"] = dict(self.headers)
        if self.headers_env:
            out["headers_env"] = dict(self.headers_env)
        return out

    def _extra_headers(self) -> dict:
        h = super()._extra_headers()
        if self.episode_headers:
            self.turn += 1
            h.update({**({"X-Episode-Id": self.episode_id} if self.episode_id is not None else {}),
                      "X-Request-Id": uuid4().hex, "X-Turn-Index": str(self.turn)})
        return h

    def _stored_call(self, tc, raw) -> dict:
        """id, type and function as OpenRouterModel stores them, plus every other field the server sent with the call
        (Gemini: extra_content.google.thought_signature), so the next request returns them unchanged."""
        out = super()._stored_call(tc, raw)
        out["function"].update({k: v for k, v in (_as_dict(tc.function) or {}).items()
                                if k not in ("name", "arguments") and v is not None})
        out.update({k: v for k, v in (_as_dict(tc) or {}).items()
                    if k not in ("id", "type", "function") and v is not None})
        return out

    def absorb(self, resp) -> ModelTurn:
        turn = super().absorb(resp)
        extra = _get(turn.raw.choices[0].message, "extra_content")
        if extra is not None:
            self.history[-1][1]["extra_content"] = extra
        if self.x_gateway:
            gw, src = _get(turn.raw, "x_gateway"), _get(turn.raw.usage, "cost_source")
            if gw is not None:
                turn.extra["x_gateway"] = gw
            if not (isinstance(gw, dict) and gw.get("revision")):
                turn.extra["gateway_revision_missing"] = True
                if not self._warned:
                    self._warned = True
                    print(f"warning: the gateway at {self.base_url} sent no model revision (x_gateway.revision); the "
                          f"run names the model only as {self.model_id!r}", file=sys.stderr, flush=True)
            if src is not None:
                turn.extra["cost_source"] = src
        if self.cost["source"] == "estimated":
            u = turn.raw.usage
            counted = u is not None and all(_get(u, k) is not None for k in ("prompt_tokens", "completion_tokens"))
            turn.usage.cost_usd = estimated_cost(self.price, turn.usage) if counted else None
            turn.extra["cost_source"] = "estimated"
        return turn


def estimated_cost(price: dict | None, usage) -> float | None:
    """USD from the usage tokens and one model's prices (per million tokens); None without a price (never invented).
    usage.input counts cached tokens too; they are priced at cached_input_per_mtok when given, else as input."""
    if price is None:
        return None
    cached = min(usage.cache_read, usage.input)
    cached_rate = price.get("cached_input_per_mtok", price["input_per_mtok"])
    return ((usage.input - cached) * price["input_per_mtok"] + cached * cached_rate
            + usage.output * price["output_per_mtok"]) / 1e6


class GatewayWorkerStartError(RuntimeError):
    """The local model gateway cannot start a model worker for this model (not transient: never retried)."""


def _start_failure(e) -> str | None:
    """The gateway's message when e is its worker_start_failed error reply, else None."""
    if getattr(e, "code", None) != START_FAILED:
        return None
    body = getattr(e, "body", None)
    msg = body.get("message") if isinstance(body, dict) else None
    return str(msg or e)


class LocalModel(OpenAICompatibleModel):
    """The local model gateway (registry entries with x_gateway: true; the built-in "local")."""
    PROVIDER = "local"
    stream = False

    def __init__(self, name, image_window=None, max_tokens=DEFAULT_MAX_TOKENS, client=None, base_url=None, sleep=time.sleep, **kw):
        if client is None:
            base_url = base_url or os.environ.get(BASE_URL_ENV)
            if not base_url:
                raise RuntimeError(f"--model local:{name} needs ${BASE_URL_ENV}: the local model gateway's URL "
                                   "(e.g. http://127.0.0.1:8098/v1, chd-gateway's default port)")
            kw.setdefault("api_key", os.environ.get(API_KEY_ENV, NO_KEY))
        kw = {"episode_headers": True, "x_gateway": True, "cost": {"source": "reported"}, **kw}
        super().__init__(name, image_window=image_window, max_tokens=max_tokens, client=client, base_url=base_url,
                         sleep=sleep, **kw)
        self._limits_resolved = False
        self.max_tokens_option: int | None = None
        self.entry_output_limit: int | None = None
        self.max_tokens_source: str | None = None
        self.timeout_s = REQUEST_TIMEOUT_S

    def request_config(self) -> dict:
        out = super().request_config()
        out["timeout_s"] = self.timeout_s
        if self.max_tokens_source is not None:
            out["max_tokens_source"] = self.max_tokens_source
        return out

    def _retryable(self, e) -> bool:
        return _start_failure(e) is None and super()._retryable(e)

    def step(self) -> ModelTurn:
        try:
            turn = super().step()
        except Exception as e:
            msg = _start_failure(e)
            if msg is None:
                raise
            raise GatewayWorkerStartError(f"the local model gateway at {self.base_url} cannot start a worker for "
                                          f"{self.model_id!r}: {msg}") from e
        if turn.stop == "max_tokens" and self.max_tokens and turn.usage.output < self.max_tokens:
            turn.extra["max_tokens_lowered"] = {"requested": self.max_tokens, "output_tokens": turn.usage.output}
        return turn

    def resolve_limits(self):
        """Once: the gateway's caps for this model (x_gateway.capabilities in GET /v1/models), before any model call.
        max_images: no --max-images, the cap applies (max_images_source "gateway"); a --max-images above it is a
        ProviderError. max_output_tokens: no --max-tokens, it is max_tokens (max_tokens_source "gateway"; a smaller
        max_output_tokens of the provider entry wins); a --max-tokens above it is a ProviderError. No cap reported,
        a gateway without capabilities, or no answer (a warning): nothing changes."""
        if self._limits_resolved:
            return
        self._limits_resolved = True
        caps = self._gateway_capabilities()
        images, output = (_whole(caps.get(k)) for k in ("max_images", "max_output_tokens"))
        gw_timeout = caps.get("_request_timeout_s")
        if isinstance(gw_timeout, (int, float)) and not isinstance(gw_timeout, bool) \
                and gw_timeout + TIMEOUT_MARGIN_S > self.timeout_s and hasattr(self.client, "with_options"):
            self.timeout_s = float(gw_timeout) + TIMEOUT_MARGIN_S
            self.client = self.client.with_options(timeout=self.timeout_s)
        from chd_client.models.registry import ProviderError
        if images is not None:
            if self.max_images is None:
                self.max_images, self.max_images_source = images, "gateway"
            elif self.max_images > images:
                raise ProviderError(f"--max-images {self.max_images} is above the gateway's cap for "
                                    f"{self.model_id!r}: {images} images per request (capabilities.max_images). Pass "
                                    f"--max-images {images} or less, or leave it out to use the gateway's cap.")
        if output is not None:
            if self.max_tokens_option is None:
                if self.entry_output_limit is None or self.entry_output_limit >= output:
                    self.max_tokens, self.max_tokens_source = output, "gateway"
            elif self.max_tokens_option > output:
                raise ProviderError(f"--max-tokens {self.max_tokens_option} is above the gateway's output cap for "
                                    f"{self.model_id!r}: {output} tokens per turn (capabilities.max_output_tokens, "
                                    f"the model creator's maximum). Pass --max-tokens {output} or less, or leave it "
                                    "out to use the gateway's cap.")

    def _gateway_capabilities(self) -> dict:
        """x_gateway.capabilities of this model's entry in GET /v1/models; {} when there is none."""
        if getattr(self.client, "models", None) is None:
            return {}
        client = self.client.with_options(timeout=MODELS_TIMEOUT_S) if hasattr(self.client, "with_options") \
            else self.client
        try:
            data = getattr(client.models.list(), "data", None) or []
        except Exception as e:  # noqa: BLE001
            print(f"warning: GET {self.base_url}/models failed ({type(e).__name__}: {str(e)[:200]}): no image cap "
                  "and no output cap from the gateway; requests keep every image the window keeps and max_tokens "
                  f"{self.max_tokens}", file=sys.stderr, flush=True)
            return {}
        for item in data:
            d = _as_dict(item) if not isinstance(item, dict) else item
            if d.get("id") == self.model_id:
                xg = d.get("x_gateway") or {}
                caps = dict(xg.get("capabilities") or {}) if isinstance(xg.get("capabilities"), dict) else {}
                caps["_request_timeout_s"] = xg.get("request_timeout_s")
                return caps
        return {}


def _whole(n) -> int | None:
    """A cap from the gateway: a whole number of at least 1, else None (no cap)."""
    return n if isinstance(n, int) and not isinstance(n, bool) and n >= 1 else None
