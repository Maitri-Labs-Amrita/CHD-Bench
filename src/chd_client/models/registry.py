"""The model providers: --model <provider>:<model> resolved through config/providers.json (configuration, not code).

    {"_comment": "...", "providers": {"<name>": {"kind": "openai_compatible", "base_url": "https://.../v1",
                                                  "api_key_env": "VENDOR_API_KEY", "cost": {"source": "reported"}}}}

The file is `path` (run-episode --providers), else $CHD_PROVIDERS, else config/providers.json. Its sibling
<stem>.local.json (config/providers.local.json, gitignored), when it exists, replaces entries of the same name and
adds new ones (a whole entry each, no field merge). The model name splits on its FIRST ':' only (a local model such as
"qwen2.5vl:7b" has its own ':'); a name without ':' is an Anthropic model (provider "anthropic").
An Anthropic model id that itself contains ':' is written anthropic:<id>. An unknown provider, an unknown
key or a wrong type is an error that names the file (never the value: it may be a pasted key).

kind is the adapter: anthropic (AnthropicModel), openrouter (OpenRouterModel: usage accounting, reasoning effort,
Claude provider routing and cache marks) or openai_compatible (models.local.OpenAICompatibleModel; with x_gateway
true, the local model gateway's LocalModel). Keys are environment variable names (api_key_env; headers_env for a
key that goes in a header: {"api-key": "AZURE_OPENAI_KEY"}); a key value is never read into the registry, provenance
or a log, and a base_url with credentials or a query is refused. cost: "reported" (the reply carries usage.cost:
kind openrouter, or an x_gateway entry such as chd-gateway) or "estimated" from per-model prices (USD per million
tokens). A model without a price is refused before any request: the runner would stop on its first turn anyway
(cost_unknown), so the price is never invented. Kind anthropic is priced by models.anthropic.PRICES, or by its own
cost.prices (required with a base_url: a proxy's prices are not Anthropic's list prices). An openai_compatible entry
may set max_output_tokens ({"<model>": n}, the model's output limit from the vendor's model page): without
--max-tokens the request asks for the smaller of the adapter's default (32000) and n; a larger --max-tokens is refused
before the run.

The built-in "anthropic" entry (api_key_env ANTHROPIC_API_KEY, no base_url, not optional) builds anthropic.Anthropic()
with no arguments: the SDK resolves the credentials and $ANTHROPIC_BASE_URL itself. The SDK
checks credentials only at its first request, so make_adapter refuses a client that resolved none (sdk_has_credentials),
before any case or Slicer work; the request does not change. One check serves run-episode, ./run.sh (both through
make_adapter) and slurm/run_batch.py (missing_env): sdk_client builds that client with the environment they check, so
all three accept what the SDK accepts (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, an SDK profile, ...). Every other
anthropic entry passes its key explicitly (a placeholder when it has none), so the SDK never adds $ANTHROPIC_API_KEY or
$ANTHROPIC_AUTH_TOKEN to a request for another host.
"""

import contextlib
import hashlib
import inspect
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[3]
DEFAULT_FILE = REPO / "config" / "providers.json"
PROVIDERS_ENV = "CHD_PROVIDERS"
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-5-5"
KEY_HINT = "export it, or put it in .env and use ./run.sh"
NO_KEY = "local"
KINDS = ("anthropic", "openrouter", "openai_compatible")
FIELDS = {"_comment", "kind", "base_url", "base_url_env", "api_key_env", "api_key_optional", "headers",
          "headers_env", "episode_headers", "stream", "extra_body", "effort", "cost", "x_gateway", "max_tokens_param",
          "max_output_tokens", "routing", "image_format"}
KIND_FIELDS = {
    "anthropic": {"base_url", "base_url_env", "api_key_env", "api_key_optional", "cost"},
    "openrouter": {"base_url", "base_url_env", "api_key_env", "api_key_optional", "headers", "headers_env", "stream",
                   "cost", "routing", "image_format"},
    "openai_compatible": {"base_url", "base_url_env", "api_key_env", "api_key_optional", "headers", "headers_env",
                          "episode_headers", "stream", "extra_body", "effort", "cost", "x_gateway",
                          "max_tokens_param", "max_output_tokens"},
}
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
AUTH_HEADERS = {"authorization", "api-key", "x-api-key", "cookie", "proxy-authorization"}
EFFORT_PARAMS = (None, "reasoning_effort")
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens")
COST_SOURCES = ("reported", "estimated")
ROUTING_FIELDS = {"order": list, "only": list, "ignore": list, "quantizations": list, "allow_fallbacks": bool,
                  "require_parameters": bool, "zdr": bool, "enforce_distillable_text": bool, "data_collection": str,
                  "sort": (str, dict), "max_price": dict, "preferred_min_throughput": (int, float, dict),
                  "preferred_max_latency": (int, float, dict)}
DATA_COLLECTION = ("allow", "deny")
PRICE_FIELDS = {"input_per_mtok", "output_per_mtok", "cached_input_per_mtok"}
ANTHROPIC_PRICE_FIELDS = PRICE_FIELDS | {"cache_write_per_mtok"}


class ProviderError(RuntimeError):
    pass


@dataclass
class Registry:
    providers: dict
    file: str
    sha256: str
    overlay: str | None = None
    overlay_sha256: str | None = None
    sources: dict = field(default_factory=dict)

    def entry(self, name: str) -> dict:
        if name not in self.providers:
            raise ProviderError(f"unknown provider {name!r}; known providers: {', '.join(sorted(self.providers))} "
                                f"({self.file})")
        return self.providers[name]

    def provenance(self) -> dict:
        return {"file": self.file, "sha256": self.sha256, "overlay": self.overlay,
                "overlay_sha256": self.overlay_sha256}


def registry_path(path=None, environ=None) -> Path:
    """`path`, else $CHD_PROVIDERS, else config/providers.json."""
    environ = os.environ if environ is None else environ
    return Path(path or environ.get(PROVIDERS_ENV) or DEFAULT_FILE).expanduser()


def overlay_path(base) -> Path:
    base = Path(base)
    return base.with_name(base.stem + ".local" + base.suffix)


def _read(path) -> tuple[dict, str]:
    try:
        data = Path(path).read_bytes()
    except OSError as e:
        raise ProviderError(f"providers file {path}: {e.strerror or e}") from None
    try:
        doc = json.loads(data, parse_constant=_no_constant)
    except ValueError as e:
        raise ProviderError(f"providers file {path}: not JSON ({e})") from None
    if not isinstance(doc, dict):
        raise ProviderError(f"providers file {path}: not a JSON object")
    unknown = set(doc) - {"_comment", "providers"}
    if unknown:
        raise ProviderError(f"providers file {path}: unknown keys {sorted(unknown)} (known: _comment, providers)")
    if not isinstance(doc.get("providers"), dict):
        raise ProviderError(f"providers file {path}: needs a `providers` object")
    return doc["providers"], hashlib.sha256(data).hexdigest()


def _no_constant(name):
    raise ValueError(f"{name} is not a JSON number")


def load_registry(path=None, environ=None) -> Registry:
    """The registry of `path` (registry_path) with its .local.json overlay, every entry checked."""
    base = registry_path(path, environ)
    providers, sha = _read(base)
    sources = {n: str(base) for n in providers}
    reg = Registry(dict(providers), str(base), sha)
    over = overlay_path(base)
    if over.exists():
        extra, reg.overlay_sha256 = _read(over)
        reg.overlay = str(over)
        reg.providers.update(extra)
        sources.update({n: str(over) for n in extra})
    if not reg.providers:
        raise ProviderError(f"providers file {base}: no providers")
    for name, e in reg.providers.items():
        check_entry(name, e, sources[name])
    reg.sources = sources
    return reg


def _json_type(v) -> str:
    """The JSON type of `v`, for an error message (the value itself is never shown: it may be a pasted key)."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true/false"
    for t, what in ((str, "a string"), ((int, float), "a number"), (list, "an array"), (dict, "an object")):
        if isinstance(v, t):
            return what
    return type(v).__name__


def _type(where, key, v, types, what):
    if not isinstance(v, types):
        raise ProviderError(f"{where}: {key} must be {what}, not {_json_type(v)}")


def _env_name(where, key, v):
    _type(where, key, v, str, "an environment variable name")
    if not ENV_NAME.fullmatch(v):
        raise ProviderError(f"{where}: {key} must be an environment variable name (A-Z, 0-9 and _, not starting with "
                            "a digit; the value is not shown here)")


def _check_url(where, key, url):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.netloc:
        raise ProviderError(f"{where}: {key} must be an http:// or https:// URL")
    if "@" in u.netloc or u.query or u.fragment:
        raise ProviderError(f"{where}: {key} must not carry credentials (user:password@), a query (?...) or a "
                            "fragment: they would be recorded. A key goes in api_key_env or headers_env.")


def check_entry(name, e, file="?"):
    """Unknown keys, keys of another kind and wrong types are errors."""
    where = f"provider {name!r} ({file})"
    if not isinstance(name, str) or not name or ":" in name:
        raise ProviderError(f"{where}: a provider name is a non-empty string without ':'")
    if not isinstance(e, dict):
        raise ProviderError(f"{where}: an entry is a JSON object")
    unknown = set(e) - FIELDS
    if unknown:
        raise ProviderError(f"{where}: unknown keys {sorted(unknown)} (known: {', '.join(sorted(FIELDS))})")
    kind = e.get("kind")
    if kind not in KINDS:
        raise ProviderError(f"{where}: kind must be one of {', '.join(KINDS)}")
    other = set(e) - KIND_FIELDS[kind] - {"_comment", "kind"}
    if other:
        raise ProviderError(f"{where}: {sorted(other)} do not apply to kind {kind}")
    if "api_key_env" not in e:
        raise ProviderError(f"{where}: needs api_key_env (an environment variable name, or null for no key)")
    if e["api_key_env"] is not None:
        _env_name(where, "api_key_env", e["api_key_env"])
    if "base_url_env" in e:
        _env_name(where, "base_url_env", e["base_url_env"])
    if "base_url" in e:
        _type(where, "base_url", e["base_url"], str, "a string")
        _check_url(where, "base_url", e["base_url"])
    if kind != "anthropic" and not (e.get("base_url") or e.get("base_url_env")):
        raise ProviderError(f"{where}: needs base_url or base_url_env")
    for k in ("api_key_optional", "episode_headers", "stream", "x_gateway"):
        if k in e:
            _type(where, k, e[k], bool, "true or false")
    if "headers" in e:
        _type(where, "headers", e["headers"], dict, "an object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in e["headers"].items()):
            raise ProviderError(f"{where}: headers are string names and string values")
        auth = sorted(k for k in e["headers"] if k.lower() in AUTH_HEADERS)
        if auth:
            raise ProviderError(f"{where}: headers {auth} carry credentials, and header values are recorded. Put "
                                "the name of the variable that holds the value in headers_env instead: "
                                "{\"<Header>\": \"<VARIABLE>\"}")
    if "headers_env" in e:
        _type(where, "headers_env", e["headers_env"], dict, "an object")
        for k, v in e["headers_env"].items():
            if not isinstance(k, str) or not k:
                raise ProviderError(f"{where}: headers_env names are non-empty strings")
            _env_name(where, f"headers_env[{k!r}]", v)
        both = sorted({k.lower() for k in e.get("headers") or {}} & {k.lower() for k in e["headers_env"]})
        if both:
            raise ProviderError(f"{where}: {both} are in both headers and headers_env")
    if "extra_body" in e:
        _type(where, "extra_body", e["extra_body"], dict, "an object")
    if "effort" in e and e["effort"] not in EFFORT_PARAMS:
        raise ProviderError(f"{where}: effort must be null or \"reasoning_effort\"")
    if "max_tokens_param" in e and e["max_tokens_param"] not in MAX_TOKENS_PARAMS:
        raise ProviderError(f"{where}: max_tokens_param must be \"max_tokens\" or \"max_completion_tokens\"")
    if "max_output_tokens" in e:
        _type(where, "max_output_tokens", e["max_output_tokens"], dict, "an object ({\"<model>\": <tokens>})")
        for model, n in e["max_output_tokens"].items():
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ProviderError(f"{where}: max_output_tokens[{model!r}] must be a whole number >= 1")
    if "routing" in e:
        check_routing(where, e["routing"])
    if "image_format" in e:
        from chd_client.models.openrouter import IMAGE_FORMATS
        if e["image_format"] not in IMAGE_FORMATS:
            raise ProviderError(f"{where}: image_format must be one of {', '.join(IMAGE_FORMATS)}")
    if kind != "anthropic" or "cost" in e or e.get("base_url") or e.get("base_url_env"):
        check_cost(where, kind, e.get("cost"), e)


def check_routing(where, routing):
    """routing: {"default": {...}, "<vendor>/": {...}, "<vendor>/<model>": {...}}; each value an OpenRouter
    `provider` object (ROUTING_FIELDS)."""
    _type(where, "routing", routing, dict, "an object ({\"default\" | \"<vendor>/\" | \"<model>\": {...}})")
    for key, rule in routing.items():
        if key == "_comment":
            continue
        at = f"{where}: routing[{key!r}]"
        if not isinstance(key, str) or not (key == "default" or "/" in key):
            raise ProviderError(f"{at}: a key is \"default\", a vendor prefix (\"google/\") or a model slug "
                                "(\"google/gemini-4-argon\")")
        if not isinstance(rule, dict):
            raise ProviderError(f"{at}: must be an object (OpenRouter's request `provider` fields)")
        unknown = sorted(set(rule) - set(ROUTING_FIELDS) - {"_comment"})
        if unknown:
            raise ProviderError(f"{at}: unknown fields {unknown} (known: {', '.join(sorted(ROUTING_FIELDS))})")
        for k, v in rule.items():
            if k == "_comment":
                continue
            t = ROUTING_FIELDS[k]
            types = t if isinstance(t, tuple) else (t,)
            if (isinstance(v, bool) and bool not in types) or not isinstance(v, types):
                raise ProviderError(f"{at}: {k} has the wrong type")
            if t is list and not all(isinstance(x, str) and x for x in v):
                raise ProviderError(f"{at}: {k} is a list of provider slugs (strings)")
        if "data_collection" in rule and rule["data_collection"] not in DATA_COLLECTION:
            raise ProviderError(f"{at}: data_collection must be \"allow\" or \"deny\"")


def routing_for(e, model) -> dict | None:
    """The `provider` object for `model` from the entry's routing: the model's own rule, else its vendor's
    ("<vendor>/": a key that ends with '/'), else "default", else None. The rule that matches is used whole (no merge with the others),
    without its _comment."""
    routing = e.get("routing") or {}
    rule = routing.get(model)
    if rule is None:
        prefixes = sorted((k for k in routing if k.endswith("/") and model.startswith(k)), key=len, reverse=True)
        rule = routing[prefixes[0]] if prefixes else routing.get("default")
    return None if rule is None else {k: v for k, v in rule.items() if k != "_comment"}


def check_cost(where, kind, cost, entry=None):
    """reported: kind openrouter or an x_gateway entry only (no other vendor sends usage.cost); estimated: prices, each
    a finite number >= 0. Kind anthropic: estimated only, with all four rates of models.anthropic.PRICES."""
    if not isinstance(cost, dict):
        if kind == "anthropic":
            raise ProviderError(f"{where}: kind anthropic with a base_url needs cost: {{\"source\": \"estimated\", "
                                "\"prices\": {...}} (the host's prices, not Anthropic's list prices)")
        raise ProviderError(f"{where}: needs cost: {{\"source\": \"reported\"}} or {{\"source\": \"estimated\", "
                            "\"prices\": {...}}")
    unknown = set(cost) - {"_comment", "source", "prices"}
    if unknown:
        raise ProviderError(f"{where}: cost: unknown keys {sorted(unknown)} (known: _comment, source, prices)")
    src = cost.get("source")
    if src not in COST_SOURCES:
        raise ProviderError(f"{where}: cost.source must be \"reported\" or \"estimated\"")
    if kind == "openrouter" and src != "reported":
        raise ProviderError(f"{where}: kind openrouter reports its cost (cost.source \"reported\")")
    if src == "reported" and not (kind == "openrouter" or (entry or {}).get("x_gateway")):
        raise ProviderError(f"{where}: cost.source \"reported\" is for kind openrouter and x_gateway entries only "
                            "(other vendors send no usage.cost); use {\"source\": \"estimated\", \"prices\": {...}}")
    if src == "reported":
        if "prices" in cost:
            raise ProviderError(f"{where}: cost.prices applies to cost.source \"estimated\" only")
        return
    prices = cost.get("prices")
    if not isinstance(prices, dict):
        raise ProviderError(f"{where}: cost.source \"estimated\" needs cost.prices (an object, may be empty)")
    fields = ANTHROPIC_PRICE_FIELDS if kind == "anthropic" else PRICE_FIELDS
    required = fields if kind == "anthropic" else PRICE_FIELDS - {"cached_input_per_mtok"}
    for model, p in prices.items():
        if not isinstance(p, dict):
            raise ProviderError(f"{where}: cost.prices[{model!r}] must be an object")
        unknown = set(p) - fields
        if unknown:
            raise ProviderError(f"{where}: cost.prices[{model!r}]: unknown keys {sorted(unknown)} "
                                f"(known: {', '.join(sorted(fields))})")
        for k in sorted(fields):
            if k not in p and k in required:
                raise ProviderError(f"{where}: cost.prices[{model!r}] needs {k}")
            if k in p and (isinstance(p[k], bool) or not isinstance(p[k], (int, float)) or not math.isfinite(p[k])
                           or p[k] < 0):
                raise ProviderError(f"{where}: cost.prices[{model!r}].{k} must be a finite number >= 0")


def split_model(name: str, registry: Registry) -> tuple[str, str]:
    """(provider, model): split on the first ':'; no ':' -> the Anthropic provider with the whole name."""
    if ":" not in name:
        if DEFAULT_PROVIDER not in registry.providers:
            raise ProviderError(f"--model {name}: a name without '<provider>:' uses provider {DEFAULT_PROVIDER!r}, "
                                f"which {registry.file} does not define; known providers: "
                                f"{', '.join(sorted(registry.providers))}")
        return DEFAULT_PROVIDER, name
    provider, model = name.split(":", 1)
    if provider not in registry.providers:
        raise ProviderError(f"--model {name}: unknown provider {provider!r}; known providers: "
                            f"{', '.join(sorted(registry.providers))} ({registry.file}). The name splits on its first "
                            f"':'; an Anthropic model id that contains ':' is written {DEFAULT_PROVIDER}:<id> "
                            f"(--model {DEFAULT_PROVIDER}:{name})")
    if not model:
        raise ProviderError(f"--model {name}: no model name after '{provider}:'")
    return provider, model


def sdk_default(entry: dict) -> bool:
    """The built-in Anthropic entry: kind anthropic, api_key_env ANTHROPIC_API_KEY, no base URL, key not optional.
    Its client is anthropic.Anthropic() with no arguments (the SDK's own credential lookup)."""
    return (entry.get("kind") == "anthropic" and entry.get("api_key_env") == "ANTHROPIC_API_KEY"
            and not entry.get("base_url") and not entry.get("base_url_env") and not entry.get("api_key_optional"))


def sdk_has_credentials(client) -> bool:
    """Whether the built-in entry's anthropic.Anthropic() resolved any credential (it reads them when it is built, but
    only checks them at the first request). True for what the SDK accepts there (anthropic 1.9 _client.py,
    _validate_headers): a credentials provider (ANTHROPIC_PROFILE, workload identity, a profile on disk), or an
    X-Api-Key or Authorization header ($ANTHROPIC_API_KEY, $ANTHROPIC_AUTH_TOKEN, $ANTHROPIC_CUSTOM_HEADERS). It reads
    the client only; nothing it sends changes."""
    if getattr(client, "credentials", None) is not None:
        return True
    headers = getattr(client, "default_headers", None) or {}
    return any(isinstance(v, str) and v and k.lower() in ("x-api-key", "authorization") for k, v in headers.items())


@contextlib.contextmanager
def as_environ(environ):
    """os.environ is `environ` while the block runs (the anthropic SDK reads its credentials, and ~ for its profiles,
    from os.environ only); nothing changes when `environ` is os.environ."""
    if environ is None or environ is os.environ:
        yield
        return
    saved = dict(os.environ)
    os.environ.clear()
    os.environ.update(environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def sdk_client(environ=None):
    """The built-in entry's client: anthropic.Anthropic() with no arguments, built with `environ` (default os.environ)
    as the environment. No request is sent. A profile the SDK cannot load (it raises at construction) is a
    ProviderError."""
    import anthropic
    with as_environ(environ):
        try:
            return anthropic.Anthropic()
        except anthropic.AnthropicError as e:
            raise ProviderError(f"the anthropic SDK could not load its credentials: {e}") from None


def base_url(entry: dict, provider: str, model: str, environ=None, label=None) -> str | None:
    """base_url_env when set and present (it wins), else base_url; None (kind anthropic: the SDK's) when neither.
    label: the --model name as typed, for the error (default <provider>:<model>)."""
    environ = os.environ if environ is None else environ
    var = entry.get("base_url_env")
    if var and environ.get(var):
        return environ[var]
    if entry.get("base_url"):
        return entry["base_url"]
    if var:
        raise ProviderError(f"--model {label or f'{provider}:{model}'} needs ${var}: the base URL of provider "
                            f"{provider!r} (e.g. http://127.0.0.1:8098/v1, chd-gateway's default port); {KEY_HINT}")
    return None


def api_key(entry: dict, provider: str, model: str, environ=None, label=None) -> str:
    """The key's value (never stored or printed), NO_KEY when the provider needs none or an optional one is unset
    (`var in environ`, not its truth). An optional key set to "" is an error before the run
    (the openai SDK refuses api_key ""), never the placeholder."""
    environ = os.environ if environ is None else environ
    var = entry.get("api_key_env")
    if var is None:
        return NO_KEY
    if entry.get("api_key_optional"):
        if var not in environ:
            return NO_KEY
        if not environ[var]:
            raise ProviderError(f"--model {label or f'{provider}:{model}'}: ${var} is set but empty; unset it (no key) "
                                "or set the key")
        return environ[var]
    if environ.get(var):
        return environ[var]
    raise ProviderError(f"--model {label or f'{provider}:{model}'} needs ${var} (the API key of provider "
                        f"{provider!r}; {KEY_HINT})")


def secret_headers(entry: dict, provider: str, model: str, environ=None, label=None) -> dict:
    """headers_env's values ({header: value}; never stored or printed), read when the client is built."""
    environ = os.environ if environ is None else environ
    out = {}
    for h, var in (entry.get("headers_env") or {}).items():
        if not environ.get(var):
            raise ProviderError(f"--model {label or f'{provider}:{model}'} needs ${var} (header {h} of provider "
                                f"{provider!r}; {KEY_HINT})")
        out[h] = environ[var]
    return out


def missing_env(name: str, environ, registry: Registry) -> list:
    """The environment variables --model `name` needs that are unset (slurm/run_batch.py checks them before a job).
    The built-in Anthropic entry, for a model name without a provider: ANTHROPIC_API_KEY when the SDK finds no
    credential in `environ` (the same check as make_adapter: sdk_has_credentials of sdk_client)."""
    provider, model = split_model(name, registry)
    e = registry.entry(provider)
    if sdk_default(e):
        if not model.startswith("claude-") or environ.get(e["api_key_env"]):
            return []
        return [] if sdk_has_credentials(sdk_client(environ)) else [e["api_key_env"]]
    out = []
    if e.get("base_url_env") and not e.get("base_url") and not environ.get(e["base_url_env"]):
        out.append(e["base_url_env"])
    var = e.get("api_key_env")
    if var and not environ.get(var) and (not e.get("api_key_optional") or var in environ):
        out.append(var)
    out += [v for v in (e.get("headers_env") or {}).values() if not environ.get(v)]
    return out


def key_env_names(registry: Registry) -> tuple:
    """Every api_key_env and headers_env variable the registry names (a scripted dry run's children get none)."""
    names = set()
    for e in registry.providers.values():
        names |= {e["api_key_env"]} if e.get("api_key_env") else set()
        names |= set((e.get("headers_env") or {}).values())
    return tuple(sorted(names))


def price(entry: dict, model: str) -> dict | None:
    cost = entry.get("cost") or {}
    return (cost.get("prices") or {}).get(model) if cost.get("source") == "estimated" else None


def provider_info(registry: Registry, provider: str, model: str, url) -> dict:
    """The provenance record of the resolved provider: its entry without _comment keys (variable NAMES only,
    never a key; headers_env as header and variable names), every URL without credentials or query, this model's
    price, the registry file's sha256."""
    from chd_client.models.local import public_url
    e = registry.entry(provider)
    entry = {k: v for k, v in e.items() if k != "_comment"}
    if entry.get("base_url"):
        entry["base_url"] = public_url(entry["base_url"])
    if "cost" in entry:
        entry["cost"] = {k: v for k, v in entry["cost"].items() if k not in ("_comment", "prices")}
        if entry["cost"].get("source") == "estimated":
            entry["cost"]["price"] = price(e, model)
    return {"name": provider, "kind": e["kind"], "base_url": public_url(url) if url else None,
            "api_key_env": e.get("api_key_env"), "entry": entry, "entry_file": registry.sources.get(provider),
            "registry": registry.provenance()}


def _no_price(name, provider, model, kind):
    fields = sorted(ANTHROPIC_PRICE_FIELDS) if kind == "anthropic" else ["input_per_mtok", "output_per_mtok"]
    rates = ", ".join(json.dumps(f) + ": ..." for f in fields)
    return ProviderError(f"--model {name}: provider {provider!r} estimates the cost from prices, and has none for "
                         f"{model!r}. Add \"{model}\": {{{rates}}} to its "
                         "cost.prices (config/providers.local.json), from the vendor's price page. Without a cost the "
                         "run would stop on its first turn (cost_unknown).")


def output_cap(entry: dict, name: str, model: str, max_tokens, cls) -> dict | None:
    """The entry's max_output_tokens for this model: without --max-tokens, {"max_tokens": min(the
    adapter's default, limit)}; a --max-tokens above the limit is an error before the run. None: no limit for this
    model, the request is unchanged."""
    limit = (entry.get("max_output_tokens") or {}).get(model)
    if limit is None:
        return None
    if max_tokens:
        if max_tokens > limit:
            raise ProviderError(f"--model {name}: --max-tokens {max_tokens} is above this model's output limit, "
                                f"{limit} (max_output_tokens in its provider entry). Pass --max-tokens {limit} or "
                                "less.")
        return {"max_tokens": max_tokens}
    default = inspect.signature(cls.__init__).parameters["max_tokens"].default
    return {"max_tokens": min(default, limit)}


def make_adapter(name: str, *, effort=None, image_window=None, max_tokens=None, registry=None, providers=None,
                 environ=None, max_images=None):
    """The adapter for --model `name` (run_episode.make_model). max_tokens None keeps the adapter's default (a local
    gateway's capabilities.max_output_tokens replaces it: models.local.LocalModel.resolve_limits).
    max_images: the --max-images cap (models.base), None for none (a local gateway's own cap applies then:
    models.local.LocalModel.resolve_limits)."""
    environ = os.environ if environ is None else environ
    reg = registry or load_registry(providers, environ)
    provider, model = split_model(name, reg)
    e = reg.entry(provider)
    kind = e["kind"]
    out = {"max_tokens": max_tokens} if max_tokens else {}
    if kind == "anthropic":
        from chd_client.models import anthropic as am
        p = price(e, model)
        if "cost" in e and p is None:
            raise _no_price(name, provider, model, kind)
        if "cost" not in e and model not in am.PRICES:
            raise ProviderError(f"--model {name}: no price for {model!r} in models/anthropic.py PRICES ("
                                f"{', '.join(sorted(am.PRICES))}); without a cost the run would stop on its first turn "
                                "(cost_unknown). Add the price there, or use a provider entry with cost.prices.")
        if sdk_default(e):
            client = sdk_client(environ)
            if not sdk_has_credentials(client):
                raise ProviderError(f"--model {name}: the anthropic SDK found no credentials. Set ANTHROPIC_API_KEY "
                                    f"(or ANTHROPIC_AUTH_TOKEN: {KEY_HINT}), or set up an anthropic SDK profile "
                                    "(ANTHROPIC_PROFILE).")
        else:
            import anthropic
            url = base_url(e, provider, model, environ, label=name)
            key = api_key(e, provider, model, environ, label=name)
            client = anthropic.Anthropic(api_key=key, **({"base_url": url} if url else {}))
        m = am.AnthropicModel(model, effort=effort, image_window=image_window, client=client, **out)
        m.provider = provider
        if p is not None:
            m.prices = {model: {"in": p["input_per_mtok"], "out": p["output_per_mtok"],
                                "cache_write": p["cache_write_per_mtok"], "cache_read": p["cached_input_per_mtok"]}}
        url = str(getattr(m.client, "base_url", "") or "") or None
    elif kind == "openrouter":
        from chd_client.models.openrouter import OpenRouterModel, openai_client
        url = base_url(e, provider, model, environ, label=name)
        key = api_key(e, provider, model, environ, label=name)
        client = openai_client(url, key, secret_headers(e, provider, model, environ, label=name))
        m = OpenRouterModel(model, image_window=image_window, effort=effort, client=client, **out)
        m.routing = routing_for(e, model)
        m.image_format = e.get("image_format", "png")
        m.PROVIDER = provider
        m.stream = e.get("stream", True)
        m.headers = dict(e.get("headers") or {})
    else:
        from chd_client.models.local import LocalModel, OpenAICompatibleModel
        from chd_client.models.openrouter import openai_client
        if e["cost"]["source"] == "estimated" and price(e, model) is None:
            raise _no_price(name, provider, model, kind)
        url = base_url(e, provider, model, environ, label=name)
        key = api_key(e, provider, model, environ, label=name)
        client = openai_client(url, key, secret_headers(e, provider, model, environ, label=name))
        cls = LocalModel if e.get("x_gateway") else OpenAICompatibleModel
        out = output_cap(e, name, model, max_tokens, cls) or out
        m = cls(model, image_window=image_window, client=client, base_url=url,
                provider=provider, effort=effort, effort_param=e.get("effort"), extra_body=e.get("extra_body"),
                headers=e.get("headers"), headers_env=e.get("headers_env"),
                episode_headers=e.get("episode_headers", False), x_gateway=e.get("x_gateway", False), cost=e["cost"],
                stream=e.get("stream", False), max_tokens_param=e.get("max_tokens_param", "max_tokens"), **out)
        if cls is LocalModel:
            m.max_tokens_option = max_tokens or None
            m.entry_output_limit = (e.get("max_output_tokens") or {}).get(model)
    if max_images is not None:
        m.max_images, m.max_images_source = max_images, "option"
    m.provider_info = provider_info(reg, provider, model, url)
    if kind == "anthropic" and "cost" not in e:
        m.provider_info["price_table"] = am.price_record(model)
    return m
