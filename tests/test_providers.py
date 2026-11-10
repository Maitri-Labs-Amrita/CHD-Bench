"""The provider registry (models/registry.py, config/providers.json) and the generic OpenAI-compatible adapter, against
fake clients only: loading, overlay and errors, the --model split, unchanged requests and provenance for the
built-in providers, estimated cost, episode headers, and slurm/run_batch.py's key check."""

import importlib.util
import json
from pathlib import Path

import anyio
import pytest
from openai.types.chat import ChatCompletion

from chd_client.models import registry
from chd_client.models.anthropic import AnthropicModel
from chd_client.models.base import Usage
from chd_client.models.local import LocalModel, OpenAICompatibleModel, estimated_cost
from chd_client.models.openrouter import OpenRouterModel
from chd_client.run_episode import make_model, run_episode
from chd_client.run_record import RequestRecorder
from tests.fakes import CASE, make_env
from tests.test_anthropic import TOOLS, obs
from tests.test_openrouter import FakeOpenAI

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_batch", ROOT / "slurm" / "run_batch.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

GATEWAY = "http://127.0.0.1:8000/v1"
RETRY = [5, 10, 20, 40, 80, 160, 300, 300, 300]
EXPECTED_CONFIG = {
    "openrouter": {"model": "vendor/model-x", "tool_choice": "auto", "max_tokens": 32000,
                   "extra_body": {"usage": {"include": True}}, "image_window": None, "timeout_s": 1800.0,
                   "retry_waits_s": RETRY, "stream": True},
    "openrouter_claude": {"model": "anthropic/claude-sonnet-5.5", "tool_choice": "auto", "max_tokens": 64000,
                          "extra_body": {"usage": {"include": True}, "reasoning": {"effort": "max"},
                                         "provider": {"order": ["amazon-bedrock", "google-vertex"],
                                                      "allow_fallbacks": False}},
                          "image_window": 3, "timeout_s": 1800.0, "retry_waits_s": RETRY, "stream": True},
    "local": {"model": "qwen2.5vl:7b", "tool_choice": "auto", "max_tokens": 32000, "extra_body": {},
              "image_window": None, "timeout_s": 1800.0, "retry_waits_s": RETRY, "stream": False,
              "base_url": GATEWAY},
    "local_window": {"model": "qwen", "tool_choice": "auto", "max_tokens": 8000, "extra_body": {}, "image_window": 2,
                     "timeout_s": 1800.0, "retry_waits_s": RETRY, "stream": False, "base_url": GATEWAY},
    "anthropic": {"model": "claude-opus-5-5", "max_tokens": 32000,
                  "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "medium"},
                  "tool_choice": {"type": "auto"}, "image_window": None},
}
ENV = {"OPENROUTER_API_KEY": "test-not-a-key", "ANTHROPIC_API_KEY": "test-not-a-key", "CHD_LOCAL_BASE_URL": GATEWAY}


def write(path, providers, **top):
    path.write_text(json.dumps({"_comment": "test", "providers": providers, **top}))
    return path


PRICE = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}


def vendor(**over):
    return {"kind": "openai_compatible", "base_url": "https://api.vendor.test/v1", "api_key_env": "VENDOR_API_KEY",
            "cost": {"source": "estimated", "prices": {"big-model": PRICE}}, **over}


def completion(cost=None, cached=400, n=[0]):
    n[0] += 1
    usage = {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930,
             "prompt_tokens_details": {"cached_tokens": cached}}
    if cost is not None:
        usage["cost"] = cost
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": f"c{n[0]}", "type": "function", "function": {"name": "get_state", "arguments": "{}"}}]}
    return ChatCompletion.model_validate({"id": f"v-{n[0]}", "object": "chat.completion", "created": 0,
                                          "model": "gpt-x", "usage": usage,
                                          "choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}]})


def started(m, responses):
    m.client = fake = FakeOpenAI(responses)
    m.sleep = lambda s: None
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    return fake


def without_routing(tmp_path):
    """The repo's registry minus the openrouter entry's routing and image_format: the plain requests."""
    data = json.loads(registry.DEFAULT_FILE.read_text())
    del data["providers"]["openrouter"]["routing"]
    del data["providers"]["openrouter"]["image_format"]
    f = tmp_path / "providers.json"
    f.write_text(json.dumps(data))
    return registry.load_registry(f, environ={})


def test_the_repo_registry_loads_with_the_four_built_ins():
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    assert {"anthropic", "openrouter", "local", "openai"} <= set(reg.providers)
    o = reg.providers["openai"]
    assert o["effort"] is None and o["cost"]["source"] == "estimated" and o["cost"]["prices"] == {}


def test_request_config_of_the_built_ins_is_pinned(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CHD_LOCAL_API_KEY", raising=False)
    reg = without_routing(tmp_path)
    mk = lambda name, **kw: registry.make_adapter(name, registry=reg, **kw)  # noqa: E731
    assert mk("openrouter:vendor/model-x").request_config() == EXPECTED_CONFIG["openrouter"]
    assert mk("openrouter:anthropic/claude-sonnet-5.5", effort="max", image_window=3,
              max_tokens=64000).request_config() == EXPECTED_CONFIG["openrouter_claude"]
    local = mk("local:qwen2.5vl:7b", effort="high")
    assert type(local) is LocalModel and local.request_config() == EXPECTED_CONFIG["local"]
    assert mk("local:qwen", image_window=2, max_tokens=8000).request_config() == EXPECTED_CONFIG["local_window"]
    claude = mk("claude-opus-5-5", effort="medium")
    assert type(claude) is AnthropicModel and claude.client is not None
    assert claude.request_config() == EXPECTED_CONFIG["anthropic"]
    assert OpenRouterModel("vendor/model-x", client=object()).request_config() == EXPECTED_CONFIG["openrouter"]
    assert LocalModel("qwen2.5vl:7b", client=object(), base_url=GATEWAY).request_config() == EXPECTED_CONFIG["local"]


def test_registry_adapters_send_what_direct_adapters_send(tmp_path, monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    reg = without_routing(tmp_path)
    via = registry.make_adapter("openrouter:anthropic/claude-sonnet-5.5", effort="max", image_window=1, registry=reg)
    direct = OpenRouterModel("anthropic/claude-sonnet-5.5", effort="max", image_window=1, client=object())
    via.recorder = RequestRecorder(tmp_path)
    f1, f2 = started(via, [completion(cost=0.01)]), started(direct, [completion(cost=0.01)])
    via.step(), direct.step()
    assert f1.requests == f2.requests and f1.requests[0]["extra_headers"] == {}
    assert json.loads((tmp_path / "requests.jsonl").read_text())["provider"] == "openrouter"
    loc = registry.make_adapter("local:qwen", registry=reg)
    old = LocalModel("qwen", client=object(), base_url=GATEWAY)
    f1, f2 = started(loc, [completion(cost=0.0)]), started(old, [completion(cost=0.0)])
    loc.step(), old.step()
    a, b = ({**r, "extra_headers": {k: v for k, v in r["extra_headers"].items() if k != "X-Request-Id"}}
            for r in (f1.requests[0], f2.requests[0]))
    assert a == b and loc.PROVIDER == "local"
    assert set(f1.requests[0]["extra_headers"]) == {"X-Request-Id", "X-Turn-Index"}


PRIVATE = {"data_collection": "deny", "zdr": True}


def test_the_repo_routing_keeps_google_for_google_models_only(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    sent = lambda m, **kw: registry.make_adapter(f"openrouter:{m}", registry=reg, **kw).request_config()[  # noqa: E731
        "extra_body"]["provider"]
    for m in ("moonshotai/kimi-k3.1-pro", "meta/muse-spark-open", "qwen/qwen3.8-27b", "deepseek/deepseek-v4.1-flash"):
        assert sent(m) == {**PRIVATE, "ignore": ["google-vertex"]}
    assert sent("google/gemini-4-argon") == PRIVATE
    assert sent("anthropic/claude-sonnet-5.5", effort="max") == {"order": ["amazon-bedrock", "google-vertex"],
                                                                 "allow_fallbacks": False, **PRIVATE}
    assert sent("openai/gpt-6.1-sol") == {"data_collection": "deny", "order": ["openai"], "allow_fallbacks": False}
    bedrock = registry.make_adapter("openrouter_bedrock:openai/gpt-6.1-sol", registry=reg).request_config()
    assert bedrock["extra_body"]["provider"] == {"data_collection": "deny", "order": ["amazon-bedrock"], "allow_fallbacks": False}
    assert bedrock["image_format"] == "webp_lossless"


def test_routing_goes_into_every_request(tmp_path):
    e = {"kind": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
         "cost": {"source": "reported"}, "routing": {"default": PRIVATE}}
    reg = registry.load_registry(write(tmp_path / "p.json", {"openrouter": e}), environ={})
    m = registry.make_adapter("openrouter:vendor/model-x", registry=reg, environ={"OPENROUTER_API_KEY": "test"})
    fake = started(m, [completion(cost=0.01), completion(cost=0.01)])
    m.step(), m.step()
    assert [r["extra_body"]["provider"] for r in fake.requests] == [PRIVATE, PRIVATE]


def test_routing_rule_order_model_then_vendor_then_default():
    e = {"routing": {"_comment": "x", "default": {"zdr": True}, "google/": {"zdr": False},
                     "google/gemini-": {"sort": "price"},
                     "google/gemini-4-argon": {"only": ["google-vertex"], "_comment": "y"}}}
    assert registry.routing_for(e, "google/gemini-4-argon") == {"only": ["google-vertex"]}
    assert registry.routing_for(e, "google/gemini-4-sol") == {"zdr": False}
    assert registry.routing_for(e, "qwen/qwen3.8-27b") == {"zdr": True}
    assert registry.routing_for({}, "qwen/qwen3.8-27b") is None


@pytest.mark.parametrize("routing, msg", [
    ([], "routing must be an object"),
    ({"qwen": {}}, "a key is \"default\""),
    ({"default": []}, "must be an object"),
    ({"default": {"zdr": "yes"}}, "zdr has the wrong type"),
    ({"default": {"ignore": "google-vertex"}}, "ignore has the wrong type"),
    ({"default": {"ignore": [""]}}, "list of provider slugs"),
    ({"default": {"preferred_max_latency": True}}, "preferred_max_latency has the wrong type"),
    ({"default": {"data_collection": "no"}}, "data_collection must be"),
    ({"default": {"zdr_only": True}}, "unknown fields ['zdr_only']"),
])
def test_bad_routing_is_refused_with_the_file_named(tmp_path, routing, msg):
    e = {"kind": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
         "cost": {"source": "reported"}, "routing": routing}
    f = write(tmp_path / "p.json", {"openrouter": e})
    with pytest.raises(registry.ProviderError) as err:
        registry.load_registry(f, environ={})
    assert msg in str(err.value) and "p.json" in str(err.value)


def test_routing_is_for_kind_openrouter_only(tmp_path):
    f = write(tmp_path / "p.json", {"vendor": vendor(routing={"default": PRIVATE})})
    with pytest.raises(registry.ProviderError, match="do not apply to kind openai_compatible"):
        registry.load_registry(f, environ={})


def _png(seed=0):
    import io

    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    a[:32] = 0
    b = io.BytesIO()
    Image.fromarray(a, "RGB").save(b, "PNG")
    return b.getvalue(), a


def test_webp_lossless_keeps_every_pixel():
    import base64
    import io

    import numpy as np
    from PIL import Image

    from chd_client.models.openrouter import encode_image
    png, a = _png()
    url = encode_image(png, "webp_lossless")
    assert url.startswith("data:image/webp;base64,")
    back = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB"))
    assert (back == a).all()
    assert encode_image(png) == "data:image/png;base64," + base64.b64encode(png).decode()
    with pytest.raises(ValueError, match="one of png, webp_lossless"):
        encode_image(png, "jpeg")


def test_the_repo_openrouter_entry_sends_webp_and_records_it(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    m = registry.make_adapter("openrouter:vendor/model-x", registry=reg)
    assert m.image_format == "webp_lossless" and m.request_config()["image_format"] == "webp_lossless"
    fake = started(m, [completion(cost=0.01)])
    m.step()
    urls = [p["image_url"]["url"] for msg in fake.requests[0]["messages"] if isinstance(msg["content"], list)
            for p in msg["content"] if p.get("type") == "image_url"]
    assert urls and all(u.startswith("data:image/webp;base64,") for u in urls)


def test_bad_image_format_is_refused(tmp_path):
    e = {"kind": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY",
         "cost": {"source": "reported"}, "image_format": "jpeg"}
    with pytest.raises(registry.ProviderError, match="image_format must be one of png, webp_lossless"):
        registry.load_registry(write(tmp_path / "p.json", {"openrouter": e}), environ={})
    with pytest.raises(registry.ProviderError, match="do not apply to kind openai_compatible"):
        registry.load_registry(write(tmp_path / "q.json", {"vendor": vendor(image_format="png")}), environ={})


def test_overlay_replaces_and_adds_entries_and_both_files_are_hashed(tmp_path):
    base = write(tmp_path / "providers.json", {"anthropic": {"kind": "anthropic", "api_key_env": "ANTHROPIC_API_KEY"},
                                               "vendor": vendor()})
    write(tmp_path / "providers.local.json", {"vendor": vendor(base_url="https://eu.vendor.test/v1"),
                                              "together": vendor(api_key_env="TOGETHER_API_KEY")})
    reg = registry.load_registry(base, environ={})
    assert reg.providers["vendor"]["base_url"] == "https://eu.vendor.test/v1" and "together" in reg.providers
    assert reg.overlay.endswith("providers.local.json") and len(reg.sha256) == len(reg.overlay_sha256) == 64
    assert reg.sources["anthropic"] == str(base)


def test_providers_file_from_the_environment(tmp_path):
    p = write(tmp_path / "other.json", {"vendor": vendor()})
    assert set(registry.load_registry(environ={"CHD_PROVIDERS": str(p)}).providers) == {"vendor"}


@pytest.mark.parametrize("entry, msg", [
    (vendor(colour="red"), r"unknown keys \['colour'\]"),
    (vendor(kind="grpc"), "kind must be one of"),
    ({**vendor(), "kind": "openrouter", "x_gateway": True}, "do not apply to kind openrouter"),
    ({k: v for k, v in vendor().items() if k != "api_key_env"}, "needs api_key_env"),
    ({k: v for k, v in vendor().items() if k != "base_url"}, "needs base_url or base_url_env"),
    ({k: v for k, v in vendor().items() if k != "cost"}, "needs cost"),
    (vendor(cost={"source": "guess"}), "cost.source must be"),
    (vendor(cost={"source": "estimated", "prices": {"m": {"input_per_mtok": 1}}}), "needs output_per_mtok"),
    (vendor(cost={"source": "estimated", "prices": {"m": {"input_per_mtok": 1, "output_per_mtok": 2, "x": 1}}}),
     "unknown keys"),
    (vendor(effort="reasoning"), "effort must be null"),
    (vendor(stream="yes"), "stream must be true or false"),
])
def test_bad_entries_are_refused(tmp_path, entry, msg):
    with pytest.raises(registry.ProviderError, match=msg):
        registry.load_registry(write(tmp_path / "p.json", {"vendor": entry}), environ={})


def test_unknown_top_level_key_is_refused(tmp_path):
    with pytest.raises(registry.ProviderError, match="unknown keys"):
        registry.load_registry(write(tmp_path / "p.json", {"vendor": vendor()}, extra=1), environ={})


def test_unknown_provider_lists_the_known_ones(tmp_path):
    reg = registry.load_registry(write(tmp_path / "p.json", {"vendor": vendor(), "zeta": vendor()}), environ={})
    with pytest.raises(registry.ProviderError, match="unknown provider 'gemini'; known providers: vendor, zeta"):
        registry.split_model("gemini:gemini-3-pro", reg)


def test_model_split_on_the_first_colon_and_bare_names_are_anthropic():
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    assert registry.split_model("local:qwen2.5vl:7b", reg) == ("local", "qwen2.5vl:7b")
    assert registry.split_model("openrouter:openai/gpt-6-astra", reg) == ("openrouter", "openai/gpt-6-astra")
    assert registry.split_model("claude-opus-5-5", reg) == ("anthropic", "claude-opus-5-5")
    m = registry.make_adapter("local:qwen2.5vl:7b", registry=reg, environ={"CHD_LOCAL_BASE_URL": GATEWAY})
    assert m.model_id == "qwen2.5vl:7b"


def test_missing_variables_name_themselves(tmp_path):
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(registry.ProviderError, match=r"\$CHD_LOCAL_BASE_URL"):
        registry.make_adapter("local:qwen", registry=reg, environ={})
    with pytest.raises(registry.ProviderError, match=r"\$OPENROUTER_API_KEY"):
        registry.make_adapter("openrouter:vendor/x", registry=reg, environ={})
    e = vendor(base_url_env="VENDOR_URL", api_key_optional=True)
    assert registry.api_key(e, "vendor", "m", {}) == registry.NO_KEY
    assert registry.base_url(e, "vendor", "m", {"VENDOR_URL": "http://x/v1"}) == "http://x/v1"
    assert registry.base_url(e, "vendor", "m", {}) == "https://api.vendor.test/v1"
    assert registry.api_key(vendor(api_key_env=None), "vendor", "m", {}) == registry.NO_KEY


def test_make_model_uses_the_providers_file(tmp_path, monkeypatch):
    monkeypatch.setenv("VENDOR_API_KEY", "test-not-a-key")
    p = write(tmp_path / "p.json", {"vendor": vendor()})
    m = make_model("vendor:big-model", "high", False, image_window=2, providers=str(p))
    assert type(m) is OpenAICompatibleModel and m.PROVIDER == "vendor" and m.effort is None
    assert m.base_url == "https://api.vendor.test/v1" and m.client.base_url == "https://api.vendor.test/v1/"


def generic(**kw):
    kw = {"provider": "vendor", "client": FakeOpenAI([]), "sleep": lambda s: None, **kw}
    m = OpenAICompatibleModel("gpt-x", **kw)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    return m


def test_estimated_cost_with_and_without_a_cached_price():
    prices = {"gpt-x": {"input_per_mtok": 2.0, "output_per_mtok": 8.0, "cached_input_per_mtok": 0.5}}
    m = generic(cost={"source": "estimated", "prices": prices}, effort="high", effort_param="reasoning_effort",
                extra_body={"service_tier": "flex"})
    m.client.responses.append(completion(cost=99.0))
    t = m.step()
    assert t.usage.cost_usd == pytest.approx((500 * 2.0 + 400 * 0.5 + 30 * 8.0) / 1e6)
    assert t.extra["cost_source"] == "estimated"
    req = m.client.requests[0]
    assert req["reasoning_effort"] == "high" and req["extra_body"] == {"service_tier": "flex"}
    assert req["max_tokens"] == 32000 and "max_completion_tokens" not in req
    assert m.request_config()["reasoning_effort"] == "high" and "stream" not in req
    no_cached = {"gpt-x": {"input_per_mtok": 2.0, "output_per_mtok": 8.0}}
    m = generic(cost={"source": "estimated", "prices": no_cached})
    m.client.responses.append(completion())
    assert m.step().usage.cost_usd == pytest.approx((900 * 2.0 + 30 * 8.0) / 1e6)
    assert "reasoning_effort" not in m.client.requests[0]


def test_no_price_gives_no_cost_and_the_run_stops(tmp_path):
    assert estimated_cost(None, Usage(900, 30)) is None
    srv, _, traces = make_env(tmp_path)
    m = OpenAICompatibleModel("gpt-x", provider="vendor", client=FakeOpenAI([completion(cost=0.5), completion()]),
                              sleep=lambda s: None, cost={"source": "estimated", "prices": {}})
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, max_steps=2, cost_cap_usd=None))
    srv.session.close("abandoned")
    assert summary["stop_reason"] == "cost_unknown" and summary["n_model_steps"] == 0


def test_make_adapter_refuses_a_model_without_a_price_before_any_request(monkeypatch):
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(registry.ProviderError, match=r"has none for 'gpt-x'.*input_per_mtok"):
        registry.make_adapter("openai:gpt-x", registry=reg, environ={"OPENAI_API_KEY": "test-not-a-key"})


def test_episode_headers_only_when_asked():
    m = generic(headers={"X-Title": "CHD-Bench"})
    m.episode_id = "ep-1"
    m.client.responses += [completion(cost=0.0)]
    m.step()
    assert m.client.requests[0]["extra_headers"] == {"X-Title": "CHD-Bench"}
    m = generic(episode_headers=True)
    m.episode_id = "ep-1"
    m.client.responses += [completion(cost=0.0)]
    m.step()
    h = m.client.requests[0]["extra_headers"]
    assert h["X-Episode-Id"] == "ep-1" and h["X-Turn-Index"] == "1" and len(h["X-Request-Id"]) == 32


def test_provenance_records_the_provider_without_the_key(tmp_path, monkeypatch):
    from chd_client.provenance import write_provenance
    monkeypatch.setenv("VENDOR_API_KEY", "sk-secret-value")
    monkeypatch.setenv("VENDOR_URL", "https://u:pw@api.vendor.test/v1?sig=1")
    p = write(tmp_path / "p.json", {"vendor": vendor(base_url_env="VENDOR_URL")})
    m = make_model("vendor:big-model", "high", False, providers=str(p))
    d = write_provenance(tmp_path / "run", system_text="sys", tools=TOOLS, model=m, repo=ROOT)
    text = (d / "provenance.json").read_text()
    prov = json.loads(text)["provider"]
    assert prov["name"] == "vendor" and prov["kind"] == "openai_compatible" and prov["api_key_env"] == "VENDOR_API_KEY"
    assert prov["base_url"] == "https://api.vendor.test/v1" and prov["registry"]["file"] == str(p)
    assert prov["registry"]["sha256"] == registry.load_registry(p, environ={}).sha256
    assert "sk-secret-value" not in text and "pw" not in prov["base_url"]


def test_run_batch_checks_the_registry_keys(tmp_path):
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    rb.check_model_env("openrouter:vendor/x", {"OPENROUTER_API_KEY": "k"}, reg)
    rb.check_model_env("local:qwen", {"CHD_LOCAL_BASE_URL": GATEWAY}, reg)
    for model, env, msg in (("openai:gpt-x", {}, "OPENAI_API_KEY not set"),
                            ("claude-opus-5-5", {"HOME": str(tmp_path)}, "ANTHROPIC_API_KEY not set"),
                            ("local:qwen", {}, "CHD_LOCAL_BASE_URL not set"),
                            ("gemini:g", {}, "known providers: anthropic, local, openai, openrouter")):
        with pytest.raises(rb.BatchError, match=msg):
            rb.check_model_env(model, env, reg)
    environ = {"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k", "PATH": "/bin"}
    names = registry.key_env_names(reg)
    assert "OPENAI_API_KEY" in names and "CHD_LOCAL_API_KEY" in names
    assert set(rb.child_env(environ, {}, "/r.json", dry_run=True, key_names=names)) == {"PATH", "SLICER_MCP_RUNTIME"}
    assert "OPENAI_API_KEY" in rb.child_env(environ, {}, "/r.json", key_names=names)


def test_run_batch_finds_the_jobs_providers_file(tmp_path):
    c = {"extra_args": ["--max-steps", "10"], "jobs": [{"extra_args": ["--providers", "config/mine.json"]}, {}]}
    assert rb.providers_file(c, 0, base=tmp_path) == tmp_path / "config" / "mine.json"
    assert rb.providers_file(c, 1, base=tmp_path) is None


def test_openai_sends_max_completion_tokens(monkeypatch):
    p = registry.load_registry(registry.DEFAULT_FILE, environ={})
    p.providers["openai"]["cost"]["prices"]["gpt-x"] = {"input_per_mtok": 1.0, "output_per_mtok": 2.0}
    p.providers["openai"]["effort"] = "reasoning_effort"
    m = registry.make_adapter("openai:gpt-x", effort="low", registry=p, environ={"OPENAI_API_KEY": "test-not-a-key"})
    started(m, [completion()])
    t = m.step()
    req = m.client.requests[0]
    assert req["max_completion_tokens"] == 32000 and "max_tokens" not in req and req["reasoning_effort"] == "low"
    assert t.usage.cost_usd == pytest.approx((900 * 1.0 + 30 * 2.0) / 1e6)
    assert m.request_config()["max_tokens_param"] == "max_completion_tokens" and m.PROVIDER == "openai"
    assert m.provider_info["entry"]["cost"] == {"source": "estimated", "price": {"input_per_mtok": 1.0,
                                                                                 "output_per_mtok": 2.0}}


def test_max_output_tokens_caps_the_default_and_refuses_a_larger_max_tokens(tmp_path):
    """A model's output limit in its entry (max_output_tokens) lowers the default 32000 sent, and a --max-tokens above
    it is refused before the run. Other models of the entry, and a value within it, are unchanged. The numbers here are
    test values, not a vendor's."""
    reg = reg_of(tmp_path, {"vendor": vendor(max_tokens_param="max_completion_tokens",
                                             max_output_tokens={"big-model": 1000, "huge-model": 64000},
                                             cost={"source": "estimated",
                                                   "prices": {"big-model": PRICE, "huge-model": PRICE,
                                                              "free-model": PRICE}})})
    env = {"VENDOR_API_KEY": "test-not-a-key"}
    mk = lambda name, **kw: registry.make_adapter(name, registry=reg, environ=env, **kw)  # noqa: E731
    m = mk("vendor:big-model")
    started(m, [completion()])
    m.step()
    assert m.client.requests[0]["max_completion_tokens"] == 1000 and m.request_config()["max_tokens"] == 1000
    assert m.provider_info["entry"]["max_output_tokens"] == {"big-model": 1000, "huge-model": 64000}
    assert mk("vendor:big-model", max_tokens=800).max_tokens == 800
    assert mk("vendor:big-model", max_tokens=1000).max_tokens == 1000
    with pytest.raises(registry.ProviderError, match=r"^--model vendor:big-model: --max-tokens 1001 is above this "
                                                     r"model's output limit, 1000 .*--max-tokens 1000 or less"):
        mk("vendor:big-model", max_tokens=1001)
    assert mk("vendor:huge-model").max_tokens == 32000
    assert mk("vendor:huge-model", max_tokens=64000).max_tokens == 64000
    assert mk("vendor:free-model", max_tokens=99999).max_tokens == 99999
    assert mk("vendor:free-model").max_tokens == 32000


@pytest.mark.parametrize("value, msg", [([1000], "max_output_tokens must be an object"),
                                        ({"m": 0}, r"max_output_tokens\['m'\] must be a whole number >= 1"),
                                        ({"m": 1.5}, r"max_output_tokens\['m'\] must be a whole number >= 1"),
                                        ({"m": True}, r"max_output_tokens\['m'\] must be a whole number >= 1")])
def test_max_output_tokens_is_checked(tmp_path, value, msg):
    with pytest.raises(registry.ProviderError, match=msg):
        reg_of(tmp_path, {"vendor": vendor(max_output_tokens=value)})
    with pytest.raises(registry.ProviderError, match=r"\['max_output_tokens'\] do not apply to kind openrouter"):
        reg_of(tmp_path, {"or": {"kind": "openrouter", "base_url": "https://or.test/v1", "api_key_env": "K",
                                 "cost": {"source": "reported"}, "max_output_tokens": {"m": 10}}})


ANTHROPIC_RATES = {"input_per_mtok": 3.0, "output_per_mtok": 15.0, "cached_input_per_mtok": 0.3,
                   "cache_write_per_mtok": 3.75}


def proxy(**over):
    return {"kind": "anthropic", "base_url": "https://proxy.test", "api_key_env": None,
            "cost": {"source": "estimated", "prices": {"claude-x": ANTHROPIC_RATES}}, **over}


def reg_of(tmp_path, providers):
    return registry.load_registry(write(tmp_path / "p.json", providers), environ={})


def test_an_anthropic_entry_for_another_host_never_gets_the_environments_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-real")
    monkeypatch.delenv("PROXY_KEY", raising=False)
    reg = reg_of(tmp_path, {"proxy": proxy(), "opt": proxy(api_key_env="PROXY_KEY", api_key_optional=True)})
    for name in ("proxy:claude-x", "opt:claude-x"):
        m = registry.make_adapter(name, registry=reg)
        assert m.client.auth_headers == {"X-Api-Key": registry.NO_KEY}, name
        assert str(m.client.base_url).startswith("https://proxy.test")


def test_estimated_cost_without_token_counts_is_unknown_and_the_run_stops(tmp_path):
    bare = completion()
    no_usage = ChatCompletion.model_validate({**bare.model_dump(exclude={"usage"}), "usage": None})
    m = generic(cost={"source": "estimated", "prices": {"gpt-x": PRICE}})
    m.client.responses.append(no_usage)
    t = m.step()
    assert t.usage.cost_usd is None and t.extra["cost_source"] == "estimated"
    srv, _, traces = make_env(tmp_path)
    m = OpenAICompatibleModel("gpt-x", provider="vendor", client=FakeOpenAI([no_usage, completion()]),
                              sleep=lambda s: None, cost={"source": "estimated", "prices": {"gpt-x": PRICE}})
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, max_steps=2, cost_cap_usd=1.0))
    srv.session.close("abandoned")
    assert summary["stop_reason"] == "cost_unknown" and summary["n_model_steps"] == 0


@pytest.mark.parametrize("raw, msg", [('NaN', "not JSON"), ('Infinity', "not JSON"), ('-Infinity', "not JSON"),
                                      ('1e999', "finite number")])
def test_nan_and_infinite_prices_are_refused(tmp_path, raw, msg):
    entry = json.dumps(vendor(cost={"source": "estimated", "prices": {"m": {"input_per_mtok": 0,
                                                                            "output_per_mtok": 1}}}))
    p = tmp_path / "p.json"
    entry = entry.replace('"output_per_mtok": 1', f'"output_per_mtok": {raw}')
    p.write_text(f'{{"providers": {{"vendor": {entry}}}}}')
    with pytest.raises(registry.ProviderError, match=msg):
        registry.load_registry(p, environ={})
    with pytest.raises(registry.ProviderError, match="finite number"):
        registry.check_cost("w", "openai_compatible", {"source": "estimated", "prices": {"m": {
            "input_per_mtok": float("nan"), "output_per_mtok": 1.0}}})


def test_anthropic_models_without_a_price_are_refused_before_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(registry.ProviderError, match=r"--model claude-unpriced-9: no price for 'claude-unpriced-9'"):
        registry.make_adapter("claude-unpriced-9", registry=reg)
    with pytest.raises(registry.ProviderError, match="kind anthropic with a base_url needs cost"):
        reg_of(tmp_path, {"proxy": {k: v for k, v in proxy().items() if k != "cost"}})
    with pytest.raises(registry.ProviderError, match="needs cache_write_per_mtok"):
        reg_of(tmp_path, {"proxy": proxy(cost={"source": "estimated", "prices": {"claude-x": PRICE}})})
    with pytest.raises(registry.ProviderError, match="reported"):
        reg_of(tmp_path, {"proxy": proxy(cost={"source": "reported"})})
    reg = reg_of(tmp_path, {"proxy": proxy()})
    with pytest.raises(registry.ProviderError, match=r"has none for 'claude-opus-5-5'.*cache_write_per_mtok"):
        registry.make_adapter("proxy:claude-opus-5-5", registry=reg)
    from tests.test_anthropic import FakeAnthropic, message, tool_use
    m = registry.make_adapter("proxy:claude-x", registry=reg)
    m.client = FakeAnthropic([message([tool_use("tu_1")])])
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    assert m.step().usage.cost_usd == pytest.approx((100 * 3.0 + 20 * 15.0 + 1000 * 3.75 + 5000 * 0.3) / 1e6)
    assert m.provider_info["entry"]["cost"]["price"] == ANTHROPIC_RATES


def test_cost_reported_only_for_openrouter_and_gateways(tmp_path):
    with pytest.raises(registry.ProviderError, match="is for kind openrouter and x_gateway entries only"):
        reg_of(tmp_path, {"vendor": vendor(cost={"source": "reported"})})
    reg = reg_of(tmp_path, {"gw": vendor(cost={"source": "reported"}, x_gateway=True)})
    assert reg.providers["gw"]["cost"] == {"source": "reported"}


@pytest.mark.parametrize("url", ["https://u:pw@api.vendor.test/v1", "https://api.vendor.test/v1?key=sk-abc",
                                 "https://api.vendor.test/v1#sk-abc", "api.vendor.test/v1"])
def test_a_base_url_with_credentials_or_a_query_is_refused(tmp_path, url):
    with pytest.raises(registry.ProviderError, match="base_url must") as ex:
        reg_of(tmp_path, {"vendor": vendor(base_url=url)})
    assert "sk-abc" not in str(ex.value) and "pw" not in str(ex.value)


def test_secret_headers_come_from_headers_env_and_are_never_recorded(tmp_path, monkeypatch):
    from chd_client.provenance import write_provenance
    monkeypatch.setenv("VENDOR_API_KEY", "test-not-a-key")
    monkeypatch.setenv("AZURE_KEY", "sk-azure-secret")
    for name in ("Authorization", "api-key", "X-Api-Key", "Cookie", "Proxy-Authorization"):
        with pytest.raises(registry.ProviderError, match="headers_env instead"):
            reg_of(tmp_path, {"vendor": vendor(headers={name: "x"})})
    with pytest.raises(registry.ProviderError, match="in both headers and headers_env"):
        reg_of(tmp_path, {"vendor": vendor(headers={"X-Tag": "a"}, headers_env={"x-tag": "AZURE_KEY"})})
    reg = reg_of(tmp_path, {"vendor": vendor(headers={"X-Title": "CHD-Bench"}, headers_env={"api-key": "AZURE_KEY"})})
    assert "AZURE_KEY" in registry.key_env_names(reg)
    assert registry.missing_env("vendor:big-model", {"VENDOR_API_KEY": "k"}, reg) == ["AZURE_KEY"]
    with pytest.raises(registry.ProviderError, match=r"--model vendor:big-model needs \$AZURE_KEY \(header api-key"):
        registry.make_adapter("vendor:big-model", registry=reg, environ={"VENDOR_API_KEY": "k"})
    m = make_model("vendor:big-model", "high", False, providers=str(tmp_path / "p.json"))
    assert m.client.default_headers["api-key"] == "sk-azure-secret"
    assert m.request_config()["headers"] == {"X-Title": "CHD-Bench"}
    assert m.request_config()["headers_env"] == {"api-key": "AZURE_KEY"}
    m.recorder = RequestRecorder(tmp_path / "rec")
    started(m, [completion()])
    m.step()
    assert m.client.requests[0]["extra_headers"] == {"X-Title": "CHD-Bench"}
    d = write_provenance(tmp_path / "run", system_text="sys", tools=TOOLS, model=m, repo=ROOT)
    assert "sk-azure-secret" not in (tmp_path / "rec" / "requests.jsonl").read_text()
    prov = (d / "provenance.json").read_text()
    assert "sk-azure-secret" not in prov
    assert json.loads(prov)["provider"]["entry"]["headers_env"] == {"api-key": "AZURE_KEY"}
    reg = reg_of(tmp_path, {"or": {"kind": "openrouter", "base_url": "https://openrouter.ai/api/v1",
                                   "api_key_env": "VENDOR_API_KEY", "headers_env": {"X-Api-Key": "AZURE_KEY"},
                                   "cost": {"source": "reported"}}})
    m = registry.make_adapter("or:vendor/x", registry=reg)
    assert m.client.default_headers["X-Api-Key"] == "sk-azure-secret" and m.headers == {}


@pytest.mark.parametrize("field, value", [("api_key_env", "sk-ant-PASTEDKEY123"), ("api_key_env", ""),
                                          ("base_url_env", "https://api.vendor.test/v1"), ("api_key_env", "lower_case"),
                                          ("stream", "sk-ant-PASTEDKEY123"), ("headers_env", {"api-key": "sk-ant-X1"})])
def test_errors_never_repeat_a_value(tmp_path, field, value):
    with pytest.raises(registry.ProviderError) as ex:
        reg_of(tmp_path, {"vendor": vendor(**{field: value})})
    text = str(ex.value)
    assert field in text and (not value or str(value if isinstance(value, str) else "sk-ant-X1") not in text)


def test_run_batch_reads_the_providers_arguments_in_order(tmp_path):
    for args, want in ((["--providers=a.json", "--providers", "b.json"], "b.json"),
                       (["--providers", "a.json", "--providers=b.json"], "b.json"),
                       (["--providers", "a.json"], "a.json"), (["--providers"], None)):
        got = rb.providers_file({"jobs": [{"extra_args": args}]}, 0, base=tmp_path)
        assert got == (tmp_path / want if want else None), args


def test_run_episode_refuses_abbreviated_options(capsys):
    from chd_client.run_episode import main
    with pytest.raises(SystemExit):
        main(["--imagechd", "1", "--labels", "gt", "--provider", "x.json"])
    assert "unrecognized arguments: --provider" in capsys.readouterr().err


def test_a_bare_model_id_with_a_colon_says_how_to_write_it():
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(registry.ProviderError, match=r"unknown provider 'anthropic\.claude-v2'.*"
                                                     r"--model anthropic:anthropic\.claude-v2:1"):
        registry.split_model("anthropic.claude-v2:1", reg)
    assert registry.split_model("anthropic:anthropic.claude-v2:1", reg) == ("anthropic", "anthropic.claude-v2:1")


def test_builtin_openai_sends_no_effort(monkeypatch):
    p = registry.load_registry(registry.DEFAULT_FILE, environ={})
    p.providers["openai"]["cost"]["prices"]["gpt-4.1"] = PRICE
    m = registry.make_adapter("openai:gpt-4.1", effort="medium", registry=p, environ={"OPENAI_API_KEY": "k"})
    started(m, [completion()])
    m.step()
    assert "reasoning_effort" not in m.client.requests[0] and "reasoning_effort" not in m.request_config()


def test_the_default_anthropic_client_takes_no_arguments(tmp_path, monkeypatch):
    import anthropic
    calls = []
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: calls.append((a, k)) or type(
        "C", (), {"base_url": "https://api.anthropic.com", "default_headers": {"X-Api-Key": "from-the-sdk"}})())
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    m = registry.make_adapter("claude-opus-5-5", registry=reg, environ={})
    assert calls == [((), {})] and m.provider == "anthropic"
    monkeypatch.undo()
    rb.check_model_env("my-model", {}, reg)
    with pytest.raises(rb.BatchError, match="ANTHROPIC_API_KEY not set .*--model claude-opus-5-5"):
        rb.check_model_env("claude-opus-5-5", {"ANTHROPIC_API_KEY": "", "HOME": str(tmp_path)}, reg)
    reg = reg_of(tmp_path, {"anthropic": {"kind": "anthropic", "api_key_env": "LAB_ANTHROPIC_KEY"}})
    with pytest.raises(registry.ProviderError, match=r"^--model claude-opus-5-5 needs \$LAB_ANTHROPIC_KEY"):
        registry.make_adapter("claude-opus-5-5", registry=reg, environ={})
    with pytest.raises(registry.ProviderError, match=r"^--model local:qwen needs \$CHD_LOCAL_BASE_URL"):
        registry.make_adapter("local:qwen", registry=registry.load_registry(registry.DEFAULT_FILE, environ={}),
                              environ={})


SDK_AUTH_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
                "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_IDENTITY_TOKEN", "ANTHROPIC_IDENTITY_TOKEN_FILE",
                "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID")


@pytest.mark.parametrize("var, value", [(None, None), ("ANTHROPIC_API_KEY", "test-not-a-key"),
                                        ("ANTHROPIC_AUTH_TOKEN", "test-not-a-token"),
                                        ("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: test-not-a-key")])
def test_the_default_anthropic_entry_needs_a_credential_before_the_run(tmp_path, monkeypatch, var, value):
    """The SDK checks credentials only at its first request, after the case is prepared. The built-in entry is refused
    before that when the real SDK resolves none; any credential it accepts passes, and the client is still
    anthropic.Anthropic() with no arguments."""
    for v in SDK_AUTH_ENV:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    if var is None:
        with pytest.raises(registry.ProviderError, match=r"^--model claude-opus-5-5: the anthropic SDK found no "
                                                         r"credentials\. Set ANTHROPIC_API_KEY \(or "
                                                         r"ANTHROPIC_AUTH_TOKEN: export it, or put it in \.env and "
                                                         r"use \./run\.sh\)"):
            registry.make_adapter("claude-opus-5-5", registry=reg)
        return
    monkeypatch.setenv(var, value)
    m = registry.make_adapter("claude-opus-5-5", registry=reg, effort="medium")
    assert type(m) is AnthropicModel and m.request_config() == EXPECTED_CONFIG["anthropic"]


def test_an_empty_optional_key_fails_before_the_run(monkeypatch):
    """CHD_LOCAL_API_KEY="" reaches openai.OpenAI(api_key=""), which refuses it ("Missing credentials"): it is refused
    before the run, never replaced by the placeholder."""
    import openai
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    env = {"CHD_LOCAL_BASE_URL": GATEWAY, "CHD_LOCAL_API_KEY": ""}
    with pytest.raises(registry.ProviderError, match=r"--model local:qwen: \$CHD_LOCAL_API_KEY is set but empty"):
        registry.make_adapter("local:qwen", registry=reg, environ=env)
    assert registry.missing_env("local:qwen", env, reg) == ["CHD_LOCAL_API_KEY"]
    env.pop("CHD_LOCAL_API_KEY")
    assert registry.missing_env("local:qwen", env, reg) == []
    assert registry.make_adapter("local:qwen", registry=reg, environ=env).client.api_key == registry.NO_KEY
    env["CHD_LOCAL_API_KEY"] = "test-not-a-key"
    assert registry.make_adapter("local:qwen", registry=reg, environ=env).client.api_key == "test-not-a-key"
    monkeypatch.setenv("CHD_LOCAL_API_KEY", "")
    with pytest.raises(openai.OpenAIError):
        LocalModel("qwen", base_url=GATEWAY)


def test_anthropic_provenance_records_the_sdks_base_url(monkeypatch):
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert registry.make_adapter("claude-opus-5-5", registry=reg).provider_info["base_url"] == \
        "https://api.anthropic.com"
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://user:pw@proxy.test/anthropic?sig=1")
    info = registry.make_adapter("claude-opus-5-5", registry=reg).provider_info
    assert info["base_url"] == "https://proxy.test/anthropic"


def test_anthropic_provenance_records_the_built_in_price_table(tmp_path, monkeypatch):
    """A run priced by models/anthropic.py PRICES records the rates it used and where they come from (PRICES_SOURCE: the
    page and the date they were checked). An entry with its own cost.prices records those instead."""
    from chd_client.models import anthropic as am
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    info = registry.make_adapter("claude-opus-5-5", registry=reg).provider_info
    assert info["price_table"] == {"table": "chd_client.models.anthropic.PRICES", "rates": am.PRICES["claude-opus-5-5"],
                                   **am.PRICES_SOURCE}
    assert am.PRICES_SOURCE["source"] and am.PRICES_SOURCE["confirmed_on"]
    info = registry.make_adapter("claude-x", registry=reg_of(tmp_path, {"anthropic": proxy()})).provider_info
    assert "price_table" not in info and info["entry"]["cost"]["price"] == ANTHROPIC_RATES


def test_the_builtin_openai_entry_caps_gpt_4o_at_its_output_limit():
    """With a price added, the built-in entry asks gpt-4o for at most its output limit (16384,
    https://developers.openai.com/api/docs/models/gpt-4o), not the default 32000."""
    p = registry.load_registry(registry.DEFAULT_FILE, environ={})
    assert p.providers["openai"]["max_output_tokens"] == {"gpt-4o": 16384}
    assert p.providers["openai"]["cost"]["prices"] == {}
    p.providers["openai"]["cost"]["prices"]["gpt-4o"] = PRICE
    m = registry.make_adapter("openai:gpt-4o", registry=p, environ={"OPENAI_API_KEY": "k"})
    started(m, [completion()])
    m.step()
    assert m.client.requests[0]["max_completion_tokens"] == 16384 and "max_tokens" not in m.client.requests[0]
    with pytest.raises(registry.ProviderError, match="--max-tokens 32000 is above this model's output limit, 16384"):
        registry.make_adapter("openai:gpt-4o", registry=p, environ={"OPENAI_API_KEY": "k"}, max_tokens=32000)


def test_every_adapter_defaults_to_the_shared_output_cap():
    import inspect
    from chd_client.models.anthropic import AnthropicModel
    from chd_client.models.base import DEFAULT_MAX_TOKENS
    from chd_client.models.local import LocalModel, OpenAICompatibleModel
    from chd_client.models.openrouter import OpenRouterModel
    assert DEFAULT_MAX_TOKENS == 32000
    for cls in (AnthropicModel, OpenRouterModel, OpenAICompatibleModel, LocalModel):
        assert inspect.signature(cls.__init__).parameters["max_tokens"].default == DEFAULT_MAX_TOKENS, cls.__name__
