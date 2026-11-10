"""The per-model output cap (each open model runs with its creator's maximum output tokens per turn). A gateway
profile's capabilities.max_output_tokens reaches GET /v1/models, and models.local.LocalModel.resolve_limits takes it as
max_tokens when --max-tokens is not given (request_config records max_tokens_source "gateway"); a larger --max-tokens is
refused before Slicer starts. Without the field, and for claude-* and openrouter: models, the requests stay
byte-identical. Fake clients only."""

import json

import pytest
from openai.types.chat import ChatCompletion

from chd_client.models import registry
from chd_client.models.local import LocalModel
from chd_client.run_episode import make_model
from tests.test_anthropic import TOOLS, obs
from tests.test_image_cap import GATEWAY, FakeGateway, multi_call_episode, two_calls

LOCAL_ENV = {"CHD_LOCAL_BASE_URL": GATEWAY}


def caps(name="qwen", max_output_tokens=None, **more):
    return [{"id": name, "object": "model", "created": 0, "owned_by": "chd-gateway",
             "x_gateway": {"capabilities": {"images": True, "tools": True, "max_images": None,
                                            "max_output_tokens": max_output_tokens, **more}}}]


def adapter(fake, max_tokens=None, providers=None):
    """local:qwen as run-episode builds it (models.registry.make_adapter), with a fake gateway client."""
    m = registry.make_adapter("local:qwen", max_tokens=max_tokens, providers=providers, environ=LOCAL_ENV)
    m.client = fake
    m.sleep = lambda s: None
    return m


def strip(rs):
    return [{**r, "extra_headers": {k: v for k, v in r["extra_headers"].items() if k != "X-Request-Id"}} for r in rs]


def test_without_max_tokens_the_gateways_cap_is_max_tokens():
    fake = FakeGateway([two_calls(k) for k in range(2)], models=caps(max_output_tokens=81920))
    m = adapter(fake)
    assert m.max_tokens == 32000
    m.resolve_limits()
    m.resolve_limits()
    assert fake.model_gets == 1 and (m.max_tokens, m.max_tokens_source) == (81920, "gateway")
    multi_call_episode(m, turns=2)
    assert [r["max_tokens"] for r in fake.requests] == [81920, 81920]
    rc = m.request_config()
    assert rc["max_tokens"] == 81920 and rc["max_tokens_source"] == "gateway"


def test_a_max_tokens_at_or_below_the_cap_is_sent_as_given():
    for given in (16384, 81920):
        fake = FakeGateway([two_calls(0)], models=caps(max_output_tokens=81920))
        m = adapter(fake, max_tokens=given)
        m.resolve_limits()
        multi_call_episode(m, turns=1)
        assert fake.requests[0]["max_tokens"] == given and m.max_tokens_source is None
        assert "max_tokens_source" not in m.request_config()


def test_a_max_tokens_above_the_cap_is_refused():
    m = adapter(FakeGateway([], models=caps(max_output_tokens=16384)), max_tokens=32000)
    with pytest.raises(registry.ProviderError,
                       match=r"--max-tokens 32000 is above the gateway's output cap for 'qwen': 16384 tokens per turn "
                             r"\(capabilities.max_output_tokens"):
        m.resolve_limits()


def test_run_sh_refuses_a_max_tokens_above_the_cap_before_slicer(tmp_path, monkeypatch):
    """./run.sh runs run_episode.prelaunch (resolve_limits included) before it starts Slicer."""
    from chd_client import run_sh
    monkeypatch.setattr(LocalModel, "_gateway_capabilities", lambda self: {"max_output_tokens": 20480})
    env = {"PATH": "/usr/bin", **LOCAL_ENV}
    run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--model", "local:qwen"], env, env_path=tmp_path / "none")
    run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--model", "local:qwen", "--max-tokens", "20480"], env,
                   env_path=tmp_path / "none")
    with pytest.raises(run_sh.RunShError, match="--max-tokens 32000 is above the gateway's output cap"):
        run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--model", "local:qwen", "--max-tokens", "32000"],
                       env, env_path=tmp_path / "none")


@pytest.mark.parametrize("models", [caps(max_output_tokens=None), [{"id": "qwen", "object": "model"}],
                                    caps("other", max_output_tokens=81920), [],
                                    caps(max_output_tokens=0), caps(max_output_tokens="81920"),
                                    caps(max_output_tokens=True)])
def test_no_cap_reported_keeps_the_requests_byte_identical(models):
    """A profile without max_output_tokens, a gateway that sends no x_gateway, another model's cap, or a
    value that is not a whole number of at least 1: requests and request_config are those of an adapter that never
    asked."""
    asked = adapter(FakeGateway([two_calls(k) for k in range(2)], models=models))
    plain = adapter(FakeGateway([two_calls(k) for k in range(2)]))
    plain.client.models = None
    before = asked.request_config()
    asked.resolve_limits(), plain.resolve_limits()
    assert asked.max_tokens == 32000 and asked.max_tokens_source is None
    assert asked.request_config() == before == plain.request_config()
    for m in (asked, plain):
        multi_call_episode(m, turns=2)
    assert strip(asked.client.requests) == strip(plain.client.requests)


def test_a_smaller_output_limit_of_the_provider_entry_wins(tmp_path):
    """A provider entry's max_output_tokens (models.registry.output_cap) below the gateway's cap stays in force."""
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"providers": {"local": {
        "kind": "openai_compatible", "base_url_env": "CHD_LOCAL_BASE_URL", "api_key_optional": True,
        "api_key_env": "CHD_LOCAL_API_KEY", "episode_headers": True, "x_gateway": True, "effort": None,
        "cost": {"source": "reported"}, "max_output_tokens": {"qwen": 8000}}}}))
    m = adapter(FakeGateway([], models=caps(max_output_tokens=81920)), providers=str(path))
    m.resolve_limits()
    assert m.max_tokens == 8000 and m.max_tokens_source is None


def test_claude_and_openrouter_requests_do_not_change():
    """Only LocalModel asks a gateway: claude-* and openrouter: adapters have no output cap step."""
    env = {"ANTHROPIC_API_KEY": "k", "OPENROUTER_API_KEY": "k"}
    for name in ("claude-opus-5-5", "openrouter:qwen/qwen3.8-27b"):
        m = registry.make_adapter(name, environ=env)
        before = m.request_config()
        m.resolve_limits()
        assert m.request_config() == before and m.max_tokens == 32000
        assert "max_tokens_source" not in before


def test_the_dry_run_model_has_no_output_cap_step():
    m = make_model("x", "medium", True)
    m.resolve_limits()
    assert "max_tokens_source" not in m.request_config()


def length_reply(output_tokens):
    return ChatCompletion.model_validate({
        "id": "r", "object": "chat.completion", "created": 0, "model": "qwen",
        "usage": {"prompt_tokens": 200000, "completion_tokens": output_tokens, "total_tokens": 200000 + output_tokens,
                  "cost": 0.0, "cost_source": "zero_api_charge"},
        "x_gateway": {"revision": "abc"},
        "choices": [{"index": 0, "finish_reason": "length", "message": {"role": "assistant", "content": "..."}}]})


@pytest.mark.parametrize("output,lowered", [(62144, True), (81920, False)])
def test_a_reply_cut_before_max_tokens_is_marked(output, lowered):
    """vLLM 0.32.0 sends min(max_tokens, max-model-len minus the prompt) to the engine and says nothing; a length stop
    with fewer output tokens than max_tokens is marked in the turn's transcript extra."""
    fake = FakeGateway([length_reply(output)], models=caps(max_output_tokens=81920))
    m = adapter(fake)
    m.resolve_limits()
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    t = m.step()
    assert t.stop == "max_tokens"
    if lowered:
        assert t.extra["max_tokens_lowered"] == {"requested": 81920, "output_tokens": output}
    else:
        assert "max_tokens_lowered" not in t.extra
