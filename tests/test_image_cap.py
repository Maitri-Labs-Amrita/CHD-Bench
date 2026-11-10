"""The image cap (models.base max_images; run-episode --max-images; a local gateway's capabilities.max_images, read from
GET /v1/models by models.local.LocalModel): a request never holds more than the cap, the newest images are the ones
kept, and without a cap nothing changes. Fake clients only."""

import json

import anyio
import pytest
from openai.types.chat import ChatCompletion

from chd_client.models import registry
from chd_client.models.anthropic import AnthropicModel
from chd_client.models.local import LocalModel, OpenAICompatibleModel
from chd_client.models.openrouter import OpenRouterModel
from chd_client.run_episode import IMAGE_NOTE, make_model, parser, run_episode
from tests.fakes import CASE, make_env
from tests.test_anthropic import TOOLS, FakeAnthropic, message, obs, tool_use
from tests.test_openrouter import FakeOpenAI

GATEWAY = "http://127.0.0.1:8098/v1"


def reply(calls, tag=None, cost=0.0, n=[0]):
    n[0] += 1
    tag = n[0] if tag is None else tag
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": f"c{tag}-{i}", "type": "function",
                           "function": {"name": name, "arguments": json.dumps(args)}}
                          for i, (name, args) in enumerate(calls)]}
    return ChatCompletion.model_validate({
        "id": f"r-{n[0]}", "object": "chat.completion", "created": 0, "model": "qwen",
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12, "cost": cost,
                  "cost_source": "zero_api_charge"},
        "x_gateway": {"revision": "abc"},
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}]})


class FakeGateway(FakeOpenAI):
    """A chd-gateway client: chat completions, and GET /v1/models with the profiles' capabilities (or none)."""

    def __init__(self, responses, models=None, fail=None):
        super().__init__(responses)
        outer, self.model_gets = self, 0

        class Models:
            def list(self):
                outer.model_gets += 1
                if fail is not None:
                    raise fail
                return type("Page", (), {"data": models or []})()
        self.models = Models()


def caps(name="qwen", max_images=None):
    return [{"id": name, "object": "model", "created": 0, "owned_by": "chd-gateway",
             "x_gateway": {"capabilities": {"images": True, "tools": True, "max_images": max_images}}}]


def n_images(req) -> int:
    return sum(1 for m in req["messages"] if isinstance(m.get("content"), list)
               for p in m["content"] if p.get("type") in ("image_url", "image"))


def multi_call_episode(m, turns=4):
    """Each turn: two tool results with 3 images each (one observation turn, 6 images)."""
    m.start(["sys"], TOOLS, "Begin.", obs(0, ("3d", "axial")))
    for k in range(turns):
        t = m.step()
        m.add_results([(c, obs(10 * k + j + 1, ("3d", "axial", "coronal"))) for j, c in enumerate(t.calls)])


def two_calls(k):
    return reply([("capture_3d", {}), ("get_state", {})], tag=k)


def test_a_capped_request_sends_only_the_newest_images():
    fake = FakeOpenAI([two_calls(k) for k in range(5)])
    m = OpenAICompatibleModel("x", client=fake, base_url=GATEWAY, sleep=lambda s: None)
    m.max_images, m.max_images_source = 4, "option"
    multi_call_episode(m, turns=5)
    assert [n_images(r) for r in fake.requests] == [2, 4, 4, 4, 4]
    last = fake.requests[-1]["messages"]
    labels = [p["text"] for msg in last if isinstance(msg.get("content"), list) for p in msg["content"]
              if p.get("type") == "text" and p["text"].startswith("[")]
    kept = [t for t in labels if not t.startswith("[image omitted")]
    assert kept == ["[capture_3d result, step 31, coronal]", "[get_state result, step 32, 3d]",
                    "[get_state result, step 32, axial]", "[get_state result, step 32, coronal]"]
    assert "[image omitted from context: step 31, axial]" in labels
    assert m.request_config()["max_images"] == 4 and m.request_config()["max_images_source"] == "option"
    assert len(m.image_substitutions()) == 2 + 5 * 6 - 4


def test_the_window_and_the_cap_together():
    fake = FakeOpenAI([two_calls(k) for k in range(4)])
    m = OpenRouterModel("vendor/x", client=fake, image_window=1, sleep=lambda s: None)
    m.stream = False
    m.max_images = 5
    multi_call_episode(m, turns=4)
    assert [n_images(r) for r in fake.requests] == [2, 5, 5, 5]


def test_anthropic_cap_counts_images_and_lets_stale_thinking_go():
    fake = FakeAnthropic([message([tool_use(f"t{k}a", "capture_3d"), tool_use(f"t{k}b")]) for k in range(3)])
    m = AnthropicModel(client=fake)
    m.max_images, m.max_images_source = 3, "option"
    multi_call_episode(m, turns=3)
    counts = [sum(1 for msg in r["messages"] if msg["role"] == "user" for b in msg["content"]
                  for c in (b["content"] if b.get("type") == "tool_result" else [b]) if c.get("type") == "image")
              for r in fake.requests]
    assert counts == [2, 3, 3]
    assert fake.requests[-1]["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert m.request_config()["max_images"] == 3


def local(fake, **kw):
    return LocalModel("qwen", client=fake, base_url=GATEWAY, sleep=lambda s: None, **kw)


def test_local_reads_the_gateways_cap_once_and_applies_it():
    fake = FakeGateway([two_calls(k) for k in range(4)], models=caps(max_images=4))
    m = local(fake)
    m.resolve_limits()
    m.resolve_limits()
    assert fake.model_gets == 1 and m.max_images == 4 and m.max_images_source == "gateway"
    multi_call_episode(m, turns=4)
    assert max(n_images(r) for r in fake.requests) == 4
    assert m.request_config()["max_images"] == 4 and m.request_config()["max_images_source"] == "gateway"


def test_a_max_images_option_below_the_gateways_cap_wins_and_above_it_is_refused():
    m = local(FakeGateway([], models=caps(max_images=16)))
    m.max_images, m.max_images_source = 8, "option"
    m.resolve_limits()
    assert m.max_images == 8 and m.max_images_source == "option"
    m = local(FakeGateway([], models=caps(max_images=16)))
    m.max_images = 20
    with pytest.raises(registry.ProviderError, match="--max-images 20 is above the gateway's cap .* 16"):
        m.resolve_limits()


@pytest.mark.parametrize("models", [caps(max_images=None), [{"id": "qwen", "object": "model"}], caps("other", 4), []])
def test_no_cap_reported_changes_nothing(models):
    """A profile without max_images, a gateway that sends no x_gateway, another model's cap, or no models: requests
    and request_config are those of an adapter that never asked."""
    asked = local(FakeGateway([two_calls(k) for k in range(3)], models=models))
    plain = local(FakeOpenAI([two_calls(k) for k in range(3)]))
    asked.resolve_limits()
    assert asked.max_images is None
    for m in (asked, plain):
        multi_call_episode(m, turns=3)
    def strip(rs):
        return [{**r, "extra_headers": {k: v for k, v in r["extra_headers"].items() if k != "X-Request-Id"}} for r in rs]
    assert strip(asked.client.requests) == strip(plain.client.requests)
    assert asked.request_config() == plain.request_config() and "max_images" not in asked.request_config()
    assert n_images(asked.client.requests[-1]) == 2 + 2 * 6


def test_a_gateway_that_cannot_be_asked_is_warned_about(capsys):
    m = local(FakeGateway([], fail=ConnectionError("refused")))
    m.resolve_limits()
    assert m.max_images is None and "no image cap and no output cap from the gateway" in capsys.readouterr().err


def _episode(tmp_path, model, steps):
    srv, _, traces = make_env(tmp_path)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, max_steps=steps))
    srv.session.close("abandoned")
    return summary, tmp_path / "runs" / summary["episode_id"]


def test_an_episode_with_more_images_than_the_gateways_cap_sends_at_most_the_cap(tmp_path):
    orbit = [("orbit", {"azimuth_deg": 10}), ("orbit", {"azimuth_deg": 20})]
    fake = FakeGateway([reply(orbit) for _ in range(4)], models=caps(max_images=6))
    summary, d = _episode(tmp_path, local(fake), steps=8)
    assert summary["stop_reason"] == "step_budget" and len(fake.requests) == 4
    counts = [n_images(r) for r in fake.requests]
    assert max(counts) == 6 and counts[-1] == 6
    lines = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines()]
    assert [len(x["images"]) for x in lines] == counts and all(x["max_images"] == 6 for x in lines)
    assert len(lines[-1]["placeholders"]) > 0
    assert IMAGE_NOTE.strip() in fake.requests[0]["messages"][0]["content"]
    prov = json.loads((d / "provenance" / "provenance.json").read_text())
    assert prov["request"]["max_images"] == 6 and prov["request"]["max_images_source"] == "gateway"
    cm = json.loads((d / "summary.json").read_text())["client_meta"]
    assert cm["max_images"] == 6 and cm["image_window"] is None


def test_an_uncapped_episode_has_no_cap_anywhere(tmp_path):
    fake = FakeGateway([reply([("orbit", {"azimuth_deg": 10})]) for _ in range(2)], models=caps(max_images=None))
    summary, d = _episode(tmp_path, local(fake), steps=2)
    assert "max_images" not in (d / "requests.jsonl").read_text()
    assert IMAGE_NOTE.strip() not in fake.requests[0]["messages"][0]["content"]


def test_the_max_images_option(capsys):
    a = parser().parse_args(["--imagechd", "1", "--labels", "gt", "--max-images", "4"])
    assert a.max_images == 4
    with pytest.raises(SystemExit):
        parser().parse_args(["--imagechd", "1", "--labels", "gt", "--max-images", "0"])
    assert "--max-images must be 1 or more" in capsys.readouterr().err
    dry = make_model("x", "medium", True, max_images=3)
    assert dry.max_images == 3 and dry.request_config()["max_images"] == 3
    assert "max_images" not in make_model("x", "medium", True).request_config()


def test_run_sh_passes_max_images_to_the_episode_and_checks_it_against_the_gateway(tmp_path, monkeypatch):
    from chd_client import run_sh
    monkeypatch.setattr(LocalModel, "_gateway_capabilities", lambda self: {"max_images": 16})
    env = {"PATH": "/usr/bin", "CHD_LOCAL_BASE_URL": GATEWAY}
    run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--model", "local:qwen", "--max-images", "8"], env,
                   env_path=tmp_path / "none")
    with pytest.raises(run_sh.RunShError, match="--max-images 20 is above the gateway's cap"):
        run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--model", "local:qwen", "--max-images", "20"], env,
                       env_path=tmp_path / "none")


def one_image_episode(m, turns):
    """One image per observation turn: the initial one, then one tool result per turn."""
    m.start(["sys"], TOOLS, "Begin.", obs(0, ("3d",)))
    for k in range(turns):
        t = m.step()
        m.add_results([(c, obs(k + 1, ("3d",))) for c in t.calls])


def _capped(block):
    fake = FakeOpenAI([reply([("capture_3d", {})], tag=f"b{block}-{k}") for k in range(8)])
    m = OpenAICompatibleModel("x", client=fake, base_url=GATEWAY, sleep=lambda s: None)
    m.max_images, m.max_images_source, m.image_cap_block = 4, "option", block
    one_image_episode(m, turns=8)
    return fake.requests, m


def test_a_block_cap_drops_the_oldest_images_a_block_at_a_time():
    reqs, m = _capped(2)
    assert [n_images(r) for r in reqs] == [1, 2, 3, 4, 3, 4, 3, 4]
    assert m.request_config()["image_cap_block"] == 2
    assert [n_images(r) for r in _capped(None)[0]] == [1, 2, 3, 4, 4, 4, 4, 4]


def test_between_drops_a_block_capped_request_only_grows_at_its_end():
    """The prompt-cache property: request 6 starts with all of request 5 (same cut), which one-by-one never gives."""
    reqs, _ = _capped(2)
    assert reqs[5]["messages"][:len(reqs[4]["messages"])] == reqs[4]["messages"]
    assert reqs[7]["messages"][:len(reqs[6]["messages"])] == reqs[6]["messages"]
    one = _capped(None)[0]
    assert one[5]["messages"][:len(one[4]["messages"])] != one[4]["messages"]


def test_no_block_and_a_default_note_change_nothing():
    _, m = _capped(None)
    assert "image_cap_block" not in m.request_config() and "image_note" not in m.request_config()


def test_no_image_note_keeps_the_uncapped_system_prompt():
    from chd_client.run_episode import PROMPT_DIR, render_system
    plain = render_system(PROMPT_DIR, 75)
    assert IMAGE_NOTE.strip() in render_system(PROMPT_DIR, 75, max_images=96)
    assert render_system(PROMPT_DIR, 75, max_images=96, image_note=False) == plain
    assert render_system(PROMPT_DIR, 75, image_window=3, image_note=False) == plain


def test_the_block_and_note_options():
    a = parser().parse_args(["--imagechd", "1", "--labels", "gt", "--max-images", "96", "--image-cap-block", "48",
                             "--no-image-note"])
    assert a.image_cap_block == 48 and a.no_image_note
    b = parser().parse_args(["--imagechd", "1", "--labels", "gt"])
    assert b.image_cap_block is None and not b.no_image_note
