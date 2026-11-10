"""AnthropicModel request building and response parsing against a fake client (canned BetaMessage objects)."""

import json

import anyio
import pytest
from anthropic.types.beta import BetaMessage

from chd_client.mcp_session import Observation, ToolDef
from chd_client.models.anthropic import AnthropicModel, cost_usd
from chd_client.run_episode import run_episode
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, NORMAL, make_env, png

TOOLS = [ToolDef("get_state", "Return the state.", {"type": "object", "properties": {}, "additionalProperties": False}),
         ToolDef("capture_3d", "Return the 3D view.", {"type": "object", "properties": {}})]


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeAnthropic:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        outer = self

        class Messages:
            def stream(self, **kw):
                outer.requests.append(kw)
                return FakeStream(outer.responses.pop(0))

        class Beta:
            messages = Messages()
        self.beta = Beta()


def message(content, stop="tool_use", usage=None, n=[0]):
    n[0] += 1
    u = usage or {"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 1000,
                  "cache_read_input_tokens": 5000}
    return BetaMessage.model_validate({"id": f"msg_{n[0]}", "type": "message", "role": "assistant",
                                       "model": "claude-opus-5-5", "content": content, "stop_reason": stop,
                                       "stop_sequence": None, "usage": u})


def tool_use(id_, name="get_state", inp=None):
    return {"type": "tool_use", "id": id_, "name": name, "input": inp or {}}


def obs(step, frames=("3d",), tool="capture_3d"):
    payload = {"step_index": step, "state": {}, "result": {}, "warnings": [], "error": None, "frames": list(frames)}
    return Observation(tool, json.dumps(payload), payload, list(frames), [png(step)] * len(frames), step=step)


def images_in(msg):
    out = []
    for b in msg["content"]:
        for c in (b["content"] if b["type"] == "tool_result" else [b]):
            if c["type"] == "image":
                out.append(c)
    return out


def cache_marks(req):
    marks = [("system", i) for i, b in enumerate(req["system"]) if "cache_control" in b]
    for i, m in enumerate(req["messages"]):
        for j, b in enumerate(m["content"] if isinstance(m["content"], list) else []):
            if isinstance(b, dict) and "cache_control" in b:
                marks.append((i, j))
    return marks


def test_request_shape_reasoning_usage_and_cost():
    first = message([{"type": "thinking", "thinking": "Look at the state first.", "signature": "sig1"},
                     {"type": "text", "text": "Checking the state."}, tool_use("tu_1"),
                     {"type": "thinking", "thinking": "", "signature": "sig2"},
                     {"type": "text", "text": "And the 3D view."}, tool_use("tu_2", "capture_3d")])
    fake = FakeAnthropic([first])
    m = AnthropicModel(client=fake)
    m.start(["SYSTEM PROMPT", "SERVER INSTRUCTIONS"], TOOLS, "first turn", obs(0, ("3d", "axial", "sagittal", "coronal")))
    turn = m.step()
    req = fake.requests[0]
    assert req["model"] == "claude-opus-5-5" and req["max_tokens"] == 32000
    assert req["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert req["output_config"] == {"effort": "medium"} and req["tool_choice"] == {"type": "auto"}
    assert "betas" not in req and "fallbacks" not in req
    assert req["tools"] == [{"name": t.name, "description": t.description, "input_schema": t.input_schema}
                            for t in TOOLS]
    assert [b["text"] for b in req["system"]] == ["SYSTEM PROMPT", "SERVER INSTRUCTIONS"]
    first_msg = req["messages"][0]
    assert first_msg["role"] == "user" and first_msg["content"][0]["text"] == "first turn"
    assert first_msg["content"][1]["text"] == obs(0).text.replace('["3d"]', '["3d", "axial", "sagittal", "coronal"]')
    assert len(images_in(first_msg)) == 4
    assert cache_marks(req) == [("system", 1), (0, 5)]

    assert [c.id for c in turn.calls] == ["tu_1", "tu_2"]
    assert turn.calls[0].reasoning == "Look at the state first.\n\nChecking the state."
    assert turn.calls[1].reasoning == "And the 3D view."
    assert turn.text == "Checking the state.And the 3D view." and turn.stop == "tool_use"
    u = turn.usage
    assert (u.input, u.output, u.cache_read, u.cache_write) == (6100, 20, 5000, 1000)
    assert u.cost_usd == pytest.approx((100 * 4 + 20 * 20 + 1000 * 5 + 5000 * 0.2) / 1e6)
    assert cost_usd("unknown-model", 1, 1, 0, 0) is None


def test_history_window_placeholders_pinning_and_cache_breakpoints():
    n_turns = 6
    fake = FakeAnthropic([message([tool_use(f"tu_{k}", "capture_3d")]) for k in range(n_turns + 1)])
    m = AnthropicModel(client=fake, image_window=3)
    m.start(["S"], TOOLS, "first", obs(0))
    responses = []
    for k in range(1, n_turns + 1):
        t = m.step()
        responses.append(m.history[-1][1])
        o = obs(k, frames=("3d_a", "3d_b") if k == 1 else ("3d",))
        m.add_results([(t.calls[0], o)])
    m.step()
    req = fake.requests[-1]
    msgs = req["messages"]
    assert [x["role"] for x in msgs] == ["user"] + ["assistant", "user"] * n_turns
    for i, r in enumerate(responses):
        assert msgs[2 * i + 1]["content"] is r.content
    with_images = [i // 2 for i, x in enumerate(msgs) if x["role"] == "user" and images_in(x)]
    assert with_images == [4, 5, 6]
    assert "[image omitted from context: step 1, 3d_b]" in json.dumps(msgs[2]["content"])
    t0 = msgs[0]["content"]
    assert t0[1]["text"] == obs(0).text and t0[2] == {"type": "text", "text": "[image omitted from context: step 0, 3d]"}
    tr = msgs[6]["content"][0]
    assert tr["type"] == "tool_result" and tr["tool_use_id"] == "tu_2" and tr["is_error"] is False
    assert tr["content"][0]["text"] == obs(3).text
    assert tr["content"][1]["text"] == "[image omitted from context: step 3, 3d]"
    assert cache_marks(req) == [("system", 0), (6, 0), (12, 0)]
    assert len(cache_marks(req)) <= 4


def test_parallel_results_share_one_user_message_and_nudge_appends():
    fake = FakeAnthropic([message([tool_use("a"), tool_use("b")]), message([{"type": "text", "text": "done?"}],
                                                                           stop="end_turn"), message([tool_use("c")])])
    m = AnthropicModel(client=fake)
    m.start(["S"], TOOLS, "first", obs(0))
    t = m.step()
    m.add_results([(t.calls[0], obs(1)), (t.calls[1], obs(2))])
    t2 = m.step()
    assert t2.calls == [] and t2.stop == "end_turn"
    m.add_user_text("nudge")
    m.step()
    msgs = fake.requests[-1]["messages"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a", "b"]
    assert msgs[-1] == {"role": "user", "content": [{"type": "text", "text": "nudge",
                                                     "cache_control": {"type": "ephemeral"}}]}
    m2 = AnthropicModel(client=FakeAnthropic([message([tool_use("a")]), message([tool_use("b")])]))
    m2.start(["S"], TOOLS, "first", obs(0))
    t = m2.step()
    m2.add_results([(t.calls[0], obs(1))])
    m2.add_user_text("note")
    m2.step()
    last = m2.client.requests[-1]["messages"][-1]
    assert last["content"][0]["type"] == "tool_result" and last["content"][1]["text"] == "note"


def run_in_process(tmp_path, responses, **kw):
    srv, fake, traces = make_env(tmp_path)
    model = AnthropicModel(client=FakeAnthropic(responses))
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, **kw))
    return srv, model, summary


def test_anthropic_episode_through_the_runner(tmp_path):
    responses = [
        message([{"type": "thinking", "thinking": "Start with the state.", "signature": "s"}, tool_use("t1")]),
        message([{"type": "text", "text": "Submitting."}, tool_use("t2", "submit_report", {"report": NORMAL})]),
    ]
    srv, model, summary = run_in_process(tmp_path, responses)
    meta, steps = read_trace(srv.session.trace_dir)
    assert summary["stop_reason"] == "submitted" and meta["status"] == "submitted"
    assert meta["model"]["id"] == "claude-opus-5-5" and meta["client_meta"]["effort"] == "medium"
    assert steps[1]["model_reasoning"] == "Start with the state." and steps[1]["tokens"] == {"in": 6100, "out": 20}
    assert steps[2]["model_reasoning"] == "Submitting."
    assert summary["tokens"] == {"in": 12200, "out": 40, "cache_read": 10000, "cache_write": 2000}
    req = model.client.requests[0]
    assert "reset_episode" not in {t["name"] for t in req["tools"]}
    assert len(req["system"]) == 1
    blob = json.dumps(req["system"]) + json.dumps(req["tools"]) + json.dumps(req["messages"])
    assert CASE not in blob and summary["episode_id"] not in blob


@pytest.mark.parametrize("stop,expected", [("refusal", "model_refusal"), ("max_tokens", "model_max_tokens")])
def test_refusal_and_max_tokens_abort(tmp_path, stop, expected):
    srv, model, summary = run_in_process(tmp_path, [message([{"type": "text", "text": "..."}], stop=stop)])
    assert summary["stop_reason"] == expected and summary["n_model_steps"] == 0
    srv.session.close("abandoned")


def test_default_keeps_every_image_append_only():
    """Default policy: every image stays; earlier messages are byte-identical request to request."""
    fake = FakeAnthropic([message([tool_use(f"tu_{k}", "capture_3d")]) for k in range(8)])
    m = AnthropicModel(client=fake)
    m.start(["S"], TOOLS, "first", obs(0))
    for k in range(1, 8):
        t = m.step()
        m.add_results([(t.calls[0], obs(k))])
    m.render()
    reqs = fake.requests
    assert all(len(images_in(msg)) > 0 for msg in reqs[-1]["messages"] if msg["role"] == "user")
    strip = lambda msgs: json.dumps([{**x, "content": [{k: v for k, v in b.items() if k != "cache_control"}
                                                       if isinstance(b, dict) else repr(b) for b in x["content"]]}
                                     for x in msgs], default=repr)
    for a, b in zip(reqs, reqs[1:]):
        assert strip(b["messages"][:len(a["messages"])]) == strip(a["messages"])
    assert "block_binding" not in reqs[-1]["thinking"]


def test_image_window_sets_the_binding_control():
    m = AnthropicModel(client=FakeAnthropic([]), image_window=3)
    m.start(["S"], TOOLS, "first", obs(0))
    req = m.render()
    assert req["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert req["betas"] == ["thinking-binding-controls-2026-08-01"]


def test_sonnet_has_a_price():
    from chd_client.models.anthropic import cost_usd
    assert cost_usd("claude-sonnet-5-5", 1_000_000, 1_000_000, 1_000_000, 1_000_000) == pytest.approx(2.0 + 10.0 + 2.5 + 0.1)
