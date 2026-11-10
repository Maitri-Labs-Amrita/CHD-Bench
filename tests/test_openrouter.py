"""OpenRouterModel request building and response parsing against a fake OpenAI-compatible client."""

import json

import anyio
import pytest
from openai.types.chat import ChatCompletion

from chd_client.models.openrouter import OpenRouterModel, _reported_cost
from chd_client.run_episode import run_episode
from tests.fakes import CASE, make_env
from tests.test_anthropic import TOOLS, obs
from tests.test_runner import SUBMIT


class FakeOpenAI:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        outer = self

        class Completions:
            def create(self, **kw):
                outer.requests.append(kw)
                r = outer.responses.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

        class Chat:
            completions = Completions()
        self.chat = Chat()


def completion(calls=(), content="", finish="tool_calls", cost=0.003, reasoning=None, n=[0]):
    n[0] += 1
    msg = {"role": "assistant", "content": content,
           "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                          for cid, name, args in calls] or None}
    if reasoning:
        msg["reasoning"] = reasoning
    usage = {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930,
             "prompt_tokens_details": {"cached_tokens": 400}}
    if cost is not None:
        usage["cost"] = cost
    return ChatCompletion.model_validate({"id": f"gen-{n[0]}", "object": "chat.completion", "created": 0,
                                          "model": "vendor/model-x", "usage": usage,
                                          "choices": [{"index": 0, "finish_reason": finish, "message": msg}]})


def test_request_shape_images_after_tool_messages_and_window():
    responses = [completion([(f"c{k}", "capture_3d", {})], content=f"look {k}", reasoning="r" if k == 0 else None)
                 for k in range(5)]
    fake = FakeOpenAI(responses)
    m = OpenRouterModel("vendor/model-x", client=fake, image_window=3)
    m.start(["SYS", "INSTR"], TOOLS, "first", obs(0, ("3d", "axial")))
    t = m.step()
    req = fake.requests[0]
    assert req["model"] == "vendor/model-x" and req["tool_choice"] == "auto"
    assert req["extra_body"] == {"usage": {"include": True}}
    assert req["tools"][0] == {"type": "function", "function": {"name": "get_state", "description": "Return the state.",
                                                                "parameters": TOOLS[0].input_schema}}
    assert req["messages"][0] == {"role": "system", "content": "SYS\n\nINSTR"}
    first = req["messages"][1]["content"]
    assert first[0]["text"] == "first" and first[1]["text"] == obs(0, ("3d", "axial")).text
    assert [p["type"] for p in first[2:]] == ["text", "image_url"] * 2
    assert first[3]["image_url"]["url"].startswith("data:image/png;base64,")
    assert t.calls[0].id == "c0" and t.calls[0].name == "capture_3d" and t.calls[0].reasoning == "r\n\nlook 0"
    assert (t.usage.input, t.usage.output, t.usage.cache_read, t.usage.cost_usd) == (900, 30, 400, 0.003)
    assert t.stop == "tool_use"
    for k in range(1, 5):
        m.add_results([(t.calls[0], obs(k))])
        t = m.step()
    msgs = fake.requests[-1]["messages"]
    tool_msgs = [x for x in msgs if x["role"] == "tool"]
    assert [x["tool_call_id"] for x in tool_msgs] == ["c0", "c1", "c2", "c3"]
    assert tool_msgs[0]["content"] == obs(1).text
    i = msgs.index(tool_msgs[-1])
    assert msgs[i + 1]["role"] == "user" and msgs[i + 1]["content"][0]["text"] == "[capture_3d result, step 4, 3d]"
    assert msgs[i - 1]["tool_calls"][0]["id"] == "c3"
    imgs = [x for x in msgs if x["role"] == "user" and any(p["type"] == "image_url" for p in x["content"])]
    assert len(imgs) == 3
    assert msgs[1]["content"][2]["text"] == "[image omitted from context: step 0, 3d]"


@pytest.mark.parametrize("cost", [None, float("nan"), float("inf"), float("-inf"), -1.0, "n/a", True])
def test_missing_or_invalid_cost_stops_the_runner(tmp_path, cost):
    """A cost that is missing, not a number, not finite or below 0 is unknown: a NaN cost would turn off every cap
    comparison and a negative one would add headroom. The turn's calls do not run, not even submit_report."""
    srv, _, traces = make_env(tmp_path)
    calls = [("c1", "reset_view", {}), ("c2", *SUBMIT)]
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([completion(calls, cost=cost),
                                                             completion([("c3", "get_state", {})], cost=0.001)]))
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, cost_cap_usd=1.0))
    assert summary["stop_reason"] == "cost_unknown" and summary["n_model_steps"] == 0
    assert summary["report"] is None and summary["cost_usd"] == 0.0
    srv.session.close("abandoned")


@pytest.mark.parametrize("cost, usd", [(0, 0.0), (0.003, 0.003), ("0.003", 0.003), (float("nan"), None),
                                       (float("inf"), None), (-0.001, None), ("abc", None), ([1], None),
                                       (False, None)])
def test_reported_cost_parsing(cost, usd):
    assert _reported_cost(cost) == usd


def test_text_only_and_content_filter():
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([completion(content="hmm", finish="stop"),
                                                             completion(finish="content_filter")]))
    m.start(["S"], TOOLS, "first", obs(0))
    t = m.step()
    assert t.calls == [] and t.stop == "end_turn" and t.text == "hmm"
    m.add_user_text("nudge")
    assert m.step().stop == "refusal"
    assert m.client.requests[-1]["messages"][-1] == {"role": "user", "content": "nudge"}


def _marks(req):
    """Indices of the messages whose last content part carries cache_control."""
    return [i for i, m in enumerate(req["messages"])
            if isinstance(m["content"], list) and m["content"][-1].get("cache_control") == {"type": "ephemeral"}]


def test_claude_is_cached_and_pinned_to_anthropic():
    responses = [completion([(f"c{k}", "capture_3d", {})]) for k in range(5)]
    fake = FakeOpenAI(responses)
    m = OpenRouterModel("anthropic/claude-fable-5.1", client=fake, image_window=1)
    m.start(["SYS", "INSTR"], TOOLS, "first", obs(0, ("3d", "axial")))
    for k in range(4):
        t = m.step()
        m.add_results([(t.calls[0], obs(k + 1, ("3d",)))])
    m.step()
    for req in fake.requests:
        assert req["extra_body"]["provider"] == {"order": ["amazon-bedrock", "google-vertex"],
                                                  "allow_fallbacks": False}
        marks = _marks(req)
        assert marks[0] == 0 and marks[-1] == len(req["messages"]) - 1 and len(marks) <= 3
    last = fake.requests[-1]
    assert len(_marks(last)) == 3
    assert last["messages"][0]["content"][0]["text"].startswith("SYS")


def test_other_models_are_not_marked():
    fake = FakeOpenAI([completion([("c0", "capture_3d", {})])])
    m = OpenRouterModel("openai/gpt-6-luna", client=fake)
    m.start(["SYS", "INSTR"], TOOLS, "first", obs(0, ("3d",)))
    m.step()
    req = fake.requests[0]
    assert _marks(req) == [] and "provider" not in req["extra_body"]
    assert isinstance(req["messages"][0]["content"], str)


def _conn_error():
    import httpx2
    import openai
    return openai.APIConnectionError(request=httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


def test_a_network_drop_is_retried_with_backoff():
    waits = []
    fake = FakeOpenAI([_conn_error(), _conn_error(), completion([("c1", "orbit", {"azimuth_deg": 30})])])
    m = OpenRouterModel("vendor/model-x", client=fake, sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    turn = m.step()
    assert [c.name for c in turn.calls] == ["orbit"] and waits == [5, 10] and len(fake.requests) == 3
    assert m.retries == [(1, "APIConnectionError"), (2, "APIConnectionError")]


def test_retries_end_and_a_bad_request_is_not_retried():
    import httpx2
    import openai
    import pytest
    from chd_client.models.openrouter import RETRY_WAITS_S
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([_conn_error()] * (len(RETRY_WAITS_S) + 1)), sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(openai.APIConnectionError):
        m.step()
    bad = openai.BadRequestError("bad", response=httpx2.Response(400, request=httpx2.Request("POST", "https://x")),
                                 body=None)
    fake = FakeOpenAI([bad])
    m = OpenRouterModel("vendor/model-x", client=fake, sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(openai.BadRequestError):
        m.step()
    assert len(fake.requests) == 1


def test_client_waits_long_and_does_not_retry_silently(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    from chd_client.models.openrouter import REQUEST_TIMEOUT_S, OpenRouterModel
    m = OpenRouterModel("anthropic/claude-opus-5.5", effort="max")
    assert m.client.timeout == REQUEST_TIMEOUT_S >= 1800 and m.client.max_retries == 0


def test_rate_limit_is_retried():
    import httpx, openai
    from chd_client.models.openrouter import _transient
    r = httpx.Response(429, request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
    assert _transient(openai.RateLimitError("429", response=r, body=None))
    r402 = httpx.Response(402, request=r.request)
    assert not _transient(openai.APIStatusError("402", response=r402, body=None))


def _chunk(delta=None, finish=None, usage=None, error=None, model="anthropic/claude-sonnet-5.5"):
    c = {"id": "gen-s", "object": "chat.completion.chunk", "created": 1, "model": model,
         "choices": [] if delta is None and finish is None else [{"index": 0, "delta": delta or {},
                                                                   "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    if error:
        c["error"] = error
    return c


def _stream_with_call():
    usage = {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930, "cost": 0.01,
             "prompt_tokens_details": {"cached_tokens": 400}}
    return iter([
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"reasoning": "think ", "reasoning_details": [{"type": "reasoning.text", "text": "think ", "index": 0}]}),
        _chunk({"reasoning": "more", "reasoning_details": [{"type": "reasoning.text", "text": "more", "index": 0}]}),
        _chunk({"reasoning_details": [{"type": "reasoning.text", "signature": "sig", "index": 0}]}),
        _chunk({"content": "Looking."}),
        _chunk({"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                "function": {"name": "orbit", "arguments": ""}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"azimuth'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '_deg": 30}'}}]}),
        _chunk(finish="tool_calls"),
        _chunk(usage=usage),
    ])


def test_a_streamed_reply_is_assembled():
    fake = FakeOpenAI([_stream_with_call()])
    m = OpenRouterModel("anthropic/claude-sonnet-5.5", client=fake, effort="max")
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    turn = m.step()
    req = fake.requests[0]
    assert req["stream"] is True and req["stream_options"] == {"include_usage": True}
    assert [(c.id, c.name, c.args) for c in turn.calls] == [("c1", "orbit", {"azimuth_deg": 30})]
    assert turn.text == "Looking." and turn.stop == "tool_use"
    assert turn.calls[0].reasoning == "think more\n\nLooking."
    assert turn.usage.cost_usd == 0.01 and turn.usage.cache_read == 400
    msg = m.history[-1][1]
    assert msg["reasoning_details"] == [{"type": "reasoning.text", "text": "think more", "index": 0,
                                         "signature": "sig"}]
    assert m.request_config()["stream"] is True


def _error_body(code, kind=None):
    from openai.types.chat import ChatCompletion
    err = {"message": "The operation was aborted", "code": code, **({"metadata": {"error_type": kind}} if kind else {})}
    return ChatCompletion.model_construct(id="gen-e", choices=None, created=None, model=None, object=None,
                                          usage=None, error=err)


def test_an_error_reply_is_retried():
    waits = []
    stream_err = iter([_chunk({"reasoning": "x"}), _chunk(error={"code": 502, "message": "upstream"})])
    fake = FakeOpenAI([_error_body(504, "timeout"), stream_err, _stream_with_call()])
    m = OpenRouterModel("anthropic/claude-sonnet-5.5", client=fake, sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    turn = m.step()
    assert [c.name for c in turn.calls] == ["orbit"] and waits == [5, 10] and len(fake.requests) == 3
    assert m.retries == [(1, "ErrorReply 504"), (2, "ErrorReply 502")]


def test_a_client_error_reply_is_not_retried():
    import pytest
    from chd_client.models.openrouter import ErrorReply
    fake = FakeOpenAI([_error_body(400)])
    m = OpenRouterModel("anthropic/claude-sonnet-5.5", client=fake, sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(ErrorReply):
        m.step()
    assert len(fake.requests) == 1


def test_a_stream_error_event_is_retried():
    import httpx2
    import openai
    from chd_client.models.openrouter import _transient
    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    assert _transient(openai.APIError("Network connection lost.", request=req, body=None))
    assert _transient(openai.APIError("x", request=req, body={"code": 502, "message": "x"}))
    assert not _transient(openai.APIError("x", request=req, body={"code": 400, "message": "x"}))
    waits = []
    fake = FakeOpenAI([openai.APIError("Network connection lost.", request=req, body=None), _stream_with_call()])
    m = OpenRouterModel("anthropic/claude-sonnet-5.5", client=fake, sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    assert [c.name for c in m.step().calls] == ["orbit"] and waits == [5]


def test_a_stream_that_ends_in_error_is_retried():
    waits = []
    bad = iter([_chunk({"content": "partial"}), _chunk(finish="error")])
    fake = FakeOpenAI([bad, _stream_with_call()])
    m = OpenRouterModel("moonshotai/kimi-k3", client=fake, sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    assert [c.name for c in m.step().calls] == ["orbit"] and waits == [5]
    assert m.retries == [(1, "ErrorReply None")]


def test_every_failed_attempt_is_logged(capsys):
    import httpx2
    import openai
    req = httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

    def dropped():
        yield _chunk({"reasoning": "a"})
        yield _chunk({"reasoning": "b"})
        raise openai.APIError("Network connection lost.", request=req, body=None)

    recorded = []
    fake = FakeOpenAI([dropped(), _stream_with_call()])
    m = OpenRouterModel("anthropic/claude-sonnet-5.5", client=fake, sleep=lambda s: None)
    m.record_response = recorded.append
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    m.step()
    fail = recorded[0]["failed_attempt"]
    assert fail["attempt"] == 1 and fail["type"] == "APIError" and fail["chunks"] == 2 and "elapsed_s" in fail
    assert "attempt 1 failed" in capsys.readouterr().err and "choices" in recorded[1]


def _rate_limit(retry_after=None):
    import httpx
    import openai
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    r = httpx.Response(429, headers=headers,
                       request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))
    return openai.RateLimitError("429", response=r, body=None)


def test_retry_after_lengthens_the_wait_and_never_shortens_it():
    from chd_client.models.openrouter import MAX_RETRY_AFTER_S
    waits = []
    fake = FakeOpenAI([_rate_limit(42), _rate_limit(1), _rate_limit(), _rate_limit(10_000), _stream_with_call()])
    m = OpenRouterModel("vendor/model-x", client=fake, sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    assert [c.name for c in m.step().calls] == ["orbit"]
    assert waits == [42, 10, 20, MAX_RETRY_AFTER_S] and MAX_RETRY_AFTER_S == 600


def test_retry_after_parsing():
    from chd_client.models.openrouter import _retry_after
    assert _retry_after(_rate_limit(7)) == 7.0 and _retry_after(_rate_limit()) is None
    assert _retry_after(_rate_limit("soon")) is None and _retry_after(RuntimeError("x")) is None


def test_a_rate_limit_after_every_retry_raises_rate_limited():
    import pytest
    from chd_client.models.openrouter import RETRY_WAITS_S, ErrorReply, RateLimited
    waits = []
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([_rate_limit()] * (len(RETRY_WAITS_S) + 1)),
                        sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(RateLimited, match=f"rate limited after {len(RETRY_WAITS_S) + 1} attempts"):
        m.step()
    assert waits == list(RETRY_WAITS_S)
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([_error_body(429)] * (len(RETRY_WAITS_S) + 1)),
                        sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(RateLimited):
        m.step()
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([_error_body(502)] * (len(RETRY_WAITS_S) + 1)),
                        sleep=lambda s: None)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(ErrorReply):
        m.step()


def test_a_rate_limited_episode_stops_rate_limited_and_is_unfinished(tmp_path):
    import importlib.util
    from pathlib import Path
    from chd_client import run_folder
    from chd_client.models.openrouter import RETRY_WAITS_S
    srv, _, traces = make_env(tmp_path)
    m = OpenRouterModel("vendor/model-x", client=FakeOpenAI([_rate_limit()] * (len(RETRY_WAITS_S) + 1)),
                        sleep=lambda s: None)
    try:
        summary = anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                                traces_dir=traces, echo=None))
    except BaseException as e:  # noqa: BLE001
        from chd_client.run_episode import summary_of
        summary = summary_of(e)
    assert summary["stop_reason"] == "rate_limited"
    assert "rate_limited" in run_folder.UNFINISHED_STOPS
    spec = importlib.util.spec_from_file_location("run_batch", Path(__file__).parents[1] / "slurm" / "run_batch.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    assert "rate_limited" in rb.INFRA_STOPS and not rb.case_done({"exit": 1, "stop": "rate_limited"})
