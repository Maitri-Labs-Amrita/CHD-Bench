"""LocalModel (models/local.py) against a fake OpenAI-compatible client: the gateway's headers, no OpenRouter fields,
the request log's provider, and the cost rules (0.0 accepted, missing stops the run)."""

import json

import anyio
from openai.types.chat import ChatCompletion

from chd_client.models.local import LocalModel
from chd_client.run_episode import run_episode
from chd_client.run_record import RequestRecorder
from tests.fakes import CASE, make_env
from tests.test_anthropic import TOOLS, obs
from tests.test_openrouter import FakeOpenAI, _conn_error


def completion(calls=(("c1", "get_state", {}),), cost=0.0, n=[0]):
    n[0] += 1
    usage = {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930, "cost_source": "zero_api_charge"}
    if cost is not None:
        usage["cost"] = cost
    msg = {"role": "assistant", "content": "",
           "tool_calls": [{"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
                          for cid, name, args in calls]}
    return ChatCompletion.model_validate({
        "id": f"gw-{n[0]}", "object": "chat.completion", "created": 0, "model": "qwen", "usage": usage,
        "x_gateway": {"revision": "abc123", "engine": "custom", "interventions": []},
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": msg}]})


def model(responses, **kw):
    fake = FakeOpenAI(responses)
    m = LocalModel("qwen", client=fake, sleep=lambda s: None, **kw)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    return m, fake


def test_headers_carry_the_episode_id():
    m, fake = model([completion()])
    m.episode_id = "ep-1"
    m.step()
    h = fake.requests[0]["extra_headers"]
    assert h["X-Episode-Id"] == "ep-1" and h["X-Turn-Index"] == "1" and len(h["X-Request-Id"]) == 32
    m2, fake2 = model([completion()])
    m2.step()
    assert "X-Episode-Id" not in fake2.requests[0]["extra_headers"]


def test_request_id_is_stable_across_retries_and_new_per_turn():
    m, fake = model([_conn_error(), _conn_error(), completion(), completion()])
    m.episode_id = "ep-1"
    t = m.step()
    ids = [r["extra_headers"]["X-Request-Id"] for r in fake.requests]
    assert len(ids) == 3 and len(set(ids)) == 1 and m.retries == [(1, "APIConnectionError"), (2, "APIConnectionError")]
    m.add_results([(t.calls[0], obs(1))])
    m.step()
    nxt = fake.requests[-1]["extra_headers"]
    assert nxt["X-Request-Id"] != ids[0] and nxt["X-Turn-Index"] == "2"


def test_extra_body_is_empty_and_provenance_matches():
    m, fake = model([completion()], image_window=3)
    m.step()
    assert fake.requests[0]["extra_body"] == {} and m.request_config()["extra_body"] == {}
    assert m.effort is None and "reasoning" not in json.dumps(fake.requests[0]["extra_body"])


def test_recorded_provider_is_local_and_gateway_details_reach_the_turn(tmp_path):
    m, _ = model([completion()])
    m.recorder = RequestRecorder(tmp_path)
    t = m.step()
    req = json.loads((tmp_path / "requests.jsonl").read_text().splitlines()[0])
    assert req["provider"] == "local" and "X-Request-Id" in req["params"]["extra_headers"]
    assert t.extra["x_gateway"]["engine"] == "custom" and t.extra["cost_source"] == "zero_api_charge"


def _run(tmp_path, responses):
    tmp_path.mkdir()
    srv, _, traces = make_env(tmp_path)
    m = LocalModel("qwen", client=FakeOpenAI(responses), sleep=lambda s: None)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, max_steps=2))
    srv.session.close("abandoned")
    return summary, m


def test_zero_cost_is_accepted_and_missing_cost_stops_the_run(tmp_path):
    summary, m = _run(tmp_path / "a", [completion(), completion()])
    assert summary["stop_reason"] == "step_budget" and summary["n_model_steps"] == 2
    assert m.episode_id == summary["episode_id"]
    summary, _ = _run(tmp_path / "b", [completion(cost=None)])
    assert summary["stop_reason"] == "cost_unknown" and summary["n_model_steps"] == 0
    for k, bad in enumerate((float("nan"), float("inf"), -1.0)):
        summary, _ = _run(tmp_path / f"c{k}", [completion(cost=bad), completion()])
        assert summary["stop_reason"] == "cost_unknown" and summary["n_model_steps"] == 0


def test_a_missing_gateway_url_says_what_to_set(monkeypatch):
    import pytest
    monkeypatch.delenv("CHD_LOCAL_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="CHD_LOCAL_BASE_URL"):
        LocalModel("qwen")


def test_provenance_names_the_gateway_and_turns_restart_per_episode():
    fake = FakeOpenAI([completion(), completion()])
    m = LocalModel("qwen", client=fake, base_url="http://user:secret@127.0.0.1:8000/v1?k=x", sleep=lambda s: None)
    assert m.request_config()["base_url"] == "http://127.0.0.1:8000/v1"
    for _ in range(2):
        m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
        m.step()
    assert [r["extra_headers"]["X-Turn-Index"] for r in fake.requests] == ["1", "1"]


def test_a_reply_without_a_model_revision_is_marked(capsys):
    r = completion()
    r.x_gateway = {"engine": "custom"}
    m, _ = model([r])
    assert m.step().extra["gateway_revision_missing"] is True
    assert "sent no model revision" in capsys.readouterr().err


def test_requests_are_not_streamed():
    m, fake = model([completion()])
    m.step()
    assert "stream" not in fake.requests[0] and m.request_config()["stream"] is False


def _gateway_reply(status, code, message, calls, max_retries=2):
    """An openai SDK client whose every request gets this gateway error reply (the gateway's own body and header)."""
    import httpx2
    import openai

    from chd_client.gateway import normalize

    def handler(request):
        calls.append(request)
        return httpx2.Response(status, json=normalize.error(status, message, code, {"profile": "qwen"}),
                               headers={"x-should-retry": "false"} if code == "worker_start_failed" else {})

    return openai.OpenAI(base_url="http://127.0.0.1:8098/v1", api_key="local", max_retries=max_retries,
                         http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def _sdk_error(status, code, message):
    """The exception the openai SDK raises for this gateway reply."""
    client = _gateway_reply(status, code, message, [], max_retries=0)
    try:
        client.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "x"}])
    except Exception as e:  # noqa: BLE001
        return e
    raise AssertionError("no error")


START_MSG = ("qwen: a model worker could not start: worker w-1 ended before it was ready; worker log: "
             "/w/pools/qwen/logs/w-1.log (it ends: ModuleNotFoundError: No module named 'torchvision')")


def test_a_failed_worker_start_is_raised_at_once_with_the_gateway_message():
    import pytest

    from chd_client.models.local import GatewayWorkerStartError
    calls, waits = [], []
    m = LocalModel("qwen", client=_gateway_reply(503, "worker_start_failed", START_MSG, calls), sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(GatewayWorkerStartError, match="logs/w-1.log") as e:
        m.step()
    assert len(calls) == 1 and waits == [] and m.retries == []
    assert "No module named 'torchvision'" in str(e.value) and "cannot start a worker for 'qwen'" in str(e.value)


def test_other_gateway_503s_are_still_retried_and_openrouter_is_unchanged():
    import openai
    import pytest

    from chd_client.models.openrouter import RETRY_WAITS_S, OpenRouterModel
    calls, waits = [], []
    m = LocalModel("qwen", client=_gateway_reply(503, "no_worker", "no qwen worker was ready", calls, max_retries=0),
                   sleep=waits.append)
    m.start(["sys"], TOOLS, "Begin.", obs(0, ["3d"]))
    with pytest.raises(openai.InternalServerError):
        m.step()
    assert len(calls) == len(RETRY_WAITS_S) + 1 and waits == list(RETRY_WAITS_S)
    e = _sdk_error(503, "worker_start_failed", START_MSG)
    assert OpenRouterModel._retryable(None, e) is True
    assert m._retryable(e) is False and m._retryable(_sdk_error(503, "no_worker", "x")) is True


def test_the_runner_stops_with_runner_exception_and_the_message(tmp_path):
    import pytest

    from chd_client.models.local import GatewayWorkerStartError
    from chd_client.run_episode import summary_of
    srv, _, traces = make_env(tmp_path)
    m = LocalModel("qwen", client=FakeOpenAI([_sdk_error(503, "worker_start_failed", START_MSG)]),
                   sleep=lambda s: None)
    try:
        with pytest.raises(GatewayWorkerStartError) as e:
            anyio.run(lambda: run_episode(CASE, "3d_seg", m, srv.server, runs_dir=tmp_path / "runs",
                                          traces_dir=traces, echo=None, max_steps=2))
    finally:
        srv.session.close("abandoned")
    s = summary_of(e.value)
    assert s["stop_reason"] == "runner_exception" and s["n_model_turns"] == 0
    assert "GatewayWorkerStartError" in s["exception"] and "logs/w-1.log" in s["exception"]
