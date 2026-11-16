"""run-episode --resume: an episode the infrastructure stopped continues from its last complete model turn
(chd_client.resume). Fake Slicer bridge, scripted model: no Slicer, no network."""

import json

import anyio
import pytest

from chd_client.models.scripted import ScriptedModel
from chd_client.resume import ResumeError, load_prior, split_turns
from chd_client.run_episode import run_episode
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, NORMAL, make_env

SUBMIT = ("submit_report", {"report": NORMAL})
TURNS = [[("reset_view", {})], [("orbit", {"azimuth_deg": 30}), ("zoom", {"factor": 1.5})], [],
         [("clear_clip", {})], [SUBMIT]]


def crash(_last):
    raise RuntimeError("the job ended")


def episode(tmp_path, script, resume=None, shift=0, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    srv, fake, traces = make_env(tmp_path)
    fake.size += shift
    model = ScriptedModel(script)
    try:
        summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                                traces_dir=traces, echo=None, make_pdf=False, resume=resume, **kw))
    except RuntimeError as e:
        if not hasattr(e, "summary"):
            raise
        summary = e.summary
    _, steps = read_trace(srv.session.trace_dir)
    return summary, model, steps, traces


def transcript(summary, tmp_path):
    return [json.loads(x) for x in (tmp_path / "runs" / summary["episode_id"] / "transcript.jsonl").read_text()
            .splitlines()]


def interrupted(tmp_path, **kw):
    summary, _, _, traces = episode(tmp_path / "a", TURNS[:4] + [crash], **kw)
    assert summary["stop_reason"] == "runner_exception" and summary["n_model_turns"] == 4
    return load_prior(tmp_path / "a" / "runs" / summary["episode_id"], traces)


def test_a_resumed_episode_ends_like_an_uninterrupted_one(tmp_path):
    whole, whole_model, whole_steps, _ = episode(tmp_path / "whole", TURNS)
    prior = interrupted(tmp_path)
    assert len(prior.turns) == 4 and prior.n_steps == 4
    summary, model, steps, _ = episode(tmp_path / "b", TURNS, resume=prior)
    assert summary["stop_reason"] == "submitted"
    for k in ("n_model_steps", "n_model_turns", "tokens", "cost_usd"):
        assert summary[k] == whole[k], k
    assert [s["tool"] for s in steps] == [s["tool"] for s in whole_steps]
    assert summary["resumed"] == {"from_episode": prior.episode_id, "from_run": str(prior.run_dir),
                                  "from_stop": "runner_exception", "replayed_turns": 4, "replayed_steps": 4,
                                  "dropped_records": prior.dropped}
    t = transcript(summary, tmp_path / "b")
    assert [r["kind"] for r in t].count("model_turn") == 5 and sum(r["kind"] == "resumed" for r in t) == 1
    assert [o.text for o in model.observations] == [o.text for o in whole_model.observations]
    assert model.texts == whole_model.texts


def test_calls_left_continue_from_the_earlier_count(tmp_path):
    prior = interrupted(tmp_path, max_steps=20)
    summary, _, _, _ = episode(tmp_path / "b", TURNS, resume=prior, max_steps=20)
    obs = [r for r in transcript(summary, tmp_path / "b") if r["kind"] == "observation" and r["step"]]
    assert [json.loads(r["text"])["calls_left"] for r in obs] == [19, 18, 17, 16, 15]


def test_a_different_frame_on_replay_stops_the_resume(tmp_path):
    prior = interrupted(tmp_path)
    summary, _, _, _ = episode(tmp_path / "b", TURNS, resume=prior, shift=7)
    assert summary["stop_reason"] == "resume_mismatch" and summary["n_model_turns"] == 0
    assert any(r["kind"] == "resume_mismatch" for r in transcript(summary, tmp_path / "b"))


def test_another_step_budget_is_another_experiment(tmp_path):
    prior = interrupted(tmp_path)
    with pytest.raises(BaseException) as e:
        episode(tmp_path / "b", TURNS, resume=prior, max_steps=33)
    assert "system_prompt_sha256" in repr(e.value)


def test_a_finished_episode_is_not_resumed(tmp_path):
    whole, _, _, traces = episode(tmp_path / "whole", TURNS)
    with pytest.raises(ResumeError, match="model's own result"):
        load_prior(tmp_path / "whole" / "runs" / whole["episode_id"], traces)


def test_split_turns_keeps_only_complete_turns():
    def obs(cid, step):
        return {"kind": "observation", "call_id": cid, "step": step}

    def turn(n, *ids, stop="tool_use"):
        return {"kind": "model_turn", "turn": n, "stop": stop, "calls": [{"id": i, "name": "orbit"} for i in ids]}
    head = [{"kind": "user_text"}, obs(None, 0)]
    recs = head + [turn(1, "a"), obs("a", 1), turn(2, "b", "c"), obs("b", 2)]
    first, turns, dropped = split_turns(recs)
    assert first == head and [t["turn"]["turn"] for t in turns] == [1] and dropped == 2
    no_nudge = head + [turn(1), turn(2, "a"), obs("a", 1)]
    assert split_turns(no_nudge)[1] == []
    cut = head + [turn(1, "a", stop="max_tokens"), obs("a", 1)]
    assert split_turns(cut)[1] == []


def test_a_local_models_history_is_rebuilt_exactly_from_its_recorded_reply():
    """restore(raw) puts a recorded reply into the history as the live step did, vendor extras and reasoning included:
    the next request after a resume is the request the interrupted run would have sent."""
    from tests.test_local import completion, model
    reply = completion(calls=(("c1", "orbit", {"azimuth_deg": 30}),))
    raw = reply.model_dump()
    raw["choices"][0]["message"].update(reasoning="I look at the apex first.", extra_content={"google": {"sig": "s"}})
    from openai.types.chat import ChatCompletion
    (live, _), (again, _) = model([ChatCompletion.model_validate(raw)]), model([])
    turn_live = live.step()
    turn_again = again.restore(raw)
    assert again.history == live.history and again.history[-1][1]["extra_content"] == {"google": {"sig": "s"}}
    assert [(c.id, c.name, c.args, c.reasoning) for c in turn_again.calls] == \
        [(c.id, c.name, c.args, c.reasoning) for c in turn_live.calls]
    assert again.render() == live.render()


def _rb():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("run_batch", Path(__file__).resolve().parents[1] / "slurm/run_batch.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_batch_resumes_an_unfinished_case_from_the_right_run(tmp_path):
    import os
    import time
    rb = _rb()
    prior = interrupted(tmp_path)
    traces = tmp_path / "a" / "traces"
    jdir = tmp_path / "job"
    (jdir / "inflight").mkdir(parents=True)
    assert rb.resume_source(jdir, 1001, "h", traces) is None
    marker = jdir / "inflight" / "ImageCHD-1001.json"
    marker.write_text(json.dumps({"run_dir": str(prior.run_dir)}))
    assert rb.resume_source(jdir, 1001, "h", traces) == str(prior.run_dir)
    row = {"case": 1001, "command_sha256": "h", "run_folder": str(prior.run_dir), "stop": "resume_mismatch"}
    (jdir / "results.jsonl").write_text(json.dumps(row) + "\n")
    old = time.time() - 60
    os.utime(marker, (old, old))
    assert rb.resume_source(jdir, 1001, "h", traces) is None
    (jdir / "results.jsonl").write_text(json.dumps({**row, "stop": "terminated"}) + "\n")
    assert rb.resume_source(jdir, 1001, "h", traces) == str(prior.run_dir)
    assert rb.resume_source(jdir, 1001, "another command", traces) == str(prior.run_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    marker.write_text(json.dumps({"run_dir": str(empty)}))
    assert rb.resume_source(jdir, 1001, "h", traces) is None


def test_a_killed_episode_without_a_summary_is_resumed(tmp_path):
    """SIGKILL or a node failure: no summary.json, only the live records of runs/<episode id>/ and the trace."""
    prior = interrupted(tmp_path)
    (prior.run_dir / "summary.json").unlink()
    again = load_prior(prior.run_dir, tmp_path / "a" / "traces")
    assert again.stop is None and len(again.turns) == 4
    summary, _, _, _ = episode(tmp_path / "b", TURNS, resume=again)
    assert summary["stop_reason"] == "submitted" and summary["resumed"]["from_stop"] is None


# ---------------------------------------------------------------------------------------------- Anthropic (Claude)
def _wire(messages):
    """The messages as the anthropic SDK puts them in the request body (an SDK block: only its set fields)."""
    from anthropic._utils._json import openapi_model_dump
    return json.loads(json.dumps(messages, default=openapi_model_dump))


def _claude_replies():
    from tests.test_anthropic import message, tool_use
    return [
        message([{"type": "thinking", "thinking": "Look at the state first.", "signature": "sig-1"}, tool_use("t1")]),
        message([{"type": "thinking", "thinking": "", "signature": "sig-2"}, {"type": "text", "text": "Now the 3D view."},
                 tool_use("t2", "orbit", {"azimuth_deg": 30})]),
        message([{"type": "redacted_thinking", "data": "enc-3"}, tool_use("t3", "reset_view")]),
        message([tool_use("t4", "submit_report", {"report": NORMAL})]),
    ]


class _FailingAnthropic:
    """FakeAnthropic whose replies may be exceptions (raised when their request is sent)."""

    def __new__(cls, replies):
        from tests.test_anthropic import FakeAnthropic, FakeStream
        fake = FakeAnthropic(replies)

        def stream(**kw):
            fake.requests.append(kw)
            r = fake.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return FakeStream(r)
        fake.beta.messages.stream = stream
        return fake


def _claude_episode(tmp_path, replies, resume=None):
    from chd_client.models.anthropic import AnthropicModel
    tmp_path.mkdir(parents=True, exist_ok=True)
    srv, _, traces = make_env(tmp_path)
    model = AnthropicModel(client=_FailingAnthropic(replies))
    try:
        summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                                traces_dir=traces, echo=None, make_pdf=False, resume=resume))
    except RuntimeError as e:
        if not hasattr(e, "summary"):
            raise
        summary = e.summary
    return summary, model, traces


def test_an_anthropic_reply_is_restored_as_it_was_sent():
    """restore(raw) from the recorded full model_dump (every optional field, null when unset) puts the reply into the
    history as the live step did: the next request body carries the same blocks, thinking signatures included."""
    from chd_client.models.anthropic import AnthropicModel
    from chd_client.run_record import to_jsonable
    from tests.test_anthropic import TOOLS, FakeAnthropic, obs
    reply = _claude_replies()[1]
    live, again = AnthropicModel(client=FakeAnthropic([reply])), AnthropicModel(client=FakeAnthropic([]))
    for m in (live, again):
        m.start(["S"], TOOLS, "first", obs(0))
    turn_live = live.step()
    raw = to_jsonable(reply)  # what responses.jsonl holds
    assert raw["content"][2]["caller"] is None  # an unset optional field, recorded as null
    turn_again = again.restore(raw)
    assert [(c.id, c.name, c.args, c.reasoning) for c in turn_again.calls] == \
        [(c.id, c.name, c.args, c.reasoning) for c in turn_live.calls]
    assert (turn_again.stop, turn_again.usage) == (turn_live.stop, turn_live.usage)
    live.add_results([(turn_live.calls[0], obs(1))])
    again.add_results([(turn_again.calls[0], obs(1))])
    a, b = live.render(), again.render()
    assert _wire(b["messages"]) == _wire(a["messages"])  # byte for byte what the API received the first time
    assert _wire(b["messages"])[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-2"}
    # the resumed episode lets the API drop a block it cannot bind instead of refusing the request
    assert "block_binding" not in a["thinking"]
    assert b["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert b["betas"] == ["thinking-binding-controls-2026-08-01"]


def test_an_interrupted_claude_episode_resumes_with_the_same_history(tmp_path):
    replies = _claude_replies()
    whole, whole_model, _ = _claude_episode(tmp_path / "whole", replies)
    assert whole["stop_reason"] == "submitted"
    cut, _, traces = _claude_episode(tmp_path / "a", replies[:2] + [RuntimeError("APIConnectionError")])
    assert cut["stop_reason"] == "runner_exception" and cut["n_model_turns"] == 2
    prior = load_prior(tmp_path / "a" / "runs" / cut["episode_id"], traces)
    assert len(prior.turns) == 2 and all(t["raw"]["type"] == "message" for t in prior.turns)
    summary, model, _ = _claude_episode(tmp_path / "b", replies[2:], resume=prior)
    assert summary["stop_reason"] == "submitted" and summary["resumed"]["replayed_turns"] == 2
    for k in ("n_model_steps", "n_model_turns", "tokens"):
        assert summary[k] == whole[k], k
    assert summary["cost_usd"] == pytest.approx(whole["cost_usd"])
    # the first request after the resume is the one the uninterrupted run sent at that point
    first, want = model.client.requests[0], whole_model.client.requests[2]
    assert _wire(first["messages"]) == _wire(want["messages"])
    assert (first["system"], first["tools"]) == (want["system"], want["tools"])
    assert first["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert len(model.client.requests) == 2  # no request for the replayed turns
