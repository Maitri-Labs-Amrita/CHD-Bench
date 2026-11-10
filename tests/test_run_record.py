"""Durable, complete run records: the request log of every adapter, files written as the episode goes, SIGTERM,
exceptions, case identity, the seed, the cost cap checked before a turn and the provenance folder."""

import hashlib
import json
import os
import signal
import threading
import time

import anyio
import pytest

from chd_client import run_record, slicer
from chd_client.mcp_session import EnvironmentAbort
from chd_client.models.anthropic import AnthropicModel
from chd_client.models.openrouter import OpenRouterModel
from chd_client.models.scripted import ScriptedModel
from chd_client.run_episode import (IdentityMismatch, IdentityTracker, Termination, dry_run_script,
                                    projected_turn_cost, run_episode)
from chd_client.run_record import RequestRecorder, read_jsonl, reconstruct_requests
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, NORMAL, make_env, png
from tests.test_anthropic import TOOLS, FakeAnthropic, message, obs, tool_use
from tests.test_openrouter import FakeOpenAI, completion

SUBMIT = ("submit_report", {"report": NORMAL})


def run(tmp_path, model, **kw):
    srv, fake, traces = make_env(tmp_path)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, make_pdf=False, **kw))
    return srv, fake, summary, tmp_path / "runs" / summary["episode_id"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def test_scripted_dry_run_logs_every_request_with_image_refs(tmp_path):
    srv, _, summary, d = run(tmp_path, ScriptedModel(dry_run_script()))
    assert summary["stop_reason"] == "submitted"
    reqs = reconstruct_requests(d / "requests.jsonl")
    assert len(reqs) == summary["n_model_turns"] == summary["n_requests"] == len(dry_run_script())
    assert [r["turn"] for r in reqs] == list(range(1, len(reqs) + 1))
    assert reqs[0]["system"][0]["text"] == (d / "provenance" / "system_prompt.txt").read_text()
    assert "system" not in read_jsonl(d / "requests.jsonl")[1]
    raw = (d / "requests.jsonl").read_text()
    assert "iVBORw0KGgo" not in raw
    tdir = srv.session.trace_dir
    for r in reqs:
        for im in r["images"]:
            f = tdir / im["frame"].removeprefix("trace/")
            assert f.is_file() and sha(f.read_bytes()) == im["sha256"], im
    last = reqs[-1]
    assert len(last["images"]) >= len(reqs[0]["images"]) > 0
    assert last["placeholders"] == [] and last["image_window"] is None
    for a, b in zip(reqs, reqs[1:]):
        assert b["messages"][:len(a["messages"])] == a["messages"]
    resp = read_jsonl(d / "responses.jsonl")
    assert [r["turn"] for r in resp] == [r["turn"] for r in reqs]
    assert resp[-1]["raw"]["calls"][0]["name"] == "submit_report"


def test_the_image_window_substitutions_are_logged(tmp_path):
    script = [[("orbit", {"azimuth_deg": 5})], [("orbit", {"azimuth_deg": 10})], [SUBMIT]]
    _, _, summary, d = run(tmp_path, ScriptedModel(script, image_window=1))
    reqs = reconstruct_requests(d / "requests.jsonl")
    assert reqs[0]["placeholders"] == []
    ph = reqs[2]["placeholders"]
    assert ph and {p["step"] for p in ph} == {0, 1} and all(p["text"].startswith("[image omitted") for p in ph)
    assert all(p["frame"].startswith("trace/frames/") for p in ph)
    texts = json.dumps(reqs[2]["messages"])
    assert all(p["text"] in texts for p in ph)
    assert {im["step"] for im in reqs[2]["images"]} == {2}


def test_anthropic_requests_and_raw_responses_are_logged(tmp_path):
    fake = FakeAnthropic([message([{"type": "thinking", "thinking": "hm", "signature": "s"}, tool_use("t1")]),
                          message([{"type": "text", "text": "done"}], stop="end_turn")])
    m = AnthropicModel(client=fake, image_window=None)
    rec = RequestRecorder(tmp_path)
    m.recorder = rec
    m.start(["SYS"], TOOLS, "first", obs(0, ("3d", "axial")))
    rec.turn = 1
    m.step()
    m.add_results([(type("C", (), {"id": "t1", "name": "get_state"})(), obs(1))])
    rec.turn = 2
    m.step()
    rec.close()
    reqs = reconstruct_requests(tmp_path / "requests.jsonl")
    assert len(reqs) == 2 and reqs[0]["provider"] == "anthropic"
    assert reqs[0]["params"]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert reqs[0]["params"]["tool_choice"] == {"type": "auto"} and reqs[0]["params"]["max_tokens"] == 32000
    assert reqs[0]["tools"][0]["name"] == "get_state"
    assert [i["frame"] for i in reqs[0]["images"]] == ["trace/frames/000_3d.png", "trace/frames/000_axial.png"]
    assert reqs[1]["images"][-1] == {"message": 2, "frame": "trace/frames/001_3d.png", "step": 1, "view": "3d",
                                     "sha256": sha(png(1))}
    assert reqs[1]["messages"][1]["content"][0]["thinking"] == "hm"
    resp = read_jsonl(tmp_path / "responses.jsonl")
    assert resp[0]["raw"]["content"][0] == {"type": "thinking", "thinking": "hm", "signature": "s"}
    assert resp[1]["raw"]["stop_reason"] == "end_turn"
    m2 = AnthropicModel(client=FakeAnthropic([]))
    rec2 = RequestRecorder(tmp_path / "b")
    m2.recorder = rec2
    m2.start(["SYS"], TOOLS, "first", obs(0))
    with pytest.raises(IndexError):
        m2.step()
    assert len(read_jsonl(tmp_path / "b" / "requests.jsonl")) == 1


def test_openrouter_requests_are_logged(tmp_path):
    fake = FakeOpenAI([completion([("c0", "capture_3d", {})], reasoning="r")])
    m = OpenRouterModel("vendor/model-x", client=fake, effort="high")
    m.recorder = RequestRecorder(tmp_path)
    m.start(["SYS"], TOOLS, "first", obs(0, ("3d",)))
    m.step()
    m.recorder.close()
    (r,) = reconstruct_requests(tmp_path / "requests.jsonl")
    assert r["provider"] == "openrouter" and r["system"]["content"] == "SYS"
    assert r["params"]["extra_body"]["reasoning"] == {"effort": "high"} and r["params"]["tool_choice"] == "auto"
    assert r["images"] == [{"message": 0, "frame": "trace/frames/000_3d.png", "step": 0, "view": "3d",
                            "sha256": sha(png(0))}]
    assert "base64" not in json.dumps(r["messages"])
    assert read_jsonl(tmp_path / "responses.jsonl")[0]["raw"]["choices"][0]["message"]["reasoning"] == "r"


def test_transcript_is_on_disk_turn_by_turn_and_summary_is_atomic(tmp_path):
    seen = []

    def peek(action):
        def turn(obs_list):
            (f,) = (tmp_path / "runs").glob("*/transcript.jsonl")
            seen.append(len(f.read_text().splitlines()))
            return [action]
        return turn
    script = [peek(("orbit", {"azimuth_deg": 5})), peek(("orbit", {"azimuth_deg": 9})), peek(SUBMIT)]
    _, _, summary, d = run(tmp_path, ScriptedModel(script))
    assert seen == sorted(seen) and seen[0] >= 2 and seen[-1] > seen[0]
    assert json.loads((d / "summary.json").read_text())["stop_reason"] == "submitted"
    assert not [p for p in d.iterdir() if ".tmp." in p.name]


def test_write_json_atomic_keeps_the_old_file_on_failure(tmp_path):
    p = tmp_path / "s.json"
    run_record.write_json_atomic(p, {"a": 1})
    with pytest.raises(TypeError):
        run_record.write_json_atomic(p, {"a": object()})
    assert json.loads(p.read_text()) == {"a": 1}


def test_termination_before_a_turn_still_writes_everything(tmp_path):
    term = Termination()

    def stop_after(obs_list):
        if len(model.turns) == 2:
            term.request("SIGTERM")
        return [("orbit", {"azimuth_deg": 5})]
    model = ScriptedModel([stop_after] * 50)
    srv, _, summary, d = run(tmp_path, model, terminate=term)
    assert summary["stop_reason"] == "terminated" and summary["terminated_by"] == "SIGTERM"
    assert summary["n_model_turns"] == 3
    meta, steps = read_trace(srv.session.trace_dir)
    assert meta["status"] == "abandoned" and "end_episode" in summary
    lines = read_jsonl(d / "transcript.jsonl")
    assert lines[-1]["kind"] == "terminated"


def test_sigterm_abandons_a_model_request_in_flight(tmp_path):
    """A real SIGTERM while model.step() blocks: the runner stops at once, closes the episode, writes the summary;
    the model's late answer is never used."""
    release = threading.Event()

    class Slow(ScriptedModel):
        def step(self):
            if len(self.turns) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                release.wait(10)
            return super().step()
    model = Slow([[("orbit", {"azimuth_deg": 5})]] * 5)
    t0 = time.perf_counter()
    try:
        srv, _, summary, d = run(tmp_path, model, handle_signals=True)
    finally:
        release.set()
    assert time.perf_counter() - t0 < 8
    assert summary["stop_reason"] == "terminated" and summary["terminated_by"] == "SIGTERM"
    assert summary["n_model_turns"] == 1
    kinds = [x["kind"] for x in read_jsonl(d / "transcript.jsonl")]
    assert "terminated" in kinds and (d / "summary.json").is_file()
    assert summary["end_episode"] is not None


def test_a_runner_exception_mid_episode_keeps_its_summary(tmp_path):
    def boom(_):
        raise RuntimeError("model adapter broke")
    with pytest.raises(RuntimeError) as ei:
        run(tmp_path, ScriptedModel([[("orbit", {"azimuth_deg": 5})], boom]))
    s = ei.value.summary
    assert s["stop_reason"] == "runner_exception" and s["n_model_turns"] == 1
    assert (tmp_path / "runs" / s["episode_id"] / "requests.jsonl").is_file()


def test_every_observation_records_its_identity_and_the_case_check_gets_the_voxel_hash(tmp_path):
    got = []

    def check(case_info, volume_sha256):
        got.append((case_info, volume_sha256))
        return {"ok": True, "method": "loaded_voxels"}
    srv, fake, summary, d = run(tmp_path, ScriptedModel([[("orbit", {"azimuth_deg": 5})], [SUBMIT]]),
                                case_check=check)
    assert got[0][1] == fake.volume_sha256 and "volume_sha256" not in got[0][0]
    obs_lines = [x for x in read_jsonl(d / "transcript.jsonl") if x["kind"] == "observation"
                 and x["tool"] != "submit_report"]
    assert all(x["identity"]["episode_id"] == summary["episode_id"] for x in obs_lines)
    assert all(x["identity"]["volume_sha256"] == fake.volume_sha256 for x in obs_lines)
    assert summary["identity"]["n_checked"] == 2 and summary["identity"]["n_without_identity"] == 1 and summary["case_identity"]["ok"] is True
    seqs = [x["identity"]["seq"] for x in obs_lines]
    assert seqs == sorted(seqs)
    for x in obs_lines:
        assert fake.volume_sha256 not in x["text"] and "seq" not in json.loads(x["text"])


def test_a_result_from_another_volume_aborts(tmp_path):
    def swap(_):
        fake.volume_sha256 = "w" * 64
        return [("orbit", {"azimuth_deg": 5})]
    srv, fake, traces = make_env(tmp_path)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", ScriptedModel([[("orbit", {"azimuth_deg": 1})], swap]),
                                            srv.server, runs_dir=tmp_path / "runs", traces_dir=traces, echo=None,
                                            make_pdf=False))
    assert summary["stop_reason"] == "environment_error"


def test_identity_tracker():
    from types import SimpleNamespace as NS
    t = IdentityTracker()
    a = {"episode_id": "E", "case_id": "c", "volume_sha256": "v", "seq": 1}
    t.check(NS(identity=a, local=False, tool="reset_episode"))
    t.check(NS(identity=None, local=False, tool="x"))
    t.check(NS(identity={**a, "seq": 2}, local=False, tool="orbit"))
    with pytest.raises(IdentityMismatch, match="seq"):
        t.check(NS(identity={**a, "seq": 2}, local=False, tool="orbit"))
    with pytest.raises(IdentityMismatch, match="episode_id"):
        t.check(NS(identity={**a, "episode_id": "F", "seq": 9}, local=False, tool="orbit"))
    assert t.summary()["n_without_identity"] == 1


def test_seed_is_marked_metadata_only(tmp_path):
    srv, _, summary, d = run(tmp_path, ScriptedModel([[SUBMIT]]), seed=7)
    cm = summary["client_meta"]
    assert cm["seed"] == 7 and "not sent" in cm["seed_use"]
    prov = json.loads((d / "provenance" / "provenance.json").read_text())
    assert prov["seed"] == {"value": 7, "use": "metadata only: not sent to the model API"}


def test_projected_turn_cost():
    assert projected_turn_cost([]) == 0.0 and projected_turn_cost([0.5]) == 0.5
    assert projected_turn_cost([0.5, 0.1, 0.2, 0.3]) == pytest.approx(0.4)
    assert projected_turn_cost([0.5, 0.3, 0.2, 0.1]) == pytest.approx(0.1)


def test_the_cost_cap_is_checked_before_sending_a_turn(tmp_path):
    """0.4 a turn, cap 1.0: after two turns (0.8) the third would cost about 1.2, so it is never sent."""
    endless = [lambda o: [("orbit", {"azimuth_deg": 5})]] * 20
    srv, _, summary, d = run(tmp_path, ScriptedModel(endless, cost_usd=0.4), cost_cap_usd=1.0)
    assert summary["stop_reason"] == "cost_cap" and summary["n_model_turns"] == 2
    assert summary["cost_usd"] == pytest.approx(0.8) and summary["n_requests"] == 2
    line = [x for x in read_jsonl(d / "transcript.jsonl") if x["kind"] == "cost_cap_projected"][0]
    assert line["projected_turn_usd"] == pytest.approx(0.4)
    srv.session.close("abandoned")


def test_provenance_folder(tmp_path):
    srv, _, summary, d = run(tmp_path, ScriptedModel([[SUBMIT]]))
    p = d / "provenance"
    assert sorted(x.name for x in p.iterdir()) == ["provenance.json", "system_prompt.txt", "tools.json"]
    prov = json.loads((p / "provenance.json").read_text())
    tools = json.loads((p / "tools.json").read_text())
    assert {t["name"] for t in tools} == {t.name for t in srv.tools()} - {"reset_episode", "end_episode", "observe"}
    assert prov["system_prompt_sha256"] == summary["client_meta"]["system_prompt_sha256"]
    assert prov["request"]["model"] == "scripted" and "python" in prov
    assert len(prov["scoring_v1_sha256"]) == 64 and "commit" in prov["harness_git"]
    assert prov["profile"] == "chd_ct_v1" and len(prov["profile_protocol_sha256"]) == 64
    assert prov["slicer_mcp"]["pinned_commit"] == slicer.PINNED_COMMIT and "commit" in prov["slicer_mcp"]


def _cli(monkeypatch, tmp_path, episode):
    """run-episode main() with the case, the episode and finish_run faked; returns (exit code, finish_run calls,
    whether the case was still there when finish_run ran)."""
    import contextlib
    from types import SimpleNamespace
    from chd_client import dicom_run as dr
    from chd_client import run_episode as re_
    nrrd = tmp_path / "case.nrrd"
    state = {"finished": [], "nrrd_at_finish": None}

    @contextlib.contextmanager
    def case(*a, **k):
        nrrd.write_bytes(b"x")
        tc = SimpleNamespace(case_id="case_nii_1", runtime=None, info={"nrrd": str(nrrd)}, has_truth=True,
                             trace_meta=lambda: {}, summary_meta=lambda: {}, check_loaded=lambda *a: None)
        try:
            yield tc
        finally:
            nrrd.unlink()

    def finish(summary, traces, **kw):
        state["finished"].append(summary)
        state["nrrd_at_finish"] = nrrd.exists()
    monkeypatch.setattr(dr, "temporary_imagechd_case", case)
    monkeypatch.setattr(re_, "finish_run", finish)
    monkeypatch.setattr(re_, "run_episode", episode)
    monkeypatch.setattr(signal, "signal", lambda *a: None)
    with pytest.raises(SystemExit) as ei:
        re_.main(["--imagechd", "1008", "--labels", "gt", "--dry-run-scripted"])
    return ei.value.code, state


def test_cli_collects_a_run_that_raised(monkeypatch, tmp_path):
    async def episode(*a, **k):
        e = RuntimeError("broke mid-episode")
        e.summary = {"episode_id": "E", "stop_reason": "runner_exception"}
        raise e
    code, st = _cli(monkeypatch, tmp_path, episode)
    assert code == 1 and st["finished"] == [{"episode_id": "E", "stop_reason": "runner_exception"}]
    assert st["nrrd_at_finish"] is True


def test_cli_collects_a_run_that_raised_inside_a_task_group(monkeypatch, tmp_path):
    async def episode(*a, **k):
        e = EnvironmentAbort("bridge connection lost")
        e.summary = {"episode_id": "E", "stop_reason": "environment_error"}
        raise BaseExceptionGroup("tg", [e])
    code, st = _cli(monkeypatch, tmp_path, episode)
    assert code == 1 and st["finished"][0]["stop_reason"] == "environment_error"


def test_cli_a_reset_failure_without_a_run_still_raises(monkeypatch, tmp_path):
    async def episode(*a, **k):
        raise EnvironmentAbort("reset_episode refused")
    with pytest.raises(EnvironmentAbort):
        _cli(monkeypatch, tmp_path, episode)


def test_scripted_turn_delay_from_the_environment(monkeypatch):
    import time
    from chd_client.models.scripted import ScriptedModel
    monkeypatch.setenv("CHD_SCRIPTED_TURN_DELAY_S", "0.2")
    m = ScriptedModel([[]])
    assert m.turn_delay_s == 0.2 and ScriptedModel([[]], turn_delay_s=0).turn_delay_s == 0
    t0 = time.monotonic()
    m.history = []
    m.step()
    assert time.monotonic() - t0 >= 0.2
