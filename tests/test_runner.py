"""The runner end to end with a scripted model over an in-process environment server (fake bridge)."""

import json
import re

import anyio
import pypdfium2
import pytest

from chd_client.run_episode import (PROMPT_DIR, PROMPT_VERSION, default_max_steps, prompt_version, render_system,
                                    run_episode)
from chd_client.models.scripted import ScriptedModel
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, NORMAL, make_env

SUBMIT = ("submit_report", {"report": NORMAL})


def run(tmp_path, script, model=None, **kw):
    srv, fake, traces = make_env(tmp_path)
    model = model or ScriptedModel(script)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, **kw))
    meta, steps = read_trace(srv.session.trace_dir)
    return srv, fake, model, summary, meta, steps


def test_tools_filtered_verbatim_and_the_prompt_alone_in_system(tmp_path):
    srv, _, model, summary, meta, _ = run(tmp_path, [[SUBMIT]])
    listed = {t.name: t for t in srv.tools()}
    got = {t.name: t for t in model.tools}
    runner_only = {n for n, t in listed.items() if (t.meta or {}).get("slicer_mcp/runner_only")}
    assert "reset_episode" not in got and set(got) == set(listed) - runner_only
    assert {"reset_episode", "end_episode"} <= runner_only
    for name, t in got.items():
        assert t.description == listed[name].description and t.input_schema == listed[name].input_schema
    assert model.system_blocks == [render_system(PROMPT_DIR, 70)]
    assert "{max_steps}" not in model.system_blocks[0] and "at most 70 tool calls" in model.system_blocks[0]
    cm = meta["client_meta"]
    assert meta["model"] == {"id": "scripted", "version": "scripted-v1"} and meta["seed"] == 0
    assert cm["prompt_version"] == "chd_ct_v1" and cm["image_window"] is None and cm["max_steps"] == 70
    assert cm["cost_cap_usd"] == 3.0 and {"commit", "dirty"} <= set(cm["client_git"])
    assert len(meta["system_prompt_sha256"]) == 64 and len(cm["server_instructions_sha256"]) == 64
    assert summary["stop_reason"] == "submitted" and meta["status"] == "submitted"


def test_reasoning_and_tokens_in_request_meta(tmp_path):
    script = [[("reset_view", {}, "checking the state"), ("clear_clip", {})], [SUBMIT]]
    *_, summary, meta, steps = run(tmp_path, script)
    assert [s["tool"] for s in steps] == ["reset_episode", "reset_view", "clear_clip", "submit_report"]
    assert steps[1]["model_reasoning"] == "checking the state" and steps[1]["tokens"] == {"in": 1000, "out": 50}
    assert steps[2]["tokens"] == {"in": 0, "out": 0}
    assert steps[3]["tokens"] == {"in": 1000, "out": 50} and steps[3]["model_reasoning"]
    assert summary["n_model_steps"] == 3 and summary["n_model_turns"] == 2
    assert summary["tokens"]["in"] == 2000 and summary["cost_usd"] == pytest.approx(0.02)
    assert set(summary["tool_latency_ms"]) == {"reset_episode", "reset_view", "clear_clip", "submit_report"}


@pytest.mark.parametrize("per_turn", [1, 2])
def test_step_budget_stops_without_report_then_abandoned(tmp_path, per_turn):
    endless = [lambda obs: [("orbit", {"azimuth_deg": 10})] * per_turn] * 100
    srv, _, _, summary, meta, steps = run(tmp_path, endless, max_steps=5)
    assert summary["stop_reason"] == "step_budget" and summary["n_model_steps"] == 5 and summary["report"] is None
    assert len(steps) == 6 and meta["status"] == "abandoned"


def test_invalid_args_go_back_to_the_model(tmp_path):
    *_, model, summary, _, steps = run(tmp_path, [[("orbit", {"azimuth_deg": "left"})], [("reset_view", {})],
                                                        [SUBMIT]])
    bad = model.observations[1]
    assert bad.is_error and bad.error_type == "invalid_argument" and "azimuth_deg" in bad.text
    assert summary["stop_reason"] == "submitted" and steps[1]["error"]["type"] == "invalid_argument"


def test_environment_error_aborts(tmp_path):
    srv, _, model, summary, _, steps = run(tmp_path, [[("orbit", {"azimuth_deg": 123})], [SUBMIT]])
    assert summary["stop_reason"] == "environment_error" and summary["n_model_steps"] == 1
    assert steps[-1]["error"]["type"] == "internal" and len(model.turns) == 1
    srv.session.close("abandoned")


def test_runner_only_tool_is_blocked_locally(tmp_path):
    script = [[("reset_episode", {"case_id": CASE})], [("reset_view", {})], [SUBMIT]]
    _, fake, model, summary, meta, steps = run(tmp_path, script)
    blocked = model.observations[1]
    assert blocked.local and blocked.is_error and blocked.error_type == "invalid_argument"
    assert "unknown tool" in blocked.text and CASE not in blocked.text
    assert [s["tool"] for s in steps] == ["reset_episode", "reset_view", "submit_report"]
    assert [c[0] for c in fake.calls].count("reset_episode") == 1 and meta["status"] == "submitted"
    lines = [json.loads(x) for x in (tmp_path / "runs" / summary["episode_id"] / "transcript.jsonl").read_text().splitlines()]
    assert any(x["kind"] == "flag" and x["tool"] == "reset_episode" for x in lines)


def test_unknown_tool_is_forwarded_and_refused(tmp_path):
    *_, model, summary, _, steps = run(tmp_path, [[("no_such_tool", {})], [SUBMIT]])
    assert steps[1]["tool"] == "no_such_tool" and steps[1]["error"]["type"] == "invalid_argument"
    assert model.observations[1].is_error and summary["stop_reason"] == "submitted"


def test_accepted_report_ends_the_episode(tmp_path):
    *_, summary, meta, steps = run(tmp_path, [[SUBMIT, ("reset_view", {})]])
    assert summary["stop_reason"] == "submitted" and summary["report"] == NORMAL
    assert [s["tool"] for s in steps] == ["reset_episode", "submit_report"] and meta["status"] == "submitted"


def test_rejected_report_then_resubmit(tmp_path):
    bad = ("submit_report", {"report": {**NORMAL, "va_connection": "?"}})
    *_, model, summary, meta, _ = run(tmp_path, [[bad], [SUBMIT]])
    assert model.observations[1].is_error and summary["stop_reason"] == "submitted" and meta["status"] == "submitted"


def test_nudges_then_no_tool_call(tmp_path):
    srv, _, model, summary, _, _ = run(tmp_path, [])
    nudge = json.loads((PROMPT_DIR / "runner_messages.json").read_text())["nudge"]
    assert summary["stop_reason"] == "no_tool_call" and summary["n_model_turns"] == 4
    assert model.texts[1:] == [nudge] * 3 and summary["n_model_steps"] == 0
    srv.session.close("abandoned")


def test_nudge_counter_resets_after_a_call(tmp_path):
    *_, summary, _, _ = run(tmp_path, [[], [], [], [("reset_view", {})], [], [], [SUBMIT]])
    assert summary["stop_reason"] == "submitted"


def test_cost_cap(tmp_path):
    endless = [lambda obs: [("reset_view", {})]] * 100
    srv, *_, summary, _, _ = run(tmp_path, None, model=ScriptedModel(endless, cost_usd=0.5), cost_cap_usd=2.0)
    assert summary["stop_reason"] == "cost_cap" and summary["n_model_turns"] == 4
    assert summary["cost_usd"] == pytest.approx(2.0)
    srv.session.close("abandoned")


def test_no_cost_cap_stops_on_the_step_limit(tmp_path):
    """--cost-cap none: no money stop and no money-based calls_left; the step limit ends the episode."""
    endless = [[("orbit", {"azimuth_deg": 5})]] * 40
    srv, *_, summary, _, _ = run(tmp_path, None, model=ScriptedModel(endless, cost_usd=0.5), cost_cap_usd=None,
                                 max_steps=12)
    assert summary["stop_reason"] == "step_budget" and summary["cost_usd"] > 2.0


def test_cli_cost_cap_none():
    from chd_client.run_episode import _cost_cap
    assert _cost_cap("none") is None and _cost_cap("OFF") is None and _cost_cap("5") == 5.0


def test_unknown_cost_stops(tmp_path):
    endless = [lambda obs: [("reset_view", {})]] * 100
    srv, *_, summary, _, _ = run(tmp_path, None, model=ScriptedModel(endless, cost_usd=None))
    assert summary["stop_reason"] == "cost_unknown" and summary["n_model_turns"] == 1
    srv.session.close("abandoned")


def test_unknown_cost_turn_with_submit_report_sends_no_call(tmp_path):
    """The accounting is checked before any of the turn's calls run. An unknown-cost turn that submits ends
    cost_unknown, not submitted, and neither of its calls reaches the environment."""
    srv, *_, model, summary, meta, steps = run(tmp_path, [[("reset_view", {}), SUBMIT]],
                                               model=ScriptedModel([[("reset_view", {}), SUBMIT]], cost_usd=None))
    assert summary["stop_reason"] == "cost_unknown" and summary["report"] is None
    assert summary["n_model_turns"] == 1 and summary["n_model_steps"] == 0
    assert [s["tool"] for s in steps] == ["reset_episode"] and meta["status"] == "abandoned"
    assert len(model.observations) == 1
    lines = [json.loads(x) for x in (tmp_path / "runs" / summary["episode_id"] / "transcript.jsonl").read_text()
             .splitlines()]
    assert lines[-1] == {"kind": "cost_unknown", "turn": 1, "calls_not_sent": ["reset_view", "submit_report"]}
    assert [x["kind"] for x in lines].count("model_turn") == 1


def test_known_cost_submit_still_ends_submitted(tmp_path):
    *_, summary, meta, steps = run(tmp_path, [[("reset_view", {}), SUBMIT]])
    assert summary["stop_reason"] == "submitted" and summary["report"] == NORMAL
    assert summary["n_model_steps"] == 2 and [s["tool"] for s in steps][1:] == ["reset_view", "submit_report"]
    assert meta["status"] == "submitted" and summary["cost_usd"] == pytest.approx(0.01)


def test_nothing_model_visible_names_the_case_or_episode(tmp_path):
    script = [[("reset_view", {}), ("clear_clip", {})], [("orbit", {"azimuth_deg": 999})],
              [("set_slice", {"view": "nope", "position_mm": 0})], [("reset_episode", {"case_id": CASE})],
              [("bookmark_view", {"label": "x"})], [("submit_report", {"report": {**NORMAL, "va_connection": "?"}})],
              [SUBMIT]]
    *_, model, summary, _, _ = run(tmp_path, script)
    blob = "".join(model.system_blocks) + "".join(model.texts)
    blob += json.dumps([vars(t) for t in model.tools]) + "".join(o.text for o in model.observations)
    for needle in (CASE, summary["episode_id"], "case_0"):
        assert needle not in blob
    assert summary["stop_reason"] == "submitted"


def test_prompt_files_name_no_case_or_dataset():
    """Lesions may be named generically; cases, datasets and patients may not. The model-facing files only:
    registry.json is never sent."""
    text = "".join(p.read_text() for p in PROMPT_DIR.iterdir() if p.is_file() and p.name != "registry.json")
    assert not re.search(r"case_\d|case_dcm|patient\s+\w+\d|\bP\d\d\b|ImageCHD", text)


def test_run_files_have_no_image_bytes(tmp_path):
    *_, summary, _, _ = run(tmp_path, [[("orbit", {"azimuth_deg": 5})], [SUBMIT]])
    d = tmp_path / "runs" / summary["episode_id"]
    s = json.loads((d / "summary.json").read_text())
    assert s["stop_reason"] == "submitted" and s["wall_s"]["total"] >= s["wall_s"]["model_s"]
    t = (d / "transcript.jsonl").read_text()
    assert "iVBORw0KGgo" not in t and '"frames": ["3d", "axial", "sagittal", "coronal"]' in t


def test_run_writes_report_pdf(tmp_path, _reports_dir_in_tmp):
    *_, summary, _, _ = run(tmp_path, [[("orbit", {"azimuth_deg": 5})], [SUBMIT]])
    d = tmp_path / "runs" / summary["episode_id"]
    assert summary["report_pdf"] == str(d / "report.pdf") and summary["report_pdf_error"] is None
    assert "report_pdf_copy" not in summary and not _reports_dir_in_tmp.exists()
    assert (d / "report.pdf").read_bytes()[:5] == b"%PDF-"
    assert json.loads((d / "summary.json").read_text())["report_pdf"] == summary["report_pdf"]
    from chd_client.report_pdf import system_tz_name
    pdf = pypdfium2.PdfDocument(str(d / "report.pdf"))
    assert f"{system_tz_name()} (" in pdf[0].get_textpage().get_text_range()
    pdf.close()


def test_pdf_failure_does_not_lose_the_run(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no fonts today")
    monkeypatch.setattr("chd_client.report_pdf.render_episode_pdf", boom)
    *_, summary, _, _ = run(tmp_path, [[SUBMIT]])
    d = tmp_path / "runs" / summary["episode_id"]
    assert summary["stop_reason"] == "submitted" and summary["report_pdf"] is None
    assert summary["report_pdf_error"] == "RuntimeError: no fonts today" and not (d / "report.pdf").exists()
    assert json.loads((d / "summary.json").read_text())["report_pdf_error"] == summary["report_pdf_error"]
    assert (d / "transcript.jsonl").exists()


def test_dry_run_script_submits(tmp_path):
    from chd_client.diagnosis import dry_normal_report
    from chd_client.run_episode import dry_run_script
    srv, _, model, summary, meta, steps = run(tmp_path, dry_run_script())
    assert summary["stop_reason"] == "submitted" and meta["profile"] == "chd_ct_v1"
    assert all(s.get("error") is None for s in steps), [s.get("error") for s in steps if s.get("error")]
    tools = [s["tool"] for s in steps]
    assert tools == ["reset_episode", "point_at", "set_segment_visibility", "set_slice_outlines", "orbit",
                     "submit_report"]
    assert summary["report"] == dry_normal_report()
    assert "set_segment_visibility" in {t.name for t in model.tools}


@pytest.mark.parametrize("arm", ["fixed", "2d"])
def test_only_the_3d_seg_and_3d_arms_run(tmp_path, arm):
    srv, _, traces = make_env(tmp_path)
    with pytest.raises(ValueError, match="not one of"):
        anyio.run(lambda: run_episode(CASE, arm, ScriptedModel([[SUBMIT]]), srv.server, runs_dir=tmp_path / "runs",
                                      traces_dir=traces, echo=None))


def test_3d_seg_arm_has_its_own_prompt():
    """The 3d_seg arm gets chd_ct_v1 (70 calls by default): the 16 classes and normal, the base tools only, every
    placeholder filled."""
    assert prompt_version() == PROMPT_VERSION == "chd_ct_v1" and default_max_steps() == 70
    t = render_system(PROMPT_DIR, 70)
    assert not re.search(r"\{[a-z_]+\}", t) and "at most 70 tool calls" in t
    order = ["## Role and task", "## The classes", "## The tools", "## The episode", "## Output"]
    assert [t.index(w) for w in order] == sorted(t.index(w) for w in order)
    from chd_client.diagnosis import CLASS_NAMES
    classes = t.split("## The classes")[1].split("\n## ")[0]
    assert all(f"- {c}\n" in classes for c in CLASS_NAMES) and "- normal" in classes and "17 classes" in classes
    tools = t.split("## The tools")[1].split("\n## ")[0]
    for name in ("set_slice", "point_at", "measure", "orbit", "set_segment_visibility", "center_on_segments",
                 "bookmark_view", "submit_report"):
        assert f"`{name}`" in tools, name
    for name in ("set_slice_plane", "set_oblique_plane", "rotate_slice_plane", "measure_area", "hu_profile",
                 "trace_path", "slice_montage", "make_blood_pool_shell", "place_structure_seeds",
                 "make_myocardium_shell", "record_signs"):
        assert name not in t, name
    out = t.split("## Output")[1]
    assert "`classes`" in out and "`normal` is present only when none of the other 16 is" in out
    assert "images from older steps are removed" not in t
    assert render_system(PROMPT_DIR, 70, image_window=2).count("images from older steps are removed") == 1


def test_every_observation_tells_the_model_its_calls_left(tmp_path):
    script = [[("set_window_preset", {"name": "CT-Angio"})], [("set_window_preset", {"name": "CT-Lung"})], [SUBMIT]]
    srv, _, model, summary, meta, _ = run(tmp_path, script, max_steps=10)
    left = [json.loads(o.text)["calls_left"] for o in model.observations]
    assert left == [10, 9, 8] and meta["client_meta"]["calls_left_shown"] is True
    first = list(json.loads(model.observations[1].text))
    assert first[first.index("step_index") + 1] == "calls_left"


def test_the_cost_cap_stops_the_run_but_the_model_never_sees_money(tmp_path):
    endless = [lambda obs: [("reset_view", {})]] * 100
    srv, _, model, summary, *_ = run(tmp_path, None, model=ScriptedModel(endless, cost_usd=0.125),
                                     cost_cap_usd=1.0, max_steps=50)
    left = [json.loads(o.text)["calls_left"] for o in model.observations]
    assert left == list(range(50, 50 - len(left), -1))
    assert summary["stop_reason"] == "cost_cap"
    seen = " ".join([o.text for o in model.observations] + model.texts).lower()
    assert "cost" not in seen and "usd" not in seen and "$" not in seen
    srv.session.close("abandoned")


def test_make_model_max_tokens(monkeypatch):
    """--max-tokens reaches both adapters; unset keeps their default."""
    from chd_client.run_episode import make_model
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    m = make_model("openrouter:google/gemini-4-argon", "high", False, max_tokens=64000)
    assert m.max_tokens == 64000 and m.effort == "high"
    assert make_model("openrouter:openai/gpt-6-astra", "max", False).max_tokens == 32000


def test_a_reply_cut_at_the_output_cap_says_why_and_what_to_change(tmp_path, capsys):
    """A turn that reaches max_tokens with no tool call ends the episode (model_max_tokens): the transcript, the
    summary and stderr say what happened and which setting to change."""
    from types import SimpleNamespace

    from chd_client.models.base import ModelTurn, Usage

    class Thinker(ScriptedModel):
        max_tokens = 20480

        def step(self):
            usage = SimpleNamespace(completion_tokens_details={"reasoning_tokens": 20480})
            return ModelTurn([], "", "max_tokens", Usage(900, 20480), SimpleNamespace(usage=usage))
    *_, summary, _, _ = run(tmp_path, [], model=Thinker([]))
    assert summary["stop_reason"] == "model_max_tokens"
    for text in (summary["stop_detail"], capsys.readouterr().err):
        assert "all of it was thinking" in text and "thinking_token_budget" in text and "max_output_tokens" in text
