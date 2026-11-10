"""The episode PDF (chd_client.report_pdf) on a synthetic trace + run directory: no Slicer, no network."""

import io
import json
import shutil

import pypdfium2
import pytest
from PIL import Image

from chd_client import report_pdf
from chd_client.report_pdf import (fmt_time, get_zone, render_episode_pdf, report_file_name, start_end, view_name,
                                   wrap_mono)

from chd_client.diagnosis import CLASS_NAMES

E = "20260101T000000Z_case_nii_0123456789_3d_seg_abcdef"
REPORT = {"classes": {**{c: {"present": c == "ventricular_septal_defect",
                             "confidence": 0.6 if c == "ventricular_septal_defect" else 0.8} for c in CLASS_NAMES},
                      "normal": {"present": False, "confidence": 0.8}},
          "evidence": {"ventricular_septal_defect": ["v001"]}}
REPORT_JSON = {"classes": {c: v for c, v in REPORT["classes"].items() if c != "normal"},
               "evidence": REPORT["evidence"]}
STATE = {"camera": {"position": [0.0, 300.0, 0.0], "zoom": 1.0}, "clip": None, "render": {"preset": "P"}}


def png(v):
    b = io.BytesIO()
    Image.new("RGB", (16, 16), (v, v, v)).save(b, format="PNG")
    return b.getvalue()


def step(k, tool, args, frames=(), result=None, error=None, warnings=(), reasoning=None, view_ref=None,
         state=None, tokens=None):
    return {"step": k, "t": f"2026-01-01T00:00:{k:02d}.000Z", "model_reasoning": reasoning, "tool": tool,
            "args": args, "args_after_validation": args, "state_after": state or STATE,
            "frames": [f"frames/{k:03d}_{v}.png" for v in frames], "view_ref": view_ref, "result": result,
            "latency_ms": {"server_ms": 10.0 + k}, "tokens": tokens, "warnings": list(warnings), "error": error}


def obs(k, tool, cid, frames=(), is_error=False):
    return {"kind": "observation", "tool": tool, "call_id": cid, "step": k, "is_error": is_error, "local": False,
            "frames": list(frames), "text": "{}"}


def turn(n, calls, text=""):
    return {"kind": "model_turn", "turn": n, "stop": "tool_use" if calls else "end_turn", "text": text,
            "calls": [{"id": c[0], "name": c[1], "args": c[2], "reasoning": c[3]} for c in calls],
            "usage": {"input": 1000, "output": 50, "cache_read": 0, "cache_write": 0, "cost_usd": 0.01}, "extra": {}}


def write_episode(root, episode_id=E, report=True, transcript=True, reset_frames=("3d", "axial", "sagittal", "coronal")):
    """traces/<E>/ and runs/<E>/ under root: reset, orbit (warning), a two-call turn with an error step, a nudge,
    a bookmark, submit_report (accepted unless report=False, then the episode ends on the step budget)."""
    td, rd = root / "traces" / episode_id, root / "runs" / episode_id
    (td / "frames").mkdir(parents=True)
    (root / "runs").mkdir(exist_ok=True)
    if transcript:
        rd.mkdir()
    st2 = {**STATE, "camera": {"position": [10.0, 300.0, 0.0], "zoom": 1.0}}
    steps = [step(0, "reset_episode", {"case_id": "case_nii_0123456789"}, reset_frames,
                  {"case_info": {"dims": [512, 512, 419]}}),
             step(1, "orbit", {"azimuth_deg": 400}, ["3d"], None, warnings=["azimuth_deg clamped to 180"],
                  reasoning="rotate to see the left side", state=st2, tokens={"in": 1000, "out": 50}),
             step(2, "set_slice", {"view": "nope", "position_mm": 0}, (), None,
                  error={"type": "invalid_argument", "message": "view must be one of axial|sagittal|coronal"},
                  reasoning="try a slice", state=st2, tokens={"in": 1000, "out": 50}),
             step(3, "set_slice", {"view": "axial", "position_mm": -400}, ["axial"], None, state=st2,
                  tokens={"in": 0, "out": 0}),
             step(4, "bookmark_view", {"label": "septum"}, ["3d", "axial", "sagittal", "coronal"],
                  {"view_ref": "v001"}, reasoning="bookmark the septum", view_ref="v001", state=st2,
                  tokens={"in": 1000, "out": 50})]
    tr = [turn(1, [("c1", "orbit", {"azimuth_deg": 400}, "rotate to see the left side")]),
          obs(1, "orbit", "c1", ["3d"]),
          turn(2, [("c2", "set_slice", {"view": "nope", "position_mm": 0}, "try a slice"),
                   ("c3", "set_slice", {"view": "axial", "position_mm": -400}, None)]),
          obs(2, "set_slice", "c2", is_error=True), obs(3, "set_slice", "c3", ["axial"]),
          turn(3, [], text="Thinking out loud without a tool call."),
          {"kind": "user_text", "text": "No tool was called. Continue by calling a tool."},
          turn(4, [("c4", "bookmark_view", {"label": "septum"}, "bookmark the septum")]),
          obs(4, "bookmark_view", "c4", ["3d", "axial", "sagittal", "coronal"])]
    k = len(steps)
    if report:
        steps.append(step(k, "submit_report", {"report": REPORT}, (), {"accepted": True},
                          reasoning="submitting", state=steps[-1]["state_after"], tokens={"in": 1000, "out": 50}))
        tr += [turn(9, [("c9", "submit_report", {"report": REPORT}, "submitting")]), obs(k, "submit_report", "c9")]
        (td / "report.json").write_text(json.dumps(REPORT_JSON))
        (td / "normal.json").write_text(json.dumps({"normal": REPORT["classes"]["normal"], "evidence": [],
                                                    "report_schema_version": "diagnosis-normal-v1"}))
    for s in steps:
        for i, f in enumerate(s["frames"]):
            (td / f).write_bytes(png(10 * s["step"] + i))
    meta = {"episode_id": episode_id, "case_id": "case_nii_0123456789", "arm": "3d_seg", "model": {"id": "m", "version": "m-1"},
            "system_prompt_sha256": "a" * 64, "client_meta": {"prompt_version": "chd_ct_v1", "max_steps": 40,
                                                             "cost_cap_usd": 2.0},
            "tool_schema_version": "slicer-mcp-tools-v1", "report_schema_version": "diagnosis-normal-v1",
            "protocol": {"config_version": "env-v1", "config_sha256": "b" * 64},
            "start_time": "2026-01-01T00:00:00.000Z", "end_time": "2026-01-01T00:01:00.000Z",
            "status": "submitted" if report else "abandoned"}
    (td / "meta.json").write_text(json.dumps(meta))
    (td / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    summary = {"episode_id": episode_id, "case": "case_nii_0123456789", "arm": "3d_seg", "model": "m", "model_version": "m-1",
               "stop_reason": "submitted" if report else "step_budget", "n_model_steps": len(steps) - 1,
               "n_model_turns": 5, "tokens": {"in": 5000, "out": 250, "cache_read": 100, "cache_write": 0},
               "cost_usd": 0.05, "wall_s": {"total": 12.0, "model_s": 10.0, "env_s": 2.0},
               "report": REPORT if report else None, "client_meta": meta["client_meta"]}
    if transcript:
        (rd / "summary.json").write_text(json.dumps(summary))
        tr = [{"kind": "user_text", "text": "The case is loaded."}, obs(0, "reset_episode", None)] + tr
        (rd / "transcript.jsonl").write_text("".join(json.dumps(x) + "\n" for x in tr))
    return root / "traces", root / "runs"


def pdf_text(path):
    d = pypdfium2.PdfDocument(str(path))
    try:
        return len(d), "\n".join(d[i].get_textpage().get_text_range() for i in range(len(d)))
    finally:
        d.close()


def test_renders_all_parts_in_order(tmp_path):
    traces, runs = write_episode(tmp_path)
    out = render_episode_pdf(E, traces_dir=traces, runs_dir=runs, tz="Asia/Kolkata")
    assert out == runs / E / "report.pdf" and out.read_bytes()[:5] == b"%PDF-"
    n, text = pdf_text(out)
    assert 2 <= n <= 8 and f"Page 1 / {n}" in text and E in text
    for needle in ("CHD episode report", "Classification: VSD", "Ventricular septal defect (VSD)", "v001 (step 4)",
                   "Score: n/a (no diagnosis flags for this case)",
                   "reset_episode {\"case_id\": \"case_nii_0123456789\"}", "orbit {\"azimuth_deg\": 400}",
                   "azimuth_deg clamped to 180", "Step 2 · ERROR", "view must be one of",
                   "model turn 2, call 1 of 2 · stop tool_use · in 1,000 · out 50", "model turn 2, call 2 of 2 · server",
                   "(no new reasoning since the previous call of this turn)", "Thinking out loud without",
                   "Model turn 3 stop end_turn · no tool call", "Runner message to the model: No tool was called",
                   "Report submitted (args.report)", "~ camera.position", "Step 0 (reset) 3D frame",
                   "view_ref v001", "server 11.0 ms · 5:30:01 AM", "rotate to see the left side"):
        assert needle in text, needle
    assert text.index("Step 1 model turn 1") < text.index("rotate to see") < text.index("orbit {")
    assert "step tokens" not in text and "1 call" not in text and "(set_slice" not in text
    order = [text.index(f"Step {k} ") for k in range(6)]
    assert order == sorted(order)
    assert text.index("Thinking out loud") < text.index("Step 4 ")


def test_deterministic_and_path_independent(tmp_path):
    traces, runs = write_episode(tmp_path / "a")
    tz = "Asia/Kolkata"
    a = render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "1.pdf", tz=tz).read_bytes()
    b = render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "2.pdf", tz=tz).read_bytes()
    shutil.copytree(tmp_path / "a", tmp_path / "moved")
    c = render_episode_pdf(E, traces_dir=tmp_path / "moved" / "traces", runs_dir=tmp_path / "moved" / "runs",
                           out=tmp_path / "3.pdf", tz=tz).read_bytes()
    assert a == b == c
    d = render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "4.pdf", tz="UTC").read_bytes()
    assert d != a


def test_no_report_path(tmp_path):
    traces, runs = write_episode(tmp_path, report=False)
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs))
    assert "No report accepted (stop reason step_budget)" in text and "submit_report" not in text


def test_view_names():
    assert view_name("frames/001_3d_anterior.png") == "3d_anterior" and view_name("frames/000_axial.png") == "axial"


def test_no_cover_image_and_trace_only(tmp_path):
    traces, runs = write_episode(tmp_path, transcript=False, reset_frames=("axial",))
    with pytest.raises(FileNotFoundError, match="no run directory"):
        render_episode_pdf(E, traces_dir=traces, runs_dir=runs)
    assert not (runs / E).exists()
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "t.pdf"))
    assert "No 3D image available" in text and "No transcript.jsonl" in text and "orbit {" in text
    assert "step tokens in 1,000 out 50" in text


def test_missing_trace_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_episode_pdf(E, traces_dir=tmp_path, runs_dir=tmp_path)


def test_truth_is_scored(tmp_path):
    traces, runs = write_episode(tmp_path)
    (tmp_path / "truth.json").write_text(json.dumps(["VSD", "DORV"]))
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, truth=tmp_path / "truth.json"))
    assert "truth: given): exact match 0, F1 0.667" in text and "truth VSD + DORV" in text
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, truth=["not_a_class"]))
    assert "Score: could not score" in text


def test_long_text_wrapped_not_cut(tmp_path):
    line = "  " + " ".join(f"w{i}" for i in range(300))
    pieces = wrap_mono(line, 60)
    assert all(len(p) <= 62 for p in pieces)
    assert "".join(p if i == 0 else p.split("» ", 1)[1] for i, p in enumerate(pieces)) == line
    block = report_pdf.mono_block("\n".join(str(i) for i in range(1000)), max_lines=50)
    assert "[truncated: showing 50 of 1000 lines" in block.lines[-1]
    assert report_pdf.clean("a → b 中") == "a -> b [U+4E2D]"


def test_cli(tmp_path, monkeypatch, capsys, _reports_dir_in_tmp):
    traces, runs = write_episode(tmp_path)
    monkeypatch.setattr("chd_client.run_episode.traces_dir_for", lambda runtime=None: traces)
    assert report_pdf.main([E, "--runs-dir", str(runs), "--out", str(tmp_path / "x.pdf"), "--tz", "UTC"]) == 0
    copy = _reports_dir_in_tmp / "2026-01-01_00-00_case_nii_0123456789_3d_seg_m_abcdef.pdf"
    printed = capsys.readouterr().out
    assert (tmp_path / "x.pdf").exists() and str(tmp_path / "x.pdf") in printed and str(copy) in printed
    assert copy.read_bytes() == (tmp_path / "x.pdf").read_bytes()
    other = tmp_path / "elsewhere"
    assert report_pdf.main([E, "--runs-dir", str(runs), "--tz", "Asia/Kolkata", "--reports-dir", str(other)]) == 0
    assert [p.name for p in other.iterdir()] == ["2026-01-01_05-30_case_nii_0123456789_3d_seg_m_abcdef.pdf"]
    capsys.readouterr()
    assert report_pdf.main([E, "--runs-dir", str(runs), "--no-copy", "--reports-dir", str(tmp_path / "none")]) == 0
    assert not (tmp_path / "none").exists() and len(capsys.readouterr().out.splitlines()) == 1


def test_cli_trace_only_writes_only_to_reports(tmp_path, monkeypatch, capsys, _reports_dir_in_tmp):
    traces, runs = write_episode(tmp_path, transcript=False)
    monkeypatch.setattr("chd_client.run_episode.traces_dir_for", lambda runtime=None: traces)
    assert report_pdf.main([E, "--runs-dir", str(runs), "--tz", "UTC"]) == 0
    assert not (runs / E).exists() and list(runs.iterdir()) == []
    pdf = _reports_dir_in_tmp / "2026-01-01_00-00_case_nii_0123456789_3d_seg_m_abcdef.pdf"
    assert pdf.read_bytes()[:5] == b"%PDF-" and capsys.readouterr().out.strip() == str(pdf)
    with pytest.raises(SystemExit):
        report_pdf.main([E, "--runs-dir", str(runs), "--no-copy"])
    assert not (runs / E).exists()


def test_report_file_name():
    meta = {"start_time": "2026-10-01T08:46:10.995Z", "case_id": "case_nii_0123456789", "arm": "3d_seg",
            "model": {"id": "claude-opus-5-5"}}
    e = "20261001T084610Z_case_nii_0123456789_3d_seg_6efcab"
    assert report_file_name(e, meta, "Asia/Kolkata") == ("2026-10-01_14-16_case_nii_0123456789_3d_seg_claude-opus-5-5"
                                                        "_6efcab.pdf")
    assert report_file_name(e, meta, "UTC") == "2026-10-01_08-46_case_nii_0123456789_3d_seg_claude-opus-5-5_6efcab.pdf"
    meta2 = {**meta, "model": {"id": "openrouter:qwen/qwen3-vl"}, "start_time": None}
    assert report_file_name(e, meta2) == "unknown-time_case_nii_0123456789_3d_seg_openrouter-qwen-qwen3-vl_6efcab.pdf"


def test_time_formatting():
    ist, ny = get_zone("Asia/Kolkata"), get_zone("America/New_York")
    assert fmt_time("2026-10-01T08:46:07.123Z", ist) == "Thu 1 Oct 2026, 2:16:07 PM (IST)"
    assert fmt_time("2026-10-01T08:46:07Z", ist, date=False) == "2:16:07 PM"
    assert fmt_time("2026-10-01T18:30:00Z", ist) == "Fri 2 Oct 2026, 12:00:00 AM (IST)"
    assert fmt_time("2026-10-01T06:30:00+00:00", ist, date=False) == "12:00:00 PM"
    assert fmt_time("2026-07-01T12:00:00Z", ny) == "Wed 1 Jul 2026, 8:00:00 AM (EDT)"
    assert fmt_time("2026-01-01T12:00:00Z", ny) == "Thu 1 Jan 2026, 7:00:00 AM (EST)"
    assert fmt_time("2026-10-01T08:46:07", get_zone("UTC")) == "Thu 1 Oct 2026, 8:46:07 AM (UTC)"
    assert fmt_time(None, ist) == "n/a" and fmt_time("not a time", ist) == "not a time"
    m = {"start_time": "2026-10-01T08:46:10.995Z", "end_time": "2026-10-01T08:48:25.304Z"}
    assert start_end(m, ist) == "Thu 1 Oct 2026, 2:16:10 PM (IST) -> 2:18:25 PM (2 min 14 s)"
    m = {"start_time": "2026-10-01T18:29:50Z", "end_time": "2026-10-01T18:30:05Z"}
    assert start_end(m, ist) == ("Thu 1 Oct 2026, 11:59:50 PM (IST) -> Fri 2 Oct 2026, 12:00:05 AM (IST) "
                                 "(15 s)")
    assert start_end({}, ist) == "n/a -> n/a"
    with pytest.raises(ValueError, match="unknown time zone 'Mars/Olympus'"):
        get_zone("Mars/Olympus")


def test_tz_on_cover_and_steps(tmp_path):
    traces, runs = write_episode(tmp_path)
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, tz="Asia/Kolkata"))
    assert "Thu 1 Jan 2026, 5:30:00 AM (IST) -> 5:31:00 AM (1 min 00 s)" in text
    assert "Asia/Kolkata (IST, UTC+05:30); all times in this report" in text
    assert "2026-01-01T00:00" not in text
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs))
    assert "Thu 1 Jan 2026, 12:00:00 AM (UTC)" in text and "UTC (UTC, UTC+00:00)" in text
    with pytest.raises(ValueError):
        render_episode_pdf(E, traces_dir=traces, runs_dir=runs, tz="Nowhere/Here")


def test_cli_tz(tmp_path, monkeypatch, capsys):
    traces, runs = write_episode(tmp_path)
    monkeypatch.setattr("chd_client.run_episode.traces_dir_for", lambda runtime=None: traces)
    out = tmp_path / "x.pdf"
    assert report_pdf.main([E, "--runs-dir", str(runs), "--out", str(out), "--tz", "America/New_York"]) == 0
    assert "Wed 31 Dec 2025, 7:00:00 PM (EST)" in pdf_text(out)[1]
    monkeypatch.setattr(report_pdf, "system_tz_name", lambda: "Europe/Berlin")
    assert report_pdf.main([E, "--runs-dir", str(runs), "--out", str(out)]) == 0
    assert "Europe/Berlin (CET, UTC+01:00)" in pdf_text(out)[1]
    with pytest.raises(SystemExit) as ex:
        report_pdf.main([E, "--runs-dir", str(runs), "--out", str(out), "--tz", "Nowhere/Here"])
    assert ex.value.code == 2 and "unknown time zone 'Nowhere/Here'" in capsys.readouterr().err


def test_system_tz_name(monkeypatch):
    monkeypatch.setenv("TZ", "Pacific/Auckland")
    assert report_pdf.system_tz_name() == "Pacific/Auckland"
    monkeypatch.setenv("TZ", ":Asia/Tokyo")
    assert report_pdf.system_tz_name() == "Asia/Tokyo"
    monkeypatch.setenv("TZ", "Not/AZone")
    get_zone(report_pdf.system_tz_name())


def test_a_segmental_report_renders_unscored(tmp_path):
    """A run with a segmental report (another report format) keeps its PDF: the report is shown as submitted, nothing is
    lost, and it is not scored."""
    traces, runs = write_episode(tmp_path)
    old = {"visceroatrial_situs": "solitus", "ventricular_loop": "D-loop", "primary_diagnosis": "VSD",
           "associated_findings": [{"type": "VSD", "subtype": "perimembranous"}],
           "free_text_impression": "old impression", "confidence": {"ventricular_loop": 0.8}}
    (traces / E / "report.json").write_text(json.dumps(old))
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "v0.pdf"))
    for needle in ("not a diagnosis report: shown as submitted", "D-loop", "perimembranous", "old impression",
                   "Score: n/a (no ground truth for this case)"):
        assert needle in text, needle
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "v1.pdf", truth=["VSD"]))
    assert "Score: n/a (only a diagnosis report is scored)" in text


def test_run_files_section(tmp_path):
    """Page 1 says whether the trace, report.json and metrics.json (its diagnosis part) are saved; a missing file is
    flagged."""
    traces, runs = write_episode(tmp_path)
    td = traces / E
    from chd_client.task.scoring.multilabel import score
    sc = {**score(REPORT_JSON, ["VSD"]), "truth_source": "imageCHD_dataset_info.xlsx"}
    (runs / E / "metrics.json").write_text(json.dumps({"diagnosis": sc}))
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "a.pdf"))
    for needle in ("Run files", "6 lines in steps.jsonl for 5 model steps + reset", "10 of 10 present",
                   "trace report.json", "metrics.json: diagnosis"):
        assert needle in text, needle
    assert "MISSING" not in text and "shells" not in text.lower() and "myocardium" not in text.lower()
    (td / "frames" / "004_axial.png").unlink()
    (td / "report.json").unlink()
    (runs / E / "metrics.json").write_text(json.dumps({"diagnosis": {"error": "no truth"}}))
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "b.pdf"))
    for needle in ("9 of 10 present; missing frames/004_axial.png", "no report.json in the trace",
                   "metrics.json: diagnosis error", "check this run before using its numbers"):
        assert needle in text, needle
    (runs / E / "metrics.json").write_text(json.dumps({"pool": {"final": {}}}))
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, out=tmp_path / "c.pdf"))
    assert "metrics.json: diagnosis absent" in text
