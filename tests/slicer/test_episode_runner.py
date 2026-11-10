"""Integration: the runner with a scripted model through the real stdio server (`slicer-mcp serve` with the chd_ct_v1
profile, a subprocess) on the live bridge, one 3d_seg episode with the registry prompt; its trace is checked and
replayed. Plus one budget-exhausted episode."""

import anyio
import pytest

from chd_client import prompt_registry
from chd_client.mcp_session import server_params
from chd_client.models.scripted import ScriptedModel
from chd_client.run_episode import (PROMPT_DIR, default_max_steps, prompt_version, run_episode,
                                    traces_dir_for)
from slicer_mcp.tracing.logger import read_trace
from slicer_mcp.tracing.replay import replay
from tests.fakes import NORMAL

pytestmark = pytest.mark.slicer
SUBMIT = ("submit_report", {"report": NORMAL})
SCRIPT = [[("point_at", {"view": "axial", "x": 400, "y": 320}, "locate the heart first")],
          [("orbit", {"azimuth_deg": 30, "elevation_deg": 10}), ("set_view_direction", {"name": "left"})],
          [("zoom", {"factor": 1.2})], [("bookmark_view", {"label": "left"})],
          [("set_segment_visibility", {"hide": ["LA", "RA"]})], [("set_slice_outlines", {"visible": False})],
          [("center_on_segments", {})], [("pan_to", {})], [SUBMIT]]


def run(runtime_file, tmp_path, model, case_id, **kw):
    traces = traces_dir_for(runtime_file)
    prompt_registry.check_runnable(prompt_registry.entry(PROMPT_DIR, prompt_version()))
    target = server_params(runtime=runtime_file)
    summary = anyio.run(lambda: run_episode(case_id, "3d_seg", model, target,
                                            runs_dir=tmp_path / "runs", traces_dir=traces, echo=None, **kw))
    return summary, traces / summary["episode_id"]


def test_scripted_episode_over_stdio(bridge, runtime_file, tmp_path, live_case):
    model = ScriptedModel(SCRIPT)
    summary, d = run(runtime_file, tmp_path, model, live_case.case_id)
    assert summary["stop_reason"] == "submitted", summary
    meta, steps = read_trace(d)
    assert meta["status"] == "submitted" and meta["profile"] == "chd_ct_v1" and meta["model"]["id"] == "scripted"
    want = (prompt_version(), default_max_steps())
    assert (meta["client_meta"]["prompt_version"], meta["client_meta"]["max_steps"]) == want
    assert all(s["model_reasoning"] and s["tokens"] for s in steps[1:])
    assert steps[1]["tokens"] == {"in": 1000, "out": 50}
    assert steps[3]["tool"] == "set_view_direction" and steps[3]["tokens"] == {"in": 0, "out": 0}
    assert len(meta["tools"]) == 22 and meta["report_schema_version"] == "diagnosis-normal-v1"
    seg = [s for s in steps if s["tool"] in ("set_segment_visibility", "set_slice_outlines", "center_on_segments")]
    assert len(seg) == 3 and all(s.get("error") is None for s in seg), [s.get("error") for s in seg]
    assert (d / "report.json").is_file() and (d / "normal.json").is_file()
    assert summary["tool_latency_ms"]["reset_episode"]["n"] == 1
    rep = replay(d, client=bridge)
    assert rep["pass"], [x for x in rep["steps"] if not x.get("skipped") and not x["pass"]]


def test_budget_exhaustion_is_abandoned(bridge, runtime_file, tmp_path, live_case):
    model = ScriptedModel([lambda obs: [("orbit", {"azimuth_deg": 15})]] * 50)
    summary, d = run(runtime_file, tmp_path, model, live_case.case_id, max_steps=3)
    assert summary["stop_reason"] == "step_budget" and summary["n_model_steps"] == 3
    meta, steps = read_trace(d)
    assert meta["status"] == "abandoned" and len(steps) == 4 and not (d / "report.json").exists()
