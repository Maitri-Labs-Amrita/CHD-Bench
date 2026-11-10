"""chd_ct_v1's report (profiles/chd_ct_v1/report.py) takes the 16 classes and normal (no defect); normal is present
exactly when none of the 16 is. report.json keeps the 16 classes (the scorer is unchanged) and normal.json the normal
call."""

import json

import pytest

from pydantic import ValidationError
from slicer_mcp.config import load_profile
from slicer_mcp.reports import format_errors
from slicer_mcp.tools.registry import tool_specs

from chd_client.task.config import PROFILE
from chd_client.task.diagnosis_schema import CHD_CLASSES, DiagnosisReport
from chd_client.task.eval_diagnosis import _with_normal

PLUGIN = load_profile(PROFILE).report


def report(defects=(), normal=False):
    classes = {c: {"present": c in defects, "confidence": 0.8} for c in CHD_CLASSES}
    return {"classes": {**classes, "normal": {"present": normal, "confidence": 0.7}}}


class Rejected(Exception):
    pass


def submit(rep):
    """The plugin's validation, with the server's rejection message (slicer_mcp.reports.format_errors)."""
    try:
        return PLUGIN.Model.model_validate(rep, context={"view_refs": set()})
    except ValidationError as e:
        raise Rejected(format_errors(e)) from None


def test_normal_or_defects_but_not_both_or_neither():
    assert submit(report(normal=True)).present() == []
    assert submit(report(["coarctation"])).present() == ["coarctation"]
    with pytest.raises(Rejected, match="normal means no defect"):
        submit(report(["coarctation"], normal=True))
    with pytest.raises(Rejected, match="no class is present and normal is absent"):
        submit(report())
    with pytest.raises(Rejected, match="normal: required field missing"):
        submit({"classes": {c: {"present": False, "confidence": 0.8} for c in CHD_CLASSES}})


def test_scored_report_is_the_16_classes():
    result, files = PLUGIN.accept(submit(report(normal=True)))
    assert result == {"accepted": True, "present": [], "normal": True}
    assert DiagnosisReport.model_validate(files["report.json"]).present() == []
    assert set(files["report.json"]["classes"]) == set(CHD_CLASSES)
    assert files["normal.json"]["normal"] == {"present": True, "confidence": 0.7}


def test_tools_and_versions():
    prof = load_profile(PROFILE)
    assert PLUGIN.SCHEMA_VERSION == "diagnosis-normal-v1" and "record_signs" not in prof.tools


def test_submit_report_text_and_schema():
    sub = {s.name: s for s in tool_specs(load_profile(PROFILE))}["submit_report"]
    assert "normal is present exactly when none of the 16 is" in sub.description
    assert '"normal"' in json.dumps(sub.input_schema)


def test_the_normal_call_is_scored_against_the_truth(tmp_path):
    (tmp_path / "trace").mkdir()
    (tmp_path / "trace" / "normal.json").write_text(json.dumps({"normal": {"present": True, "confidence": 0.9}}))
    assert _with_normal(tmp_path, {"truth": []})["normal_call"] == {"present": True, "confidence": 0.9,
                                                                    "truth": True, "correct": True}
    assert _with_normal(tmp_path, {"truth": ["coarctation"]})["normal_call"]["correct"] is False
    assert "normal_call" not in _with_normal(tmp_path / "x", {"truth": []})


def test_session_writes_both_files(tmp_path):
    """Through slicer-mcp's session with the chd_ct_v1 profile (a fake bridge): the model's result, report.json (the 16
    classes) and normal.json."""
    from tests.fakes import CASE, make_env
    srv, _, _ = make_env(tmp_path)
    sess = srv.session
    sess.reset(CASE)
    out = sess.call("submit_report", {"report": report(normal=True)})
    assert out.ok and out.result == {"accepted": True, "present": [], "normal": True}
    d = sess.last_trace_dir
    assert set(json.loads((d / "report.json").read_text())["classes"]) == set(CHD_CLASSES)
    assert json.loads((d / "normal.json").read_text())["normal"] == {"present": True, "confidence": 0.7}
    assert json.loads((d / "meta.json").read_text())["report_schema_version"] == "diagnosis-normal-v1"
