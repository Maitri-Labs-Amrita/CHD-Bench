"""slurm/check_batch.py --rescore compares the diagnosis only (no segmentation numbers)."""

import importlib.util

from chd_client.task.config import REPO

spec = importlib.util.spec_from_file_location("check_batch", REPO / "slurm" / "check_batch.py")
check_batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_batch)


def dx(**kw):
    base = {"truth": ["ventricular_septal_defect"], "predicted": ["ventricular_septal_defect"],
            "per_class": {"ventricular_septal_defect": "tp"}, "exact_match": 1.0, "f1": 1.0, "hamming_accuracy": 1.0}
    return {"diagnosis": {**base, **kw}, "tool_use": {"steps": 3}}


def test_rescore_compares_the_diagnosis():
    assert check_batch.score_parts(dx()) == check_batch.score_parts(dx())
    assert check_batch.score_parts(dx()) != check_batch.score_parts(dx(f1=0.0, exact_match=0.0))
    assert check_batch.score_parts(dx()) != check_batch.score_parts(dx(predicted=[]))
    assert check_batch.score_parts(dx()) != check_batch.score_parts({"diagnosis": {"skipped": "no report submitted"}})


def test_no_segmentation_numbers_and_no_source_fields():
    m = dx(run="a", data_source="dataset", truth_source="sheet.xlsx")
    m["pool"] = {"final": {"dice": 0.9}}
    assert set(check_batch.score_parts(m)) == set(dx()["diagnosis"])


def test_confidence_only_changes_are_caught():
    assert check_batch.score_parts(dx(brier=0.01)) != check_batch.score_parts(dx(brier=0.2))
    assert check_batch.score_parts(dx(normal_call={"correct": True})) != \
        check_batch.score_parts(dx(normal_call={"correct": False}))


def test_a_submitted_run_must_be_scored(tmp_path, monkeypatch):
    import json
    import subprocess
    skip = {"diagnosis": {"skipped": "diagnosis sheet not found"}}
    (tmp_path / "metrics.json").write_text(json.dumps(skip))
    monkeypatch.setattr(check_batch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=json.dumps(skip) + "\n", stderr=""))
    errs, _ = check_batch.rescore(tmp_path, tmp_path / "rt.json", "submitted")
    assert any("no diagnosis scores (diagnosis sheet not found)" in e for e in errs)
    assert check_batch.rescore(tmp_path, tmp_path / "rt.json", "step_budget")[0] == []


def test_a_run_with_no_scores_on_either_side_is_a_problem(tmp_path, monkeypatch):
    import json
    import subprocess
    (tmp_path / "metrics.json").write_text(json.dumps({"diagnosis": {"error": "sheet missing"}}))
    out = json.dumps({"diagnosis": {"error": "sheet missing"}})
    monkeypatch.setattr(check_batch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out + "\n", stderr=""))
    errs, _ = check_batch.rescore(tmp_path, tmp_path / "rt.json")
    assert any("metrics.json: no diagnosis scores (sheet missing)" in e for e in errs)


def test_a_rescore_error_is_reported_once(tmp_path, monkeypatch):
    import json
    import subprocess
    (tmp_path / "metrics.json").write_text(json.dumps(dx()))
    out = json.dumps({"diagnosis": {"error": "boom"}})
    monkeypatch.setattr(check_batch.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out + "\n", stderr=""))
    errs, _ = check_batch.rescore(tmp_path, tmp_path / "rt.json")
    assert sum("boom" in e for e in errs) == 1
