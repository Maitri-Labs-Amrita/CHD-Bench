"""The 3d_seg arm's classification on the client side: the class list matches the task profile's, the dry run's report is valid
there, and the PDF shows the classification and its score."""

import json

from chd_client.task.diagnosis_schema import CHD_CLASSES, IMAGECHD_FLAG, DiagnosisNormalReport, DiagnosisReport

from chd_client import diagnosis
from chd_client.report_pdf import render_episode_pdf
from chd_client.run_episode import dry_run_script
from tests.test_report_pdf import E, pdf_text, write_episode


def dx(*present):
    return {"classes": {c: {"present": c in present, "confidence": 0.8} for c in diagnosis.CLASS_NAMES}}


def test_classes_are_the_envs():
    assert diagnosis.CLASS_NAMES == CHD_CLASSES
    assert diagnosis.SHORT == IMAGECHD_FLAG


def test_dry_run_submits_a_valid_classification():
    assert DiagnosisReport.model_validate(diagnosis.dry_report()).present() == []
    normal = DiagnosisNormalReport.model_validate(diagnosis.dry_normal_report())
    assert normal.present() == [] and normal.normal_call() == {"present": True, "confidence": 0.5}
    assert dry_run_script()[-1] == [("submit_report", {"report": diagnosis.dry_normal_report()})]


def test_helpers():
    r = dx("double_outlet_right_ventricle", "ventricular_septal_defect")
    assert diagnosis.is_diagnosis_report(r) and not diagnosis.is_diagnosis_report({"visceroatrial_situs": "solitus"})
    assert diagnosis.present(r) == ["ventricular_septal_defect", "double_outlet_right_ventricle"]
    assert diagnosis.present_short(r) == "VSD + DORV" and diagnosis.present_short(dx()) == "none of 16"


def _with_dx(tmp_path, report, metrics=None):
    traces, runs = write_episode(tmp_path)
    (traces / E / "report.json").write_text(json.dumps(report))
    s = json.loads((runs / E / "summary.json").read_text())
    (runs / E / "summary.json").write_text(json.dumps({**s, "report": report}))
    if metrics is not None:
        (runs / E / "metrics.json").write_text(json.dumps(metrics))
    return traces, runs


def test_pdf_shows_the_classification_and_the_flags_score(tmp_path):
    from chd_client.task.scoring.multilabel import score
    rep = {**dx("double_outlet_right_ventricle"), "evidence": {"double_outlet_right_ventricle": ["v001"]}}
    sc = {**score(rep, ["DORV", "VSD"]), "truth_source": "imageCHD_dataset_info.xlsx"}
    traces, runs = _with_dx(tmp_path, rep, {"diagnosis": sc})
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs))
    assert "Classification: DORV" in text and "Double outlet right ventricle (DORV)" in text
    assert "PRESENT" in text and "WRONG (missed)" in text and "Segmental analysis" not in text
    assert "Score (multilabel-2, truth: imageCHD_dataset_info.xlsx): exact match 0, F1 0.667" in text
    assert "v001 (step 4)" in text


def test_pdf_scores_a_given_truth_and_says_when_there_is_none(tmp_path):
    traces, runs = _with_dx(tmp_path, dx())
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs))
    assert "Classification: none of 16" in text and "Score: n/a (no diagnosis flags for this case)" in text
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs, truth=["ASD"]))
    assert "truth: given): exact match 0, F1 0.000" in text and "truth ASD" in text


def test_pdf_marks_unverified_truth_and_unscored_classes(tmp_path):
    """A case without flags is unverified; AVSD is not scored on ImageCHD."""
    from chd_client.task.scoring.multilabel import score
    sc = {**score(dx(), [], exclude=["AVSD"]), "truth_verified": False, "truth_source": "imageCHD_dataset_info.xlsx"}
    traces, runs = _with_dx(tmp_path, dx(), {"diagnosis": sc})
    _, text = pdf_text(render_episode_pdf(E, traces_dir=traces, runs_dir=runs))
    text = " ".join(text.split())
    assert "UNVERIFIED truth" in text and "Not scored: AVSD" in text and "not scored" in text
    from chd_client.run_episode import diagnosis_line
    assert diagnosis_line(sc) == "diagnosis none of 16 vs truth none of 16 (unverified: no flags): exact 1, F1 1.00"
