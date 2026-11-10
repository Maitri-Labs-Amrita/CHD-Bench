"""The multi-label classification: schema, scorer, eval_diagnosis, and the episode."""

import json

import pytest
from pydantic import ValidationError

from chd_client.task import eval_diagnosis, imagechd
from slicer_mcp.config import load_profile

from chd_client.task.config import PROFILE
from chd_client.task.diagnosis_schema import CHD_CLASSES, IMAGECHD_FLAG, DiagnosisNormalReport, DiagnosisReport, format_errors, is_diagnosis_report
from chd_client.task.scoring.multilabel import score, truth_classes
from tests.task.report_examples import DX_CLASSES, NORMAL, NORMAL_DX, dx

N = 1008


def test_sixteen_classes_map_one_to_one_onto_the_imagechd_flags():
    assert CHD_CLASSES == DX_CLASSES and len(CHD_CLASSES) == 16
    assert tuple(IMAGECHD_FLAG.values()) == imagechd.DIAGNOSES
    assert IMAGECHD_FLAG["pulmonary_atresia"] == "PA" and IMAGECHD_FLAG["coarctation"] == "CA"


def test_the_profile_report_is_the_scorers_schema():
    """The model-facing report (profiles/chd_ct_v1/report.py, loaded by slicer-mcp) and the scorer's copy
    (task.diagnosis_schema) are the same schema: the same classes, flags and JSON schema."""
    plugin = load_profile(PROFILE).report
    assert plugin.SCHEMA_VERSION == "diagnosis-normal-v1" and plugin.CHD_CLASSES == CHD_CLASSES
    assert plugin.Model.model_json_schema() == DiagnosisNormalReport.model_json_schema()
    def body(m):
        return {k: v for k, v in m.model_json_schema().items() if k != "description"}
    assert body(plugin.DiagnosisReport) == body(DiagnosisReport)


def test_valid_reports():
    r = DiagnosisReport.model_validate(dx("double_outlet_right_ventricle", "ventricular_septal_defect"))
    assert r.present() == ["ventricular_septal_defect", "double_outlet_right_ventricle"]
    assert DiagnosisReport.model_validate(NORMAL_DX).present() == []
    ok = {**NORMAL_DX, "evidence": {"double_svc": ["v001", "v002"]}}
    assert DiagnosisReport.model_validate(ok, context={"view_refs": {"v001", "v002"}}).view_refs() == {"v001", "v002"}


def test_every_problem_is_reported_at_once():
    classes = dict(NORMAL_DX["classes"])
    del classes["double_svc"]
    classes["single_ventricle"] = {"present": True, "confidence": 0.9}
    classes["coarctation"] = {"present": True, "confidence": 0.3}
    classes["tetralogy_of_fallot"] = {"present": True}
    bad = {"classes": classes, "evidence": {"segmental_set": ["v001"], "coarctation": ["v009", "x1"]},
           "impression": "text"}
    with pytest.raises(ValidationError) as e:
        DiagnosisReport.model_validate(bad, context={"view_refs": {"v001"}})
    msg = format_errors(e.value)
    for part in ("classes.double_svc: required field missing", "classes.single_ventricle: unknown field",
                 "classes.coarctation.confidence: Input should be greater than or equal to 0.5",
                 "classes.tetralogy_of_fallot.confidence: required field missing",
                 "evidence: keys must be among", "bad view_refs ['x1']", "cites unknown view_refs ['v009']",
                 "impression: unknown field"):
        assert part in msg, part


def test_segmental_report_is_not_a_diagnosis_report():
    assert is_diagnosis_report(NORMAL_DX) and not is_diagnosis_report(NORMAL)
    with pytest.raises(ValidationError):
        DiagnosisReport.model_validate(NORMAL)


def test_score_perfect_partial_and_normal():
    s = score(dx("double_outlet_right_ventricle", "ventricular_septal_defect", confidence=0.9), ["DORV", "VSD"])
    assert s["exact_match"] == 1.0 and s["f1"] == 1.0 and s["hamming_accuracy"] == 1.0 and s["jaccard"] == 1.0
    assert s["brier"] == pytest.approx(0.01)
    assert s["per_class"]["double_outlet_right_ventricle"] == "tp" and s["per_class"]["double_svc"] == "tn"

    s = score(dx("tetralogy_of_fallot", "ventricular_septal_defect"), ["DORV", "VSD"])
    assert s["exact_match"] == 0.0 and s["precision"] == 0.5 and s["recall"] == 0.5 and s["f1"] == 0.5
    assert s["jaccard"] == pytest.approx(1 / 3, abs=1e-6) and s["hamming_accuracy"] == 14 / 16
    assert s["per_class"]["tetralogy_of_fallot"] == "fp" and s["per_class"]["double_outlet_right_ventricle"] == "fn"
    assert s["predicted"] == ["ventricular_septal_defect", "tetralogy_of_fallot"]
    assert s["truth"] == ["ventricular_septal_defect", "double_outlet_right_ventricle"]

    s = score(NORMAL_DX, [])
    assert s["exact_match"] == s["f1"] == s["precision"] == s["recall"] == s["jaccard"] == 1.0
    s = score(NORMAL_DX, ["ASD"])
    assert s["exact_match"] == s["f1"] == s["recall"] == s["precision"] == s["jaccard"] == 0.0
    s = score(dx("atrial_septal_defect"), [])
    assert s["exact_match"] == s["f1"] == s["precision"] == s["recall"] == 0.0


def test_score_accepts_class_names_and_rejects_unknown_truth():
    assert truth_classes(["PA", "pulmonary_atresia", "VSD"]) == {"pulmonary_atresia", "ventricular_septal_defect"}
    with pytest.raises(ValueError, match="unknown class or flag 'HLHS'"):
        score(NORMAL_DX, ["HLHS"])


def _run(tmp_path, report, case_source=None):
    run = tmp_path / "run"
    (run / "trace").mkdir(parents=True)
    meta = {"episode_id": "e", "case_id": "case_nii_0123456789", "arm": "3d", "status": "submitted",
            "client_meta": {"case_source": case_source or {"kind": "imagechd_temporary", "imagechd_id": N}}}
    (run / "trace" / "meta.json").write_text(json.dumps(meta))
    (run / "trace" / "steps.jsonl").write_text("")
    if report is not None:
        (run / "trace" / "report.json").write_text(json.dumps(report))
    return run


def test_eval_diagnosis_scores_an_imagechd_run_against_its_flags(tmp_path, capsys):
    pytest.importorskip("SimpleITK")
    from tests.task.imagechd_fake import make_dataset
    root = make_dataset(tmp_path / "ds", flags={1008: ["DORV", "VSD"]})
    out = eval_diagnosis.evaluate_run(_run(tmp_path, dx("double_outlet_right_ventricle")), root)
    assert out["dataset"] == "ImageCHD" and out["imagechd_id"] == N and out["truth_source"] == imagechd.DIAGNOSIS_INFO
    assert out["truth"] == ["ventricular_septal_defect", "double_outlet_right_ventricle"]
    assert out["recall"] == 0.5 and out["precision"] == 1.0 and out["scorer_version"] == "multilabel-2"
    eval_diagnosis.main(["--run", str(tmp_path / "run"), "--dir", str(root), "--json"])
    assert json.loads(capsys.readouterr().out)["f1"] == out["f1"]


def test_eval_diagnosis_given_truth_and_skips(tmp_path):
    no_dataset = {"kind": "other_temporary"}
    out = eval_diagnosis.evaluate_run(_run(tmp_path, dx("double_outlet_right_ventricle"), no_dataset), truth=["DORV"])
    assert out["truth_source"] == "given" and out["exact_match"] == 1.0
    assert "no ground truth" in eval_diagnosis.evaluate_run(tmp_path / "run")["skipped"]
    (tmp_path / "run" / "trace" / "report.json").write_text(json.dumps(NORMAL))
    assert "segmental" in eval_diagnosis.evaluate_run(tmp_path / "run", truth=[])["skipped"]
    (tmp_path / "run" / "trace" / "report.json").unlink()
    assert eval_diagnosis.evaluate_run(tmp_path / "run")["skipped"] == "no report submitted"


def test_excluded_classes_leave_every_number():
    s = score(dx("atrioventricular_septal_defect", "double_outlet_right_ventricle"), ["DORV"], exclude=["AVSD"])
    assert s["exact_match"] == 1.0 and s["precision"] == 1.0 and "atrioventricular_septal_defect" not in s["per_class"]
    assert len(s["per_class"]) == 15 and s["excluded"] == ["atrioventricular_septal_defect"]
    assert s["hamming_accuracy"] == 1.0 and score(NORMAL_DX, [])["excluded"] == []


def test_imagechd_excludes_avsd_and_unflagged_truth_is_normal(tmp_path):
    """config/scoring_v1.json: AVSD is not scored (the sheet's AVSD column is ASD AND VSD), and a case without flags
    is a normal heart."""
    pytest.importorskip("SimpleITK")
    from tests.task.imagechd_fake import make_dataset
    root = make_dataset(tmp_path / "ds", flags={1008: ["DORV"]})
    out = eval_diagnosis.evaluate_run(_run(tmp_path, dx("double_outlet_right_ventricle",
                                                        "atrioventricular_septal_defect")), root)
    assert out["truth_verified"] is True and out["excluded"] == ["atrioventricular_septal_defect"]
    assert "atrioventricular_septal_defect" not in out["per_class"] and out["exact_match"] == 1.0
    run2 = tmp_path / "r2"
    _run(run2, NORMAL_DX, {"kind": "imagechd_temporary", "imagechd_id": 1003})
    out = eval_diagnosis.evaluate_run(run2 / "run", root)
    assert out["truth_verified"] is True and "truth_note" not in out and out["exact_match"] == 1.0
