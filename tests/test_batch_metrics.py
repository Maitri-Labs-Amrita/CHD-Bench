"""slurm/batch_metrics.py on a hand-made batch: truth rows, the two Normal columns, a missing report, planned cases
counted once at their last attempt, and scikit-learn's numbers."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
spec = importlib.util.spec_from_file_location("batch_metrics", Path(__file__).parents[1] / "slurm" / "batch_metrics.py")
bm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bm)

VSD, ASD = "ventricular_septal_defect", "atrial_septal_defect"
FLAGS = {1: ["VSD"], 2: ["VSD", "ASD"], 3: [], 4: ["ASD"]}


def truth(case):
    return bm.imagechd_truth(FLAGS[int(case)])


def report(present, conf=0.9):
    return {"classes": {c: {"present": c in present, "confidence": conf} for c in bm.CHD_CLASSES}}


def batch(tmp_path, episodes, plan=None, job="job0_m-a"):
    """episodes: [(case, report or None, normal call or None)] of one model, in attempt order. plan: the planned
    cases (plan.json), or None for no plan.json."""
    jdir = tmp_path / job
    jdir.mkdir()
    lines = []
    for i, (case, rep, normal) in enumerate(episodes):
        rf = tmp_path / f"{job}_run{i}"
        (rf / "trace").mkdir(parents=True)
        if rep is not None:
            (rf / "trace" / "report.json").write_text(json.dumps(rep))
        if normal is not None:
            (rf / "trace" / "normal.json").write_text(json.dumps({"normal": normal}))
        lines.append(json.dumps({"model": "m", "name": "m-a", "case": case, "exit": 0 if rep else 1,
                                 "run_folder": str(rf)}))
    if lines:
        (jdir / "results.jsonl").write_text("\n".join(lines) + "\n")
    if plan is not None:
        (jdir / "plan.json").write_text(json.dumps({"batch": "b", "job": 0, "name": "m-a", "dataset": "imagechd",
                                                    "model": "m", "cases": [{"case": c} for c in plan]}))
    return tmp_path


def matrices(d):
    return (np.array([r[i] for r in d["rows"]]) for i in (2, 3, 4))


def test_columns_and_truth_rows_have_both_normal_columns():
    assert bm.columns([VSD, ASD]) == [VSD, ASD, bm.NORMAL, bm.NORMAL_EXPLICIT]
    assert bm.truth_row([], [VSD, ASD]).tolist() == [0, 0, 1, 1]
    assert bm.truth_row([VSD], [VSD, ASD]).tolist() == [1, 0, 0, 0]
    assert bm.truth_row(truth(2)).sum() == 2 and len(bm.truth_row([])) == len(bm.CHD_CLASSES) + 2


def test_prediction_rows_derive_normal_and_take_the_explicit_call_from_normal_json():
    p, s = bm.report_rows(report(set(), 0.8))
    assert p[-2:].tolist() == [1, 1] and p[:-2].sum() == 0
    assert s[-2] == pytest.approx(0.8) and s[-1] == pytest.approx(0.8)
    p, s = bm.report_rows(report(set(), 0.8), {"present": False, "confidence": 0.7})
    assert p[-2:].tolist() == [1, 0] and s[-1] == pytest.approx(0.3)
    p, s = bm.report_rows(report({VSD}, 0.9), {"present": True, "confidence": 0.6})
    assert p[-2:].tolist() == [0, 1] and s[-2] == pytest.approx(0.1) and s[-1] == pytest.approx(0.6)
    p, s = bm.report_rows(None)
    assert p.sum() == 0 and (s == 0.5).all() and len(p) == len(bm.CHD_CLASSES) + 2


def test_batch_numbers(tmp_path):
    d = bm.load(batch(tmp_path, [(1, report({VSD}), None), (2, report({VSD}), None), (3, report(set()), None),
                                 (4, None, None)]), truth)["m"]
    assert d["missing"] == [{"case": 4, "job": "m-a", "exit": 1}] and d["attempts"] == 4
    m = bm.model_metrics(*matrices(d))
    v, a, n = (m["per_class"][c] for c in (VSD, ASD, bm.NORMAL))
    assert (v["tp"], v["fp"], v["fn"], v["tn"], v["sensitivity"], v["specificity"]) == (2, 0, 0, 2, 1.0, 1.0)
    assert (a["tp"], a["fn"], a["sensitivity"], a["specificity"], a["balanced_accuracy"]) == (0, 2, 0.0, 1.0, 0.5)
    assert (n["n_pos"], n["tp"], n["fp"], n["tn"]) == (1, 1, 0, 3)
    assert set(m["macro"]["classes"]) == {ASD, VSD, bm.NORMAL}
    assert m["macro"]["sensitivity"] == pytest.approx((1.0 + 0.0 + 1.0) / 3, abs=1e-4)
    assert m["exact_match"] == pytest.approx(2 / 4) and m["n_episodes"] == 4
    assert m["per_class"]["tetralogy_of_fallot"]["auc"] is None


def test_a_case_with_a_failed_then_a_successful_attempt_counts_once(tmp_path):
    d = bm.load(batch(tmp_path, [(1, None, None), (1, report({VSD}), None), (3, report(set()), None)],
                      plan=[1, 3]), truth)["m"]
    assert [r[0] for r in d["rows"]] == [1, 3] and d["missing"] == [] and d["attempts"] == 3
    assert d["rows"][0][1].endswith("_run1")
    assert bm.model_metrics(*matrices(d))["exact_match"] == 1.0


def test_the_last_attempt_counts_even_when_it_failed(tmp_path):
    d = bm.load(batch(tmp_path, [(1, report({VSD}), None), (1, None, None)], plan=[1]), truth)["m"]
    assert len(d["rows"]) == 1 and d["missing"] == [{"case": 1, "job": "m-a", "exit": 1}]


def test_a_planned_case_that_never_ran_is_missing(tmp_path):
    d = bm.load(batch(tmp_path, [(1, report({VSD}), None)], plan=[1, 2]), truth)["m"]
    assert [r[0] for r in d["rows"]] == [1, 2] and d["attempts"] == 1
    assert d["missing"] == [{"case": 2, "job": "m-a", "exit": "not run"}]
    case, rf, y, p, s = d["rows"][1]
    assert rf is None and p.sum() == 0 and (s == 0.5).all() and y.sum() == 2


def test_a_job_with_a_plan_and_no_results_counts_every_case_as_missing(tmp_path):
    d = bm.load(batch(tmp_path, [], plan=[3, 4]), truth)["m"]
    assert d["attempts"] == 0 and [x["case"] for x in d["missing"]] == [3, 4]


def test_normal_explicit_uses_normal_json(tmp_path):
    d = bm.load(batch(tmp_path, [(3, report(set()), {"present": False, "confidence": 0.9}),
                                 (1, report({VSD}), {"present": False, "confidence": 0.8})], plan=[3, 1]),
                truth)["m"]
    m = bm.model_metrics(*matrices(d))
    n, ex = m["per_class"][bm.NORMAL], m["per_class"][bm.NORMAL_EXPLICIT]
    assert (n["tp"], n["fn"], n["tn"]) == (1, 0, 1)
    assert (ex["tp"], ex["fn"], ex["tn"]) == (0, 1, 1)
    assert bm.NORMAL_EXPLICIT not in m["macro"]["classes"]


def test_custom_classes_shape_every_row(tmp_path):
    classes = (VSD, ASD)
    rep = {"classes": {VSD: {"present": True, "confidence": 0.9}, ASD: {"present": False, "confidence": 0.9}}}
    d = bm.load(batch(tmp_path, [(2, rep, None)], plan=[2]), truth, classes)["m"]
    Y, P, S = matrices(d)
    assert Y.tolist() == [[1, 1, 0, 0]] and P.tolist() == [[1, 0, 0, 0]]
    assert bm.model_metrics(Y, P, S, classes)["per_class"].keys() == set(bm.columns(classes))
