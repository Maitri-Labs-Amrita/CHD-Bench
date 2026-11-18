"""What a person sees around a run (chd_client.ux, run.sh): the run plan with its warnings and the question before a
paid run, the checks that stop a run before it costs anything, and the RESULT box."""

import io
import json

import numpy as np
import pytest

from chd_client import run_sh, ux
from chd_client.run_episode import parser
from tests.test_dicom_run import write_dicom_series

ARGON = "openrouter:google/gemini-4-argon"


def nifti(path, values, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.asarray(values))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    sitk.WriteImage(img, str(path))
    return path


def ct(tmp_path, name="scan.nii.gz", shape=(4, 5, 6)):
    v = np.full(shape, 40, np.int16)
    v[0] = -1000
    return nifti(tmp_path / name, v)


def seg(tmp_path, labels, name="seg.nii.gz", shape=(4, 5, 6)):
    v = np.zeros(shape, np.uint8)
    for i, lab in enumerate(labels):
        v.flat[i] = lab
    return nifti(tmp_path / name, v)


def plan(argv, **kw):
    lines, warns = ux.run_plan(parser().parse_args(argv), **kw)
    return "\n".join(lines), " ".join(warns)


def test_own_scan_without_labels_or_truth_warns_about_both_and_about_privacy(tmp_path):
    text, warns = plan(["--volume", str(ct(tmp_path)), "--model", ARGON])
    assert "Segmentation  none: the model sees the scan only" in text
    assert "Truth         not given" in text and f"{ARGON} (PAID API)" in text
    assert "No segmentation (--labels not given)" in warns and "No true diagnosis (--truth not given)" in warns
    assert "Images of your scan are sent to OpenRouter. Use de-identified scans only" in warns


def test_a_scored_own_scan_shows_the_truth_in_words(tmp_path):
    s = ct(tmp_path)
    text, warns = plan(["--volume", str(s), "--labels", str(seg(tmp_path, [1, 2, 3, 4, 6, 7])), "--truth", "vsd",
                        "TOF", "--dry-run-scripted"])
    assert "Ventricular septal defect, Tetralogy of Fallot (from --truth): the run is scored" in text
    assert "Label values  1=LV, 2=RV, 3=LA, 4=RA, 6=Ao, 7=PA (ImageCHD codes, the default)" in text
    assert "scripted test model: no API calls, free" in text and warns == ""


def test_label_values_that_do_not_match_the_map_are_flagged(tmp_path):
    s = ct(tmp_path)
    hvsmr_coded = str(seg(tmp_path, [1, 2, 3, 4, 5, 6]))
    _, warns = plan(["--volume", str(s), "--labels", hvsmr_coded, "--dry-run-scripted"])
    assert "the label map expects 7 (PA), which the file does not have" in warns and "--label-map" in warns
    _, warns = plan(["--volume", str(s), "--labels", hvsmr_coded, "--label-map", "1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA",
                     "--dry-run-scripted"])
    assert "label map" not in warns


@pytest.mark.parametrize("make, says", [
    (lambda t: seg(t, [1, 2], shape=(4, 5, 7)), "the segmentation's grid (7x5x4 voxels) is not the scan's (6x5x4)"),
    (lambda t: seg(t, []), "has no labelled voxels"),
    (lambda t: t / "nope.nii.gz", "does not exist"),
])
def test_a_bad_segmentation_stops_the_run_before_slicer_starts(tmp_path, make, says):
    a = parser().parse_args(["--volume", str(ct(tmp_path)), "--labels", str(make(tmp_path)), "--model", ARGON])
    err = io.StringIO()
    assert run_sh.show_plan(a, yes=True, tty=True, err=err) is False
    assert says in err.getvalue() and "Nothing ran" in err.getvalue()


def test_a_ct_without_hu_values_and_an_mri_in_hu_are_flagged(tmp_path):
    flat = nifti(tmp_path / "flat.nii.gz", np.full((3, 4, 5), 200, np.int16))
    assert "lowest value is 200" in plan(["--volume", str(flat), "--dry-run-scripted"])[1]
    assert "looks like a CT in HU" in plan(["--volume", str(ct(tmp_path)), "--modality", "mr", "--labels",
                                            str(seg(tmp_path, [1])), "--dry-run-scripted"])[1]


def test_a_dicom_folder_shows_its_series_and_an_mr_header_makes_the_run_mri(tmp_path):
    root = tmp_path / "patient"
    write_dicom_series(root / "study" / "mr", np.zeros((5, 6, 7), np.int16), modality="MR")
    args, info, note = run_sh.auto_modality(["--volume", str(root), "--model", ARGON])
    assert args[-2:] == ["--modality", "mr"] and note == "from the DICOM header"
    assert info["n_files"] == 5 and info["folder"].endswith("study/mr")
    text, _ = plan(args, dicom=info, modality_note=note)
    assert "(a DICOM folder, converted to NIfTI first)" in text and "Modality      MRI (from the DICOM header)" in text
    assert "DICOM series  " in text and "(5 files)" in text
    _, info, note = run_sh.auto_modality(["--volume", str(root), "--modality", "ct"])
    assert note == "WARNING: the DICOM header says MR, but --modality is ct"


def test_the_plan_asks_before_a_paid_run_only(tmp_path):
    s = str(ct(tmp_path))
    paid = parser().parse_args(["--volume", s, "--model", ARGON])
    free = parser().parse_args(["--volume", s, "--dry-run-scripted"])
    for answer, ok in (("y\n", True), ("n\n", False), ("", False)):
        err = io.StringIO()
        assert run_sh.show_plan(paid, yes=False, tty=True, stdin=io.StringIO(answer), err=err) is ok
        assert "Start this paid run? [y/N]" in err.getvalue()
    err = io.StringIO()
    assert run_sh.show_plan(paid, yes=False, tty=False, err=err) is False and "Add --yes" in err.getvalue()
    for a, yes in ((paid, True), (free, False)):
        err = io.StringIO()
        assert run_sh.show_plan(a, yes=yes, tty=True, stdin=io.StringIO(""), err=err) is True
        assert "Start this paid run" not in err.getvalue() and "CHD-Bench run plan" in err.getvalue()


def test_a_missing_dataset_names_the_setup_command(tmp_path, monkeypatch):
    monkeypatch.setenv("CHD_IMAGECHD_DIR", str(tmp_path / "nowhere"))
    assert "./setup.sh --imagechd-dir" in ux.missing_data(parser().parse_args(["--imagechd", "1001"]))
    (tmp_path / "nowhere").mkdir()
    assert "./run.sh --list imagechd" in ux.missing_data(parser().parse_args(["--imagechd", "1001"]))


def run_folder(tmp_path, present=("ventricular_septal_defect",), truth=None, stop="submitted"):
    d = tmp_path / "reports" / "runs" / "r1"
    (d / "trace").mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"stop_reason": stop, "cost_usd": 1.8412, "n_model_steps": 41,
                                                "wall_s": {"total": 1320.0}, "model": ARGON}))
    if stop == "submitted":
        from chd_client.diagnosis import CLASS_NAMES
        cls = {c: {"present": c in present, "confidence": 0.8 if c in present else 0.9} for c in CLASS_NAMES}
        cls["normal"] = {"present": not present, "confidence": 0.95}
        (d / "trace" / "report.json").write_text(json.dumps({"classes": cls}))
    m = {"diagnosis": {"skipped": "no truth given (--truth)"}}
    if truth is not None:
        f1 = 1.0 if set(truth) == set(present) else 0.5
        m = {"diagnosis": {"exact_match": float(set(truth) == set(present)), "f1": f1, "truth": truth,
                           "predicted": list(present)}}
    (d / "metrics.json").write_text(json.dumps(m))
    (d / "report.pdf").write_bytes(b"%PDF")
    return d


def test_the_result_box_has_the_diagnosis_the_score_the_cost_and_the_pdf(tmp_path):
    d = run_folder(tmp_path, truth=["ventricular_septal_defect", "tetralogy_of_fallot"])
    box = "\n".join(ux.result_box(d))
    assert "RESULT" in box and "finished: the model submitted its diagnosis" in box
    assert "Model diagnosis Ventricular septal defect (confidence 80%)" in box
    assert "heart called normal: no (confidence 95%)" in box
    assert "True diagnosis  Ventricular septal defect, Tetralogy of Fallot" in box
    assert "Score           partly right (exact match 0, F1 0.50)" in box
    assert "$1.84 · 22 min · 41 steps" in box and f"Report (PDF)    {d / 'report.pdf'}" in box
    assert "not a clinical diagnosis" in box


def test_the_result_box_without_truth_or_without_a_diagnosis(tmp_path):
    box = "\n".join(ux.result_box(run_folder(tmp_path)))
    assert "Model diagnosis Ventricular septal defect" in box
    assert "not scored: no true diagnosis given (--truth)" in box
    box = "\n".join(ux.result_box(run_folder(tmp_path / "b", stop="cost_cap")))
    assert "the cost limit was reached before a diagnosis (raise --cost-cap)" in box
    assert "none: the model did not submit one" in box


def test_the_result_box_says_when_nothing_was_charged():
    assert "nothing was charged" in "\n".join(ux.result_box(None, "3D Slicer did not start", before_model=True))
    box = "\n".join(ux.result_box(None, "killed"))
    assert "nothing was charged" not in box and "may have been charged" in box
