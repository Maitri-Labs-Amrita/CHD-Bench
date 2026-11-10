"""HVSMR-2.0 and custom scans (chd_client.task.cohorts): the prompt, the profile's report and the scorer use one class
list per modality; the HVSMR files, label map and MRI scaling; run-episode and run_batch pick the dataset's prompt,
profile and labels."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from chd_client import diagnosis, paths, run_episode
from chd_client.run_episode import PROMPT_DIR
from chd_client.task import cohorts
from chd_client.task.diagnosis_schema import CHD_CLASSES
from chd_client.task.scoring.multilabel import score_classes
from slicer_mcp.config import load_profile

REG = json.loads((PROMPT_DIR / "registry.json").read_text())
HVSMR_DATA = paths.get(cohorts.HVSMR_CLINICAL).is_file() and paths.get(cohorts.HVSMR_IMAGES).is_dir()
CLINICAL_COLUMNS = ["Pat", "Normal", *cohorts.HVSMR_COLUMNS.values()]


def entry(version):
    return next(x for x in REG["prompts"] if x["version"] == version)


def prompt_classes(version):
    text = (PROMPT_DIR / entry(version)["file"]).read_text()
    block = text.split("## The classes")[1].split("Mark every class")[0]
    return [re.sub(r" - .*", "", ln[2:]).strip() for ln in block.splitlines() if ln.startswith("- ")]


@pytest.mark.parametrize("modality", ["ct", "mr"])
def test_prompt_profile_and_scorer_share_one_class_list(modality):
    c = cohorts.custom(modality)
    assert prompt_classes(c.prompt) == [*c.classes, "normal"]
    rep = load_profile(c.profile).report
    fields = list(rep.Model.model_fields["classes"].annotation.model_fields)
    assert fields == [*c.classes, "normal"]
    assert rep.CHD_CLASSES == c.classes
    n = len(c.classes) + 1
    text = (PROMPT_DIR / entry(c.prompt)["file"]).read_text()
    assert f"these {n} classes" in text and f"every one of the {n} classes" in text and f"other {n - 1} is" in text
    e = entry(c.prompt)
    assert e["modality"] == modality and e["profile"] == f"profiles/{c.profile.name}"


def test_hvsmr_uses_the_mr_profile_prompt_and_classes():
    assert cohorts.HVSMR.modality == "mr" and cohorts.HVSMR.prompt == "chd_mr_v1"
    assert cohorts.HVSMR.profile.name == "chd_mr_v1" and cohorts.HVSMR.classes == tuple(cohorts.HVSMR_COLUMNS)
    assert cohorts.custom("ct").classes == tuple(CHD_CLASSES) and cohorts.custom("ct").prompt == "chd_ct_v1"


def test_mr_prompt_says_mri_and_ct_prompt_says_ct():
    mr = (PROMPT_DIR / "chd_mr_v1.md").read_text()
    assert "cardiac MRI" in mr and "contrast-enhanced chest CT" not in mr and "not HU" in mr
    assert "contrast-enhanced chest CT" in (PROMPT_DIR / "chd_ct_v1.md").read_text()


def test_mr_profile_has_mri_presets():
    p = json.loads((cohorts.HVSMR.profile / "protocol.json").read_text())
    assert p["window_level"]["default_preset"] == "MR-Default" and not any(
        k.startswith("CT-") for k in p["window_level"]["presets"])
    assert p["render"]["presets"] == ["MR-BloodPool", "MR-Soft"]


def write_clinical(f, rows):
    """rows: {pat: set of clinical column names marked X}."""
    lines = [",".join(CLINICAL_COLUMNS)]
    for n, marked in rows.items():
        lines.append(",".join([str(n)] + ["X" if c in marked else "" for c in CLINICAL_COLUMNS[1:]]))
    f.write_text("\n".join(lines) + "\n")
    return f


@pytest.fixture
def hvsmr_files(tmp_path, monkeypatch):
    """A fake HVSMR-2.0 release in tmp_path, named in a paths file given by $CHD_PATHS."""
    images = tmp_path / "HVSMR-2.0" / "orig"
    images.mkdir(parents=True)
    clinical = write_clinical(tmp_path / "HVSMR-2.0" / "hvsmr_clinical.csv",
                              {1: {"Normal"}, 6: set(), 24: {"TortuousVessels"}, 30: {"VSD", "DORV"}})
    pf = tmp_path / "paths.json"
    pf.write_text(json.dumps({"paths": {cohorts.HVSMR_IMAGES: str(images), cohorts.HVSMR_CLINICAL: str(clinical)}}))
    monkeypatch.setenv(paths.PATHS_ENV, str(pf))
    return images


def test_hvsmr_truth_and_case_list_from_the_clinical_table(hvsmr_files):
    assert cohorts.hvsmr_cases() == [1, 24, 30]
    assert cohorts.hvsmr_truth(1) == [] and cohorts.hvsmr_truth(24) == ["tortuous_vessels"]
    assert cohorts.hvsmr_truth(30) == ["ventricular_septal_defect", "double_outlet_right_ventricle"]
    with pytest.raises(cohorts.CohortError, match="not in the cohort"):
        cohorts.hvsmr_truth(6)
    with pytest.raises(cohorts.CohortError, match="no row in hvsmr_clinical.csv"):
        cohorts.hvsmr_truth(99)


def test_hvsmr_volume_and_gt_files(hvsmr_files):
    assert cohorts.HVSMR.raw(24) == hvsmr_files / "pat24_orig.nii.gz"
    assert cohorts.HVSMR.gt_labels(24) == hvsmr_files / "pat24_orig_seg.nii.gz"
    with pytest.raises(cohorts.CohortError, match="no ground-truth segmentation"):
        cohorts.custom("ct").gt_labels("/x.nii.gz")


def test_hvsmr_gt_preparation_uses_the_seg_file_and_the_hvsmr_label_map(hvsmr_files, tmp_path):
    from chd_client import dicom_run
    sitk = pytest.importorskip("SimpleITK")
    img = sitk.GetImageFromArray(np.arange(64, dtype=np.int16).reshape(4, 4, 4))
    sitk.WriteImage(img, str(hvsmr_files / "pat24_orig.nii.gz"))
    (hvsmr_files / "pat24_orig_seg.nii.gz").write_bytes(b"seg")
    work = tmp_path / "work"
    work.mkdir()
    args, extra = dicom_run.cohort_prepare_args(cohorts.HVSMR, 24, "gt", work)
    i = args.index("--labels")
    assert args[i + 1] == hvsmr_files / "pat24_orig_seg.nii.gz"
    assert args[i + 3] == cohorts.HVSMR_LABEL_MAP == "1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA"
    assert extra["labels_file"] == str(hvsmr_files / "pat24_orig_seg.nii.gz")
    assert extra["label_map"] == cohorts.HVSMR_LABEL_MAP and extra["shell_source"] == "gt"
    assert extra["hvsmr_id"] == 24 and extra["volume"] == str(work / "volume_mr_scaled.nii.gz")
    assert args[args.index("--profile") + 1] == cohorts.HVSMR.profile
    assert args[args.index("--case-key") + 1].startswith("hvsmr:24:")


def test_normalize_mr_scales_the_995th_percentile_to_1000(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    a = np.linspace(0, 400, 1000).reshape(10, 10, 10)
    img = sitk.GetImageFromArray(a)
    img.SetSpacing((0.7, 0.8, 0.9))
    img.SetOrigin((1.0, 2.0, 3.0))
    sitk.WriteImage(img, str(tmp_path / "in.nii.gz"))
    rec = cohorts.normalize_mr(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")
    p = float(np.percentile(a, 99.5))
    assert rec == {"percentile": 99.5, "value": pytest.approx(p), "to": 1000.0, "clip": [0, 2000]}
    out = sitk.ReadImage(str(tmp_path / "out.nii.gz"))
    b = sitk.GetArrayFromImage(out)
    assert b.dtype == np.int16 and out.GetSpacing() == pytest.approx((0.7, 0.8, 0.9))
    assert out.GetOrigin() == pytest.approx((1.0, 2.0, 3.0))
    assert np.percentile(b, 99.5) == pytest.approx(1000, abs=1) and b.min() == 0
    assert np.abs(b - np.round(a * 1000 / p)).max() <= 1


def test_normalize_mr_clips_and_refuses_an_empty_volume(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    a = np.zeros((10, 10, 10))
    a[0, 0, :] = [-50, 5000, 1, 1, 1, 1, 1, 1, 1, 1]
    a[1:] = 1
    sitk.WriteImage(sitk.GetImageFromArray(a), str(tmp_path / "in.nii.gz"))
    cohorts.normalize_mr(tmp_path / "in.nii.gz", tmp_path / "out.nii.gz")
    b = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "out.nii.gz")))
    assert b.min() == 0 and b.max() == 2000
    sitk.WriteImage(sitk.GetImageFromArray(np.zeros((4, 4, 4))), str(tmp_path / "zero.nii.gz"))
    with pytest.raises(cohorts.CohortError, match="cannot rescale"):
        cohorts.normalize_mr(tmp_path / "zero.nii.gz", tmp_path / "out2.nii.gz")


@pytest.mark.skipif(not HVSMR_DATA, reason="needs the HVSMR-2.0 release (config/paths.json hvsmr_images)")
def test_the_real_hvsmr_release_has_its_files():
    cases = cohorts.hvsmr_cases()
    assert cases and all(cohorts.HVSMR.raw(n).is_file() and cohorts.HVSMR.gt_labels(n).is_file() for n in cases)


def test_parse_truth():
    classes = cohorts.custom("ct").classes
    assert cohorts.parse_truth("normal", classes) == []
    assert cohorts.parse_truth("tetralogy_of_fallot, ventricular_septal_defect", classes) == \
        ["ventricular_septal_defect", "tetralogy_of_fallot"]
    for bad in ("VSD", "normal,ventricular_septal_defect", ""):
        with pytest.raises(cohorts.CohortError, match="--truth"):
            cohorts.parse_truth(bad, classes)


def test_custom_cohort_truth_and_case_source():
    c = cohorts.custom("mr", ["tortuous_vessels"])
    assert c.has_truth and c.truth(None) == ["tortuous_vessels"] and c.key == "custom" and c.modality == "mr"
    assert not cohorts.custom("ct").has_truth
    with pytest.raises(cohorts.CohortError, match="--truth"):
        cohorts.custom("ct").truth(None)
    with pytest.raises(cohorts.CohortError, match="modality 'pet'"):
        cohorts.custom("pet")
    c, n = cohorts.for_case_source({"dataset": "custom", "name": "scan1", "modality": "mr", "truth": []})
    assert (c.key, c.modality, c.truth(n), n) == ("custom", "mr", [], "scan1")
    c, n = cohorts.for_case_source({"dataset": "hvsmr", "hvsmr_id": "24"})
    assert c is cohorts.HVSMR and n == 24
    assert cohorts.for_case_source({"imagechd_id": 1001}) == (None, None)


def test_score_classes():
    cls = cohorts.HVSMR.classes
    rep = {"classes": {c: {"present": c == "tortuous_vessels", "confidence": 0.8} for c in cls}}
    s = score_classes(rep, ["tortuous_vessels"], cls)
    assert s["exact_match"] == 1.0 and s["f1"] == 1.0 and s["predicted"] == ["tortuous_vessels"]
    assert score_classes(rep, [], cls)["exact_match"] == 0.0
    with pytest.raises(ValueError, match="differ"):
        score_classes({"classes": {}}, [], cls)


def test_display_helpers_follow_the_report():
    rep = diagnosis.dry_normal_report(cohorts.HVSMR.classes)
    assert diagnosis.class_names(rep) == cohorts.HVSMR.classes
    assert diagnosis.present_short(rep) == "none of 20"
    assert diagnosis.present_short(diagnosis.dry_report()) == "none of 16"


def test_cli_dataset_options():
    ap = run_episode.parser()
    a = ap.parse_args(["--hvsmr", "24", "--labels", "/x.nii.gz"])
    assert run_episode.cohort_of(a) is cohorts.HVSMR
    assert run_episode.cohort_of(ap.parse_args(["--imagechd", "1001", "--labels", "gt"])) is None
    with pytest.raises(SystemExit):
        ap.parse_args(["--imagechd", "1001", "--hvsmr", "1", "--labels", "gt"])
    with pytest.raises(SystemExit):
        ap.parse_args(["--hvsmr", "1", "--volume", "/x.nii.gz", "--labels", "gt"])


def test_run_batch_dataset_commands():
    import sys
    sys.path.insert(0, str(Path(run_episode.__file__).resolve().parents[2] / "slurm"))
    import run_batch
    cfg = {"arm": "3d_seg", "cost_cap": 10, "labels": "/m/hvsmr/{stem}_pred.nii.gz", "dataset": "hvsmr",
           "jobs": [{"name": "t0", "model": "local:x", "cases": [24]}]}
    cmd = run_batch.episode_command(cfg, 0, 24, "/rt.json")
    assert cmd[3:5] == ["--hvsmr", "24"] and "/m/hvsmr/hvsmr_24_pred.nii.gz" in cmd
    assert "--profile" in run_batch.slicer_command("/rt.json", profile=run_batch.slicer_profile(cfg))
    assert run_batch.slicer_profile(cfg) == str(cohorts.HVSMR.profile)
    assert run_batch.slicer_profile({"dataset": "imagechd"}) is None
    assert run_batch.DATASETS == ("imagechd", "hvsmr")
    assert run_batch.case_name(cfg, 24) == "HVSMR-24" and run_batch.case_name({}, 1001) == "ImageCHD-1001"
