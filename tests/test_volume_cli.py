"""run-episode --volume: any NIfTI scan, with --modality, --truth and --label-map. Parsing and the prelaunch checks
(no Slicer, no model call), and what main hands to the case preparation."""

import contextlib
import types

import pytest

from chd_client import run_episode as re_mod
from chd_client.task import cohorts
from chd_client.task.config import NOMASK_PROFILE

VSD, TOF = "ventricular_septal_defect", "tetralogy_of_fallot"


def parse(*argv):
    return re_mod.parser().parse_args([*argv, "--dry-run-scripted"])


def prelaunch(*argv):
    return re_mod.prelaunch(parse(*argv))


def test_volume_parses_with_its_options():
    a = parse("--volume", "/s/scan.nii.gz", "--modality", "mr", "--truth", "normal", "--labels", "/s/mask.nii.gz",
              "--label-map", "1=LV,2=RV")
    assert (a.volume, a.modality, a.truth, a.labels, a.label_map) == ("/s/scan.nii.gz", "mr", "normal",
                                                                      "/s/mask.nii.gz", "1=LV,2=RV")
    assert parse("--volume", "/s/scan.nii.gz", "--labels", "/m.nii.gz").modality == "ct"
    with pytest.raises(SystemExit):
        parse("--volume", "/s/scan.nii.gz", "--modality", "pet", "--labels", "/m.nii.gz")


@pytest.mark.parametrize("modality, prompt", [("ct", "chd_ct_v1"), ("mr", "chd_mr_v1")])
def test_a_custom_scan_gets_its_modalitys_prompt(modality, prompt):
    version, model = prelaunch("--volume", "/s/scan.nii.gz", "--modality", modality, "--labels", "/m.nii.gz")
    assert version == prompt
    c = re_mod.cohort_of(parse("--volume", "/s/scan.nii.gz", "--modality", modality, "--labels", "/m.nii.gz"))
    assert c.key == "custom" and c.modality == modality and not c.has_truth and c.profile == cohorts.PROFILES[modality]


def test_truth_is_parsed_against_the_modalitys_classes():
    c = re_mod.cohort_of(parse("--volume", "/s.nii.gz", "--truth", f"{TOF},{VSD}", "--labels", "/m.nii.gz"))
    assert c.has_truth and c.truth(None) == [VSD, TOF]
    c = re_mod.cohort_of(parse("--volume", "/s.nii.gz", "--modality", "mr", "--truth", "tortuous_vessels",
                               "--labels", "/m.nii.gz"))
    assert c.truth(None) == ["tortuous_vessels"]
    assert re_mod.cohort_of(parse("--volume", "/s.nii.gz", "--truth", "normal", "--labels", "/m.nii.gz")).truth(
        None) == []


@pytest.mark.parametrize("argv, msg", [
    (["--volume", "/s.nii.gz", "--truth", "VSD", "--labels", "/m.nii.gz"], r"--truth 'VSD': unknown classes \['VSD'\]"),
    (["--volume", "/s.nii.gz", "--truth", "tortuous_vessels", "--labels", "/m.nii.gz"], "unknown classes"),
    (["--volume", "/s.nii.gz", "--labels", "gt"], "--volume has no ground-truth segmentation"),
    (["--imagechd", "1008", "--modality", "mr", "--labels", "gt"], "--modality and --truth go with --volume"),
    (["--hvsmr", "24", "--truth", "normal", "--labels", "gt"], "--modality and --truth go with --volume"),
    (["--imagechd", "1008", "--labels", "gt", "--label-map", "1=LV"], "--label-map goes with a --labels file"),
    (["--volume", "/s.nii.gz", "--arm", "3d", "--labels", "/m.nii.gz"], "leave out --labels"),
])
def test_prelaunch_refuses_a_wrong_volume_command(argv, msg):
    with pytest.raises(re_mod.PrelaunchError, match=msg):
        prelaunch(*argv)


def test_the_cli_reports_a_refusal_as_a_usage_error(capsys):
    with pytest.raises(SystemExit) as ex:
        re_mod.main(["--volume", "/s.nii.gz", "--labels", "gt", "--dry-run-scripted"])
    assert ex.value.code == 2 and "--volume has no ground-truth segmentation" in capsys.readouterr().err


def test_a_ct_volume_can_run_the_no_mask_arm():
    version, _ = prelaunch("--volume", "/s.nii.gz", "--arm", "3d")
    assert version == "chd_ct_nomask_v1"
    a = parse("--volume", "/s.nii.gz", "--arm", "3d")
    assert re_mod.arm_profile(a.arm, re_mod.cohort_of(a)) == NOMASK_PROFILE


def fake_main(monkeypatch, tmp_path):
    from chd_client import dicom_run
    seen = {}

    @contextlib.contextmanager
    def fake_case(*a, **k):
        seen["case"] = (a, k)
        yield types.SimpleNamespace(runtime=None, case_id="case_test", has_truth=False, info={},
                                    trace_meta=lambda: {}, summary_meta=lambda: {}, check_loaded=None)

    def fake_episode(case_id, arm, model, target, **k):
        seen["arm"], seen["prompt"] = arm, k["prompt"]
        return {"stop_reason": "submitted"}

    monkeypatch.setattr(dicom_run, "temporary_cohort_case", fake_case)
    monkeypatch.setattr(re_mod, "server_params", lambda **k: seen.setdefault("params", k))
    monkeypatch.setattr(re_mod, "traces_dir_for", lambda runtime=None: tmp_path)
    monkeypatch.setattr(re_mod, "finish_run", lambda *a, **k: seen.setdefault("finish", (a, k)))
    monkeypatch.setattr(re_mod, "run_episode", fake_episode)
    monkeypatch.setattr(re_mod.anyio, "run", lambda fn: fn())
    return seen


def test_main_prepares_a_custom_mr_volume_with_its_profile(monkeypatch, tmp_path):
    seen = fake_main(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as ex:
        re_mod.main(["--volume", "/s/scan.nii.gz", "--modality", "mr", "--labels", "/s/mask.nii.gz",
                     "--label-map", "1=LV,2=RV", "--truth", "normal", "--dry-run-scripted"])
    assert ex.value.code == 0 and seen["prompt"] == "chd_mr_v1" and seen["arm"] == "3d_seg"
    (cohort, volume, labels, runtime), kw = seen["case"]
    assert cohort.key == "custom" and cohort.modality == "mr" and cohort.truth(None) == []
    assert (volume, labels, runtime) == ("/s/scan.nii.gz", "/s/mask.nii.gz", None)
    assert kw == {"label_map": "1=LV,2=RV", "profile": cohorts.PROFILES["mr"]}
    assert seen["params"]["profile"] == cohorts.PROFILES["mr"]
    assert seen["finish"][1]["cohort"].key == "custom" and seen["finish"][1]["imagechd"] is False


def test_main_prepares_hvsmr_gt_with_the_hvsmr_cohort(monkeypatch, tmp_path):
    seen = fake_main(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as ex:
        re_mod.main(["--hvsmr", "24", "--labels", "gt", "--dry-run-scripted"])
    assert ex.value.code == 0 and seen["prompt"] == "chd_mr_v1"
    (cohort, n, labels, _), kw = seen["case"]
    assert cohort is cohorts.HVSMR and n == 24 and labels == "gt" and kw == {"label_map": None}
