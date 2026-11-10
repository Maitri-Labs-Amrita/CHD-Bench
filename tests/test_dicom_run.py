"""run-episode --imagechd: convert into a temporary case with slicer-mcp, run, delete (subprocesses faked, on a tiny
synthetic ImageCHD folder)."""

import json
import subprocess
from pathlib import Path

import pytest

from chd_client import dicom_run, slicer
from chd_client.run_episode import main
from chd_client.task.config import PROFILE
from tests.task.imagechd_fake import make_dataset

NII = {"case_id": "case_nii_0123456789", "nrrd": "/x.nrrd", "sha256": "cd", "dims": [512, 512, 200],
       "spacing_mm": [0.3, 0.3, 0.75],
       "flags": ["DORV"]}
PREPARE, REMOVE = "slicer_mcp.cli case prepare", "slicer_mcp.cli case remove"


def sha12(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


@pytest.fixture(autouse=True)
def _imagechd(tmp_path, monkeypatch):
    """A synthetic ImageCHD folder (cases 1003, 1008) as the default dataset, and a pinned slicer-eval-mcp checkout."""
    d = make_dataset(tmp_path / "ImageCHD")
    monkeypatch.setenv("CHD_IMAGECHD_DIR", str(d))
    monkeypatch.setattr(slicer, "check_pin", lambda *a, **k: {"pinned": True})
    monkeypatch.setattr(slicer, "check_launched_pin", lambda *a, **k: {"pinned": True, "launched": True})
    return d
IMAGECHD_LIST = {"dir": "/data/ImageCHD", "cases": [{"id": 1003, "flags": [], "spacing_mm": [0.33, 0.33, 0.75]},
                                                    {"id": 1008, "flags": ["DORV"], "spacing_mm": [0.3, 0.3, 0.75]}]}
DX_EVAL = {"trace": "t", "imagechd_id": 1008, "scorer_version": "multilabel-1", "predicted": [],
           "truth": ["double_outlet_right_ventricle"], "exact_match": 0.0, "hamming_accuracy": 0.9375, "jaccard": 0.0,
           "precision": 0.0, "recall": 0.0, "f1": 0.0, "brier": 0.0156,
           "per_class": {"double_outlet_right_ventricle": "fn"}, "truth_source": "imageCHD_dataset_info.xlsx"}


def fake_slicer_bundle(args) -> dict:
    """The env's chd_client.task.slicer_bundle: a dummy <run>/slicer/ and its JSON line."""
    d = Path(args[args.index("--run") + 1]) / "slicer"
    d.mkdir(parents=True, exist_ok=True)
    files = ["ct.nii.gz", "open_me.mrml", "README.txt"]
    import gzip
    for f in files:
        (d / f).write_bytes(gzip.compress(b"x") if f.endswith(".gz") else b"x")
    if "--imagechd" in args:
        (d / "truth").mkdir(exist_ok=True)
        (d / "truth" / "case.json").write_text(json.dumps({"format": "run-truth-v1", "spacing": [0.3, 0.3, 0.75]}))
        (d / "truth" / "labels.npz").write_bytes(b"PK\x03\x04 fake npz")
        files += ["truth/case.json", "truth/labels.npz"]
    return {"dir": str(d), "files": files, "scene": "open_me.mrml"}


def fake_input_manifest(args) -> dict:
    """task_scripts/input_manifest.py: the dataset files' names and sha256."""
    assert "--imagechd" in args
    return {"dataset": "imagechd", "files": {"labels": {"name": "labels.nii.gz", "sha256": "f" * 64}}}


class FakeRun:
    """The subprocesses (`task_python`): slicer-mcp case preparation (-> NII) and removal, the ImageCHD list and the
    diagnosis evaluation."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def __call__(self, cmd, **kw):
        args = cmd[1:]
        what = " ".join(args[1:4]) if args[1] == "slicer_mcp.cli" else args[1]
        self.calls.append((what, args))
        if what == self.fail_on:
            return subprocess.CompletedProcess(cmd, 1, "", "kaput")
        out = {"removed": True}
        if what == PREPARE:
            out = NII
        elif what == "chd_client.task.imagechd":
            out = IMAGECHD_LIST
        elif what == "chd_client.task.eval_diagnosis":
            out = DX_EVAL
        elif what == "chd_client.task.slicer_bundle":
            out = fake_slicer_bundle(args)
        return subprocess.CompletedProcess(cmd, 0, "log line\n" + json.dumps(out) + "\n", "")


def test_temporary_case_lifecycle(tmp_path, _imagechd):
    fake = FakeRun()
    with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None) as tc:
        assert tc.case_id == NII["case_id"] and tc.imagechd_id == 1008 and tc.info["source_dir"] == str(_imagechd)
        rt = json.loads(tc.runtime.read_text())
        assert Path(rt["traces_dir"]).is_absolute() and Path(rt["cases_dir"]).is_absolute()
        tmp = tc.runtime.parent
    assert [w for w, _ in fake.calls] == [PREPARE, REMOVE]
    assert fake.calls[1][1][4] == NII["case_id"] and not tmp.exists()


def test_imagechd_rules_are_case_prepare_arguments(_imagechd):
    """The ImageCHD quirks as slicer-mcp arguments: the sheet's spacing, HU + 1024, the LPS-identity header, the case
    key (the source plus the sha256 of the shells' label file) and the label codes."""
    fake = FakeRun()
    with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None, labels="gt"):
        pass
    a = fake.calls[0][1]
    assert a[a.index("--volume") + 1] == str(_imagechd / "ct_1008_image.nii.gz")
    assert a[a.index("--spacing") + 1:a.index("--spacing") + 4] == ["0.3", "0.32", "0.75"]
    assert a[a.index("--intensity-offset") + 1] == "-1024" and "--lps-identity" in a
    gt = _imagechd / "ct_1008_label.nii.gz"
    assert a[a.index("--case-key") + 1] == f"imagechd:1008:{sha12(gt)}"
    assert a[a.index("--labels") + 1] == str(_imagechd / "ct_1008_label.nii.gz")
    assert a[a.index("--label-map") + 1] == "1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA"
    assert a[a.index("--profile") + 1] == str(PROFILE)


def test_a_missing_case_fails_before_slicer(_imagechd):
    fake = FakeRun()
    with pytest.raises(dicom_run.DicomRunError, match="ct_1099_image.nii.gz not found"):
        with dicom_run.temporary_imagechd_case(1099, run=fake, echo=None):
            pass
    assert fake.calls == []


def test_the_runtime_copy_lives_beside_the_runs_runtime_not_in_tmp(tmp_path):
    """Under Slurm's job_container/tmpfs the job's /tmp can vanish while the job runs; the copy goes beside the job's
    runtime.json and is removed with the case. Its paths are absolute (relative = relative to CHD-Bench)."""
    from chd_client.task.config import REPO, load_runtime
    job = tmp_path / "job0"
    job.mkdir()
    (job / "runtime.json").write_text(json.dumps({**load_runtime(), "traces_dir": "traces", "cases_dir": "data/cases"}))
    with dicom_run.temporary_imagechd_case(1008, runtime=job / "runtime.json", run=FakeRun(), echo=None) as tc:
        assert tc.runtime.parent.parent == job.resolve() and tc.runtime.parent.name.startswith(".chd_dicom_run_")
        assert json.loads(tc.runtime.read_text())["cases_dir"] == str(REPO / "data" / "cases")
    assert sorted(p.name for p in job.iterdir()) == ["runtime.json"]


def test_remove_passes_the_runs_lease(tmp_path, monkeypatch):
    """prepare returns a lease; remove deletes the case only when no other run's lease is left."""
    monkeypatch.setitem(NII, "lease", "tok123")
    fake = FakeRun()
    with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None):
        pass
    assert fake.calls[1][1][-2:] == ["--lease", "tok123"] and NII["case_id"] in fake.calls[1][1]


def test_task_python_forwards_the_runtime(tmp_path):
    """The CLI's --runtime reaches every scoring script (scoring, bundle) as SLICER_MCP_RUNTIME."""
    seen = {}

    def run(cmd, **kw):
        seen.update(kw["env"])
        return subprocess.CompletedProcess(cmd, 0, "{}\n", "")
    rt = tmp_path / "rt.json"
    dicom_run.task_python(["-m", "x"], run=run, runtime=rt)
    assert seen["SLICER_MCP_RUNTIME"] == str(rt.resolve())


def test_cleanup_after_episode_error(tmp_path):
    fake = FakeRun()
    with pytest.raises(RuntimeError, match="episode broke"):
        with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None) as tc:
            tmp = tc.runtime.parent
            raise RuntimeError("episode broke")
    assert fake.calls[-1][0] == REMOVE and not tmp.exists()


def test_failed_conversion_raises_before_anything_else(tmp_path):
    fake = FakeRun(fail_on=PREPARE)
    with pytest.raises(dicom_run.DicomRunError, match="kaput"):
        with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None):
            pass
    assert len(fake.calls) == 1


def test_an_unpinned_slicer_eval_mcp_is_refused(monkeypatch):
    monkeypatch.undo()
    monkeypatch.delenv(slicer.ALLOW_ENV, raising=False)
    with pytest.raises(slicer.SlicerMCPVersionError, match="not the pinned"):
        slicer.check_pin(state={"path": "/x", "commit": "0" * 40, "dirty": False})
    with pytest.raises(slicer.SlicerMCPVersionError, match="uncommitted changes"):
        slicer.check_pin(state={"path": "/x", "commit": slicer.PINNED_COMMIT, "dirty": True})
    assert slicer.check_pin(state={"path": "/x", "commit": slicer.PINNED_COMMIT, "dirty": False})["pinned"]
    monkeypatch.setenv(slicer.ALLOW_ENV, "1")
    assert slicer.check_pin(state={"path": "/x", "commit": "0" * 40, "dirty": False})["pinned"] is False


def test_trace_gets_no_path_summary_does():
    tc = dicom_run.TemporaryCase(NII["case_id"], Path("/r"), {**NII, "source_dir": "/data/ImageCHD", "imagechd_id": 1008})
    blob = json.dumps(tc.trace_meta())
    assert "/data/ImageCHD" not in blob and '"sha256": "cd"' in blob and "DORV" not in blob
    assert tc.summary_meta()["source_dir"] == "/data/ImageCHD" and tc.summary_meta()["imagechd_id"] == 1008


CASE_REQUIRED = "one of the arguments --imagechd --hvsmr --volume is required"


@pytest.mark.parametrize("argv,msg", [
    ([], "required: --labels"),
    (["--imagechd", "1008"], "required: --labels"),
    (["--labels", "gt"], CASE_REQUIRED),
    (["--imagechd", "1008", "--labels", "gt", "--dicom", "d"], "unrecognized arguments: --dicom"),
    (["--imagechd", "1008", "--labels", "gt", "--hvsmr", "12"], "not allowed with argument --imagechd"),
    (["--imagechd", "1008", "--labels", "gt", "--series", "1"], "unrecognized arguments: --series"),
    (["--imagechd", "1008", "--labels", "gt", "--preseg-from", "x"], "unrecognized arguments: --preseg-from"),
    (["--case", "c", "--labels", "gt"], CASE_REQUIRED),
])
def test_cli_needs_a_case_and_its_labels(argv, msg, capsys):
    """One case (ImageCHD, HVSMR or a --volume scan) and --labels are required; no stored or DICOM case."""
    with pytest.raises(SystemExit):
        main(argv + ["--dry-run-scripted"])
    assert msg in capsys.readouterr().err


def test_imagechd_case_lifecycle_and_meta(tmp_path):
    fake = FakeRun()
    other = make_dataset(tmp_path / "other")
    with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None, dataset_dir=other) as tc:
        meta, summary, tmp = tc.trace_meta(), tc.summary_meta(), tc.runtime.parent
    prep = fake.calls[0][1]
    assert prep[prep.index("--volume") + 1] == str(other.resolve() / "ct_1008_image.nii.gz")
    assert meta == {"case_source": {"kind": "imagechd_temporary", "imagechd_id": 1008, "sha256": "cd",
                                    "dims": [512, 512, 200], "spacing_mm": [0.3, 0.3, 0.75]}}
    assert summary["kind"] == "imagechd_temporary" and summary["source_dir"] == str(other.resolve())
    assert "DORV" not in json.dumps(meta) + json.dumps(summary)
    assert [w for w, _ in fake.calls] == [PREPARE, REMOVE]
    assert fake.calls[1][1][4] == NII["case_id"] and not tmp.exists()


@pytest.mark.parametrize("arm", ["fixed", "2d"])
def test_cli_has_only_the_3d_seg_and_3d_arms(arm, capsys):
    with pytest.raises(SystemExit):
        main(["--imagechd", "1008", "--labels", "gt", "--arm", arm, "--dry-run-scripted"])
    assert "invalid choice" in capsys.readouterr().err


def _tc(tmp_path, content=b"the prepared volume"):
    import hashlib
    from chd_client.dicom_run import TemporaryCase
    nrrd = tmp_path / "case_nii_0123456789.nrrd"
    nrrd.write_bytes(content)
    info = {"nrrd": str(nrrd), "sha256": hashlib.sha256(content).hexdigest(), "dims": [512, 512, 206],
            "spacing_mm": [0.308594, 0.308594, 0.75], "imagechd_id": 1008}
    return TemporaryCase("case_nii_0123456789", tmp_path / "rt.json", info), nrrd


def test_the_prepared_case_passes_the_identity_check(tmp_path):
    tc, _ = _tc(tmp_path)
    tc.check_loaded({"dims": [512, 512, 206], "spacing_mm": [0.3086, 0.3086, 0.75]})


@pytest.mark.parametrize("case_info,overwrite,msg", [
    ({"dims": [512, 512, 275], "spacing_mm": [0.3086, 0.3086, 0.75]}, False, "dims"),
    ({"dims": [512, 512, 206], "spacing_mm": [0.2441, 0.2441, 0.75]}, False, "spacing"),
    ({"dims": [512, 512, 206], "spacing_mm": [0.3086, 0.3086, 0.75]}, True, "sha256"),
])
def test_another_volume_is_refused(tmp_path, case_info, overwrite, msg):
    from chd_client.dicom_run import CaseIdentityError
    tc, nrrd = _tc(tmp_path)
    if overwrite:
        nrrd.write_bytes(b"a different case with the same geometry")
    with pytest.raises(CaseIdentityError, match=msg):
        tc.check_loaded(case_info)


def test_a_refused_case_stops_before_the_first_model_turn(tmp_path):
    import anyio
    from chd_client.models.scripted import ScriptedModel
    from chd_client.run_episode import dry_run_script, run_episode
    from tests.fakes import CASE, make_env

    def refuse(case_info, volume_sha256=None):
        raise RuntimeError("not this case")
    srv, _, traces = make_env(tmp_path)
    model = ScriptedModel(dry_run_script())
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, case_check=refuse))
    assert summary["stop_reason"] == "case_identity_mismatch" and summary["n_model_turns"] == 0
    assert summary["cost_usd"] == 0 and model.observations == [] and model.turns == []


def test_imagechd_case_records_its_labels(tmp_path):
    """A 3d_seg case's shells are traceable: "gt", or the label map's path and sha256."""
    m = tmp_path / "ct_1008_pred.nii.gz"
    m.write_bytes(b"mask")
    with dicom_run.temporary_imagechd_case(1008, run=FakeRun(), echo=None, labels=str(m)) as tc:
        meta, summary = tc.trace_meta()["case_source"], tc.summary_meta()
    assert meta["labels"] == str(m) and len(meta["labels_sha256"]) == 64 and summary["labels"] == str(m)
    with dicom_run.temporary_imagechd_case(1008, run=FakeRun(), echo=None, labels="gt") as tc:
        assert tc.trace_meta()["case_source"]["labels"] == "gt" and "labels_sha256" not in tc.summary_meta()
    with dicom_run.temporary_imagechd_case(1008, run=FakeRun(), echo=None) as tc:
        assert "labels" not in tc.trace_meta()["case_source"]


def test_the_loaded_voxel_hash_is_compared_not_the_file(tmp_path):
    from chd_client.dicom_run import CaseIdentityError
    tc, nrrd = _tc(tmp_path)
    tc.info["volume_sha256"] = "a" * 64
    info = {"dims": [512, 512, 206], "spacing_mm": [0.3086, 0.3086, 0.75]}
    nrrd.unlink()
    check = tc.check_loaded(info, "a" * 64)
    assert check == {"method": "loaded_voxels", "loaded_volume_sha256": "a" * 64,
                     "prepared_volume_sha256": "a" * 64, "ok": True} and tc.identity_check == check
    with pytest.raises(CaseIdentityError, match="loaded voxels sha256 bbbbbbbbbbbb != prepared aaaaaaaaaaaa"):
        tc.check_loaded(info, "b" * 64)
    assert tc.identity_check["ok"] is False


def test_the_prepared_voxel_hash_is_computed_once(tmp_path):
    from chd_client.dicom_run import TASK_SCRIPTS
    tc, _ = _tc(tmp_path)
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"volume_sha256": "c" * 64}) + "\n", "")
    tc.run = run
    info = {"dims": [512, 512, 206], "spacing_mm": [0.3086, 0.3086, 0.75]}
    tc.check_loaded(info, "c" * 64)
    tc.check_loaded(info, "c" * 64)
    assert len(calls) == 1 and str(TASK_SCRIPTS / "voxel_sha256.py") in calls[0]


def test_an_env_without_the_voxel_hash_falls_back_to_the_file(tmp_path):
    tc, _ = _tc(tmp_path)
    check = tc.check_loaded({"dims": [512, 512, 206], "spacing_mm": [0.3086, 0.3086, 0.75]})
    assert check["ok"] and check["method"].startswith("case_file")


def test_the_voxel_script_hashes_as_the_bridge_does(tmp_path):
    """sha256 of the (k, j, i) array in C order and its own dtype, as SimpleITK reads the NRRD."""
    np = pytest.importorskip("numpy")
    sitk = pytest.importorskip("SimpleITK")
    import hashlib
    import runpy
    import sys
    a = (np.arange(4 * 3 * 2, dtype=np.int16).reshape(4, 3, 2) - 7)
    p = tmp_path / "v.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(a), str(p))
    from chd_client.dicom_run import TASK_SCRIPTS
    argv, sys.argv = sys.argv, ["voxel_sha256.py", str(p)]
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(TASK_SCRIPTS / "voxel_sha256.py"))
    finally:
        sys.argv = argv
    assert json.loads(buf.getvalue())["volume_sha256"] == hashlib.sha256(a.tobytes()).hexdigest()


def test_case_key_names_the_shells_by_their_file_hash(tmp_path):
    gt, a, b, a_copy = (tmp_path / f"{n}.nii.gz" for n in ("gt", "a", "b", "a_copy"))
    gt.write_bytes(b"ground truth")
    a.write_bytes(b"model A mask")
    b.write_bytes(b"model B mask")
    a_copy.write_bytes(b"model A mask")
    keys = {"gt": dicom_run.case_key("imagechd:1008", "gt", gt),
            "a": dicom_run.case_key("imagechd:1008", str(a)),
            "b": dicom_run.case_key("imagechd:1008", str(b)),
            "none": dicom_run.case_key("imagechd:1008", None)}
    assert len(set(keys.values())) == 4
    assert keys["gt"] == f"imagechd:1008:{sha12(gt)}" and keys["a"] == f"imagechd:1008:{sha12(a)}"
    assert keys["none"] == "imagechd:1008:nolabels"
    assert dicom_run.case_key("imagechd:1008", str(a_copy)) == keys["a"]


def test_two_preparations_with_different_labels_get_different_case_keys(tmp_path, _imagechd):
    m1, m2 = tmp_path / "m1.nii.gz", tmp_path / "m2.nii.gz"
    m1.write_bytes(b"mask one")
    m2.write_bytes(b"mask two")
    keys = []
    for labels in ("gt", str(m1), str(m2), None):
        args, _ = dicom_run.imagechd_prepare_args(1008, labels=labels)
        keys.append(args[args.index("--case-key") + 1])
    assert len(set(keys)) == 4 and keys[-1] == "imagechd:1008:nolabels"


def write_volume(path, values=None):
    sitk = pytest.importorskip("SimpleITK")
    np = pytest.importorskip("numpy")
    a = np.arange(64, dtype=np.int16).reshape(4, 4, 4) if values is None else values
    sitk.WriteImage(sitk.GetImageFromArray(a), str(path))
    return path


def test_a_custom_ct_volume_is_prepared_as_it_is(tmp_path):
    from chd_client.task import cohorts
    vol = write_volume(tmp_path / "my_scan.nii.gz")
    mask = tmp_path / "my_mask.nii.gz"
    mask.write_bytes(b"mask")
    c = cohorts.custom("ct", ["ventricular_septal_defect"])
    args, extra = dicom_run.cohort_prepare_args(c, str(vol), str(mask), tmp_path / "work", label_map="1=LV,2=RV")
    assert args[args.index("--volume") + 1] == vol.resolve()
    assert args[args.index("--profile") + 1] == cohorts.PROFILES["ct"]
    assert args[args.index("--labels") + 1] == mask.resolve() and args[args.index("--label-map") + 1] == "1=LV,2=RV"
    assert args[args.index("--case-key") + 1] == f"custom:my_scan:{sha12(mask)}"
    assert extra["name"] == "my_scan" and extra["modality"] == "ct" and "scaling" not in extra
    assert extra["truth"] == ["ventricular_septal_defect"] and extra["dataset"] == "custom"
    args, extra = dicom_run.cohort_prepare_args(cohorts.custom("ct"), str(vol), str(mask), tmp_path / "work")
    assert args[args.index("--label-map") + 1] == cohorts.IMAGECHD_LABEL_MAP and extra["truth"] is None


def test_a_custom_mr_volume_is_rescaled_into_the_work_folder(tmp_path):
    from chd_client.task import cohorts
    vol = write_volume(tmp_path / "heart.nii")
    work = tmp_path / "work"
    work.mkdir()
    args, extra = dicom_run.cohort_prepare_args(cohorts.custom("mr"), str(vol), None, work)
    assert args[args.index("--volume") + 1] == work / "volume_mr_scaled.nii.gz" and (work / "volume_mr_scaled.nii.gz").is_file()
    assert extra["scaling"]["to"] == 1000.0 and extra["name"] == "heart"
    assert args[args.index("--profile") + 1] == cohorts.PROFILES["mr"] and "--labels" not in args
    assert args[args.index("--case-key") + 1] == "custom:heart:nolabels"


@pytest.mark.parametrize("name, msg", [("missing.nii.gz", "volume not found"), ("scan.nrrd", "must be a .nii")])
def test_a_bad_custom_volume_is_refused(tmp_path, name, msg):
    from chd_client.task import cohorts
    if not name.startswith("missing"):
        (tmp_path / name).write_bytes(b"x")
    with pytest.raises(dicom_run.DicomRunError, match=msg):
        dicom_run.cohort_prepare_args(cohorts.custom("ct"), str(tmp_path / name), None, tmp_path)


def test_a_custom_case_lifecycle_records_its_source_and_labels(tmp_path):
    from chd_client.task import cohorts
    vol = write_volume(tmp_path / "my_scan.nii.gz")
    mask = tmp_path / "my_mask.nii.gz"
    mask.write_bytes(b"mask")
    fake = FakeRun()
    c = cohorts.custom("ct", [])
    with dicom_run.temporary_cohort_case(c, str(vol), str(mask), run=fake, echo=None) as tc:
        src, meta = tc.summary_meta(), tc.trace_meta()["case_source"]
        assert tc.has_truth and tc.cohort.key == "custom"
    assert src["kind"] == "custom_temporary" and src["name"] == "my_scan" and src["truth"] == []
    assert src["labels"] == str(mask.resolve()) and src["labels_sha256"] == hashlib_hex(mask)
    assert src["labels_file"] == str(mask.resolve()) and src["label_map"] == cohorts.IMAGECHD_LABEL_MAP
    assert meta["volume"] == str(vol.resolve()) and "source_dir" not in meta
    assert [w for w, _ in fake.calls] == [PREPARE, REMOVE]


def hashlib_hex(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_a_case_records_the_slicer_mcp_checkout_and_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(slicer, "check_pin", lambda *a, **k: {"commit": "c1", "pinned": True})
    monkeypatch.setattr(slicer, "check_launched_pin", lambda rt: {"commit": "c2", "pinned": True, "rt": str(rt)})
    with dicom_run.temporary_imagechd_case(1008, run=FakeRun(), echo=None) as tc:
        assert tc.info["slicer_mcp_checkout"] == {"commit": "c1", "pinned": True}
        assert tc.info["slicer_mcp_launched"]["commit"] == "c2"


def test_an_unpinned_launch_is_refused_before_the_case_is_prepared(monkeypatch):
    def refuse(rt):
        raise slicer.SlicerMCPVersionError("launched from another commit")
    monkeypatch.setattr(slicer, "check_launched_pin", refuse)
    fake = FakeRun()
    with pytest.raises(slicer.SlicerMCPVersionError, match="another commit"):
        with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None):
            pass
    assert fake.calls == []
