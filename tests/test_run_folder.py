"""One folder per run in reports/runs/: chd_client.run_folder and the CLI's finish_run."""

import json
import shutil
from pathlib import Path

import anyio
import pytest

from chd_client import run_folder
from chd_client.models.scripted import ScriptedModel
from chd_client.report_pdf import main as report_main
from chd_client.run_episode import dry_run_script, finish_run, metrics_line, run_episode
from tests.fakes import CASE, make_env
from tests.test_dicom_run import fake_input_manifest, fake_slicer_bundle

DX = {"scorer_version": "multilabel-1", "predicted": [], "truth": ["double_outlet_right_ventricle"],
      "exact_match": 0.0, "hamming_accuracy": 0.9375, "jaccard": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
      "brier": 0.016, "per_class": {"double_outlet_right_ventricle": "fn"}, "truth_source": "imageCHD_dataset_info.xlsx"}


IMAGECHD_SRC = {"kind": "imagechd_temporary", "source_dir": "/data/ImageCHD", "imagechd_id": 1008}
CT_PROMPT = "chd_ct_v1"


def run_seg(tmp_path, case_source=IMAGECHD_SRC):
    srv, fake, traces = make_env(tmp_path)
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", ScriptedModel(dry_run_script()), srv.server,
                                            runs_dir=tmp_path / "runs", traces_dir=traces, echo=None,
                                            case_source=case_source))
    return summary, traces, fake


def fake_env_python(calls, bundle_fails=False):
    def run(args, timeout_s=None):
        calls.append(args)
        if str(args[0]).endswith("input_manifest.py"):
            return json.dumps(fake_input_manifest(args)) + "\n"
        if args[1] == "chd_client.task.slicer_bundle":
            if bundle_fails:
                raise run_folder.DicomRunError("chd_client.task.slicer_bundle failed (exit 1)")
            return "noise\n" + json.dumps(fake_slicer_bundle(args)) + "\n"
        out = {"chd_client.task.eval_diagnosis": DX}[args[1]]
        return "noise\n" + json.dumps(out) + "\n"
    return run


def test_folder_name_case_first_24_hour_time():
    meta = {"start_time": "2026-10-13T07:49:10.995Z", "case_id": "case_nii_488f9c7705", "arm": "3d_seg",
            "model": {"id": "claude-opus-5-5"}}
    e = "20261013T074910Z_case_nii_488f9c7705_3d_seg_78b876"
    imagechd = {"case_source": {"kind": "imagechd_temporary", "imagechd_id": 1025}}
    assert run_folder.folder_name(e, meta, imagechd, "Asia/Kolkata") == \
        "ImageCHD-1025_2026-10-13_13-19_3d_seg_opus-5-5_78b876"
    assert run_folder.folder_name(e, meta, imagechd, "UTC") == "ImageCHD-1025_2026-10-13_07-49_3d_seg_opus-5-5_78b876"
    tmeta = {**meta, "client_meta": {"case_source": {"kind": "imagechd_temporary", "imagechd_id": 1025}}}
    assert run_folder.folder_name(e, tmeta, None, "UTC").startswith("ImageCHD-1025_")
    other = {**meta, "case_id": "case_nii_ab", "model": {"id": "openrouter:qwen/qwen3-vl"}}
    assert run_folder.folder_name(e, other, {}, "UTC") == \
        "case_nii_ab_2026-10-13_07-49_3d_seg_openrouter-qwen-qwen3-vl_78b876"
    summ = {"case": "case_nii_ab", "arm": "3d_seg", "model": "claude-sonnet-5-5"}
    assert run_folder.folder_name(e, {}, summ, "UTC") == "case_nii_ab_2026-10-13_07-49_3d_seg_sonnet-5-5_78b876"
    assert run_folder.folder_name("junk_x", {}, summ, "UTC") == "case_nii_ab_unknown-time_3d_seg_sonnet-5-5_x"
    assert run_folder.folder_name(e, {**meta, "start_time": "2026-10-13T20:05:00Z"}, imagechd,
                                  "UTC").split("_")[2] == "20-05"


def test_run_folders_default_to_reports_runs(_reports_dir_in_tmp):
    assert run_folder.reports_dir_default() == _reports_dir_in_tmp / "runs"


def test_no_test_can_reach_the_real_reports_folder():
    """Every test (and work left running after one) sees a tmp reports dir, never the real reports folder."""
    from tests.conftest import REAL_REPORTS_DIR
    from chd_client import run_episode
    assert Path(run_episode.REPORTS_DIR) != REAL_REPORTS_DIR
    assert REAL_REPORTS_DIR not in run_folder.reports_dir_default().parents


def test_a_finished_imagechd_run_is_one_folder_with_everything(tmp_path, _reports_dir_in_tmp):
    summary, traces, fake = run_seg(tmp_path)
    e = summary["episode_id"]
    calls, lines = [], []
    dest = finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=lines.append,
                      run=fake_env_python(calls))
    assert dest.parent == _reports_dir_in_tmp / "runs" and dest.name.startswith("ImageCHD-1008_")
    assert dest.name.endswith(f"_3d_seg_{CT_PROMPT.removeprefix('system_')}_scripted_" + e.rsplit("_", 1)[-1])
    assert sorted(p.name for p in dest.iterdir()) == ["COMPLETE.json", "metrics.json", "provenance", "report.pdf",
                                                       "requests.jsonl", "responses.jsonl", "slicer",
                                                       "summary.json", "trace", "transcript.jsonl"]
    done = json.loads((dest / "COMPLETE.json").read_text())
    assert done["episode_id"] == e and done["stop_reason"] == "submitted" and done["files_sha256"]["metrics.json"]
    assert set(run_folder.REQUIRED_FILES + run_folder.TRUTH_FILES) == set(done["required"])
    assert calls[-1][0].name == "input_manifest.py" and calls[-1][calls[-1].index("--imagechd") + 1] == 1008
    prov = json.loads((dest / "provenance" / "provenance.json").read_text())
    assert prov["inputs"]["dataset"] == "imagechd" and prov["environment"]["slicer"]["version"] == "5.10.0"
    calls = calls[1:-1] + calls[:1]
    m = json.loads((dest / "metrics.json").read_text())
    assert m.pop("tool_use")["n_calls"] >= 1
    assert m == {"diagnosis": DX}
    assert [c[1] for c in calls] == ["chd_client.task.eval_diagnosis", "chd_client.task.slicer_bundle"]
    assert calls[0][calls[0].index("--run") + 1] == str(dest) and "--trace" not in calls[0]
    assert calls[1] == ["-m", "chd_client.task.slicer_bundle", "--run", str(dest), "--imagechd", 1008,
                        "--dataset-dir", "/data/ImageCHD", "--json"]
    assert (dest / "slicer" / "open_me.mrml").is_file()
    meta = json.loads((dest / "trace" / "meta.json").read_text())
    assert meta["episode_id"] == e and meta["profile"] == "chd_ct_v1" and "shells" not in meta
    assert (dest / "trace" / "steps.jsonl").is_file() and any((dest / "trace" / "frames").iterdir())
    assert (dest / "trace" / "report.json").is_file() and (dest / "trace" / "normal.json").is_file()
    assert not (dest / "trace" / "shells").exists() and not (dest / "shells").exists()
    assert not (tmp_path / "runs" / e).exists() and not (traces / e).exists()
    s = json.loads((dest / "summary.json").read_text())
    assert s["run_dir"] == str(dest) and s["trace_dir"] == str(dest / "trace")
    assert s["report_pdf"] == str(dest / "report.pdf") and "report_pdf_copy" not in s
    assert s["slicer_bundle"]["scene"] == "open_me.mrml" and "slicer_bundle_error" not in s
    assert lines == [f"slicer bundle: {dest / 'slicer'}",
                     "metrics: diagnosis none of 16 vs truth DORV: exact 0, F1 0.00",
                     f"run folder: {dest}", "COMPLETE.json written"]
    (dest / "report.pdf").unlink()
    assert report_main([str(dest), "--tz", "UTC"]) == 0
    assert (dest / "report.pdf").read_bytes()[:5] == b"%PDF-"


def test_a_failed_slicer_bundle_is_recorded_not_raised(tmp_path, _reports_dir_in_tmp):
    summary, traces, _ = run_seg(tmp_path)
    lines = []
    dest = finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=lines.append,
                      run=fake_env_python([], bundle_fails=True))
    assert dest.parent == _reports_dir_in_tmp / "runs" and (dest / "trace" / "meta.json").is_file()
    s = json.loads((dest / "summary.json").read_text())
    assert "slicer_bundle failed" in s["slicer_bundle_error"] and "slicer_bundle" not in s
    assert lines[-2] == f"run folder: {dest}" and lines[0].startswith("slicer bundle failed")
    assert lines[-1].startswith("INCOMPLETE") and "slicer_bundle_error" in lines[-1]
    assert not (dest / "COMPLETE.json").exists()
    problems = json.loads((dest / "INCOMPLETE.json").read_text())["problems"]
    assert "missing slicer/ct.nii.gz" in problems and "missing slicer/truth/labels.npz" in problems
    write_json = json.loads((dest / "summary.json").read_text())
    write_json.pop("case_source")
    (dest / "summary.json").write_text(json.dumps(write_json))
    assert run_folder.add_slicer_bundle(dest, run=fake_env_python([])) is None
    assert "not an ImageCHD case" in json.loads((dest / "summary.json").read_text())["slicer_bundle_error"]


def test_a_failed_collect_leaves_the_files_where_they_were(tmp_path, monkeypatch):
    summary, traces, _ = run_seg(tmp_path)
    e = summary["episode_id"]

    def boom(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(run_folder, "collect_run", boom)
    lines = []
    assert finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=lines.append) is None
    assert (tmp_path / "runs" / e / "summary.json").is_file() and (traces / e / "meta.json").is_file()
    assert "its files stay in runs/ and traces/" in lines[-1]
    assert lines[-1].endswith(f"{tmp_path / 'runs' / e} and {traces / e}")


def test_the_summary_line_prints_no_path_that_finish_run_moves(tmp_path, _reports_dir_in_tmp):
    """run-episode's summary line comes before finish_run moves runs/<E>/ into reports/runs/: it names no runs/ path
    (runs/<E>/report.pdf is gone after the move); finish_run's "run folder:" line is the one path."""
    srv, _, traces = make_env(tmp_path)
    lines = []
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", ScriptedModel(dry_run_script()), srv.server,
                                            runs_dir=tmp_path / "runs", traces_dir=traces, echo=lines.append,
                                            case_source=IMAGECHD_SRC))
    assert len(lines) == 1 and lines[0].endswith(" | pdf written") and str(tmp_path) not in lines[0]
    dest = finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=lines.append,
                      run=fake_env_python([]))
    assert f"run folder: {dest}" in lines and (dest / "report.pdf").is_file()


def test_read_metrics_reads_only_metrics_json_and_metrics_line_handles_skips(tmp_path):
    assert run_folder.read_metrics(tmp_path) is None
    (tmp_path / "pool_eval.json").write_text(json.dumps({"final": {"dice": 0.9}}))
    assert run_folder.read_metrics(tmp_path) is None
    run_folder.write_metrics(tmp_path, {"diagnosis": DX})
    assert run_folder.read_metrics(tmp_path) == {"diagnosis": DX}
    assert run_folder.METRICS == (("diagnosis", "chd_client.task.eval_diagnosis"),)
    assert metrics_line({"diagnosis": {"skipped": "no flags"}}) == "diagnosis: no flags"
    assert metrics_line({"diagnosis": {"error": "x"}}) == "diagnosis: x"
    assert metrics_line({}) == "diagnosis: n/a"
    assert metrics_line({"pool": {"final": {"dice": 0.9}}, "diagnosis": DX}) == \
        "diagnosis none of 16 vs truth DORV: exact 0, F1 0.00"


def test_folder_name_says_the_kind_of_run():
    """eval (report submitted) or eval-stopped, after the time; left out when unknown (there is no chat kind)."""
    e = "20261013T074902Z_case_nii_488f9c7705_3d_seg_78b876"
    meta = {"start_time": "2026-10-13T07:49:02Z", "arm": "3d_seg", "model": {"id": "claude-opus-5-5"}}
    src = {"case_source": {"kind": "imagechd_temporary", "imagechd_id": 1025}}
    name = lambda s, m=meta: run_folder.folder_name(e, m, {**src, **s}, "UTC")
    assert name({"stop_reason": "submitted"}) == "ImageCHD-1025_2026-10-13_07-49_eval_3d_seg_opus-5-5_78b876"
    assert name({"stop_reason": "user_stop"}).split("_")[3] == "eval-stopped"
    assert name({"stop_reason": "max_steps"}).split("_")[3] == "eval-stopped"
    assert name({}, {**meta, "status": "submitted"}).split("_")[3] == "eval"
    assert name({}) == "ImageCHD-1025_2026-10-13_07-49_3d_seg_opus-5-5_78b876"
    assert name({"mode": "chat", "stop_reason": None}) == name({})


def _finished(tmp_path, src=IMAGECHD_SRC):
    summary, traces, _ = run_seg(tmp_path, src)
    return finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=None,
                      run=fake_env_python([]))


@pytest.mark.parametrize("damage,problem", [
    (lambda d: (d / "requests.jsonl").unlink(), "missing requests.jsonl"),
    (lambda d: (d / "metrics.json").write_text("{"), "metrics.json: "),
    (lambda d: next((d / "trace" / "frames").iterdir()).write_bytes(b"other"), "sha256 differs from what was sent"),
    (lambda d: (d / "slicer" / "truth" / "labels.npz").unlink(), "missing slicer/truth/labels.npz"),
    (lambda d: run_folder.write_metrics(d, {"diagnosis": {"error": "boom"}}), "metrics diagnosis: boom"),
    (lambda d: shutil.rmtree(d / "trace" / "frames"), "frame frames/"),
])
def test_complete_json_only_when_everything_validates(tmp_path, _reports_dir_in_tmp, damage, problem):
    dest = _finished(tmp_path)
    assert (dest / "COMPLETE.json").is_file()
    damage(dest)
    ok, problems = run_folder.write_completion(dest, has_truth=True)
    assert not ok and any(problem in p for p in problems), problems
    assert not (dest / "COMPLETE.json").exists() and (dest / "INCOMPLETE.json").is_file()


def test_an_unfinished_episode_is_never_complete(tmp_path, _reports_dir_in_tmp):
    dest = _finished(tmp_path)
    s = json.loads((dest / "summary.json").read_text())
    (dest / "summary.json").write_text(json.dumps({**s, "stop_reason": "terminated"}))
    ok, problems = run_folder.write_completion(dest, has_truth=True)
    assert not ok and problems == ["episode did not finish: stop_reason 'terminated'"]


def test_rescore_writes_metrics_by_scorer_commit(tmp_path, _reports_dir_in_tmp, monkeypatch):
    dest = _finished(tmp_path)
    before = (dest / "metrics.json").read_text()
    calls = []

    def run(args, timeout_s=None):
        calls.append(args)
        return json.dumps({"diagnosis": DX}) + "\n"
    monkeypatch.setattr(run_folder, "scorer_commit", lambda repo=None: "abc123def456")
    p = run_folder.rescore(dest, run=run)
    assert p == dest / "metrics_abc123def456.json" and (dest / "metrics.json").read_text() == before
    assert calls == [["-m", "chd_client.task.rescore", "--run", str(dest), "--source", "run", "--json"]]
    m = json.loads(p.read_text())
    assert m["diagnosis"] == DX and m["rescored"]["source"] == "run" and m["tool_use"]["n_calls"] >= 1
    monkeypatch.setattr(run_folder, "task_python", lambda args, **k: run(args))
    assert run_folder.rescore_main([str(dest)]) == 0


def test_scoring_uses_the_folder_the_case_was_prepared_from(tmp_path, monkeypatch):
    """CHD_IMAGECHD_DIR (dataset A) and the runtime's imagechd_dir (B) can differ: the bundle, the scores and the input
    hashes use A, where the model's CT came from (case_source.source_dir)."""
    seen = []
    monkeypatch.setattr(run_folder, "collect_run", lambda e, **k: tmp_path)
    monkeypatch.setattr(run_folder, "add_slicer_bundle", lambda d, dataset_dir=None, **k: seen.append(dataset_dir))
    monkeypatch.setattr(run_folder, "input_manifest", lambda s, dataset_dir=None, **k: seen.append(dataset_dir) or {})
    monkeypatch.setattr(run_folder, "dataset_dir_for", lambda *a: "/B")
    summary = {"episode_id": "e", "case_source": {"imagechd_id": 1001, "source_dir": "/A"}}
    try:
        finish_run(summary, tmp_path, imagechd=False, runs_dir=tmp_path, runtime=tmp_path / "rt.json",
                   run=lambda *a, **k: "{}", echo=None)
    except Exception:  # noqa: BLE001
        pass
    assert seen and set(seen) == {"/A"}


def test_a_submitted_report_that_was_not_scored_is_incomplete(tmp_path, _reports_dir_in_tmp):
    dest = _finished(tmp_path)
    run_folder.write_metrics(dest, {"diagnosis": {"skipped": "no truth given (--truth)"}})
    ok, problems = run_folder.write_completion(dest, has_truth=True)
    assert not ok and "the submitted report was not scored: no truth given (--truth)" in problems
    run_folder.write_metrics(dest, {"tool_use": {"n_calls": 1}})
    ok, problems = run_folder.write_completion(dest, has_truth=True)
    assert not ok and "the submitted report was not scored" in problems


def test_a_run_with_no_truth_expected_is_complete_without_a_score(tmp_path, _reports_dir_in_tmp):
    dest = _finished(tmp_path)
    run_folder.write_metrics(dest, {"diagnosis": {"skipped": "no truth given (--truth)"}})
    ok, problems = run_folder.write_completion(dest, has_truth=False)
    assert ok and problems == [] and (dest / "COMPLETE.json").is_file()


def test_a_failed_pdf_refresh_is_recorded_and_cleared_by_a_good_one(tmp_path, _reports_dir_in_tmp, monkeypatch):
    from chd_client import report_pdf
    dest = _finished(tmp_path)
    real = report_pdf.render_episode_pdf

    def boom(*a, **k):
        raise RuntimeError("no fonts")
    monkeypatch.setattr(report_pdf, "render_episode_pdf", boom)
    lines = []
    assert run_folder.refresh_pdf(dest, echo=lines.append) is False
    s = json.loads((dest / "summary.json").read_text())
    assert s["report_pdf_refresh_error"] == "RuntimeError: no fonts" and "no fonts" in lines[0]
    ok, problems = run_folder.write_completion(dest, has_truth=True)
    assert not ok and "summary report_pdf_refresh_error: RuntimeError: no fonts" in problems
    monkeypatch.setattr(report_pdf, "render_episode_pdf", real)
    assert run_folder.refresh_pdf(dest) is True
    assert "report_pdf_refresh_error" not in json.loads((dest / "summary.json").read_text())
    assert run_folder.write_completion(dest, has_truth=True)[0]


def test_case_labels_of_hvsmr_and_custom_scans():
    assert run_folder.case_label({}, {"case_source": {"dataset": "hvsmr", "hvsmr_id": 24}}) == "HVSMR-24"
    custom = {"dataset": "custom", "name": "my_scan", "modality": "ct", "truth": None}
    assert run_folder.case_label({}, {"case_source": custom}) == "my_scan"
    assert run_folder.case_label({}, {"case_source": {"imagechd_id": 1001}}) == "ImageCHD-1001"


def test_cohort_metrics_skip_a_custom_scan_without_truth(tmp_path):
    from chd_client.task import cohorts
    src = {"dataset": "custom", "name": "my_scan", "modality": "ct", "truth": None}
    (tmp_path / "summary.json").write_text(json.dumps({"case_source": src}))
    m = run_folder.cohort_metrics(tmp_path, cohorts.custom("ct"))
    assert m["diagnosis"]["skipped"] == "no truth given (--truth)" and m["diagnosis"]["name"] == "my_scan"
