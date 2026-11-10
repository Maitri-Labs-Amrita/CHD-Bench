"""task.config: the runtime pre-check, the resolved copy (one per runtime file, never the source), and the launch
state: a new launch runs on a private copy published only when it succeeds; later commands keep its port and nonce."""

import json
import subprocess

import pytest

from chd_client import slicer
from chd_client.task import config


def source(tmp_path, port="auto"):
    p = tmp_path / "runtime.local.json"
    p.write_text(json.dumps({"bridge_host": "127.0.0.1", "bridge_port": port, "cases_dir": "data/cases",
                             "traces_dir": "traces", "client_timeout_s": 300,
                             "case_prep": {"nrrd_compression": False}}))
    return p


def launch(src, port, nonce, ok=True):
    """What `chd-slicer launch` does around slicer-mcp launch (slicer.run), with launch.write_runtime_port faked."""
    staging, canon = config.launch_runtime(src)
    assert json.loads(staging.read_text())["bridge_port"] == json.loads(src.read_text())["bridge_port"]
    rt = json.loads(staging.read_text())
    rt["bridge_port"], rt["launch_nonce"] = port, nonce
    staging.write_text(json.dumps(rt))
    if ok:
        config.publish_launch(staging, canon)
    else:
        staging.unlink()
    return canon


def given(src):
    return json.loads(config.resolved_runtime(src).read_text())


def test_resolved_copy_is_in_the_tests_tmp_dir_with_absolute_paths(tmp_path):
    src = source(tmp_path)
    canon = config.resolved_runtime(src)
    assert canon.parent == config.RESOLVED_DIR and str(tmp_path) in str(canon)
    assert given(src)["cases_dir"] == str(config.REPO / "data" / "cases")
    assert json.loads(src.read_text())["cases_dir"] == "data/cases"


def test_launch_state_survives_resolves_and_reaches_a_runs_copy(tmp_path):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    assert given(src)["bridge_port"] == 45678
    run_copy = config.resolved_runtime(src, out=tmp_path / "run" / "runtime.json")
    assert json.loads(run_copy.read_text())["launch_nonce"] == "n1"
    assert config.resolved_runtime(run_copy) == run_copy


def test_a_new_launch_starts_from_the_sources_port(tmp_path):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    launch(src, 45999, "n2")
    assert (given(src)["bridge_port"], given(src)["launch_nonce"]) == (45999, "n2")


def test_a_port_override_survives(tmp_path):
    src = source(tmp_path, port=2017)
    launch(src, 2020, "n1")
    assert (given(src)["bridge_port"], given(src)["launch_nonce"]) == (2020, "n1")


def test_a_failed_launch_leaves_the_running_one(tmp_path):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    launch(src, 45999, "n2", ok=False)
    assert given(src)["launch_nonce"] == "n1"


def test_stop_forgets_the_launch(tmp_path):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    config.forget_launch(src)
    assert given(src)["bridge_port"] == "auto" and "launch_nonce" not in given(src)


def test_an_absolute_runtime_is_not_written_by_a_launch(tmp_path):
    src = tmp_path / "job.json"
    src.write_text(json.dumps(config.load_runtime(source(tmp_path))))
    launch(src, 45678, "n1")
    assert json.loads(src.read_text())["bridge_port"] == "auto"
    assert given(src)["bridge_port"] == 45678


def test_an_unchanged_copy_is_not_rewritten(tmp_path):
    src = source(tmp_path)
    canon = launch(src, 45678, "n1")
    before = canon.stat().st_mtime_ns
    config.resolved_runtime(src)
    assert canon.stat().st_mtime_ns == before


def test_two_runtime_files_get_two_copies(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert config.resolved_runtime(source(tmp_path / "a")) != config.resolved_runtime(source(tmp_path / "b"))


def test_scratch_dir_is_made_absolute(tmp_path):
    src = source(tmp_path)
    rt = json.loads(src.read_text())
    rt["eval_launch"] = {"scratch_dir": "data/home"}
    src.write_text(json.dumps(rt))
    assert given(src)["eval_launch"]["scratch_dir"] == str(config.REPO / "data" / "home")


def test_replay_gets_the_resolved_runtime(tmp_path, monkeypatch):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    seen = {}
    monkeypatch.setattr(slicer.subprocess, "run",
                        lambda cmd, env: seen.update(cmd=cmd, env=env) or subprocess.CompletedProcess(cmd, 0))
    monkeypatch.setenv(config.RUNTIME_ENV, str(src))
    assert slicer.run(["replay", "some/trace"]) == 0
    assert json.loads(open(seen["env"][config.RUNTIME_ENV]).read())["bridge_port"] == 45678


PINNED_STATE = {"path": "/mcp", "commit": slicer.PINNED_COMMIT, "dirty": False}


def test_a_failed_slicer_mcp_launch_is_not_published(tmp_path, monkeypatch):
    src = source(tmp_path)
    launch(src, 45678, "n1")
    monkeypatch.setattr(slicer, "git_state", lambda *a, **k: dict(PINNED_STATE))
    monkeypatch.setattr(slicer.subprocess, "run", lambda cmd, env: subprocess.CompletedProcess(cmd, 1))
    assert slicer.run(["launch", "--runtime", str(src)]) == 1
    assert given(src)["launch_nonce"] == "n1"
    assert not list(config.RESOLVED_DIR.glob(".*.launch-*.json"))
    assert not slicer.launch_record(config.resolved_runtime(src)).exists()


def test_a_successful_launch_records_the_slicer_mcp_state_and_stop_removes_it(tmp_path, monkeypatch):
    src = source(tmp_path)
    monkeypatch.setattr(slicer, "git_state", lambda *a, **k: {**PINNED_STATE, "dirty": True})
    monkeypatch.setattr(slicer.subprocess, "run", lambda cmd, env: subprocess.CompletedProcess(cmd, 0))
    assert slicer.run(["launch", "--runtime", str(src)]) == 0
    rec = slicer.launch_record(config.resolved_runtime(src))
    assert json.loads(rec.read_text()) == {**PINNED_STATE, "dirty": True}
    assert slicer.run(["launch", "--stop", "--runtime", str(src)]) == 0
    assert not rec.exists()


def write_record(tmp_path, **state):
    rt = tmp_path / "runtime.json"
    slicer.launch_record(rt).write_text(json.dumps({**PINNED_STATE, **state}))
    return rt


@pytest.mark.parametrize("state, msg", [
    ({"dirty": True}, "with uncommitted changes, not the pinned"),
    ({"commit": "0" * 40}, "launched from slicer-eval-mcp 0000"),
    ({"commit": None}, "an unknown commit"),
])
def test_check_launched_pin_refuses_an_unpinned_or_dirty_launch(tmp_path, monkeypatch, state, msg):
    monkeypatch.delenv(slicer.ALLOW_ENV, raising=False)
    rt = write_record(tmp_path, **state)
    with pytest.raises(slicer.SlicerMCPVersionError, match=msg):
        slicer.check_launched_pin(rt)
    monkeypatch.setenv(slicer.ALLOW_ENV, "1")
    assert slicer.check_launched_pin(rt)["pinned"] is False


def test_check_launched_pin_passes_a_pinned_launch_and_no_record(tmp_path, monkeypatch):
    monkeypatch.delenv(slicer.ALLOW_ENV, raising=False)
    assert slicer.check_launched_pin(tmp_path / "never_launched.json") is None
    assert slicer.check_launched_pin(write_record(tmp_path))["pinned"] is True


def test_stop_and_check_run_on_an_unpinned_checkout(monkeypatch):
    def unpinned(*a, **k):
        raise slicer.SlicerMCPVersionError("not pinned")
    monkeypatch.setattr(slicer, "check_pin", unpinned)
    monkeypatch.setattr(slicer, "run", lambda argv: 0)
    for argv in (["launch", "--stop"], ["launch", "--check"]):
        with pytest.raises(SystemExit) as e:
            slicer.main(argv)
        assert e.value.code == 0
    with pytest.raises(SystemExit) as e:
        slicer.main(["launch"])
    assert e.value.code == 2


def test_a_runtime_without_a_required_key_is_refused_before_anything_runs(tmp_path, capsys):
    from chd_client import run_episode
    rt = json.loads(config.DEFAULT_RUNTIME.read_text())
    del rt["traces_dir"]
    del rt["case_prep"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rt))
    with pytest.raises(config.RuntimeConfigError, match="has no traces_dir, case_prep.nrrd_compression"):
        config.load_runtime(bad)
    with pytest.raises(SystemExit) as e:
        run_episode.main(["--imagechd", "1112", "--labels", "gt", "--runtime", str(bad), "--dry-run-scripted"])
    assert e.value.code == 2 and "has no traces_dir" in capsys.readouterr().err


def test_a_runtime_that_is_not_json_is_refused_before_anything_runs(tmp_path, capsys):
    from chd_client import run_episode
    bad = tmp_path / "bad.json"
    bad.write_text("{not json,}")
    with pytest.raises(SystemExit) as e:
        run_episode.main(["--imagechd", "1112", "--labels", "gt", "--runtime", str(bad), "--dry-run-scripted"])
    assert e.value.code == 2 and "runtime:" in capsys.readouterr().err


def test_an_untracked_file_makes_the_checkout_dirty(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    (tmp_path / ".gitignore").write_text("data/\n")
    git("add", ".gitignore")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "log.txt").write_text("ignored")
    assert slicer.git_state(tmp_path)["dirty"] is False
    (tmp_path / "shadow.py").write_text("")
    assert slicer.git_state(tmp_path)["dirty"] is True
