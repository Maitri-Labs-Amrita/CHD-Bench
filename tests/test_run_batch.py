"""slurm/run_batch.py: config checks, the per-job runtime and the commands (no Slurm, no Slicer, no API)."""

import importlib.util
import json
from pathlib import Path

import pytest

from chd_client import slicer

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_batch", ROOT / "slurm" / "run_batch.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)


def config(tmp_path, **over):
    c = json.loads((ROOT / "slurm" / "batch_config.example.json").read_text())
    c.update(over)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    return p


def test_example_config_is_six_jobs_three_models_two_runs(tmp_path):
    c = rb.load_config(config(tmp_path))
    assert len(c["jobs"]) == 6 and c["cost_cap"] == 8.0
    models = [j["model"] for j in c["jobs"]]
    assert sorted(set(models)) == ["claude-opus-5-5", "openrouter:google/gemini-4-argon",
                                   "openrouter:openai/gpt-6-astra"] and all(models.count(m) == 2 for m in models)
    for m in set(models):
        a, b = [j["cases"] for j in c["jobs"] if j["model"] == m]
        assert len(a) == len(b) == 5 and not set(a) & set(b)


@pytest.mark.parametrize("bad, msg", [
    ({"jobs": []}, "non-empty"),
    ({"batch": ""}, "batch name"),
    ({"jobs": [{"name": "x", "model": "m", "cases": ["1001"]}]}, "list of ImageCHD ids"),
    ({"jobs": [{"name": "x", "model": "m", "cases": [1], "gpu": 1}]}, "unknown keys"),
    ({"jobs": [{"name": "x", "model": "m", "cases": [1]}, {"name": "x", "model": "m", "cases": [2]}]}, "twice"),
])
def test_bad_configs_are_refused(tmp_path, bad, msg):
    with pytest.raises(rb.BatchError, match=msg):
        rb.load_config(config(tmp_path, **bad))


def test_job_index_from_slurm_array():
    assert rb.job_index(None, {"SLURM_ARRAY_TASK_ID": "4"}) == 4 and rb.job_index(2, {}) == 2
    with pytest.raises(rb.BatchError):
        rb.job_index(None, {})


def test_each_job_gets_its_own_port_cases_dir_and_launch_files(tmp_path):
    c = rb.load_config(config(tmp_path))
    base = {"bridge_host": "127.0.0.1", "bridge_port": 2037, "cases_dir": "data/cases", "traces_dir": "traces",
            "eval_launch": {"startup_timeout_s": 180, "caffeinate": True, "state_file": "data/dev/eval_slicer.json"}}
    rts = [rb.job_runtime(base, c, k, rb.job_dir(c, k, tmp_path)) for k in range(6)]
    for key in (lambda r: r["cases_dir"], lambda r: r["traces_dir"], lambda r: r["eval_launch"]["state_file"],
                lambda r: r["eval_launch"]["log"]):
        assert len({key(r) for r in rts}) == 6
    assert all(r["bridge_port"] == "auto" for r in rts)
    assert all(Path(r["cases_dir"]).is_absolute() and Path(r["traces_dir"]).is_absolute() for r in rts)
    forced = rb.load_config(config(tmp_path, port_base=2101))
    assert [rb.job_runtime(base, forced, k, tmp_path)["bridge_port"] for k in range(6)] == list(range(2101, 2107))
    assert rts[0]["eval_launch"]["startup_timeout_s"] == 180 and not rts[0]["eval_launch"]["caffeinate"]
    assert base["bridge_port"] == 2037 and base["eval_launch"]["caffeinate"]


def test_episode_commands(tmp_path):
    p = config(tmp_path, extra_args=["--max-steps", 10], labels="gt")
    raw = json.loads(p.read_text())
    del raw["arm"]
    p.write_text(json.dumps(raw))
    c = rb.load_config(p)
    c["jobs"][2]["extra_args"] = ["--max-tokens", 64000]
    cmd = rb.episode_command(c, 2, 1003, "/x/runtime.json")
    assert cmd[:3] == ["uv", "run", "run-episode"] and cmd[cmd.index("--imagechd") + 1] == "1003"
    assert cmd[cmd.index("--model") + 1] == "openrouter:openai/gpt-6-astra" and cmd[cmd.index("--effort") + 1] == "max"
    assert cmd[cmd.index("--cost-cap") + 1] == "8" and cmd[cmd.index("--runtime") + 1] == "/x/runtime.json"
    assert cmd[-4:] == ["--max-steps", "10", "--max-tokens", "64000"] and cmd[cmd.index("--arm") + 1] == "3d_seg"
    assert cmd[cmd.index("--labels") + 1] == "gt"
    dry = rb.episode_command(c, 2, 1003, "/x/runtime.json", dry_run=True)
    assert "--dry-run-scripted" in dry and "--model" not in dry and "--effort" not in dry


def test_dotenv_parsing_tolerates_a_space_after_the_equals(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\n\nA=1\nB= two \nexport C='three'\nnot a line\n")
    assert rb.read_dotenv(p) == {"A": "1", "B": "two", "C": "three"}
    env = rb.child_env({"A": "outer", "VIRTUAL_ENV": "/v"}, {"A": "1", "B": "2"}, "/r.json")
    assert env == {"A": "outer", "B": "2", "SLICER_MCP_RUNTIME": "/r.json"}


def test_plan_prints_and_runs_nothing(tmp_path, capsys):
    base = tmp_path / "rt.json"
    base.write_text(json.dumps({"bridge_host": "127.0.0.1", "bridge_port": 2037, "cases_dir": "c", "traces_dir": "t",
                                "client_timeout_s": 300, "case_prep": {"nrrd_compression": False}}))
    p = config(tmp_path, runtime_base=str(base), out_dir=str(tmp_path / "out"))
    assert rb.main(["--config", str(p), "--job", "5", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["runtime"]["bridge_port"] == "auto" and len(plan["episodes"]) == 5
    assert plan["session"][:3] == ["uv", "run", "python"] and plan["session"][3].endswith("tools/xvfb_session.py")
    assert "--inner" in plan["session"] and plan["session"][5] == str(slicer.xvfb_runtime())
    assert plan["runtime"]["cases_dir"] != "c" and Path(plan["runtime"]["traces_dir"]).is_absolute()
    assert not (tmp_path / "out").exists()
    assert rb.main(["--config", str(p), "--job", "6", "--plan"]) == 2


def test_3d_seg_labels_per_case(tmp_path):
    """Arm 3d_seg passes each case's own label map ({case} in the path), or gt."""
    c = rb.load_config(config(tmp_path, arm="3d_seg", labels="/m/ct_{case}_pred.nii.gz"))
    cmd = rb.episode_command(c, 0, 1066, "/x/runtime.json")
    assert cmd[cmd.index("--arm") + 1] == "3d_seg" and cmd[cmd.index("--labels") + 1] == "/m/ct_1066_pred.nii.gz"
    gt = rb.episode_command(rb.load_config(config(tmp_path, arm="3d_seg", labels="gt")), 0, 7, "/r")
    assert gt[gt.index("--labels") + 1] == "gt"
    for bad in ({"arm": "3d_seg", "labels": None}, {"arm": "3d", "labels": "gt"},
                {"arm": "3d_seg", "labels": "/m/one.nii.gz"}, {"arm": "2d", "labels": None}):
        with pytest.raises(rb.BatchError):
            rb.load_config(config(tmp_path, **bad))


def test_a_dry_run_child_gets_no_api_key():
    environ = {"ANTHROPIC_API_KEY": "k", "OPENROUTER_API_KEY": "k", "PATH": "/bin"}
    assert not set(rb.API_KEYS) & set(rb.child_env(environ, {}, "/r.json", dry_run=True))
    assert set(rb.API_KEYS) <= set(rb.child_env(environ, {}, "/r.json"))


def test_sigterm_is_passed_on_to_the_running_child(tmp_path):
    """The child traps TERM and writes a marker: run_forwarding must deliver it, not die first."""
    import os
    import signal
    import sys
    import threading
    marker = tmp_path / "got_term"
    child = ("import signal, sys, time, pathlib\n"
             f"signal.signal(signal.SIGTERM, lambda *a: (pathlib.Path({str(marker)!r}).write_text('1'), sys.exit(7)))\n"
             "print('ready', flush=True)\ntime.sleep(30)\n")
    old = signal.signal(signal.SIGTERM, rb._on_term)
    try:
        threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        rc = rb.run_forwarding([sys.executable, "-c", child], timeout=20)
    finally:
        signal.signal(signal.SIGTERM, old)
        rb._terminated = False
    assert rc == 7 and marker.read_text() == "1"


def test_finished_cases_are_not_run_again(tmp_path):
    (tmp_path / "results.jsonl").write_text("\n".join([
        '{"case": 1001, "exit": 0}', '{"case": 1002, "exit": 1}', 'not json', '{"case": 1003, "exit": "timeout"}',
        '{"case": 1002, "exit": 0}']) + "\n")
    assert rb.finished_cases(tmp_path) == {1001, 1002}
    assert rb.finished_cases(tmp_path / "missing") == set()


def test_a_dry_run_does_not_count_for_a_real_model(tmp_path):
    real = rb.command_sha256(["run-episode", "--imagechd", "1001", "--model", "openrouter:x"])
    dry = rb.command_sha256(["run-episode", "--imagechd", "1001", "--dry-run-scripted"])
    (tmp_path / "results.jsonl").write_text(json.dumps({"case": 1001, "exit": 0, "command_sha256": dry}) + "\n")
    assert rb.finished_cases(tmp_path, {1001: real}) == set()
    assert rb.finished_cases(tmp_path, {1001: dry}) == {1001}
