"""./run.sh (chd_client.run_sh), the .env file, setup.sh's option errors and the first-run error messages (no Slicer,
no network, no API): the episode shell is run with a fake `uv` that records its commands."""

import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from chd_client import env_file, run_sh
from chd_client.models import registry
from chd_client.models.local import LocalModel
from chd_client.task.config import REPO, RUNTIME_ENV

KEYS = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "CHD_LOCAL_BASE_URL", "CHD_PROVIDERS", "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR", "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_IDENTITY_TOKEN",
        "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID",
        "SLICER_MCP_LAUNCH_NONCE", run_sh.NONCE_ENV)


@pytest.fixture(autouse=True)
def _no_sdk_profile(tmp_path, monkeypatch):
    """No anthropic SDK profile on disk (~/.config/anthropic) for the environments clean_env makes."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


def clean_env(**over):
    env = {k: v for k, v in os.environ.items() if k not in KEYS and k != RUNTIME_ENV}
    env.update(over)
    return env


def test_env_file_parsing_and_the_environment_wins(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# keys\nexport A='1'\nB = \"two\"\nC=three\nno equals sign\n")
    assert env_file.read_env_file(p) == {"A": "1", "B": "two", "C": "three"}
    assert env_file.read_env_file(tmp_path / "missing") == {}
    env, taken = env_file.merged({"A": "shell", "PATH": "/bin"}, env_file.read_env_file(p))
    assert env == {"A": "shell", "PATH": "/bin", "B": "two", "C": "three"} and taken == ["B", "C"]


def test_batch_runner_uses_the_shared_env_parser(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_batch", REPO / "slurm" / "run_batch.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    p = tmp_path / ".env"
    p.write_text("K='v'\n")
    assert rb.read_dotenv(p) == env_file.read_env_file(p) == {"K": "v"}
    assert rb.ENV_FILE == env_file.ENV_FILE


def test_prepare_reads_env_file_names_only_and_checks_the_key(tmp_path):
    p = tmp_path / ".env"
    p.write_text("ANTHROPIC_API_KEY=sk-secret-value\n")
    env, notes = run_sh.prepare(["--imagechd", "1001", "--labels", "gt"], clean_env(), env_path=p)
    assert env["ANTHROPIC_API_KEY"] == "sk-secret-value"
    assert notes == [f"run.sh: from {p}: ANTHROPIC_API_KEY (values not shown)"]
    env, notes = run_sh.prepare(["--imagechd", "1001", "--labels", "gt"],
                                clean_env(ANTHROPIC_API_KEY="from-shell"), env_path=p)
    assert env["ANTHROPIC_API_KEY"] == "from-shell" and notes == []


@pytest.mark.parametrize("args, var", [
    (["--imagechd", "1", "--labels", "gt"], "ANTHROPIC_API_KEY"),
    (["--imagechd", "1", "--labels", "gt", "--model", "openrouter:openai/gpt-6-astra"], "OPENROUTER_API_KEY"),
    (["--imagechd", "1", "--labels", "gt", "--model=local:qwen2.5-vl-3b"], "CHD_LOCAL_BASE_URL"),
])
def test_prepare_refuses_a_missing_key_before_slicer_starts(tmp_path, args, var):
    with pytest.raises(run_sh.RunShError) as e:
        run_sh.prepare(args, clean_env(), env_path=tmp_path / "none")
    assert var in str(e.value) and "put it in .env and use ./run.sh" in str(e.value) and "Nothing ran" in str(e.value)
    assert str(tmp_path / "none") in str(e.value)


def test_prepare_dry_run_reads_no_env_file_and_needs_no_key(tmp_path):
    p = tmp_path / ".env"
    p.write_text("OPENROUTER_API_KEY=x\n")
    env, notes = run_sh.prepare(["--imagechd", "1", "--labels", "gt", "--dry-run-scripted"], clean_env(), env_path=p)
    assert "OPENROUTER_API_KEY" not in env and notes == []


def test_prepare_unknown_provider_runtime_and_slicer_url(tmp_path):
    with pytest.raises(run_sh.RunShError, match="unknown provider 'nope'"):
        run_sh.prepare(["--imagechd", "1", "--labels", "gt", "--model", "nope:x"], clean_env(),
                       env_path=tmp_path / "none")
    rt = str(REPO / "config" / "runtime.json")
    env, _ = run_sh.prepare(["--imagechd", "1", "--labels", "gt", "--dry-run-scripted", "--runtime", rt], clean_env(),
                            env_path=tmp_path / "none")
    assert env[RUNTIME_ENV] == rt
    with pytest.raises(run_sh.RunShError, match=r"^runtime: .*my\.json.*Nothing ran"):
        run_sh.prepare(["--imagechd", "1", "--labels", "gt", "--dry-run-scripted", "--runtime",
                        str(tmp_path / "my.json")], clean_env(), env_path=tmp_path / "none")
    with pytest.raises(run_sh.RunShError, match="starts its own Slicer"):
        run_sh.prepare(["--imagechd", "1", "--dry-run-scripted", "--slicer-url", "http://x"], clean_env(),
                       env_path=tmp_path / "none")


def _run_batch():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_batch", REPO / "slurm" / "run_batch.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    return rb


@pytest.mark.parametrize("line", ["ANTHROPIC_AUTH_TOKEN=test-not-a-token", "ANTHROPIC_API_KEY=test-not-a-key",
                                  "ANTHROPIC_CUSTOM_HEADERS=X-Api-Key: test-not-a-key"])
def test_run_sh_run_episode_and_run_batch_accept_what_the_anthropic_sdk_accepts(tmp_path, line):
    """A claude- model takes any credential the anthropic SDK resolves, not only $ANTHROPIC_API_KEY. run.sh, run-episode
    and run_batch use one check (registry.sdk_client + sdk_has_credentials) with the environment the episode gets, .env
    included."""
    p = tmp_path / ".env"
    p.write_text(line + "\n")
    before = dict(os.environ)
    env, _ = run_sh.prepare(["--imagechd", "1001", "--labels", "gt"], clean_env(), env_path=p)
    assert dict(os.environ) == before
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    assert registry.missing_env("claude-opus-5-5", env, reg) == []
    _run_batch().check_model_env("claude-opus-5-5", env, reg)
    assert registry.make_adapter("claude-opus-5-5", registry=reg, environ=env).provider == "anthropic"


def test_run_batch_still_names_the_api_key_when_the_sdk_finds_no_credential():
    rb = _run_batch()
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(rb.BatchError, match="ANTHROPIC_API_KEY not set .*--model claude-opus-5-5"):
        rb.check_model_env("claude-opus-5-5", clean_env(), reg)
    with pytest.raises(registry.ProviderError, match=r"found no credentials\. Set ANTHROPIC_API_KEY \(or "
                                                     r"ANTHROPIC_AUTH_TOKEN: export it, or put it in \.env and use "
                                                     r"\./run\.sh\)"):
        registry.make_adapter("claude-opus-5-5", registry=reg, environ=clean_env())


def vendor_registry(tmp_path):
    prov = tmp_path / "providers.json"
    prov.write_text(json.dumps({"providers": {"vendor": {
        "kind": "openai_compatible", "base_url": "https://api.vendor.test/v1", "api_key_env": "VENDOR_API_KEY",
        "max_output_tokens": {"big": 16384},
        "cost": {"source": "estimated", "prices": {"big": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}}}}}))
    return ["--imagechd", "1001", "--labels", "gt", "--providers", str(prov)]


@pytest.mark.parametrize("extra, msg", [
    (["--model", "vendor:small"], "has none for 'small'"),
    (["--model", "vendor:big", "--max-tokens", "32000"],
     "--max-tokens 32000 is above this model's output limit, 16384"),
    (["--model", "vendor:big", "--prompt", "no-such-version"], "no-such-version"),
])
def test_prepare_checks_the_model_as_run_episode_builds_it(tmp_path, monkeypatch, capsys, extra, msg):
    """A missing price or a --max-tokens above a provider's output limit stops run.sh before it starts Slicer: run.sh
    runs run-episode's own prelaunch first."""
    env = clean_env(VENDOR_API_KEY="test-not-a-key")
    run_sh.prepare([*vendor_registry(tmp_path), "--model", "vendor:big", "--max-tokens", "16384"], env,
                   env_path=tmp_path / "none")
    with pytest.raises(run_sh.RunShError, match=msg):
        run_sh.prepare([*vendor_registry(tmp_path), *extra], env, env_path=tmp_path / "none")
    monkeypatch.setattr(run_sh.os, "execvpe", lambda *a: pytest.fail("Slicer must not start"))
    monkeypatch.setattr(run_sh.os, "environ", env)
    with pytest.raises(SystemExit) as e:
        run_sh.main([*vendor_registry(tmp_path), *extra])
    assert e.value.code == 2 and msg in capsys.readouterr().err


def test_prepare_refuses_an_option_run_episode_does_not_know(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--dry-run-scripted", "--max-token", "5"],
                       clean_env(), env_path=tmp_path / "none")
    assert e.value.code == 2 and "unrecognized arguments: --max-token" in capsys.readouterr().err


def test_main_gives_the_episode_shell_a_launch_nonce(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(run_sh.os, "execvpe", lambda f, cmd, env: seen.append(env))
    monkeypatch.setattr(run_sh.os, "chdir", lambda p: None)
    monkeypatch.setattr(run_sh.os, "environ", clean_env())
    args = ["--imagechd", "1001", "--labels", "gt", "--dry-run-scripted"]
    run_sh.main(args)
    run_sh.main(args)
    a, b = (e[run_sh.NONCE_ENV] for e in seen)
    assert len(a) == 32 and a != b
    monkeypatch.setattr(run_sh.os, "environ", clean_env(SLICER_MCP_LAUNCH_NONCE="mine"))
    run_sh.main(args)
    assert seen[-1][run_sh.NONCE_ENV] == "mine"


def launch_state(tmp_path, launch_id):
    """A runtime whose eval_launch.state_file is in tmp_path, with a state file that records `launch_id`."""
    rt = json.loads((REPO / "config" / "runtime.json").read_text())
    rt["eval_launch"] = {**rt.get("eval_launch", {}), "state_file": str(tmp_path / "eval_slicer.json")}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(rt))
    if launch_id is not None:
        (tmp_path / "eval_slicer.json").write_text(json.dumps({"pid": 1, "launch_id": launch_id}))
    return path


def test_own_launch_is_the_state_files_launch_id(tmp_path, monkeypatch):
    rt = launch_state(tmp_path, "abc123")
    assert run_sh.own_launch("abc123", rt)
    assert not run_sh.own_launch("other", rt)
    assert not run_sh.own_launch("", rt)
    (tmp_path / "eval_slicer.json").unlink()
    assert not run_sh.own_launch("abc123", rt)
    launch_state(tmp_path, "abc123")
    monkeypatch.setenv(RUNTIME_ENV, str(rt))
    for nonce, code in (("abc123", 0), ("other", 1)):
        with pytest.raises(SystemExit) as e:
            run_sh.main(["--own-launch", nonce])
        assert e.value.code == code


@pytest.mark.parametrize("args, msg", [
    (["--labels", "gt", "--dry-run-scripted"], "a case is required"),
    (["--imagechd", "", "--labels", "gt", "--dry-run-scripted"], "--imagechd '' is not a case id"),
    (["--imagechd=abc", "--labels", "gt", "--dry-run-scripted"], "--imagechd 'abc' is not a case id"),
])
def test_prepare_refuses_a_bad_case_id_before_slicer_starts(tmp_path, args, msg):
    with pytest.raises(run_sh.RunShError, match=msg) as e:
        run_sh.prepare(args, clean_env(), env_path=tmp_path / "none")
    assert "chd_client.task.imagechd list" in str(e.value) and "Nothing ran" in str(e.value)
    run_sh.prepare(["--imagechd=1001", "--labels", "gt", "--dry-run-scripted"], clean_env(),
                   env_path=tmp_path / "none")


def test_the_eval_slicer_runs_the_case_profile(tmp_path, monkeypatch):
    """An MRI case launches Slicer with chd_mr_v1, so the bridge serves the profile the episode uses."""
    from chd_client.task import cohorts
    vol = tmp_path / "scan.nii.gz"
    vol.write_bytes(b"x")
    lab = tmp_path / "seg.nii.gz"
    lab.write_bytes(b"x")
    env, _ = run_sh.prepare(["--volume", str(vol), "--modality", "mr", "--labels", str(lab), "--dry-run-scripted"],
                            clean_env(), env_path=tmp_path / "none")
    assert env[run_sh.PROFILE_ENV] == str(cohorts.PROFILES["mr"])
    env, _ = run_sh.prepare(["--imagechd", "1001", "--labels", "gt", "--dry-run-scripted"], clean_env(),
                            env_path=tmp_path / "none")
    assert run_sh.PROFILE_ENV not in env


def test_main_bad_case_id_exits_2_without_slicer(monkeypatch, capsys):
    monkeypatch.setattr(run_sh.os, "execvpe", lambda *a: pytest.fail("Slicer must not start"))
    with pytest.raises(SystemExit) as e:
        run_sh.main(["--imagechd", "", "--labels", "gt", "--dry-run-scripted"])
    assert e.value.code == 2 and "is not a case id" in capsys.readouterr().err


def test_command_runs_the_episode_shell_under_xvfb_session_on_linux():
    args = ["--imagechd", "1001", "--labels", "gt", "--model", "openrouter:a/b"]
    assert run_sh.command(args, linux=False) == ["bash", "-c", run_sh.INNER, "run.sh", *args]
    cmd = run_sh.command(args, linux=True, xvfb_session=Path("/m/tools/xvfb_session.py"),
                         xvfb_runtime=Path("/m/config/runtime.local.json"))
    assert cmd == [sys.executable, "/m/tools/xvfb_session.py", "--runtime", "/m/config/runtime.local.json", "--",
                   "bash", "-c", run_sh.INNER, "run.sh", *args]
    from chd_client import slicer
    cmd = run_sh.command(args, linux=True)
    assert cmd[1] == str(slicer.repo() / "tools" / "xvfb_session.py") and cmd[3] == str(slicer.xvfb_runtime())


def test_main_usage(capsys):
    with pytest.raises(SystemExit) as e:
        run_sh.main([])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        run_sh.main(["--help"])
    assert e.value.code == 0 and "./run.sh --imagechd" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        run_sh.main(["--imagechd", "1001", "-h"])
    assert e.value.code == 0 and "uv run run-episode --help" in capsys.readouterr().out


def fake_uv(tmp_path, launch=0, episode=0, stop=0, stop_says="", own=1):
    """A `uv` that appends its arguments to calls.log and exits with the given code per command; the stop prints
    stop_says; --own-launch (is the recorded launch this run.sh's?) exits `own`. The launch nonce each command sees
    goes to nonce.log."""
    log = tmp_path / "calls.log"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f"""#!/bin/bash
echo "$*" >> {log}
echo "${{SLICER_MCP_LAUNCH_NONCE-unset}} ${{CHD_RUN_SH_LAUNCH_NONCE-unset}}" >> {tmp_path / "nonce.log"}
case "$*" in
  *"--own-launch"*) exit {own} ;;
  *"launch --no-main-window"*) exit {launch} ;;
  *"launch --stop"*) echo "{stop_says}"; exit {stop} ;;
  *run-episode*) exit {episode} ;;
esac
""")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    return log, {**os.environ, "PATH": f"{uv.parent}:{os.environ['PATH']}"}


def inner(tmp_path, **codes):
    log, env = fake_uv(tmp_path, **codes)
    env = {k: v for k, v in env.items() if k not in KEYS}
    env[run_sh.NONCE_ENV] = "n0nce"
    p = subprocess.run(run_sh.command(["--imagechd", "1001", "--labels", "gt"], linux=False), env=env,
                       capture_output=True, text=True)
    return p.returncode, log.read_text().splitlines(), p.stderr, p.stdout


@pytest.mark.parametrize("codes, rc, ran_episode, stopped", [
    ({}, 0, True, True),
    ({"episode": 1}, 1, True, True),
    ({"launch": 1}, 10, False, False),
    ({"stop": 1}, 11, True, True),
    ({"episode": 1, "stop": 1}, 1, True, True),
])
def test_episode_shell_exit_codes(tmp_path, codes, rc, ran_episode, stopped):
    got, calls, err, _ = inner(tmp_path, **codes)
    assert got == rc
    assert calls[0] == "run --no-sync chd-slicer launch --no-main-window"
    assert ("run --no-sync run-episode --imagechd 1001 --labels gt" in calls) == ran_episode
    assert ("run --no-sync chd-slicer launch --stop" in calls) == stopped
    if codes.get("stop"):
        assert "launch --stop failed" in err
    if codes.get("launch"):
        assert calls[1:] == ["run --no-sync python -m chd_client.run_sh --own-launch n0nce"]


def test_episode_shell_gives_the_nonce_to_the_launch_only(tmp_path):
    got, calls, _, _ = inner(tmp_path)
    nonces = (tmp_path / "nonce.log").read_text().splitlines()
    assert got == 0 and calls[0].endswith("launch --no-main-window")
    assert calls[1].startswith("run --no-sync run-episode")
    assert nonces[0] == "n0nce unset"
    assert nonces[1:] == ["unset unset", "unset unset"]


def test_episode_shell_stops_its_own_slicer_after_a_failed_launch(tmp_path):
    """After a failed launch, the Slicer this run.sh started (the state file records its nonce) is stopped; the exit
    code is 10."""
    got, calls, err, _ = inner(tmp_path, launch=1, own=0)
    assert got == 10 and calls == ["run --no-sync chd-slicer launch --no-main-window",
                                   "run --no-sync python -m chd_client.run_sh --own-launch n0nce",
                                   "run --no-sync chd-slicer launch --stop"]
    assert "stopping the Slicer this run.sh started" in err


def slow_launch_uv(tmp_path, launch_s):
    """A `uv` whose launch takes launch_s seconds (touching `started` first) and then succeeds; the recorded launch is
    this run.sh's (--own-launch exits 0)."""
    log = tmp_path / "calls.log"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f"""#!/bin/bash
echo "$*" >> {log}
case "$*" in
  *"launch --no-main-window"*) touch {tmp_path}/started; sleep {launch_s} ;;
esac
exit 0
""")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    return log, {**env, "PATH": f"{uv.parent}:{os.environ['PATH']}", run_sh.NONCE_ENV: "n0nce"}


@pytest.mark.parametrize("group, sig, rc", [(True, signal.SIGINT, 130), (True, signal.SIGTERM, 143),
                                            (False, signal.SIGTERM, 143), (False, signal.SIGHUP, 129)])
def test_episode_shell_stops_its_own_slicer_after_a_signal_during_the_launch(tmp_path, group, sig, rc):
    """A signal while Slicer starts (macOS: no xvfb_session cleans up), to the whole process group (Ctrl-C) or to the
    shell alone (the launch then ends normally), stops this run.sh's Slicer and runs no episode; the exit code is 128 +
    the signal number."""
    log, env = slow_launch_uv(tmp_path, 30 if group else 2)
    p = subprocess.Popen(run_sh.command(["--imagechd", "1001", "--labels", "gt"], linux=False), env=env,
                         start_new_session=True, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if (tmp_path / "started").exists():
            break
        time.sleep(0.1)
    (os.killpg if group else os.kill)(p.pid, sig)
    _, err = p.communicate(timeout=60)
    assert p.returncode == rc and log.read_text().splitlines() == [
        "run --no-sync chd-slicer launch --no-main-window",
        "run --no-sync python -m chd_client.run_sh --own-launch n0nce",
        "run --no-sync chd-slicer launch --stop"]
    assert "signal received: Slicer's start ends" in err


NOT_STOPPED = "pid 42 is no longer the recorded Slicer (start time or command differ): not stopped"


@pytest.mark.parametrize("episode, rc", [(0, 11), (1, 1)])
def test_episode_shell_treats_not_stopped_as_a_failed_stop(tmp_path, episode, rc):
    """slicer-mcp's `launch --stop` prints "not stopped" and still exits 0 (slicer-eval-mcp launch.py stop())."""
    got, calls, err, out = inner(tmp_path, episode=episode, stop_says=NOT_STOPPED)
    assert got == rc and calls[-1] == "run --no-sync chd-slicer launch --stop"
    assert NOT_STOPPED in out and "did not stop Slicer" in err


def test_episode_shell_shows_the_stop_output(tmp_path):
    got, _, err, out = inner(tmp_path, stop_says="stopped eval Slicer (pid 42)")
    assert got == 0 and "stopped eval Slicer (pid 42)" in out and "run.sh:" not in err


def test_episode_shell_stops_slicer_after_a_signal(tmp_path):
    """SIGTERM to the shell and the runner (xvfb_session signals the process group): the shell waits for the runner,
    then stops Slicer; the runner's exit code is kept."""
    log = tmp_path / "calls.log"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f"""#!/bin/bash
echo "$*" >> {log}
case "$*" in
  *run-episode*) trap 'echo terminated >> {log}; exit 1' TERM; touch {tmp_path}/started; sleep 30 & wait ;;
esac
exit 0
""")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{uv.parent}:{os.environ['PATH']}"}
    p = subprocess.Popen(run_sh.command(["--imagechd", "1001", "--labels", "gt"], linux=False), env=env,
                         start_new_session=True, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if (tmp_path / "started").exists():
            break
        time.sleep(0.1)
    os.killpg(p.pid, 15)
    _, err = p.communicate(timeout=30)
    calls = log.read_text().splitlines()
    assert p.returncode == 1 and calls[-2:] == ["terminated", "run --no-sync chd-slicer launch --stop"]
    assert "signal received" in err


@pytest.mark.parametrize("args, msg", [
    (["--slicer"], "--slicer requires the path to the Slicer launcher"),
    (["--slicer", "--skip-check"], "--slicer requires the path to the Slicer launcher"),
    (["--imagechd-dir", ""], "--imagechd-dir requires the ImageCHD dataset folder"),
    (["--skip-check", "--check-case"], "--check-case requires an ImageCHD case id"),
    (["--xvfb"], "--xvfb requires"), (["--lib-dir"], "--lib-dir requires"), (["--shared"], "--shared requires"),
])
def test_setup_option_without_value_is_a_usage_error(args, msg):
    p = subprocess.run(["bash", str(REPO / "setup.sh"), *args], capture_output=True, text=True, timeout=30)
    assert p.returncode == 2 and msg in p.stderr and "unbound variable" not in p.stderr and p.stdout == ""


def test_setup_help_prints_the_header_only():
    p = subprocess.run(["bash", str(REPO / "setup.sh"), "--help"], capture_output=True, text=True, timeout=30)
    assert p.returncode == 0 and p.stdout.startswith("# Set up CHD-Bench") and "Exit code:" in p.stdout
    assert "set -euo" not in p.stdout


def test_run_sh_is_executable_and_runs_the_module():
    text = (REPO / "run.sh").read_text()
    assert os.access(REPO / "run.sh", os.X_OK) and "python -m chd_client.run_sh" in text


def test_missing_key_messages_say_how_to_set_it():
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    with pytest.raises(registry.ProviderError, match=r"export it, or put it in \.env and use \./run\.sh"):
        registry.make_adapter("openrouter:openai/gpt-6-astra", registry=reg, environ={})
    with pytest.raises(registry.ProviderError, match="127.0.0.1:8098/v1"):
        registry.make_adapter("local:qwen2.5-vl-3b", registry=reg, environ={})


def test_local_model_error_names_the_gateway_default_port(monkeypatch):
    monkeypatch.delenv("CHD_LOCAL_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="127.0.0.1:8098/v1"):
        LocalModel("qwen")


def test_gateway_default_port_is_the_one_the_docs_and_errors_use(monkeypatch):
    from chd_client.gateway import cli
    seen = {}
    monkeypatch.setattr(cli, "cmd_serve", lambda a: seen.setdefault("port", a.port) and 0)
    cli.main(["serve", "--profile", "p.yaml"])
    assert seen["port"] == 8098
    for doc in ("README.md", "docs/GATEWAY.md"):
        text = (REPO / doc).read_text()
        assert "127.0.0.1:8098/v1" in text and "127.0.0.1:8000" not in text, doc
