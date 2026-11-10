"""One job of a Slurm batch: its own Xvfb display and eval Slicer, then run-episode on each of its cases in turn.

    uv run python slurm/run_batch.py --config slurm/batch_config.json --job 0 [--dry-run-scripted] [--plan]

The job index is --job, else $SLURM_ARRAY_TASK_ID (a Slurm array task per job). The job:

1. writes <out_dir>/<batch>/job<k>_<name>/runtime.json: a copy of the runtime (`runtime_base`, paths made absolute) with its own
   bridge_port (port_base + k), its own cases_dir (the same ImageCHD case prepared by two jobs at once would share
   one <case_id>.nrrd, and the first to finish deletes it) and its own eval-launch state file and log;
2. re-runs itself under slicer-eval-mcp's tools/xvfb_session.py (a free display number; every process on it is stopped
   when the job ends), starts the eval Slicer there (`chd-slicer launch`, the chd_ct_v1 profile; arm 3d:
   chd_ct_nomask_v1; HVSMR: chd_mr_v1) and checks it;
3. runs `uv run run-episode --imagechd <id> ...` for each case, one after the other (a failed case is logged and
   the next one runs; a Slicer that died is started again first), each with a time limit;
4. stops the eval Slicer. Every case appends one line to results.jsonl (case, exit code, wall time, the summary
   line run-episode printed); its full output is in logs/<case>.log.

API keys come from the workspace .env (KEY=VALUE lines, parsed here; never printed). Which variables a job's model
needs comes from the provider registry (chd_client.models.registry: the provider's api_key_env and a required
base_url_env); the job stops before Slicer starts when one is unset. --dry-run-scripted runs the scripted model (no API
calls) for a plumbing check; --plan prints the job's runtime and commands and runs nothing.

Arms (`arm`): 3d_seg (default; `labels` required: "gt" or a path with {case} / {stem}) or 3d (no mask: the CT only,
profile chd_ct_nomask_v1; `labels` must be absent or null; dataset imagechd only).
"""

import argparse
import datetime
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

CLIENT_REPO = Path(__file__).resolve().parents[1]
ENV_FILE = CLIENT_REPO / ".env"
DEFAULTS = {"out_dir": "slurm/batches", "port_base": None, "cost_cap": 8.0, "episode_timeout_s": 5400,
            "extra_args": [], "runtime_base": str(CLIENT_REPO / "config" / "runtime.local.json"), "arm": "3d_seg",
            "labels": None,
            "reports_dir": None,
            "wandb": None,
            "dataset": "imagechd"}
DATASETS = ("imagechd", "hvsmr")
ARMS = ("3d_seg", "3d")
NOMASK_ARM = "3d"
JOB_KEYS = {"name", "model", "effort", "cases", "extra_args"}


class BatchError(Exception):
    pass


_child = None
_terminated = False


def _on_term(signum, frame):
    """Slurm's TERM before the time limit: pass it on to the running child and start no further case."""
    global _terminated
    _terminated = True
    if _child is not None and _child.poll() is None:
        _child.send_signal(signal.SIGTERM)


def run_forwarding(cmd, timeout=None, **kw) -> int:
    """subprocess.run that passes a SIGTERM on to the child (and waits for it) instead of dying first."""
    global _child
    with subprocess.Popen(cmd, **kw) as p:
        _child = p
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()
            raise
        finally:
            _child = None


def load_config(path) -> dict:
    """The batch config with DEFAULTS filled in; checked (pure apart from reading the file)."""
    c = {**DEFAULTS, **json.loads(Path(path).read_text())}
    if not isinstance(c.get("batch"), str) or not c["batch"]:
        raise BatchError("config needs a batch name (`batch`)")
    if c["arm"] not in ARMS:
        raise BatchError(f"arm {c['arm']!r}: one of {ARMS} (3d_seg: with the segmentation of `labels`; 3d: no mask)")
    if c["dataset"] not in DATASETS:
        raise BatchError(f"dataset {c['dataset']!r}: one of {DATASETS}")
    if c["arm"] == NOMASK_ARM:
        if c["labels"] is not None:
            raise BatchError(f"arm {NOMASK_ARM} shows no segmentation: remove `labels` ({c['labels']!r}) from the "
                             "config, or use arm 3d_seg")
        if c["dataset"] != "imagechd":
            raise BatchError(f"arm {NOMASK_ARM} (no mask) runs dataset imagechd only: {c['dataset']} has no no-mask "
                             "profile or prompt; use arm 3d_seg with `labels`")
    if not c["labels"] and c["arm"] != NOMASK_ARM:
        raise BatchError("the config needs `labels` (\"gt\" or a path with {case})")
    if c["labels"] and c["labels"] != "gt" and "{case}" not in c["labels"] and "{stem}" not in c["labels"]:
        raise BatchError("labels must be \"gt\" or a path with {case} or {stem} (one label map per case)")
    jobs = c.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise BatchError("config needs a non-empty `jobs` list")
    names = set()
    for k, j in enumerate(jobs):
        unknown = set(j) - JOB_KEYS
        if unknown:
            raise BatchError(f"job {k}: unknown keys {sorted(unknown)}")
        if not j.get("name") or not j.get("model"):
            raise BatchError(f"job {k}: needs name and model")
        if j["name"] in names:
            raise BatchError(f"job {k}: name {j['name']!r} used twice")
        names.add(j["name"])
        cases = j.get("cases")
        if not isinstance(cases, list) or not cases or not all(isinstance(n, int) for n in cases):
            raise BatchError(f"job {k} ({j['name']}): cases must be a non-empty list of ImageCHD ids")
    w = c["wandb"]
    if w is not None and (not isinstance(w, dict) or not w.get("project") or set(w) - {"project", "entity", "group"}):
        raise BatchError("wandb must be {\"project\": ..., \"entity\"?: ..., \"group\"?: ...}")
    return c


def job_index(arg, environ=os.environ) -> int:
    v = arg if arg is not None else environ.get("SLURM_ARRAY_TASK_ID")
    if v is None:
        raise BatchError("give --job or run as a Slurm array task (SLURM_ARRAY_TASK_ID)")
    return int(v)


def job_dir(config, k, base=CLIENT_REPO) -> Path:
    out = Path(config["out_dir"])
    out = out if out.is_absolute() else base / out
    return out / config["batch"] / f"job{k}_{config['jobs'][k]['name']}"


def job_runtime(base_rt: dict, config, k, jdir) -> dict:
    """The job's runtime.json (pure): its own bridge port, cases_dir, traces_dir and eval-launch files. The port is
    "auto" (eval_slicer binds a free one and writes it back here) unless the config sets port_base."""
    rt = json.loads(json.dumps(base_rt))
    rt["bridge_port"] = "auto" if config.get("port_base") is None else int(config["port_base"]) + k
    rt["cases_dir"] = str(Path(jdir) / "cases")
    rt["traces_dir"] = str(Path(jdir) / "traces")
    el = dict(rt.get("eval_launch") or {})
    el["state_file"] = str(Path(jdir) / "eval_slicer.json")
    el["log"] = str(Path(jdir) / "eval_slicer.log")
    el["caffeinate"] = False
    el.pop("scratch_dir", None)
    rt["eval_launch"] = el
    rt["_comment"] = f"Written by slurm/run_batch.py for batch {config['batch']} job {k}; do not edit."
    return rt


def episode_command(config, k, case, runtime_path, dry_run=False) -> list:
    """The run-episode command of one case (pure)."""
    j = config["jobs"][k]
    cmd = ["uv", "run", "run-episode", f"--{config.get('dataset') or 'imagechd'}", str(case), "--arm", config["arm"],
           "--runtime", str(runtime_path),
           "--cost-cap", f"{float(config['cost_cap']):g}"]
    if config["labels"]:
        cmd += ["--labels", config["labels"].format(case=case, stem=case_stem(config, case))]
    if dry_run:
        cmd.append("--dry-run-scripted")
    else:
        cmd += ["--model", j["model"]]
        if j.get("effort"):
            cmd += ["--effort", j["effort"]]
    return cmd + [str(a) for a in config.get("extra_args") or []] + [str(a) for a in j.get("extra_args") or []]


def case_stem(config, case) -> str:
    """The case's file stem for `labels` {stem}: ct_<id> (ImageCHD), else the cohort's stem (hvsmr_<n>)."""
    ds = config.get("dataset") or "imagechd"
    if ds == "imagechd":
        return f"ct_{case}"
    from chd_client.task.cohorts import COHORTS
    return COHORTS[ds].stem(case)


def slicer_profile(config) -> str | None:
    """The eval Slicer's profile: the cohort's (HVSMR: chd_mr_v1), arm 3d's (chd_ct_nomask_v1: the profile
    run-episode serves for that arm, chd_client.run_episode.arm_profile), else None (chd-slicer's chd_ct_v1)."""
    ds = (config or {}).get("dataset") or "imagechd"
    if ds == "imagechd" and (config or {}).get("arm") == NOMASK_ARM:
        from chd_client.task.config import NOMASK_PROFILE
        return str(NOMASK_PROFILE)
    if ds == "imagechd":
        return None
    from chd_client.task.cohorts import COHORTS
    return str(COHORTS[ds].profile)


def slicer_command(runtime_path, action=None, profile=None) -> list:
    """`chd-slicer launch` (slicer-mcp launch with the chd_ct_v1 profile, or `profile`) on the job's runtime."""
    cmd = ["uv", "run", "chd-slicer", "launch", "--runtime", str(runtime_path), "--no-main-window"]
    cmd += ["--profile", profile] if profile else []
    return cmd + ([f"--{action}"] if action else [])


def xvfb_wrap(inner) -> list:
    """slicer-eval-mcp's tools/xvfb_session.py with its own runtime (the Xvfb settings are the environment's, not the
    job's; chd_client.slicer.xvfb_runtime)."""
    from chd_client import slicer
    return ["uv", "run", "python", str(slicer.repo() / "tools" / "xvfb_session.py"), "--runtime",
            str(slicer.xvfb_runtime()), "--"] + list(inner)


def read_dotenv(path=ENV_FILE) -> dict:
    """KEY=VALUE lines (chd_client.env_file, shared with ./run.sh). Values are never printed."""
    from chd_client.env_file import read_env_file
    return read_env_file(path)


API_KEYS = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")


def child_env(environ, dotenv, runtime_path, dry_run=False, key_names=(), reports_dir=None) -> dict:
    """The children's environment; a dry run gets no API key at all (API_KEYS and every key variable the provider
    registry names, `key_names`), so a mistake fails instead of spending."""
    drop = set(API_KEYS) | set(key_names) if dry_run else set()
    env = {k: v for k, v in environ.items() if k != "VIRTUAL_ENV" and k not in drop}
    for k, v in dotenv.items():
        env.setdefault(k, v)
    env["SLICER_MCP_RUNTIME"] = str(runtime_path)
    if reports_dir:
        env["CHD_REPORTS_DIR"] = str(reports_dir)
    return env


def port_free(port, host="127.0.0.1") -> bool:
    with socket.socket() as s:
        return s.connect_ex((host, port)) != 0


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _log(jdir, msg):
    line = f"{_now()} {msg}"
    print(line, flush=True)
    with open(Path(jdir) / "job.log", "a") as f:
        f.write(line + "\n")


def start_slicer(rt_path, env, jdir, profile=None):
    r = subprocess.run(slicer_command(rt_path, profile=profile), cwd=CLIENT_REPO, env=env, capture_output=True,
                       text=True)
    _log(jdir, f"eval Slicer start: exit {r.returncode}: {(r.stdout + r.stderr).strip()[-500:]}")
    return r.returncode == 0


def slicer_ok(rt_path, env, profile=None):
    return subprocess.run(slicer_command(rt_path, "check", profile), cwd=CLIENT_REPO, env=env,
                          capture_output=True).returncode == 0


def command_sha256(cmd) -> str:
    """The identity of one planned episode: its run-episode command (model, effort, scripted or not, labels, prompt
    and extra arguments, the case)."""
    return hashlib.sha256(json.dumps([str(c) for c in cmd]).encode()).hexdigest()


INFRA_STOPS = {"runner_exception", "rate_limited", "environment_error", "terminated", "case_identity_mismatch", "cost_unknown"}


def case_done(rec: dict) -> bool:
    """Whether a results.jsonl row is a finished case: exit 0, or a recorded stop reason that is not an infra stop."""
    return rec.get("exit") == 0 or (rec.get("stop") is not None and rec.get("stop") not in INFRA_STOPS)


def stop_reason(folder) -> str | None:
    """The run folder's summary.json stop_reason; None when there is none (the episode did not end normally)."""
    try:
        return json.loads((Path(folder) / "summary.json").read_text()).get("stop_reason")
    except (OSError, ValueError, TypeError):
        return None


def resume_source(jdir, case, expected_sha, traces_dir) -> str | None:
    """The run folder to resume an unfinished case from, or None (run it from step 0): the case's last results.jsonl
    row when it stopped for an infra reason and has a run folder, else the run folder its episode announced when it
    started (inflight/<dataset>-<case>.json: the job was killed before it wrote a row). Never a run that a resume
    already failed on (stop resume_mismatch), and only a run with a complete model turn (chd_client.resume)."""
    from chd_client.resume import ResumeError, load_prior
    folder, stop = None, None
    res = Path(jdir) / "results.jsonl"
    for ln in _lines(res):
        if ln.get("case") == case and ln.get("command_sha256") == expected_sha:
            folder, stop = ln.get("run_folder"), ln.get("stop")
    mpath = next(iter(sorted((Path(jdir) / "inflight").glob(f"*-{case}.json"))), Path(jdir) / "inflight" / "none")
    marker = _read_json(mpath)
    if marker and (folder is None or not res.exists() or mpath.stat().st_mtime > res.stat().st_mtime):
        folder, stop = marker.get("run_dir"), None
    if not folder or stop == "resume_mismatch":
        return None
    try:
        load_prior(folder, traces_dir)
    except ResumeError:
        return None
    return str(folder)


def _lines(path) -> list:
    out = []
    try:
        for ln in Path(path).read_text().splitlines():
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def finished_cases(jdir, expected=None) -> set:
    """Cases this job folder already finished (case_done in results.jsonl) with the same command as planned (`expected`:
    {case: command_sha256}); a scripted dry run never counts for a real model, nor one model for another. A spot job
    that Slurm preempts is requeued (PreemptMode=REQUEUE) and runs again from the start: these are skipped, so only
    unfinished cases restart. A row without a command hash never counts when `expected` is given."""
    out = set()
    try:
        lines = (Path(jdir) / "results.jsonl").read_text().splitlines()
    except OSError:
        return out
    for ln in lines:
        try:
            rec = json.loads(ln)
        except ValueError:
            continue
        if case_done(rec) and (expected is None or
                                     rec.get("command_sha256") == expected.get(rec.get("case"))):
            out.add(rec.get("case"))
    return out


def providers_file(config, k, base=CLIENT_REPO):
    """The --providers file the job's episodes get (its last one in the config's then the job's extra_args), relative
    to the repo (the episodes' working directory); None: $CHD_PROVIDERS, else config/providers.json."""
    args = [str(a) for a in (config.get("extra_args") or []) + (config["jobs"][k].get("extra_args") or [])]
    found = None
    for i, a in enumerate(args):
        if a == "--providers" and i + 1 < len(args):
            found = args[i + 1]
        elif a.startswith("--providers="):
            found = a.split("=", 1)[1]
    if found is None:
        return None
    p = Path(found).expanduser()
    return p if p.is_absolute() else base / p


def load_providers(config, k, env):
    """The provider registry of job k's episodes (BatchError when it is broken)."""
    from chd_client.models import registry
    try:
        return registry.load_registry(providers_file(config, k), env)
    except registry.ProviderError as e:
        raise BatchError(str(e)) from None


def check_model_env(model, env, reg):
    """The model's provider must be known, and its key (unless optional or null) and a required base URL variable set
    (the registry's api_key_env / base_url_env)."""
    from chd_client.models import registry
    try:
        missing = registry.missing_env(model, env, reg)
    except registry.ProviderError as e:
        raise BatchError(str(e)) from None
    if missing:
        raise BatchError(f"{', '.join(missing)} not set (workspace .env), needed by --model {model}")


def log_wandb(config, j, case, folder, rec, dry_run, jdir, log=None) -> str:
    """Log the case's run folder to W&B (chd_client.wandb_log): its URL, or "failed: ..." (the batch goes on; the
    folder can be backfilled with python -m chd_client.wandb_log)."""
    w = config["wandb"]
    model = "scripted" if dry_run else (w.get("group") or j["model"])
    arm = config.get("arm") or "3d_seg"
    try:
        from chd_client import wandb_log
        url = (log or wandb_log.log_episode)(
            folder, model=model, case=case, project=w["project"], entity=w.get("entity") or wandb_log.ENTITY,
            config={"batch": config["batch"], "job": j["name"], "exit": rec["exit"],
                    "command_sha256": rec["command_sha256"], "extra_args": config.get("extra_args"),
                    **({"arm": arm} if arm != "3d_seg" else {})},
            **({"arm": arm} if arm != "3d_seg" else {}))
        return url or "logged"
    except Exception as e:  # noqa: BLE001
        _log(jdir, f"case {case}: W&B logging FAILED ({type(e).__name__}: {str(e)[:300]}); backfill later")
        return f"failed: {type(e).__name__}"


def write_plan(jdir, config, k, expected, dry_run) -> Path:
    """plan.json: the job's planned episodes, so the batch metrics count every planned case once."""
    j = config["jobs"][k]
    plan = {"batch": config["batch"], "job": k, "name": j["name"], "dataset": config.get("dataset") or "imagechd",
            "model": "scripted" if dry_run else j["model"],
            "cases": [{"case": c, "command_sha256": expected[c]} for c in j["cases"]]}
    f = Path(jdir) / "plan.json"
    f.write_text(json.dumps(plan, indent=1) + "\n")
    return f


def case_name(config, case) -> str:
    return "HVSMR-" + str(case) if (config.get("dataset") or "imagechd") == "hvsmr" else f"ImageCHD-{case}"


def run_cases(config, k, rt_path, jdir, dry_run, environ=os.environ):
    """Inside the Xvfb session: start Slicer, run every case, stop Slicer."""
    from chd_client.models.registry import key_env_names
    dotenv = {} if dry_run else read_dotenv()
    reg = load_providers(config, k, {**dotenv, **environ})
    env = child_env(environ, dotenv, rt_path, dry_run, key_env_names(reg), config.get("reports_dir"))
    j = config["jobs"][k]
    if not dry_run:
        check_model_env(j["model"], env, reg)
    (jdir / "logs").mkdir(exist_ok=True)
    expected = {c: command_sha256(episode_command(config, k, c, rt_path, dry_run)) for c in j["cases"]}
    write_plan(jdir, config, k, expected, dry_run)
    done = finished_cases(jdir, expected)
    todo = [c for c in j["cases"] if c not in done]
    if done:
        _log(jdir, f"already finished here (exit 0 in results.jsonl), not run again: {sorted(done)}")
    if not todo:
        _log(jdir, f"job done: {len(j['cases'])} of {len(j['cases'])} cases exited 0 (nothing left to run)")
        return 0
    prof = slicer_profile(config)
    if not start_slicer(rt_path, env, jdir, prof):
        raise BatchError(f"the eval Slicer did not start; see {jdir / 'eval_slicer.log'}")
    failed = 0
    try:
        for case in todo:
            if _terminated:
                _log(jdir, f"SIGTERM: case {case} and later not started")
                break
            if not slicer_ok(rt_path, env, prof):
                _log(jdir, "eval Slicer check failed: starting it again")
                subprocess.run(slicer_command(rt_path, "stop", prof), cwd=CLIENT_REPO, env=env, capture_output=True)
                if not start_slicer(rt_path, env, jdir, prof):
                    raise BatchError("the eval Slicer could not be started again")
            cmd = episode_command(config, k, case, rt_path, dry_run)
            src = resume_source(jdir, case, expected[case], json.loads(rt_path.read_text()).get("traces_dir"))
            if src:
                cmd = cmd + ["--resume", src]
            (jdir / "inflight").mkdir(exist_ok=True)
            marker = jdir / "inflight" / f"{case_name(config, case)}.json"
            marker.unlink(missing_ok=True)
            env_case = {**env, "CHD_RUN_DIR_FILE": str(marker)}
            _log(jdir, f"case {case}: {' '.join(cmd)}")
            t0 = time.monotonic()
            with open(jdir / "logs" / f"{case_name(config, case)}.log", "w") as out:
                try:
                    rc = run_forwarding(cmd, cwd=CLIENT_REPO, env=env_case, stdout=out, stderr=subprocess.STDOUT,
                                        timeout=float(config["episode_timeout_s"]))
                except subprocess.TimeoutExpired:
                    rc = "timeout"
            wall = round(time.monotonic() - t0, 1)
            lines = (jdir / "logs" / f"{case_name(config, case)}.log").read_text(errors="replace").strip().splitlines()
            folder = next((ln[len("run folder: "):] for ln in reversed(lines) if ln.startswith("run folder: ")), None)
            rec = {"batch": config["batch"], "job": k, "name": j["name"], "model": "scripted" if dry_run else j["model"],
                   "effort": None if dry_run else j.get("effort"), "case": case, "exit": rc, "wall_s": wall,
                   "ended": _now(), "run_folder": folder, "stop": stop_reason(folder) if folder else None,
                   "tail": lines[-3:], "command_sha256": expected[case], **({"resumed_from": src} if src else {})}
            if config.get("wandb") and folder:
                rec["wandb"] = log_wandb(config, j, case, folder, rec, dry_run, jdir)
            with open(jdir / "results.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            failed += not case_done(rec)
            _log(jdir, f"case {case}: exit {rc}, stop {rec['stop']} in {wall} s")
    finally:
        r = subprocess.run(slicer_command(rt_path, "stop", prof), cwd=CLIENT_REPO, env=env, capture_output=True,
                           text=True)
        _log(jdir, f"eval Slicer stop: exit {r.returncode}")
    _log(jdir, f"job done: {len(j['cases']) - failed} of {len(j['cases'])} cases finished (case_done)")
    return 0 if failed == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="run_batch", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--job", type=int, help="job index (default $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--dry-run-scripted", action="store_true", help="scripted model, no API calls")
    ap.add_argument("--plan", action="store_true", help="print the runtime and commands; run nothing")
    ap.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _on_term)
    try:
        config = load_config(a.config)
        k = job_index(a.job)
        if not 0 <= k < len(config["jobs"]):
            raise BatchError(f"job {k} not in the config (0..{len(config['jobs']) - 1})")
        jdir = job_dir(config, k)
        rt_path = jdir / "runtime.json"
        if a.inner:
            return run_cases(config, k, rt_path, jdir, a.dry_run_scripted)
        base = Path(config["runtime_base"]).expanduser()
        base = base if base.is_absolute() else CLIENT_REPO / base
        from chd_client.task.config import load_runtime
        rt = job_runtime(load_runtime(base), config, k, jdir)
        inner = ["uv", "run", "--project", str(CLIENT_REPO), "python", str(Path(__file__).resolve()), "--config",
                 str(Path(a.config).resolve()), "--job", str(k), "--inner"] + (
                    ["--dry-run-scripted"] if a.dry_run_scripted else [])
        if a.plan:
            print(json.dumps({"job": k, "dir": str(jdir), "runtime": rt,
                              "session": xvfb_wrap(inner),
                              "episodes": [episode_command(config, k, n, rt_path, a.dry_run_scripted)
                                           for n in config["jobs"][k]["cases"]]}, indent=1))
            return 0
        if rt["bridge_port"] != "auto" and not port_free(rt["bridge_port"]):
            raise BatchError(f"bridge port {rt['bridge_port']} is in use on {socket.gethostname()}")
        (jdir / "cases").mkdir(parents=True, exist_ok=True)
        rt_path.write_text(json.dumps(rt, indent=2) + "\n")
        _log(jdir, f"batch {config['batch']} job {k} ({config['jobs'][k]['name']}) on {socket.gethostname()}, "
                   f"port {rt['bridge_port']} (auto: the bound port is written back to runtime.json), {'scripted' if a.dry_run_scripted else config['jobs'][k]['model']}")
        return run_forwarding(xvfb_wrap(inner), cwd=CLIENT_REPO, env=child_env(os.environ, {}, rt_path))
    except BatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
