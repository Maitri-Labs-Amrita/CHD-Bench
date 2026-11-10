"""./run.sh: one CHD episode with one command. It starts the eval Slicer, runs the episode and always stops Slicer.

    ./run.sh --imagechd <ID> | --hvsmr <N> | --volume <FILE> [--labels ...] [--model <MODEL>] [any run-episode option]

run-episode's options: uv run run-episode --help.

Examples (from the CHD-Bench folder; pick a case with `uv run python -m chd_client.task.imagechd list`):

    ./run.sh --imagechd 1001 --labels gt --dry-run-scripted                      # no model, no cost
    ./run.sh --imagechd 1001 --labels gt --model claude-opus-5-5                 # needs ANTHROPIC_API_KEY (or SDK auth)
    ./run.sh --imagechd 1001 --labels gt --model 'openrouter:openai/gpt-6-astra' # needs OPENROUTER_API_KEY

What it does:
1. Reads the API keys from .env in the CHD-Bench folder (KEY=value lines; gitignored), when the file exists. A
   variable already set in your environment wins. The names it took are printed, never the values. A scripted dry
   run (--dry-run-scripted) reads no .env.
2. Checks, before Slicer starts, that a case is given, then runs run-episode's own pre-launch checks with
   that environment (run_episode.prelaunch: the options, the prompt, the runtime and the model as run-episode builds
   it: the provider registry, config/providers.json, the price, the output limit, the keys or the anthropic SDK's
   credentials, the base URL). No request is sent.
3. Runs, in the CHD-Bench folder (relative paths in the options are relative to it):
       uv run chd-slicer launch --no-main-window
       uv run run-episode <your options>
       uv run chd-slicer launch --stop        (always, also after an error, Ctrl-C or SIGTERM)
   When the launch fails or a signal comes during it, the stop runs only for the Slicer this run.sh started: the
   launch gets a nonce of its own ($SLICER_MCP_LAUNCH_NONCE), and the stop runs only when the launch state file
   records that nonce (a Slicer started by something else is never stopped).
   On Linux all three run on a virtual display: slicer-eval-mcp's tools/xvfb_session.py with slicer-eval-mcp's own
   config/runtime.local.json (./setup.sh writes it). On macOS they run directly.

Exit code: run-episode's (0 only when the model's report was accepted), 10 when Slicer did not start, 11 when the
episode succeeded but Slicer did not stop (the stop failed, or it printed "not stopped"), 2 for a usage error or a
model that cannot run (a missing key or price, ...); 128 + the signal number when a signal came while Slicer was
starting, and on Linux also when the run was stopped by a signal (tools/xvfb_session.py).
--runtime <file> is used for the launch and the stop too. The eval Slicer runs the episode's task profile
(run_episode.arm_profile: chd_mr_v1 for MRI cases, chd_ct_nomask_v1 for --arm 3d; $CHD_RUN_SH_PROFILE in the episode
shell).
--slicer-url is refused: run.sh starts its own Slicer (for a Slicer you started yourself, use `uv run run-episode`).
"""

import argparse
import json
import os
import secrets
import sys

from chd_client import env_file
from chd_client.task.config import REPO, RUNTIME_ENV


class RunShError(Exception):
    pass


NONCE_ENV = "CHD_RUN_SH_LAUNCH_NONCE"
PROFILE_ENV = "CHD_RUN_SH_PROFILE"
LAUNCH_NONCE_ENV = "SLICER_MCP_LAUNCH_NONCE"

INNER = r'''sig=0
on_signal="Slicer's start ends, and a Slicer this run.sh started is stopped"
trap 'sig=130; echo "run.sh: signal received: $on_signal" >&2' INT
trap 'sig=143; echo "run.sh: signal received: $on_signal" >&2' TERM
trap 'sig=129; echo "run.sh: signal received: $on_signal" >&2' HUP
nonce=${CHD_RUN_SH_LAUNCH_NONCE:-}
unset CHD_RUN_SH_LAUNCH_NONCE
lrc=0
SLICER_MCP_LAUNCH_NONCE="$nonce" uv run --no-sync chd-slicer launch --no-main-window ${CHD_RUN_SH_PROFILE:+--profile "$CHD_RUN_SH_PROFILE"} || lrc=$?
if [ "$lrc" != 0 ] || [ "$sig" != 0 ]; then
  [ "$lrc" = 0 ] || echo "run.sh: chd-slicer launch failed (output above)" >&2
  if uv run --no-sync python -m chd_client.run_sh --own-launch "$nonce"; then
    echo "run.sh: stopping the Slicer this run.sh started" >&2
    uv run --no-sync chd-slicer launch --stop ${CHD_RUN_SH_PROFILE:+--profile "$CHD_RUN_SH_PROFILE"} ||
      echo "run.sh: chd-slicer launch --stop failed: a Slicer may still run (uv run chd-slicer launch --stop)" >&2
  fi
  [ "$sig" = 0 ] || exit "$sig"
  exit 10
fi
on_signal="run-episode finishes its run folder, then Slicer stops"
rc=0
uv run --no-sync run-episode "$@" || rc=$?
if ! out=$(uv run --no-sync chd-slicer launch --stop ${CHD_RUN_SH_PROFILE:+--profile "$CHD_RUN_SH_PROFILE"} 2>&1); then
  [ -z "$out" ] || printf '%s\n' "$out"
  echo "run.sh: chd-slicer launch --stop failed: a Slicer may still run (uv run chd-slicer launch --stop)" >&2
  [ "$rc" != 0 ] || rc=11
else
  [ -z "$out" ] || printf '%s\n' "$out"
  case "$out" in *": not stopped"*)  # slicer-mcp prints this and exits 0; it forgets the launch, so no retry
    echo "run.sh: chd-slicer launch --stop did not stop Slicer (line above): a Slicer may still run; find it with ps" \
      "and stop it (on Linux, tools/xvfb_session.py stops what is left on its display)" >&2
    [ "$rc" != 0 ] || rc=11 ;;
  esac
fi
exit "$rc"'''


def options(args):
    """The run-episode options run.sh itself needs (the rest go to run-episode unchanged)."""
    ap = argparse.ArgumentParser(prog="run.sh", add_help=False, allow_abbrev=False)
    ap.add_argument("--imagechd")
    ap.add_argument("--hvsmr")
    ap.add_argument("--volume")
    ap.add_argument("--runtime")
    ap.add_argument("--slicer-url")
    ap.add_argument("--dry-run-scripted", action="store_true")
    return ap.parse_known_args(args)[0]


def own_launch(nonce, runtime=None) -> bool:
    """Whether the eval launch the state file records is the one started with `nonce` (this run.sh's): its launch_id.
    The state file is slicer-mcp's (slicer_mcp.launch._state_path: the runtime's eval_launch.state_file, else
    slicer-eval-mcp's data/dev/eval_slicer.json); the runtime is the one chd-slicer launch read ($SLICER_MCP_RUNTIME,
    else the default). False without a nonce, a state file or a runtime."""
    if not nonce:
        return False
    from slicer_mcp.launch import _state_path
    from chd_client.task.config import load_runtime
    try:
        state = json.loads(_state_path(load_runtime(runtime)).read_text())
    except (OSError, ValueError):
        return False
    return isinstance(state, dict) and state.get("launch_id") == nonce


def command(args, linux, xvfb_session=None, xvfb_runtime=None) -> list:
    """The command to exec: the episode shell, on Linux under slicer-eval-mcp's xvfb_session with its own runtime
    (chd_client.slicer.xvfb_runtime, as slurm/run_batch.py)."""
    inner = ["bash", "-c", INNER, "run.sh", *map(str, args)]
    if not linux:
        return inner
    from chd_client import slicer
    session = xvfb_session or slicer.repo() / "tools" / "xvfb_session.py"
    return [sys.executable, str(session), "--runtime", str(xvfb_runtime or slicer.xvfb_runtime()), "--", *inner]


def prepare(args, environ, env_path=env_file.ENV_FILE):
    """(env, notes) for the episode; RunShError with the reason when it cannot run. run-episode's own pre-launch checks
    (run_episode.prelaunch) run here with `env` as the environment, so the model is built exactly as the episode will
    build it; argparse exits 2 for an option run-episode does not know."""
    a = options(args)
    if a.imagechd is None and a.hvsmr is None and a.volume is None:
        raise RunShError("a case is required: --imagechd ID (see uv run python -m chd_client.task.imagechd list), "
                         "--hvsmr N or --volume FILE. Nothing ran.")
    for opt, val in (("--imagechd", a.imagechd), ("--hvsmr", a.hvsmr)):
        if val is not None:
            try:
                int(val)
            except ValueError:
                hint = " (see uv run python -m chd_client.task.imagechd list)" if opt == "--imagechd" else ""
                raise RunShError(f"{opt} {val!r} is not a case id{hint}. Nothing ran.") from None
    if a.slicer_url:
        raise RunShError("--slicer-url: run.sh starts its own Slicer; for a Slicer you started yourself use "
                         "uv run run-episode")
    env, notes = dict(environ), []
    if not a.dry_run_scripted:
        env, taken = env_file.merged(env, env_file.read_env_file(env_path))
        if taken:
            notes.append(f"run.sh: from {env_path}: {', '.join(taken)} (values not shown)")
    if a.runtime:
        env[RUNTIME_ENV] = a.runtime
    from chd_client import run_episode
    from chd_client.models import registry
    parsed = run_episode.parser(prog="run.sh").parse_args(args)
    try:
        with registry.as_environ(env):
            run_episode.prelaunch(parsed)
    except run_episode.PrelaunchError as e:
        raise RunShError(f"{e}. Nothing ran.") from None
    except registry.ProviderError as e:
        where = f" (run.sh reads KEY=value lines from {env_path} when it exists)" if registry.KEY_HINT in str(e) else ""
        raise RunShError(f"{e}{where}. Nothing ran.") from None
    profile = run_episode.arm_profile(parsed.arm, run_episode.cohort_of(parsed))
    env.pop(PROFILE_ENV, None)
    if profile is not None:
        env[PROFILE_ENV] = str(profile)
    return env, notes


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--own-launch"]:
        sys.exit(0 if own_launch(args[1] if len(args) > 1 else "") else 1)
    if not args or {"-h", "--help"} & set(args):
        print(__doc__.strip())
        sys.exit(0 if args else 2)
    try:
        env, notes = prepare(args, os.environ)
    except RunShError as e:
        print(f"run.sh: {e}", file=sys.stderr)
        sys.exit(2)
    for n in notes:
        print(n, file=sys.stderr)
    env[NONCE_ENV] = env.get(LAUNCH_NONCE_ENV) or secrets.token_hex(16)
    cmd = command(args, sys.platform.startswith("linux"))
    os.chdir(REPO)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
