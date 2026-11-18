"""./run.sh: one CHD episode with one command. It starts the eval Slicer, runs the episode and always stops Slicer.

    ./run.sh --imagechd <ID> | --hvsmr <N> | --volume <SCAN> [--labels ...] [--model <MODEL>] [any run-episode option]

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
   On Linux all three run on a virtual display: slicer-mcp's xvfb-session with the Xvfb settings of CHD-Bench's
   config/runtime.local.json (./setup.sh writes it). On macOS they run directly.

Exit code: run-episode's (0 only when the model's report was accepted), 10 when Slicer did not start, 11 when the
episode succeeded but Slicer did not stop (the stop failed, or it printed "not stopped"), 2 for a usage error or a
model that cannot run (a missing key or price, ...); 128 + the signal number when a signal came while Slicer was
starting, and on Linux also when the run was stopped by a signal (slicer-mcp xvfb-session).
--runtime <file> is used for the launch and the stop too. The eval Slicer runs the episode's task profile
(run_episode.arm_profile: chd_mr_v1 for MRI cases, chd_ct_nomask_v1 for --arm 3d, the *_no3d_v1 copy for --arm
2d_seg; $CHD_RUN_SH_PROFILE in the episode shell).
--slicer-url is refused: run.sh starts its own Slicer (for a Slicer you started yourself, use `uv run run-episode`).
"""

import argparse
from pathlib import Path
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
  why="3D Slicer did not start (its messages are above)"
  [ "$sig" = 0 ] || why="stopped by a signal while 3D Slicer was starting"
  if uv run --no-sync python -m chd_client.run_sh --own-launch "$nonce"; then
    echo "run.sh: stopping the Slicer this run.sh started" >&2
    uv run --no-sync chd-slicer launch --stop ${CHD_RUN_SH_PROFILE:+--profile "$CHD_RUN_SH_PROFILE"} ||
      echo "run.sh: chd-slicer launch --stop failed: a Slicer may still run (uv run chd-slicer launch --stop)" >&2
  fi
  uv run --no-sync python -m chd_client.run_sh --show-result "" "$why" before-model || true
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
      "and stop it (on Linux, slicer-mcp xvfb-session stops what is left on its display)" >&2
    [ "$rc" != 0 ] || rc=11 ;;
  esac
fi
uv run --no-sync python -m chd_client.run_sh --show-result "${CHD_RUN_SH_RESULT:-}" || true
exit "$rc"'''


def options(args):
    """The run-episode options run.sh itself needs (the rest go to run-episode unchanged)."""
    ap = argparse.ArgumentParser(prog="run.sh", add_help=False, allow_abbrev=False)
    ap.add_argument("--imagechd")
    ap.add_argument("--hvsmr")
    ap.add_argument("--volume")
    ap.add_argument("--modality")
    ap.add_argument("--runtime")
    ap.add_argument("--slicer-url")
    ap.add_argument("--dry-run-scripted", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true")
    return ap.parse_known_args(args)[0]


def without_yes(args) -> list:
    """The arguments for run-episode: run.sh's own --yes / -y removed."""
    return [x for x in args if x not in ("--yes", "-y")]


def own_launch(nonce, runtime=None) -> bool:
    """Whether the eval launch the state file records is the one started with `nonce` (this run.sh's): its launch_id.
    The state file is slicer-mcp's (slicer_mcp.launch._state_path: the runtime's eval_launch.state_file, else
    data/dev/eval_slicer.json); the runtime is the one chd-slicer launch read ($SLICER_MCP_RUNTIME,
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


def command(args, linux) -> list:
    """The command to exec: the episode shell, on Linux under slicer-mcp's xvfb-session with the runtime that holds
    the Xvfb settings (chd_client.slicer.xvfb_session_command, as slurm/run_batch.py)."""
    inner = ["bash", "-c", INNER, "run.sh", *map(str, args)]
    if not linux:
        return inner
    from chd_client import slicer
    return slicer.xvfb_session_command(inner)


def prepare(args, environ, env_path=env_file.ENV_FILE):
    """(env, notes) for the episode; RunShError with the reason when it cannot run. run-episode's own pre-launch checks
    (run_episode.prelaunch) run here with `env` as the environment, so the model is built exactly as the episode will
    build it; argparse exits 2 for an option run-episode does not know."""
    a = options(args)
    if a.imagechd is None and a.hvsmr is None and a.volume is None:
        raise RunShError("a scan is required: --imagechd ID (list: ./run.sh --list imagechd), --hvsmr N (list: "
                         "./run.sh --list hvsmr) or --volume SCAN (your own scan: a DICOM folder or a NIfTI file). "
                         "Every option: ./run.sh --help. Nothing ran.")
    for opt, val in (("--imagechd", a.imagechd), ("--hvsmr", a.hvsmr)):
        if val is not None:
            try:
                int(val)
            except ValueError:
                hint = " (list: ./run.sh --list imagechd)" if opt == "--imagechd" else " (list: ./run.sh --list hvsmr)"
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
    parsed = run_episode.parser(prog="run.sh").parse_args(without_yes(args))
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


def auto_modality(args) -> tuple[list, dict | None, str | None]:
    """(args, dicom_info, modality note). A DICOM --volume without --modality takes the modality from its header:
    --modality mr is added for an MR series. A --modality that the header contradicts is kept, with a note."""
    from chd_client import ux
    a = options(args)
    if a.volume is None:
        return args, None, None
    try:
        info = ux.dicom_info(a.volume)
    except Exception:  # noqa: BLE001
        info = None
    if info is None or info["modality"] not in ("CT", "MR"):
        return args, info, None
    given = next((args[i + 1] for i, x in enumerate(args[:-1]) if x == "--modality"), None)
    header = "ct" if info["modality"] == "CT" else "mr"
    if given is None:
        return (args + ["--modality", "mr"] if header == "mr" else args), info, "from the DICOM header"
    if given != header:
        return args, info, (f"WARNING: the DICOM header says {info['modality']}, but --modality is {given}")
    return args, info, None


def preflight_problem(parsed, dicom=None) -> str | None:
    """Why the run cannot work, found before Slicer starts (a missing scan or dataset folder, a bad segmentation
    file), as a sentence with the fix; None when there is nothing."""
    from chd_client import ux
    return ux.missing_data(parsed) or ux.label_check(parsed, dicom)[2]


def show_plan(parsed, yes: bool, tty: bool, stdin=None, err=None, dicom=None, modality_note=None,
              check=True) -> bool:
    """Print the run plan and its warnings; for a paid model ask before starting (unless --yes). False: do not run.
    check: run preflight_problem first (main runs it itself, to exit 2 for a problem and 1 for a no)."""
    from chd_client import ux
    err = err or sys.stderr
    problem = preflight_problem(parsed, dicom) if check else None
    if problem:
        print(f"run.sh: {problem}. Nothing ran.", file=err)
        return False
    lines, warns = ux.run_plan(parsed, dicom, modality_note)
    ux.print_lines(lines, err)
    for w in warns:
        print(f"  WARNING: {w}", file=err)
    print("", file=err)
    if not ux.is_paid(parsed) or yes:
        return True
    if not tty:
        print("run.sh: a paid run needs --yes when there is no terminal to ask. Nothing ran.", file=err)
        return False
    if ux.confirm("Start this paid run? [y/N] ", stdin=stdin, stdout=err):
        return True
    print("Not started.", file=err)
    return False


def show_result(pointer, reason=None, before_model=False) -> None:
    """After Slicer stops: the result box of the run that run-episode wrote to `pointer` ($CHD_RUN_SH_RESULT); with no
    pointer, a box with `reason`."""
    from chd_client import ux
    try:
        info = json.loads(open(pointer).read()) if pointer else None
    except (OSError, ValueError):
        info = None
    if info is None:
        ux.print_lines(ux.result_box(None, reason or "the run stopped before it finished (see the messages above)",
                                     before_model))
    else:
        ux.print_lines(ux.result_box(info.get("run_dir"), info.get("error"), bool(info.get("before_model"))))
    if pointer:
        try:
            os.unlink(pointer)
        except OSError:
            pass


def list_things(what) -> int:
    """./run.sh --list imagechd | hvsmr | classes: the cases a run can take, or the --truth class names."""
    if what == "imagechd":
        from chd_client.task import imagechd
        try:
            root = imagechd.dataset_dir()
        except Exception:  # noqa: BLE001
            root = None
        if root is None or not Path(root).is_dir():
            print(f"run.sh: ImageCHD is not set up on this machine (no folder at {root}). Tell CHD-Bench where it "
                  "is: ./setup.sh --imagechd-dir /path/to/ImageCHD (README, section 2)", file=sys.stderr)
            return 2
        try:
            imagechd.main(["list"])
        except SystemExit as e:  # its own error line is printed
            print(f"run.sh: check the folder with ./setup.sh --imagechd-dir {root}", file=sys.stderr)
            return 2 if e.code else 0
        return 0
    if what == "hvsmr":
        from chd_client import paths
        from chd_client.chd_classes import code
        from chd_client.task import cohorts
        try:
            table = paths.get(cohorts.HVSMR_CLINICAL)
        except Exception:  # noqa: BLE001
            table = None
        if table is None or not Path(table).is_file():
            print(f"run.sh: HVSMR-2.0 is not set up on this machine (no hvsmr_clinical.csv at {table}). Tell "
                  "CHD-Bench where it is: ./setup.sh --hvsmr-dir /path/to/HVSMR-2.0 (README, section 2)",
                  file=sys.stderr)
            return 2
        try:
            cases = cohorts.hvsmr_cases()
            rows = [(n, " ".join(code(c) for c in cohorts.hvsmr_truth(n)) or "normal") for n in cases]
        except Exception as e:  # noqa: BLE001
            print(f"run.sh: cannot read the HVSMR-2.0 diagnoses in {table} ({e}). Check the folder with "
                  "./setup.sh --hvsmr-dir /path/to/HVSMR-2.0", file=sys.stderr)
            return 2
        print(f"{'case':<6}diagnosis (class codes: ./run.sh --list classes)")
        for n, dx in rows:
            print(f"{n:<6}{dx}")
        print(f"{len(rows)} cases (use: ./run.sh --hvsmr N --labels gt ...)")
        return 0
    if what == "classes":
        from chd_client.chd_classes import MANIFEST, SEGMENT_NAMES, load
        print(f"Class codes for --truth (from {MANIFEST.relative_to(REPO)}). Any case; several: --truth VSD ToF;")
        print("a normal heart: --truth normal; unknown diagnosis: leave --truth out.")
        for title, pick in (("CT scans (ImageCHD's 16 classes)", "ct"), ("MRI scans (HVSMR-2.0's 20 classes)", "mri")):
            print(f"\n{title}:")
            for c in load():
                if getattr(c, pick):
                    also = f"  (also: {', '.join(c.also)})" if c.also else ""
                    print(f"  {c.code:<7} {c.full_name}{also}")
        print("\nNot class codes: " + ", ".join(SEGMENT_NAMES) + " are the segmentation structures "
              "(e.g. PA = pulmonary artery).")
        return 0
    print("run.sh --list: one of imagechd, hvsmr, classes", file=sys.stderr)
    return 2


def help_text() -> str:
    """./run.sh --help: run-episode's manual (run_episode.parser) with run.sh's own options."""
    from chd_client import run_episode
    ap = run_episode.parser(prog="./run.sh")
    g = ap.add_argument_group("run.sh only")
    g.add_argument("--yes", "-y", action="store_true",
                   help="start a paid run without asking (for scripts)")
    g.add_argument("--list", metavar="WHAT", choices=("imagechd", "hvsmr", "classes"),
                   help="list the ImageCHD cases, the HVSMR-2.0 cases or the class codes")
    return ap.format_help()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--own-launch"]:
        sys.exit(0 if own_launch(args[1] if len(args) > 1 else "") else 1)
    if args[:1] == ["--list"]:
        sys.exit(list_things(args[1] if len(args) > 1 else ""))
    if args[:1] == ["--show-result"]:
        show_result(args[1] if len(args) > 1 else "", args[2] if len(args) > 2 else None,
                    len(args) > 3 and args[3] == "before-model")
        sys.exit(0)
    if not args or {"-h", "--help"} & set(args):
        print(help_text())
        sys.exit(0 if args else 2)
    args, dicom, modality_note = auto_modality(args)
    try:
        env, notes = prepare(args, os.environ)
    except RunShError as e:
        print(f"run.sh: {e}", file=sys.stderr)
        sys.exit(2)
    for n in notes:
        print(n, file=sys.stderr)
    from chd_client import run_episode
    parsed = run_episode.parser(prog="run.sh").parse_args(without_yes(args))
    problem = preflight_problem(parsed, dicom)
    if problem:
        print(f"run.sh: {problem}. Nothing ran.", file=sys.stderr)
        sys.exit(2)
    if not show_plan(parsed, options(args).yes, sys.stdin.isatty(), dicom=dicom, modality_note=modality_note,
                     check=False):
        sys.exit(1)
    env[NONCE_ENV] = env.get(LAUNCH_NONCE_ENV) or secrets.token_hex(16)
    import tempfile
    fd, env["CHD_RUN_SH_RESULT"] = tempfile.mkstemp(prefix="chd_result_", suffix=".json")
    os.close(fd)
    os.unlink(env["CHD_RUN_SH_RESULT"])
    cmd = command(without_yes(args), sys.platform.startswith("linux"))
    os.chdir(REPO)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
