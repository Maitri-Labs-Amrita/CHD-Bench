"""The environment: slicer-mcp (the slicer-eval-mcp repo), run with CHD-Bench's task profile.

CHD-Bench holds no environment code. slicer-mcp is an editable install of a slicer-eval-mcp checkout (pyproject
[tool.uv.sources]): it runs only from a checkout, because its Slicer bridge and case-preparation scripts are not in its
wheel. PINNED_COMMIT is the slicer-eval-mcp commit this release is tested with; `check_pin` refuses another commit (or uncommitted
changes) unless CHD_ALLOW_SLICER_MCP_COMMIT=1.

    chd-slicer launch [--no-main-window] [--check | --stop]     the Slicer that serves the bridge (slicer-mcp launch)
    chd-slicer case prepare ... | case remove <case_id> ...     slicer-mcp case, with the CHD profile
    chd-slicer serve | replay <trace dir>                       the other slicer-mcp commands
    chd-slicer tools-md [--check]                               docs/TOOLS.md: the tools as the model receives them

Every command gets `--profile profiles/chd_ct_v1` (where it takes one) and `--runtime <the runtime with absolute
paths>` unless given (task.config.resolved_runtime): slicer-mcp would read relative paths against its own checkout.
Options take their value as the next argument (`--runtime R`); `--runtime=R` / `--profile=P` are refused.
The virtual display of a Linux server (Xvfb) is the environment's: slicer-eval-mcp's tools/xvfb_session.py with its own
config/runtime.local.json (tools/setup_headless.sh writes it).
"""

import os
import subprocess
import sys
from pathlib import Path

from chd_client.task import config
from chd_client.task.config import PROFILE, resolved_runtime

TOOLS_MD = PROFILE.parents[1] / "docs" / "TOOLS.md"
PINNED_COMMIT = "cb8f380fddbe60c274cca6e19374ad1d6a594d12"
ALLOW_ENV = "CHD_ALLOW_SLICER_MCP_COMMIT"
TAKES_PROFILE = {"launch", "serve", "case prepare"}
TAKES_RUNTIME = {"launch", "serve", "case prepare", "case remove"}


class SlicerMCPVersionError(RuntimeError):
    pass


def repo() -> Path:
    """The slicer-eval-mcp checkout slicer-mcp is installed from."""
    from slicer_mcp.config import REPO
    return Path(REPO)


def git_state(path=None) -> dict:
    """{commit, dirty} of the slicer-eval-mcp checkout (None values when it is not a git checkout). dirty: a modified
    or an untracked file (an untracked module can shadow a pinned one); gitignored files (data/, logs) do not count."""
    path = Path(path or repo())

    def git(*a):
        p = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    return {"path": str(path), "commit": commit, "dirty": bool(status) if status is not None else None}


def check_pin(path=None, state=None) -> dict:
    """The checkout's git state; raises SlicerMCPVersionError when it is not PINNED_COMMIT or has uncommitted changes,
    unless $CHD_ALLOW_SLICER_MCP_COMMIT=1 (then the state says `pinned: false` and the run records it)."""
    st = state or git_state(path)
    st["pinned"] = st["commit"] == PINNED_COMMIT and st["dirty"] is False
    if not st["pinned"] and os.environ.get(ALLOW_ENV) != "1":
        raise SlicerMCPVersionError(
            f"slicer-eval-mcp at {st['path']} is at {st['commit'] or 'an unknown commit'}"
            f"{' with uncommitted changes' if st['dirty'] else ''}, not the pinned {PINNED_COMMIT[:12]}. "
            f"Check out {PINNED_COMMIT[:12]} there, or set {ALLOW_ENV}=1 to run anyway (the run records it).")
    return st


def launch_record(canon) -> Path:
    """The file beside a resolved runtime that records which slicer-eval-mcp code the running Slicer loaded."""
    canon = Path(canon)
    return canon.with_name(f"{canon.name}.slicer_mcp.json")


def check_launched_pin(runtime) -> dict | None:
    """The launch-time git state of the running Slicer bridge (None when no launch was recorded). Raises
    SlicerMCPVersionError when the bridge was launched from a commit other than PINNED_COMMIT or from a dirty
    checkout, even if the checkout on disk is pinned now, unless $CHD_ALLOW_SLICER_MCP_COMMIT=1."""
    import json
    f = launch_record(runtime)
    if not f.is_file():
        return None
    st = json.loads(f.read_text())
    st["pinned"] = st.get("commit") == PINNED_COMMIT and st.get("dirty") is False
    if not st["pinned"] and os.environ.get(ALLOW_ENV) != "1":
        raise SlicerMCPVersionError(
            f"the running Slicer bridge was launched from slicer-eval-mcp {st.get('commit') or 'an unknown commit'}"
            f"{' with uncommitted changes' if st.get('dirty') else ''}, not the pinned {PINNED_COMMIT[:12]}: "
            f"relaunch it (chd-slicer launch --stop, then chd-slicer launch), or set {ALLOW_ENV}=1")
    return st


CLI = ("-m", "slicer_mcp.cli")


def command(args) -> list:
    """`python -m slicer_mcp.cli <args>` in this venv."""
    return [sys.executable, *CLI, *map(str, args)]


def with_defaults(args, runtime=None) -> list:
    """args plus --profile (the CHD profile, unless given) and --runtime (the given or default runtime, resolved to
    absolute paths) where the command takes them."""
    args = list(map(str, args))
    for a in args:
        if a.startswith(("--runtime=", "--profile=")):
            raise ValueError(f"write {a.split('=')[0]} <value>, not {a}")
    cmd = " ".join(args[:2]) if args[:1] == ["case"] else (args[0] if args else "")
    if cmd in TAKES_PROFILE and "--profile" not in args:
        args += ["--profile", str(PROFILE)]
    if cmd in TAKES_RUNTIME:
        if "--runtime" in args:
            i = args.index("--runtime")
            runtime = args[i + 1]
            del args[i:i + 2]
        args += ["--runtime", str(resolved_runtime(runtime))]
    return args


def _pop_runtime(args) -> tuple[list, str | None]:
    args = list(map(str, args))
    if "--runtime" not in args:
        return args, None
    i = args.index("--runtime")
    return args[:i] + args[i + 2:], args[i + 1]


def run(argv) -> int:
    """One chd-slicer passthrough command. A new launch runs on a private runtime copy published only when it succeeds
    (task.config.launch_runtime); a successful `launch --stop` forgets the launch state; commands that take no
    --runtime (replay) get the resolved runtime as $SLICER_MCP_RUNTIME."""
    for a in argv:
        if str(a).startswith(("--runtime=", "--profile=")):
            raise ValueError(f"write {str(a).split('=')[0]} <value>, not {a}")
    rest, runtime = _pop_runtime(argv)
    env = dict(os.environ)
    if rest[:1] == ["launch"] and not {"--stop", "--check"} & set(rest):
        staging, canon = config.launch_runtime(runtime)
        state = git_state()
        try:
            rc = subprocess.run(command(with_defaults([*rest, "--runtime", staging])), env=env).returncode
            if rc == 0:
                config.publish_launch(staging, canon)
                import json
                launch_record(canon).write_text(json.dumps({k: state[k] for k in ("path", "commit", "dirty")}))
            return rc
        finally:
            staging.unlink(missing_ok=True)
    args = with_defaults(argv)
    if "--runtime" not in args:
        env[config.RUNTIME_ENV] = str(resolved_runtime(runtime))
    rc = subprocess.run(command(args), env=env).returncode
    if rc == 0 and rest[:1] == ["launch"] and "--stop" in rest:
        config.forget_launch(runtime)
        launch_record(resolved_runtime(runtime)).unlink(missing_ok=True)
    return rc


def xvfb_runtime() -> Path:
    """slicer-eval-mcp's own runtime, which holds the Xvfb settings (xvfb_executable, extra_lib_dirs, xvfb_grace_s):
    config/runtime.local.json when it exists, else config/runtime.json."""
    local = repo() / "config" / "runtime.local.json"
    return local if local.exists() else repo() / "config" / "runtime.json"


def tools_md() -> str:
    """The profile's model-facing tool reference (slicer-mcp's tools-md renderer, written into CHD-Bench)."""
    from slicer_mcp.tools_md import render
    return render(str(PROFILE)).replace(
        "<!-- regenerate: uv run slicer-mcp tools-md (tests/unit/test_tools_md.py checks it) -->",
        "<!-- regenerate: uv run chd-slicer tools-md (tests/test_contract.py checks it) -->")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        sys.exit(0 if argv else 2)
    cleanup = argv[0] == "launch" and {"--stop", "--check"} & set(argv)
    try:
        check_pin()
    except SlicerMCPVersionError as e:
        if not cleanup:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        print(f"warning: {e}", file=sys.stderr)
    if argv[0] == "tools-md":
        text = tools_md()
        if "--check" in argv:
            ok = TOOLS_MD.is_file() and TOOLS_MD.read_text() == text
            print(f"{TOOLS_MD.name} {'is current' if ok else 'differs from the profile: run chd-slicer tools-md'}")
            sys.exit(0 if ok else 1)
        TOOLS_MD.parent.mkdir(parents=True, exist_ok=True)
        TOOLS_MD.write_text(text)
        print(f"wrote {TOOLS_MD} ({len(text)} bytes)")
        sys.exit(0)
    try:
        sys.exit(run(argv))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
