"""The environment: slicer-mcp (the slicer-eval-mcp repo), run with CHD-Bench's task profile.

CHD-Bench holds no environment code. slicer-mcp is a dependency installed from its git repository (pyproject
[tool.uv.sources]; pinned to a release tag; uv.lock records its exact commit, and `uv sync` installs that one). The package ships its Slicer
bridge and case-preparation scripts. Every run records the slicer-mcp version and commit it used (source()).

    chd-slicer launch [--no-main-window] [--check | --stop]     the Slicer that serves the bridge (slicer-mcp launch)
    chd-slicer case prepare ... | case remove <case_id> ...     slicer-mcp case, with the CHD profile
    chd-slicer serve | replay <trace dir>                       the other slicer-mcp commands
    chd-slicer tools-md [--check]                               docs/TOOLS.md: the tools as the model receives them

Every command gets `--profile profiles/chd_ct_v1` (where it takes one) and `--runtime <the runtime with absolute
paths>` unless given (task.config.resolved_runtime), so relative paths always mean CHD-Bench's folder.
Options take their value as the next argument (`--runtime R`); `--runtime=R` / `--profile=P` are refused.
The virtual display of a Linux server (Xvfb): slicer-mcp's xvfb-session, with CHD-Bench's own runtime, which holds the
Xvfb settings (xvfb_executable, extra_lib_dirs; ./setup.sh writes them with `slicer-mcp setup-headless`).
"""

import os
import subprocess
import sys
from pathlib import Path

from chd_client.task import config
from chd_client.task.config import PROFILE, resolved_runtime

TOOLS_MD = PROFILE.parents[1] / "docs" / "TOOLS.md"
TAKES_PROFILE = {"launch", "serve", "case prepare"}
TAKES_RUNTIME = {"launch", "serve", "case prepare", "case remove"}


def repo() -> Path:
    """Where slicer-mcp's own files are: its source checkout, or the copy shipped in the installed package."""
    from slicer_mcp.config import REPO
    return Path(REPO)


def source() -> dict:
    """The slicer-mcp code in use: {version, commit, dirty, from_checkout, path}. commit: the git commit it was
    installed from (or the checkout's HEAD); dirty: uncommitted changes in a checkout (False for an installed package)."""
    import slicer_mcp
    from slicer_mcp.config import FROM_CHECKOUT
    from slicer_mcp.tracing.logger import git_info
    return {"version": slicer_mcp.__version__, **git_info(), "from_checkout": FROM_CHECKOUT, "path": str(repo())}


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
        try:
            rc = subprocess.run(command(with_defaults([*rest, "--runtime", staging])), env=env).returncode
            if rc == 0:
                config.publish_launch(staging, canon)
            return rc
        finally:
            staging.unlink(missing_ok=True)
    args = with_defaults(argv)
    if "--runtime" not in args:
        env[config.RUNTIME_ENV] = str(resolved_runtime(runtime))
    rc = subprocess.run(command(args), env=env).returncode
    if rc == 0 and rest[:1] == ["launch"] and "--stop" in rest:
        config.forget_launch(runtime)
    return rc


def xvfb_runtime() -> Path:
    """The runtime that holds the Xvfb settings (xvfb_executable, extra_lib_dirs, xvfb_grace_s): CHD-Bench's
    config/runtime.local.json when it exists, else config/runtime.json (task.config.runtime_path)."""
    return config.runtime_path()


def xvfb_session_command(inner, python=None) -> list:
    """`slicer-mcp xvfb-session --runtime <xvfb_runtime> -- <inner>`: inner on its own virtual display."""
    return [python or sys.executable, *CLI, "xvfb-session", "--runtime", str(xvfb_runtime()), "--", *map(str, inner)]


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
