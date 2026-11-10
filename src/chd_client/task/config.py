"""The harness's paths and its runtime config.

One runtime.json serves the harness and slicer-mcp: slicer-mcp's keys (bridge host and port, Slicer, cases_dir,
traces_dir, eval_launch) plus the harness's (imagechd_dir, imagechd_diagnosis_sheet). Relative paths are relative to
CHD-Bench. slicer-mcp would read them relative to its own checkout, so the harness never hands it a runtime with relative
paths: `resolved_runtime` keeps one absolute copy per source file (data/dev/<name>.<hash>.resolved.json) and that copy
is what slicer-mcp gets. The source file is never written.

The launch state: `slicer-mcp launch` writes the port it bound and its launch nonce into the runtime file it was given.
A new launch gets a private copy of the source (`launch_runtime`: the source's own port setting, so "auto" stays auto),
which is published as the resolved copy only when the launch succeeds (`publish_launch`); a failed launch leaves the
running one's state alone. Every later resolve keeps the published port and nonce, until `forget_launch` (after
`launch --stop`). Each read-modify-write of a resolved copy holds a file lock. A resolved copy carries RESOLVED_MARK
and is used as it is (a run's own copy, `out`).
"""

import fcntl
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PROFILE = REPO / "profiles" / "chd_ct_v1"
NOMASK_PROFILE = REPO / "profiles" / "chd_ct_nomask_v1"
DEFAULT_RUNTIME = REPO / "config" / "runtime.json"
RUNTIME_ENV = "SLICER_MCP_RUNTIME"
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
PATH_KEYS = ("cases_dir", "traces_dir", "imagechd_dir", "imagechd_diagnosis_sheet", "slicer_executable")
EVAL_LAUNCH_PATH_KEYS = ("log", "state_file", "scratch_dir")
RESOLVED_DIR = REPO / "data" / "dev"
LAUNCH_KEYS = ("bridge_port", "launch_nonce")
RESOLVED_MARK = "_resolved_from"
REQUIRED_KEYS = ("bridge_host", "bridge_port", "cases_dir", "traces_dir", "client_timeout_s")
REQUIRED_NESTED = (("case_prep", "nrrd_compression"),)


class RuntimeConfigError(ValueError):
    pass


def runtime_path(path=None) -> Path:
    """`path`, else $SLICER_MCP_RUNTIME, else config/runtime.local.json when it exists, else config/runtime.json."""
    if path or os.environ.get(RUNTIME_ENV):
        return Path(path or os.environ[RUNTIME_ENV]).expanduser()
    local = REPO / "config" / "runtime.local.json"
    return local if local.exists() else DEFAULT_RUNTIME


def _abs(p) -> str:
    p = Path(os.path.expanduser(str(p)))
    return str(p if p.is_absolute() else REPO / p)


def load_runtime(path=None) -> dict:
    """The runtime with every path absolute (relative ones are relative to CHD-Bench). Refused (RuntimeConfigError): a
    missing REQUIRED_KEYS key, or a non-loopback host."""
    src = runtime_path(path)
    rt = json.loads(src.read_text())
    missing = [k for k in REQUIRED_KEYS if rt.get(k) in (None, "")]
    missing += [".".join(ks) for ks in REQUIRED_NESTED
                if not isinstance(rt.get(ks[0]), dict) or rt[ks[0]].get(ks[1]) is None]
    if missing:
        raise RuntimeConfigError(f"runtime {src} has no {', '.join(missing)} (slicer-mcp needs them; "
                                 f"see config/runtime.json)")
    if rt["bridge_host"] not in LOOPBACK_HOSTS:
        raise RuntimeConfigError(f"bridge_host {rt['bridge_host']!r} is not loopback; refusing")
    for k in PATH_KEYS:
        if rt.get(k):
            rt[k] = _abs(rt[k])
    el = rt.get("eval_launch")
    if isinstance(el, dict):
        for k in EVAL_LAUNCH_PATH_KEYS:
            if el.get(k):
                el[k] = _abs(el[k])
    return rt


def _write_atomic(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


@contextmanager
def _locked(path: Path):
    """An exclusive lock for a resolved copy's read-modify-write (a lock file beside it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(f".{path.name}.lock"), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def _source(path=None) -> tuple[Path, dict, bool]:
    """(source path, its JSON, whether it is a resolved copy already)."""
    src = runtime_path(path)
    raw = json.loads(src.read_text())
    return src, raw, bool(raw.get(RESOLVED_MARK))


def _canon(src: Path) -> Path:
    key = hashlib.sha256(str(src.resolve()).encode()).hexdigest()[:8]
    return RESOLVED_DIR / f"{src.stem}.{key}.resolved.json"


def _fresh(src: Path) -> dict:
    """The source with absolute paths and the mark: no launch state."""
    return {**load_runtime(src), RESOLVED_MARK: str(src.resolve())}


def resolved_runtime(path=None, out=None) -> Path:
    """The runtime slicer-mcp gets, and its path: the source's resolved copy (paths absolute, the published launch
    state kept), updated under its lock when the source changed. A resolved copy is used as it is. out: also write a
    copy of that to `out` (a run's own runtime) and return `out`."""
    src, _, done = _source(path)
    if done:
        load_runtime(src)
        canon = src.resolve()
    else:
        canon = _canon(src)
        rt = _fresh(src)
        with _locked(canon):
            old = _read(canon)
            new = {**rt, **({k: old[k] for k in LAUNCH_KEYS if k in old} if "launch_nonce" in old else {})}
            if new != old:
                _write_atomic(canon, new)
    if out is None:
        return canon
    with _locked(canon):
        data = json.loads(canon.read_text())
    _write_atomic(Path(out), data)
    return Path(out)


def launch_runtime(path=None) -> tuple[Path, Path]:
    """For a new launch: (a private copy of the source for slicer-mcp launch to write its port and nonce into, the
    resolved copy to publish it as). The private copy has the source's own port setting and no earlier launch."""
    src, raw, done = _source(path)
    canon = src.resolve() if done else _canon(src)
    rt = {k: v for k, v in raw.items() if k != "launch_nonce"} if done else _fresh(src)
    staging = canon.with_name(f".{canon.name}.launch-{uuid.uuid4().hex[:8]}.json")
    _write_atomic(staging, rt)
    return staging, canon


def publish_launch(staging: Path, canon: Path):
    """A launch succeeded: its runtime (with the bound port and nonce) becomes the resolved copy."""
    with _locked(canon):
        Path(staging).replace(canon)


def forget_launch(path=None):
    """After `launch --stop`: the resolved copy without the stopped launch's port and nonce."""
    src, _, done = _source(path)
    if done:
        return
    canon = _canon(src)
    with _locked(canon):
        _write_atomic(canon, _fresh(src))
