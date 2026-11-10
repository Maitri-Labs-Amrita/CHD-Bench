"""Worker wrapper: runs one engine on a free port, waits until the engine answers its health route, then publishes
the worker's URL in <workdir>/workers/<worker_id>.json for the gateway.

It stops the engine (its whole process group) on SIGTERM / SIGINT, when the engine's main process exits, and when
the content of the gateway's heartbeat file has not changed for --orphan-timeout-s by this node's clock (the gateway
died, so nobody would ever stop this worker). The content is read, not the mtime: another NFS client's mtime-only
update may never be seen here. It removes its URL file when it ends.

  python -m chd_client.gateway.worker --profile P --profile-sha256 S --worker-id W --workdir D --bind local|all \\
      --heartbeat F [--profile-dir DIR] [--orphan-timeout-s 900]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from chd_client.gateway.engines import build_command
from chd_client.gateway.profile import ProfileError, load_profile


class Stop(Exception):
    pass


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[worker]", msg, flush=True)


def free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def probe(url: str, key: str | None, timeout: float = 5.0):
    """(ok, parsed JSON body or None)."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok, body = 200 <= r.status < 300, r.read(65536)
    except (urllib.error.URLError, OSError, ValueError):
        return False, None
    try:
        return ok, json.loads(body) if body else None
    except ValueError:
        return ok, None


def stop_engine(proc, why: str):
    """SIGTERM the engine's process group (the engine and every process it started), wait up to 30 s, then
    SIGKILL whatever is left of the group (children can outlive the engine's main process)."""
    if proc.poll() is None:
        log(f"stopping the engine ({why})")
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def write_json(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chd_client.gateway.worker")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--profile-sha256", default=None)
    ap.add_argument("--profile-dir", default="")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--bind", choices=["local", "all"], default="local")
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--orphan-timeout-s", type=float, default=900.0)
    a = ap.parse_args(argv)
    try:
        profile = load_profile(a.profile)
    except ProfileError as e:
        log(str(e))
        return 2
    if a.profile_sha256 and profile.sha256 != a.profile_sha256:
        log(f"the profile changed since the gateway started this worker ({profile.sha256[:12]} != "
            f"{a.profile_sha256[:12]}); not starting")
        return 2
    workdir = Path(a.workdir)
    ready = workdir / "workers" / f"{a.worker_id}.json"
    ready.parent.mkdir(parents=True, exist_ok=True)
    bind = "127.0.0.1" if a.bind == "local" else "0.0.0.0"
    advertise = "127.0.0.1" if a.bind == "local" else socket.getfqdn()
    port = free_port(bind)
    argv_e, env_add = build_command(profile, host=bind, port=port, worker_id=a.worker_id, workdir=str(workdir),
                                    profile_dir=a.profile_dir)
    log(f"worker {a.worker_id} ({profile.name}) on {advertise}:{port}: {shlex.join(argv_e)}")

    def on_signal(signum, _frame):
        raise Stop(f"signal {signum}")

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    proc = None
    why = "engine exited"
    try:
        proc = subprocess.Popen(argv_e, env={**os.environ, **env_add}, start_new_session=True)
        key = os.environ.get("CHD_GATEWAY_WORKER_KEY")
        health = f"http://127.0.0.1:{port}{profile.engine.health_path}"
        started, published = time.time(), False
        beat, beat_seen = None, time.monotonic()
        while proc.poll() is None:
            if not published:
                ok, data = probe(health, key)
                if ok:
                    cap = data.get("capacity") if isinstance(data, dict) else None
                    write_json(ready, {
                        "url": f"http://{advertise}:{port}{profile.engine.api_prefix}",
                        "health_url": f"http://{advertise}:{port}{profile.engine.health_path}",
                        "capacity": cap if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 1 else None,
                        "host": advertise, "port": port, "pid": os.getpid(), "ready_at": time.time(),
                        "startup_s": round(time.time() - started, 1), "profile_sha256": profile.sha256})
                    published = True
                    log(f"ready after {time.time() - started:.0f} s")
                    continue
                time.sleep(1)
            else:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if a.heartbeat:
                try:
                    with open(a.heartbeat) as f:
                        now_beat = f.read().strip()
                except OSError:
                    now_beat = None
                if now_beat and now_beat != beat:
                    beat, beat_seen = now_beat, time.monotonic()
                if time.monotonic() - beat_seen > a.orphan_timeout_s:
                    raise Stop(f"the gateway heartbeat has not changed for {time.monotonic() - beat_seen:.0f} s")
        log(f"engine exited with code {proc.returncode}")
    except Stop as e:
        why = str(e)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            ready.unlink()
        except FileNotFoundError:
            pass
        if proc is not None:
            stop_engine(proc, why)
    return proc.returncode if proc is not None and proc.returncode is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
