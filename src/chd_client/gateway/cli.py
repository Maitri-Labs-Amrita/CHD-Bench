"""chd-gateway command line.

  chd-gateway serve --profile P.yaml [--profile Q.yaml] [--port 8098] [--host 127.0.0.1] [--workdir DIR]
  chd-gateway check P.yaml [Q.yaml ...]     validate profiles; print each engine command and SLURM job script
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import socket
import sys
from pathlib import Path

from chd_client.gateway import __version__
from chd_client.gateway.profile import ProfileError, load_profile


def _loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def default_workdir(port: int) -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "chd-gateway" / f"{socket.gethostname()}-{port}"


def free_port(host: str) -> int:
    with socket.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def cmd_check(a) -> int:
    from chd_client.gateway.engines import build_command
    from chd_client.gateway.launchers import SlurmLauncher
    bad = 0
    for path in a.profiles:
        try:
            p = load_profile(path)
        except ProfileError as e:
            print(f"error: {e}", file=sys.stderr)
            bad += 1
            continue
        print(f"{path}: ok  name={p.name}  engine={p.engine.kind}  launcher={p.launcher.kind}  "
              f"sha256={p.sha256[:12]}")
        iv = [k for k in ("tool_call_repair", "forced_tool_call") if getattr(p.interventions, k)]
        iv += [f"{k} (by the engine)" for k in p.engine.declared_interventions]
        print(f"  interventions: {', '.join(iv) if iv else 'none'}   cost: {p.cost.source}")
        if p.launches:
            argv, env = build_command(p, host="HOST", port=0, worker_id="WORKER", workdir="WORKDIR",
                                      profile_dir=str(p.path.parent))
            print("  engine command:", " ".join(argv).replace(" --port 0", " --port PORT"))
            if env:
                print("  engine env:", ", ".join(sorted(env)))
        if p.launcher.kind == "slurm":
            work = Path("WORKDIR")
            script = SlurmLauncher(p, work, work / "profile.yaml", "<worker key>", work / "gateway.alive").script(
                "WORKER")
            print("  job script:\n    " + script.rstrip().replace("\n", "\n    "))
    return 1 if bad else 0


def cmd_serve(a) -> int:
    import uvicorn

    from chd_client.gateway.app import Gateway, create_app, log
    try:
        profiles = [load_profile(p) for p in a.profile]
    except ProfileError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    token = None
    if a.token_env:
        token = os.environ.get(a.token_env)
        if not token:
            print(f"error: --token-env {a.token_env}: that environment variable is empty", file=sys.stderr)
            return 2
    if not _loopback(a.host) and not token and not a.allow_no_auth:
        print(f"error: --host {a.host} accepts connections from other machines; give a token with --token-env, "
              "or pass --allow-no-auth on a trusted network", file=sys.stderr)
        return 2
    port = a.port or free_port(a.host)
    workdir = Path(a.workdir).expanduser() if a.workdir else default_workdir(port)
    gw = Gateway(profiles, workdir, token=token, keep_workers=a.keep_workers,
                 orphan_timeout_s=a.orphan_timeout_s, max_body_mb=a.max_body_mb)
    shown = "127.0.0.1" if _loopback(a.host) else (socket.getfqdn() if a.host in ("0.0.0.0", "::") else a.host)
    url = f"http://{shown}:{port}/v1"
    (workdir / "gateway.url").write_text(url + "\n")
    if a.url_file:
        Path(a.url_file).write_text(url + "\n")
    log(f"chd-gateway {__version__} at {url}; models: {', '.join(gw.pools)}; workdir {workdir}")
    uvicorn.run(create_app(gw), host=a.host, port=port, log_level=a.log_level)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="chd-gateway", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="run the gateway")
    s.add_argument("--profile", action="append", required=True, help="a model profile (YAML); repeat for more")
    s.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this machine only)")
    s.add_argument("--port", type=int, default=8098, help="0 = any free port (see <workdir>/gateway.url)")
    s.add_argument("--workdir", help="state, logs and worker files (default ~/.cache/chd-gateway/<host>-<port>); "
                                     "for SLURM workers it must be on a file system the compute nodes share")
    s.add_argument("--url-file", help="also write the gateway's base URL here")
    s.add_argument("--token-env", help="environment variable with a token clients must send (Bearer)")
    s.add_argument("--allow-no-auth", action="store_true", help="allow a non-loopback --host without a token")
    s.add_argument("--keep-workers", action="store_true",
                   help="leave workers running when the gateway stops (the next run takes them over; they stop "
                        "themselves after --orphan-timeout-s without a gateway)")
    s.add_argument("--orphan-timeout-s", type=float, default=900.0)
    s.add_argument("--max-body-mb", type=float, default=512.0, help="largest request accepted (default 512 MB)")
    s.add_argument("--log-level", default="warning")
    c = sub.add_parser("check", help="validate profiles and show what they would run")
    c.add_argument("profiles", nargs="+")
    a = ap.parse_args(argv)
    return cmd_check(a) if a.cmd == "check" else cmd_serve(a)


if __name__ == "__main__":
    raise SystemExit(main())
