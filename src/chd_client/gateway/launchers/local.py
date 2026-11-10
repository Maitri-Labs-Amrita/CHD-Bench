"""Local launcher: each worker is a subprocess on this machine, optionally pinned to GPUs with CUDA_VISIBLE_DEVICES.

Every process of a worker (the wrapper, its engine in the engine's own session, the engine's children) carries
CHD_GATEWAY_WORKER_ID=<worker id> in its environment. A worker counts as alive while any process with that tag runs,
and stopping it signals all of them, so an engine whose wrapper was killed (e.g. by the OOM killer) is still found
and stopped (Linux; elsewhere the wrapper's process group is used). GPUs are owned per worker id and released once,
when the worker is confirmed gone.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time

from chd_client.gateway.launchers.base import LaunchError, Launcher

TAG = "CHD_GATEWAY_WORKER_ID"


def members(worker_id: str):
    """The live processes of this user tagged with the worker id; None when /proc cannot be read."""
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    uid, me, tag = os.getuid(), os.getpid(), f"{TAG}={worker_id}".encode()
    out = []
    for e in entries:
        if not e.isdigit() or int(e) == me:
            continue
        try:
            if os.stat(f"/proc/{e}").st_uid != uid:
                continue
            with open(f"/proc/{e}/stat") as f:
                if f.read().rsplit(")", 1)[-1].split()[0] == "Z":
                    continue
            with open(f"/proc/{e}/environ", "rb") as f:
                if tag in f.read().split(b"\0"):
                    out.append(int(e))
        except (OSError, ValueError, IndexError):
            continue
    return out


def handle_status(handle: dict, proc=None) -> str | None:
    """"alive", "gone", or None (unknown) for a local worker handle; proc: its Popen, when this process started it."""
    if handle.get("host") != socket.gethostname():
        return None
    if proc is not None:
        proc.poll()
    found = members(handle.get("worker_id", "")) if handle.get("worker_id") else None
    if found is not None:
        return "alive" if found else "gone"
    try:
        os.killpg(handle["pid"], 0)
    except (ProcessLookupError, PermissionError):
        return "gone"
    return "alive"


def _signal(handle: dict, sig):
    found = members(handle.get("worker_id", "")) if handle.get("worker_id") else None
    if found is None:
        try:
            os.killpg(handle["pid"], sig)
        except (ProcessLookupError, PermissionError):
            pass
        return
    for pid in found:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def kill(handle: dict, proc=None, grace_s: float = 40.0) -> bool:
    """SIGTERM every process of the worker, SIGKILL what is left after grace_s. True once it is gone."""
    st = handle_status(handle, proc)
    if st != "alive":
        return st == "gone"
    _signal(handle, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and handle_status(handle, proc) == "alive":
        time.sleep(0.2)
    if handle_status(handle, proc) == "alive":
        _signal(handle, signal.SIGKILL)
        time.sleep(0.5)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    return handle_status(handle, proc) == "gone"


class LocalLauncher(Launcher):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.free_gpus = list(self.spec.gpus) if self.spec.gpus is not None else None
        self.owned: dict = {}
        self.procs: dict = {}
        self._lock = threading.Lock()

    def can_start(self) -> bool:
        with self._lock:
            return self.free_gpus is None or len(self.free_gpus) >= self.spec.gpus_per_worker

    def start(self, worker_id: str) -> dict:
        gpus = None
        with self._lock:
            if self.free_gpus is not None:
                if len(self.free_gpus) < self.spec.gpus_per_worker:
                    raise LaunchError("no free GPUs")
                gpus = self.free_gpus[:self.spec.gpus_per_worker]
                del self.free_gpus[:self.spec.gpus_per_worker]
                self.owned[worker_id] = gpus
        env = dict(os.environ, CHD_GATEWAY_WORKER_KEY=self.worker_key, **{TAG: worker_id})
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        logs = self.pool_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{worker_id}.log"
        try:
            with open(log_path, "ab") as log:
                p = subprocess.Popen(self.worker_argv(worker_id, "local"), stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        except OSError as e:
            self._release(worker_id)
            raise LaunchError(f"cannot start the worker: {e}") from None
        self.procs[p.pid] = p
        return {"pid": p.pid, "gpus": gpus, "log": str(log_path), "host": socket.gethostname(),
                "worker_id": worker_id}

    def _release(self, worker_id: str):
        """Give the worker's GPUs back, once (a second call does nothing)."""
        with self._lock:
            gpus = self.owned.pop(worker_id, None)
            if gpus is not None and self.free_gpus is not None:
                self.free_gpus.extend(g for g in gpus if g not in self.free_gpus)

    def status(self, handle: dict) -> str | None:
        return handle_status(handle, self.procs.get(handle.get("pid")))

    def stop(self, handle: dict) -> None:
        kill(handle, self.procs.get(handle.get("pid")))

    def forget(self, handle: dict) -> None:
        """The worker is confirmed gone: reap it and free its GPUs (idempotent)."""
        p = self.procs.pop(handle.get("pid"), None)
        if p is not None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._release(handle.get("worker_id", ""))

    def adopt(self, handle: dict) -> None:
        """A worker of a previous run (running, or still stopping): its GPUs stay taken until it is gone."""
        gpus = handle.get("gpus")
        if gpus is not None and self.free_gpus is not None:
            with self._lock:
                self.free_gpus = [g for g in self.free_gpus if g not in gpus]
                self.owned[handle.get("worker_id", "")] = list(gpus)
