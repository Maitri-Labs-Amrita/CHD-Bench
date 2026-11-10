"""Launchers start, check and stop workers. Every call blocks (the scheduler runs them in a thread).

A handle is a small JSON-able dict that the launcher returns from start(); it is saved in the gateway state file,
so a restarted gateway can check and stop the workers of the previous run.
"""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class LaunchError(RuntimeError):
    pass


class Launcher(ABC):
    launches = True

    def __init__(self, profile, pool_dir: Path, frozen_profile: Path, worker_key: str, heartbeat: Path,
                 orphan_timeout_s: float = 900.0):
        self.profile = profile
        self.spec = profile.launcher
        self.pool_dir = pool_dir
        self.frozen_profile = frozen_profile
        self.worker_key = worker_key
        self.heartbeat = heartbeat
        self.orphan_timeout_s = orphan_timeout_s

    def worker_argv(self, worker_id: str, bind: str) -> list:
        return [self.spec.worker_python or sys.executable, "-m", "chd_client.gateway.worker",
                "--profile", str(self.frozen_profile), "--profile-sha256", self.profile.sha256,
                "--profile-dir", str(self.profile.path.parent) if self.profile.path else "",
                "--worker-id", worker_id, "--workdir", str(self.pool_dir), "--bind", bind,
                "--heartbeat", str(self.heartbeat), "--orphan-timeout-s", str(self.orphan_timeout_s)]

    def ready(self, worker_id: str) -> dict | None:
        """The worker's published URL file, once its engine answers."""
        f = self.pool_dir / "workers" / f"{worker_id}.json"
        try:
            return json.loads(f.read_text())
        except (OSError, ValueError):
            return None

    def can_start(self) -> bool:
        return True

    def log_path(self, handle: dict) -> str | None:
        """The worker's log file (the worker wrapper's and its engine's output), or None when not known."""
        return handle.get("log")

    @abstractmethod
    def start(self, worker_id: str) -> dict:
        ...

    @abstractmethod
    def status(self, handle: dict) -> str | None:
        """"alive", "gone", or None when it cannot be known now (a failed squeue must never mean gone)."""

    def status_many(self, handles: dict) -> dict:
        return {k: self.status(h) for k, h in handles.items()}

    @abstractmethod
    def stop(self, handle: dict) -> None:
        """Ask the worker to end. Not a confirmation: the scheduler checks status() and retries."""

    def forget(self, handle: dict) -> None:
        """The worker is confirmed gone: free what the launcher kept for it (e.g. its GPUs)."""

    def adopt(self, handle: dict) -> None:
        """Take back the resources of a worker kept from a previous gateway run (e.g. its GPUs)."""
