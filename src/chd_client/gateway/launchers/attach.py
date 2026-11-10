"""Attach launcher: the workers are servers that already run (vLLM, Ollama, LM Studio, ...), one per URL. The gateway
neither starts nor stops them; it checks that each answers GET <url>/models before it sends work there."""

from __future__ import annotations

from chd_client.gateway.launchers.base import LaunchError, Launcher


class AttachLauncher(Launcher):
    launches = False

    def start(self, worker_id: str) -> dict:
        raise LaunchError("an attached server is not started by the gateway")

    def status(self, handle: dict) -> str | None:
        return "alive"

    def stop(self, handle: dict) -> None:
        pass
