import sys
import textwrap
from pathlib import Path

import httpx
import pytest
import yaml

from chd_client.gateway.app import Gateway, create_app
from chd_client.gateway.profile import load_profile

FAKE = Path(__file__).with_name("fake_engine.py")

TOOLS = [{"type": "function", "function": {"name": "look", "description": "look at the scene",
                                           "parameters": {"type": "object", "properties": {}}}}]


def chat(text="hello", **extra) -> dict:
    return {"model": "fake", "messages": [{"role": "system", "content": "sys"},
                                          {"role": "user", "content": text}], "tools": TOOLS, **extra}


def write_profile(tmp_path: Path, body: str, name: str = "fake.yaml") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def fake_profile(tmp_path: Path, *, name="fake", mode="call", args=(), engine=None, launcher=None, scheduler=None,
                 **sections) -> Path:
    data = {"name": name,
            "engine": {"kind": "custom", "command": [sys.executable, str(FAKE), "--port", "{port}", "--host", "{host}",
                                                     "--mode", mode, *args], **(engine or {})},
            "launcher": {"kind": "local", "startup_timeout_s": 30, **(launcher or {})},
            "scheduler": {"request_timeout_s": 20, **(scheduler or {})}, **sections}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


class Harness:
    """A gateway in this process (its workers are real subprocesses) and an HTTP client for it."""

    def __init__(self, paths, workdir, **kw):
        kw.setdefault("tick_s", 0.2)
        kw.setdefault("poll_s", 0.1)
        self.gw = Gateway([load_profile(p) for p in paths], workdir, **kw)
        self.app = create_app(self.gw)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gw",
                                        timeout=60)

    async def __aenter__(self):
        await self.gw.start()
        return self

    async def __aexit__(self, *exc):
        await self.gw.stop()
        await self.client.aclose()

    async def post(self, body, episode=None, request_id=None, headers=None):
        h = dict(headers or {})
        if episode:
            h["X-Episode-Id"] = episode
        if request_id:
            h["X-Request-Id"] = request_id
        return await self.client.post("/v1/chat/completions", json=body, headers=h)
