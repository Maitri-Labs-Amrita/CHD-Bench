"""One MCP connection to the environment (slicer-mcp with the chd_ct_v1 profile, chd_client.slicer), as the runner sees
it.

    async with EnvMCP(server_params()) as env:          # or EnvMCP(srv.server) in-process (unit tests)
        tools = env.model_tools()                             # runner-only tools removed; description/schema verbatim
        episode_id, obs0 = await env.reset("case_nii_<hex>", client_meta)
        obs = await env.call("orbit", {"azimuth_deg": 30}, reasoning="...", tokens={"in": 1200, "out": 40})

Each result becomes an Observation: the server's JSON text block verbatim, its parsed payload, and the PNG frames
in `frames` order. error.type "environment" raises EnvironmentAbort; every other error goes back to the model.
A runner-only tool named by the model (reset_episode) is never forwarded: it would restart the episode.
Leaving the context first calls the runner-only end_episode (when the server has it): the episode closes inside that
call, not in the 2 s the SDK gives a server to exit. `ended` holds its result.
"""

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import anyio
from mcp import Client, StdioServerParameters

META_PREFIX = "slicer_mcp/"
RUNNER_ONLY_META = META_PREFIX + "runner_only"
META_REASONING = META_PREFIX + "model_reasoning"
META_TOKENS = META_PREFIX + "tokens"
META_EPISODE = META_PREFIX + "episode_id"
META_IDENTITY = META_PREFIX + "identity"
META_CASE_INFO = META_PREFIX + "case_info"
END_TIMEOUT_S = 180.0
log = logging.getLogger(__name__)


class EnvironmentAbort(RuntimeError):
    """The environment failed (bridge internal / connection error, or a refused reset): abort the episode."""


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict


@dataclass
class Observation:
    tool: str
    text: str
    payload: dict
    frames: list = field(default_factory=list)
    images: list = field(default_factory=list)
    is_error: bool = False
    step: int | None = None
    local: bool = False
    identity: dict | None = None
    meta: dict = field(default_factory=dict)

    @property
    def error_type(self) -> str | None:
        e = self.payload.get("error")
        return e.get("type") if isinstance(e, dict) else None


def server_params(runtime=None, slicer_url=None, profile=None) -> StdioServerParameters:
    """`slicer-mcp serve --profile <chd_ct_v1> [--runtime R] [--slicer-url U]` over stdio, in this venv. The profile
    fixes the arm (3d_seg), the toolset (base), the report (classes_normal) and the state (full), so the prompt's
    registry entry is checked against it (prompt_registry.check_profile), not passed. runtime: a runtime.json; it is
    resolved to absolute paths first (task.config.resolved_runtime)."""
    from chd_client.slicer import command, with_defaults
    args = ["serve", *(["--profile", str(profile)] if profile else []),
            *(["--runtime", str(runtime)] if runtime else []), *(["--slicer-url", slicer_url] if slicer_url else [])]
    cmd = command(with_defaults(args, runtime))
    drop = {"VIRTUAL_ENV"}
    if runtime and not slicer_url:
        drop.add("SLICER_URL")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    from chd_client.task.config import REPO
    return StdioServerParameters(command=cmd[0], args=cmd[1:], env=env, cwd=str(REPO))


def to_observation(tool: str, res) -> Observation:
    text = res.content[0].text
    payload = json.loads(text)
    images = [base64.b64decode(c.data) for c in res.content[1:]]
    frames = list(payload.get("frames") or [])
    if len(images) != len(frames):
        raise EnvironmentAbort(f"{tool}: {len(images)} images for frames {frames}")
    meta = {k: v for k, v in (getattr(res, "meta", None) or {}).items() if str(k).startswith(META_PREFIX)}
    ident = meta.get(META_IDENTITY)
    return Observation(tool, text, payload, frames, images, bool(res.is_error), step=payload.get("step_index"),
                       identity=ident if isinstance(ident, dict) else None, meta=meta)


def loaded_volume_sha256(obs: Observation) -> str | None:
    """The sha256 of the voxels Slicer loaded, from a reset's result: its _meta slicer_mcp/case_info or slicer_mcp/identity
    (trace only), else a case_info in the payload; None when the result has neither."""
    for src in ((obs.meta.get(META_CASE_INFO) or {}), (obs.identity or {}),
                ((obs.payload.get("result") or {}).get("case_info") or {})):
        if isinstance(src, dict) and src.get("volume_sha256"):
            return src["volume_sha256"]
    return None


def local_error(tool: str, message: str, kind: str = "invalid_argument") -> Observation:
    payload = {"step_index": None, "state": None, "result": {}, "warnings": [],
               "error": {"type": kind, "message": message}, "frames": []}
    return Observation(tool, json.dumps(payload), payload, is_error=True, local=True)


class EnvMCP:
    def __init__(self, target):
        self.target = target
        self._client = None
        self._tools = []
        self.runner_only = set()
        self.flags = []
        self.ended = None

    async def __aenter__(self):
        self._client = Client(self.target)
        await self._client.__aenter__()
        self._tools = (await self._client.list_tools()).tools
        self.runner_only = {t.name for t in self._tools if (t.meta or {}).get(RUNNER_ONLY_META) is True}
        return self

    async def __aexit__(self, *exc):
        c = self._client
        await self.end_episode()
        self._client = None
        if c is not None:
            await c.__aexit__(*exc)

    async def end_episode(self):
        """Close the open episode now; idempotent. Leaving the context calls it too."""
        c = self._client
        if c is None or "end_episode" not in self.runner_only or self.ended is not None:
            return
        with anyio.CancelScope(shield=True), anyio.move_on_after(END_TIMEOUT_S):
            try:
                self.ended = json.loads((await c.call_tool("end_episode", {})).content[0].text)
            except Exception as e:  # noqa: BLE001
                log.warning("end_episode failed: %s", e)

    @property
    def instructions(self) -> str | None:
        return self._client.instructions

    def model_tools(self) -> list[ToolDef]:
        return [ToolDef(t.name, t.description or "", t.input_schema) for t in self._tools
                if t.name not in self.runner_only]

    async def reset(self, case_id: str, client_meta: dict) -> tuple[str, Observation]:
        self.ended = None
        res = await self._client.call_tool("reset_episode", {"case_id": case_id, "client_meta": client_meta},
                                           meta=None)
        obs = to_observation("reset_episode", res)
        if obs.is_error:
            raise EnvironmentAbort(f"reset_episode refused: {obs.payload.get('error')}")
        return (res.meta or {})[META_EPISODE], obs

    async def call(self, name: str, args, reasoning: str | None = None, tokens: dict | None = None) -> Observation:
        if name in self.runner_only:
            self.flags.append({"event": "blocked_runner_only_tool", "tool": name})
            return local_error(name, f"unknown tool {name!r}")
        meta = {}
        if reasoning:
            meta[META_REASONING] = reasoning
        if tokens is not None:
            meta[META_TOKENS] = tokens
        res = await self._client.call_tool(name, args if isinstance(args, dict) else {"_raw": args}, meta=meta)
        obs = to_observation(name, res)
        if obs.error_type == "environment":
            raise EnvironmentAbort(f"{name}: {obs.payload['error'].get('message')}")
        return obs
