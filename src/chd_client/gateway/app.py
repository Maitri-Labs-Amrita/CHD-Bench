"""The gateway: one process, one OpenAI-compatible API, one pool of workers per profile.

Routes: POST /v1/chat/completions, GET /v1/models, GET /health.
GET /v1/models lists each profile with a non-standard x_gateway.capabilities ({images, tools, max_images,
max_output_tokens}), so a client can keep its requests within the profile's limits before it sends one (models.local
reads max_images, and takes max_output_tokens as its max_tokens when the run sets none).
Request headers the gateway reads (all optional):
- X-Episode-Id: the conversation this request belongs to; requests of one episode go to the same worker.
- X-Request-Id: one id per model turn, the same on every retry of that turn (deduplication).
- X-Turn-Index: the turn number, for the request log.
Every reply carries x_gateway (profile, its sha256, worker, waits, interventions) and usage.cost_source.
A worker start that failed for good gives 503 with error code worker_start_failed and the header x-should-retry: false
(the openai SDK then does not retry; models.local.LocalModel stops the episode): see chd_client.gateway.scheduler.
The request log (<workdir>/requests.jsonl) holds timings and counts, never message text or images.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from chd_client.gateway import __version__, normalize
from chd_client.gateway.launchers import stop_orphan
from chd_client.gateway.scheduler import START_FAILED, Pool

IMAGE_PARTS = ("image_url", "input_image", "image")
MAX_HEADER = 256


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), "[gateway]", msg, flush=True)


class Gateway:
    def __init__(self, profiles: list, workdir: Path, *, token: str | None = None, keep_workers: bool = False,
                 tick_s: float = 10.0, poll_s: float = 2.0, orphan_timeout_s: float = 900.0,
                 max_body_mb: float = 512.0):
        names = [p.name for p in profiles]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"two profiles have the same name: {', '.join(sorted(dup))}")
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.workdir, 0o700)
        self.token = token
        self.keep_workers = keep_workers
        self.tick_s = tick_s
        self.max_body_bytes = int(max_body_mb * 1024 * 1024)
        self.state_path = self.workdir / "state.json"
        self.heartbeat = self.workdir / "gateway.alive"
        self._beats = 0
        self.request_log = self.workdir / "requests.jsonl"
        try:
            saved = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            saved = {}
        self.orphans = dict(saved.get("orphans") or {})
        shas = {p.name: p.sha256 for p in profiles}
        pools_saved = {}
        for name, sp in (saved.get("pools") or {}).items():
            sp = dict(sp or {})
            keep = {}
            for wid, d in (sp.get("workers") or {}).items():
                if name in shas and d.get("profile_sha256") == shas[name]:
                    keep[wid] = d
                else:
                    self.orphans[f"{name}/{wid}"] = d.get("handle") or {}
            sp["workers"] = keep
            pools_saved[name] = sp
        self._orphan_logged: dict = {}
        self.pools = {}
        for p in profiles:
            pool = Pool(p, self.workdir / "pools" / p.name, self.heartbeat, pools_saved.get(p.name),
                        poll_s=poll_s, orphan_timeout_s=orphan_timeout_s, log=log)
            pool.on_change = self.save
            self.pools[p.name] = pool
        self._ticker = None
        self._beater = None

    def touch(self):
        self._beats += 1
        with open(self.heartbeat, "w") as f:
            f.write(f"{self._beats} {time.time():.0f} {os.getpid()}\n")

    def save(self):
        data = {"version": 1, "saved_at": time.time(), "pools": {n: p.saved() for n, p in self.pools.items()},
                "orphans": self.orphans}
        tmp = self.state_path.with_name("state.json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)

    async def _stop_orphans(self):
        for k, handle in list(self.orphans.items()):
            if await asyncio.to_thread(stop_orphan, handle):
                del self.orphans[k]
                self._orphan_logged.pop(k, None)
                log(f"stopped worker {k} of a previous run: this run does not serve its profile version")
            elif time.time() - self._orphan_logged.get(k, 0) > 600:
                self._orphan_logged[k] = time.time()
                log(f"could not confirm that worker {k} of a previous run ended (handle {handle}); trying again")
        self.save()

    async def start(self):
        self.touch()
        if self.orphans:
            await self._stop_orphans()
        for pool in self.pools.values():
            await pool.start()
        self._beater = asyncio.create_task(self._beat_loop())
        self._ticker = asyncio.create_task(self._tick_loop())

    async def _beat_loop(self):
        """The heartbeat on its own: slow upkeep (scancel, squeue, stops) must never delay it, or healthy workers
        would take the gateway for dead."""
        while True:
            await asyncio.sleep(self.tick_s)
            try:
                self.touch()
            except OSError as e:
                log(f"heartbeat write failed: {e!r}")

    async def _tick_loop(self):
        while True:
            await asyncio.sleep(self.tick_s)
            if self.orphans:
                await self._stop_orphans()
            for pool in self.pools.values():
                try:
                    await pool.tick()
                except Exception as e:  # noqa: BLE001
                    log(f"{pool.name}: upkeep failed: {e!r}")

    async def stop(self):
        for t in (self._ticker, self._beater):
            if t is not None:
                t.cancel()
        for pool in self.pools.values():
            await pool.close(stop_workers=not self.keep_workers)
        self.save()

    def log_request(self, rec: dict):
        with open(self.request_log, "a") as f:
            f.write(json.dumps(rec) + "\n")


def count_images(body: dict) -> tuple:
    """(number of images, the first image URL that is not a data: URL or None)."""
    n, remote = 0, None
    for m in body.get("messages") or []:
        content = m.get("content") if isinstance(m, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in IMAGE_PARTS:
                n += 1
                url = part.get("image_url")
                url = url.get("url") if isinstance(url, dict) else url
                if isinstance(url, str) and not url.startswith("data:") and remote is None:
                    remote = url[:80]
    return n, remote


def check_request(profile, body: dict) -> str | None:
    caps = profile.capabilities
    if not all(isinstance(m, dict) and isinstance(m.get("role"), str) for m in body["messages"]):
        return "each message must be an object with a role"
    n_images, remote = count_images(body)
    if n_images and not caps.images:
        return f"{profile.name} does not take images (capabilities.images is false)"
    if caps.max_images and n_images > caps.max_images:
        return f"{profile.name} takes at most {caps.max_images} images per request; this one has {n_images}"
    if remote and not caps.remote_images:
        return f"only data: image URLs are accepted (capabilities.remote_images is false): {remote}"
    if body.get("tools") and not caps.tools:
        return f"{profile.name} does not take tools (capabilities.tools is false)"
    if body.get("stream"):
        return "streaming is not supported"
    if body.get("n") not in (None, 1):
        return "only n = 1 is supported"
    return None


def _header(req: Request, name: str) -> str | None:
    v = req.headers.get(name)
    if v is None:
        return None
    v = v.strip()
    return v if v and len(v) <= MAX_HEADER and v.isprintable() else None


def create_app(gw: Gateway) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        await gw.start()
        try:
            yield
        finally:
            await gw.stop()

    app = FastAPI(title="chd-gateway", version=__version__, lifespan=lifespan)

    def denied(req: Request):
        if gw.token and req.headers.get("authorization") != f"Bearer {gw.token}":
            return JSONResponse(normalize.error(401, "missing or wrong gateway token", "unauthorized"), 401)
        return None

    @app.get("/health")
    async def health(req: Request):
        if (r := denied(req)) is not None:
            return r
        return {"ok": True, "version": __version__, "models": {n: p.summary() for n, p in gw.pools.items()}}

    @app.get("/v1/models")
    async def models(req: Request):
        if (r := denied(req)) is not None:
            return r
        return {"object": "list", "data": [
            {"id": n, "object": "model", "created": 0, "owned_by": "chd-gateway",
             "x_gateway": {"capabilities": {"images": p.profile.capabilities.images,
                                            "tools": p.profile.capabilities.tools,
                                            "max_images": p.profile.capabilities.max_images,
                                            "max_output_tokens": p.profile.capabilities.max_output_tokens},
                           "request_timeout_s": p.profile.scheduler.request_timeout_s}}
            for n, p in gw.pools.items()]}

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        if (r := denied(req)) is not None:
            return r
        too_big = JSONResponse(normalize.error(413, f"the request is larger than {gw.max_body_bytes >> 20} MB "
                                                    "(--max-body-mb)", "request_too_large"), 413)
        length = req.headers.get("content-length")
        if length and length.isdigit() and int(length) > gw.max_body_bytes:
            return too_big
        chunks, size = [], 0
        async for chunk in req.stream():
            size += len(chunk)
            if size > gw.max_body_bytes:
                return too_big
            chunks.append(chunk)
        try:
            body = json.loads(b"".join(chunks))
        except ValueError:
            return JSONResponse(normalize.error(400, "the body is not JSON", "bad_json"), 400)
        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return JSONResponse(normalize.error(400, "the body needs a non-empty messages list", "bad_request"), 400)
        name = str(body.get("model") or "")
        name = name[len("local:"):] if name.startswith("local:") else name
        pool = gw.pools.get(name)
        if pool is None:
            return JSONResponse(normalize.error(404, f"no model {name!r} here; models: {', '.join(gw.pools)}",
                                                "model_not_found"), 404)
        problem = check_request(pool.profile, body)
        if problem:
            return JSONResponse(normalize.error(400, problem, "unsupported_request"), 400)
        key = _header(req, "x-episode-id")
        if key is None and pool.profile.scheduler.legacy_prompt_hash:
            first = json.dumps(body["messages"][:2], sort_keys=True).encode()
            key = "hash:" + hashlib.sha1(first).hexdigest()[:20]
        request_id = _header(req, "x-request-id")
        turn = _header(req, "x-turn-index")
        t0 = time.time()
        status, data = await pool.complete(body, key, request_id, turn)
        meta = (data.get("x_gateway") or {}) if isinstance(data, dict) else {}
        usage = data.get("usage") if isinstance(data, dict) else None
        gw.log_request({
            "t": round(t0, 3), "model": name, "status": status, "episode": key, "request_id": request_id,
            "turn": turn, "worker": meta.get("worker"), "affinity": meta.get("affinity"),
            "wait_s": meta.get("wait_s"), "inference_s": meta.get("inference_s"), "total_s": round(time.time() - t0, 2),
            "attempts": meta.get("attempts"), "replayed": meta.get("replayed", False),
            "n_messages": len(body["messages"]), "n_images": count_images(body)[0],
            "interventions": [r.get("kind") for r in meta.get("interventions") or []],
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "error": (data.get("error") or {}).get("code") if isinstance(data, dict) else None})
        failed_start = isinstance(data, dict) and (data.get("error") or {}).get("code") == START_FAILED
        return JSONResponse(data, status_code=status, headers={"x-should-retry": "false"} if failed_start else None)

    return app
