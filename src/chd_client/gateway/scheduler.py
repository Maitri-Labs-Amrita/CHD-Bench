"""A pool serves one profile: its workers, the episodes they hold, and the requests waiting for them.

Placement of one request:
- with an episode key (X-Episode-Id): the episode's own worker while its lease lives (sticky, so the engine's prefix
  cache stays warm); else the least-loaded worker with a free slot (ready ones first); else a new worker, if the
  profile allows one more (max_workers) and not too many are starting (max_starting); else wait for a slot.
  A lease lives while the episode has a request in flight on its worker, and for lease_s after its last request.
- without a key: the same, and nothing is kept after the request (stateless).
Stickiness is an optimization only. Every request carries the whole conversation, so a lost worker costs its cache,
never correctness.

Worker states: starting -> ready -> stopping -> gone. A worker leaves the pool only when its launcher confirms that
it ended; until then it stays "stopping" (in the state file too) and the stop is retried, so a failed scancel never
leaves a GPU job that nobody tracks.

Requests with an X-Request-Id are deduplicated: a retry that arrives while the first attempt runs gets that
attempt's reply; a retry after a successful reply gets the same reply again (DEDUP_TTL_S, at most MAX_RECENT ids);
the same id with a different body is refused (409). A retry after a FAILED attempt runs again, and after a timeout
the engine may still be generating the first attempt: the guarantee is one reply per turn, not one generation.

A worker start that fails for good (the worker ended before it was ready, or was not ready by startup_timeout_s) ends
the requests that wait for it at once, when no other worker is ready or starting: HTTP 503 with error code
worker_start_failed (START_FAILED), the worker's log path in the message, and the header x-should-retry: false. A
retry would only start the same broken worker again. After START_FAILURES_TO_FAIL failed starts in a row the pool is
"failing" for FAILING_COOLDOWN_S: it starts no worker, and a request that no worker can take gets that error at once.
A worker that gets ready resets the count.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from chd_client.gateway import __version__, interventions, normalize
from chd_client.gateway.launchers import LAUNCHERS, LaunchError

DEDUP_TTL_S = 900.0
MAX_RECENT = 2048
LAUNCH_BACKOFF_S = 30.0
STOP_RETRY_S = 30.0
START_FAILED = "worker_start_failed"
START_FAILURES_TO_FAIL = 2
FAILING_COOLDOWN_S = 300.0
ACTIVE = ("starting", "ready")


@dataclass
class Worker:
    id: str
    handle: dict
    profile_sha256: str
    state: str = "starting"
    url: str | None = None
    health_url: str | None = None
    capacity: int | None = None
    since: float = field(default_factory=time.time)
    ready_at: float | None = None
    last_active: float = field(default_factory=time.time)
    inflight: int = 0
    anon: int = 0
    requests: int = 0
    stop_at: float | None = None


@dataclass
class Lease:
    worker: str
    inflight: int = 0
    last: float = field(default_factory=time.time)


def body_sha256(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Pool:
    def __init__(self, profile, pool_dir: Path, heartbeat: Path, saved: dict | None = None, *, poll_s: float = 2.0,
                 orphan_timeout_s: float = 900.0, log=print):
        self.profile = profile
        self.name = profile.name
        self.pool_dir = pool_dir
        for d in (pool_dir, pool_dir / "workers", pool_dir / "logs"):
            d.mkdir(parents=True, exist_ok=True)
        self.frozen = pool_dir / "profile.yaml"
        self.frozen.write_bytes(profile.raw)
        self._saved = saved or {}
        self.worker_key = self._saved.get("worker_key") or secrets.token_urlsafe(32)
        self.launcher = LAUNCHERS[profile.launcher.kind](profile, pool_dir, self.frozen, self.worker_key, heartbeat,
                                                         orphan_timeout_s)
        self.poll_s = poll_s
        self.log = log
        self.workers: dict = {}
        self.leases: dict = {}
        self.lock = asyncio.Lock()
        self.tasks: set = set()
        self.last_launch_error: str | None = None
        self.next_launch_at = 0.0
        self.start_failures = 0
        self.start_error: str | None = None
        self.start_fail_seq = 0
        self.failing_until = 0.0
        self.pending: dict = {}
        self.recent: dict = {}
        self.client: httpx.AsyncClient | None = None
        self.on_change = None
        self.launching: set = set()
        self.started = False
        self.closing = False

    def changed(self):
        if self.on_change is not None:
            self.on_change()

    def _spawn(self, coro):
        t = asyncio.create_task(coro)
        self.tasks.add(t)
        t.add_done_callback(self._task_done)

    def _task_done(self, t):
        self.tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            self.log(f"{self.name}: background task failed: {t.exception()!r}")

    async def start(self):
        self.client = httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None,
                                                                          max_keepalive_connections=32))
        if not self.launcher.launches:
            for i, url in enumerate(self.profile.launcher.urls):
                w = Worker(f"attach-{i}", {"url": url}, self.profile.sha256, url=url, health_url=url + "/models")
                self.workers[w.id] = w
                self._spawn(self._watch_start(w))
            return
        for wid, d in (self._saved.get("workers") or {}).items():
            handle = d.get("handle") or {}
            w = Worker(wid, handle, d.get("profile_sha256", ""), since=d.get("since") or time.time(),
                       capacity=d.get("capacity"), requests=d.get("requests", 0))
            if d.get("state") == "stopping":
                self.log(f"{self.name}: stopping worker {wid} of the previous run (it was stopping)")
                self.launcher.adopt(handle)
                w.state = "stopping"
                self.workers[wid] = w
                await self._stop(w)
                continue
            if await asyncio.to_thread(self.launcher.status, handle) == "gone":
                continue
            self.launcher.adopt(handle)
            self.workers[wid] = w
            self._spawn(self._watch_start(w))
        for k, d in (self._saved.get("leases") or {}).items():
            w = self.workers.get(d.get("worker"))
            if w is not None and w.state in ACTIVE:
                self.leases[k] = Lease(w.id, 0, d.get("last") or time.time())
        kept = sum(w.state in ACTIVE for w in self.workers.values())
        if kept:
            self.log(f"{self.name}: kept {kept} worker(s) of the previous run")
        self.started = True
        self.changed()

    async def close(self, stop_workers: bool = True):
        self.closing = True
        for t in list(self.tasks):
            t.cancel()
        if self.launching:
            await asyncio.wait(list(self.launching), timeout=90)
            await asyncio.sleep(0.1)
        if stop_workers and self.launcher.launches:
            ws = [w for w in self.workers.values() if w.state != "gone"]
            if ws:
                self.log(f"{self.name}: stopping {len(ws)} worker(s)")
            for w in ws:
                w.state = "stopping"
            await asyncio.gather(*(self._stop(w) for w in ws))
            if self.workers:
                self.log(f"{self.name}: could not confirm that worker(s) {', '.join(self.workers)} ended; they stay "
                         "in the state file and the next gateway run stops them")
            self.leases.clear()
        if self.client is not None:
            await self.client.aclose()

    def saved(self) -> dict:
        if not self.started:
            return self._saved
        if not self.launcher.launches:
            return {}
        return {"worker_key": self.worker_key,
                "workers": {w.id: {"handle": w.handle, "profile_sha256": w.profile_sha256, "since": w.since,
                                   "capacity": w.capacity, "requests": w.requests, "state": w.state}
                            for w in self.workers.values() if w.state != "gone"},
                "leases": {k: {"worker": l.worker, "last": l.last} for k, l in self.leases.items()}}

    def summary(self) -> dict:
        return {"launcher": self.profile.launcher.kind, "engine": self.profile.engine.kind,
                "workers": [{"id": w.id, "state": w.state, "load": f"{self.load(w)}/{self.cap(w)}",
                             "requests": w.requests, "since": round(w.since),
                             "startup_s": round(w.ready_at - w.since, 1) if w.ready_at else None}
                            for w in self.workers.values()],
                "episodes": sum(1 for l in self.leases.values() if self._live(l, time.time())),
                "last_launch_error": self.last_launch_error, "start_failures": self.start_failures,
                "failing_until": round(self.failing_until) if self.failing() else None,
                "last_start_failure": self.start_error}

    def cap(self, w: Worker) -> int:
        fixed = self.profile.capacity_fixed()
        return fixed if fixed is not None else (w.capacity or 1)

    def _live(self, lease: Lease, t: float) -> bool:
        return lease.inflight > 0 or t - lease.last < self.profile.scheduler.lease_s

    def load(self, w: Worker, exclude: str | None = None) -> int:
        t = time.time()
        return w.anon + sum(1 for k, l in self.leases.items()
                            if l.worker == w.id and k != exclude and self._live(l, t))

    def _active(self) -> list:
        return [w for w in self.workers.values() if w.state in ACTIVE]

    def _auth(self) -> dict:
        if self.launcher.launches:
            return {"Authorization": f"Bearer {self.worker_key}"}
        env = self.profile.launcher.api_key_env
        key = os.environ.get(env) if env else None
        return {"Authorization": f"Bearer {key}"} if key else {}

    async def _healthy(self, url: str | None) -> bool:
        if not url:
            return False
        try:
            r = await self.client.get(url, headers=self._auth(), timeout=10)
            return 200 <= r.status_code < 300
        except httpx.HTTPError:
            return False

    async def _watch_start(self, w: Worker):
        while w.state == "starting":
            if self.launcher.launches:
                info = await asyncio.to_thread(self.launcher.ready, w.id)
                ok = bool(info) and info.get("profile_sha256") == self.profile.sha256 and \
                    await self._healthy(info.get("health_url"))
                if ok and w.state == "starting":
                    w.url, w.health_url = info["url"], info.get("health_url")
                    w.capacity = info.get("capacity") or w.capacity
            else:
                ok = await self._healthy(w.health_url)
            if ok and w.state == "starting":
                w.state, w.ready_at, w.last_active = "ready", time.time(), time.time()
                self.start_failures = 0
                self.log(f"{self.name}: worker {w.id} ready after {w.ready_at - w.since:.0f} s")
                self.changed()
                return
            await asyncio.sleep(self.poll_s)

    def _can_launch(self) -> bool:
        la = self.profile.launcher
        active = self._active()
        return (self.launcher.launches and not self.closing and len(active) < la.max_workers
                and sum(w.state == "starting" for w in active) < la.max_starting
                and time.time() >= self.next_launch_at and not self.failing() and self.launcher.can_start())

    def failing(self) -> bool:
        return time.time() < self.failing_until

    def _start_failed(self, w: Worker, why: str):
        """A worker start failed for good: say why and where its log is, wake the requests that wait for it, and after
        START_FAILURES_TO_FAIL in a row stop starting workers for FAILING_COOLDOWN_S."""
        log = self.launcher.log_path(w.handle) or self.pool_dir / "logs"
        last = _last_log_line(log)
        self.start_failures += 1
        self.start_error = f"worker {w.id} {why}; worker log: {log}" + (f" (it ends: {last})" if last else "")
        self.last_launch_error = self.start_error
        self.next_launch_at = time.time() + LAUNCH_BACKOFF_S
        if self.start_failures >= START_FAILURES_TO_FAIL:
            self.failing_until = time.time() + FAILING_COOLDOWN_S
            self.log(f"{self.name}: {self.start_failures} worker starts failed in a row; no new worker for "
                     f"{FAILING_COOLDOWN_S:.0f} s")
        self.start_fail_seq += 1
        self.log(f"{self.name}: {self.start_error}")

    def start_failure_error(self, meta: dict) -> tuple:
        """(503, error body) of a failed start: not transient, so a client must not retry it (x-should-retry)."""
        msg = f"{self.name}: a model worker could not start: {self.start_error}"
        if self.failing():
            msg += (f". {self.start_failures} starts failed in a row: the gateway starts no {self.name} worker until "
                    f"{time.strftime('%H:%M:%S', time.localtime(self.failing_until))}; fix the cause (see the log), "
                    "then restart the gateway or wait")
        meta.update(start_failures=self.start_failures)
        return 503, normalize.error(503, msg, START_FAILED, meta)

    def _register(self, wid: str, handle: dict) -> Worker:
        w = Worker(wid, handle, self.profile.sha256)
        self.workers[wid] = w
        where = {k: v for k, v in handle.items() if k in ("pid", "jobid", "gpus")}
        self.log(f"{self.name}: starting worker {wid} {where}")
        self._spawn(self._watch_start(w))
        self.changed()
        return w

    def _late_start(self, wid: str, task):
        """A start that finished after its request was cancelled: track the worker anyway (if nothing uses it, the
        idle stop ends it), so no started worker is ever untracked."""
        if task.cancelled() or task.exception() is not None:
            return
        if self.closing:
            self.launcher.stop(task.result())
            return
        try:
            self._register(wid, task.result())
        except RuntimeError:
            self.launcher.stop(task.result())

    async def _launch(self) -> Worker | None:
        wid = "w-" + secrets.token_hex(4)
        task = asyncio.ensure_future(asyncio.to_thread(self.launcher.start, wid))
        self.launching.add(task)
        task.add_done_callback(self.launching.discard)
        try:
            handle = await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(functools.partial(self._late_start, wid))
            raise
        except Exception as e:  # noqa: BLE001
            self.last_launch_error = str(e) if isinstance(e, LaunchError) else f"{type(e).__name__}: {e}"
            self.next_launch_at = time.time() + LAUNCH_BACKOFF_S
            self.log(f"{self.name}: cannot start a worker: {self.last_launch_error}")
            return None
        if self.closing:
            await asyncio.to_thread(self.launcher.stop, handle)
            return None
        return self._register(wid, handle)

    def _gone(self, w: Worker, why: str):
        """The worker is confirmed ended: forget it and its leases (once)."""
        if w.state == "gone":
            return
        w.state = "gone"
        self.workers.pop(w.id, None)
        for k in [k for k, l in self.leases.items() if l.worker == w.id]:
            del self.leases[k]
        if self.launcher.launches:
            self.launcher.forget(w.handle)
        self.log(f"{self.name}: worker {w.id} removed ({why})")
        self.changed()

    async def _stop(self, w: Worker):
        """Ask the launcher to stop the worker; it leaves the pool only once the launcher confirms it ended."""
        w.stop_at = time.time()
        try:
            await asyncio.to_thread(self.launcher.stop, w.handle)
            st = await asyncio.to_thread(self.launcher.status, w.handle)
        except Exception as e:  # noqa: BLE001
            self.log(f"{self.name}: stopping worker {w.id} failed: {e!r}")
            return
        if st == "gone":
            self._gone(w, "stopped")
        else:
            self.log(f"{self.name}: worker {w.id} is not confirmed stopped yet; trying again in {STOP_RETRY_S:.0f} s")

    async def _drop(self, w: Worker, why: str):
        """Take a worker out of service and stop it."""
        if w.state not in ACTIVE:
            return
        w.state = "stopping"
        for k in [k for k, l in self.leases.items() if l.worker == w.id]:
            del self.leases[k]
        self.log(f"{self.name}: stopping worker {w.id} ({why})")
        self.changed()
        if self.launcher.launches:
            await self._stop(w)
        else:
            self._gone(w, why)

    async def tick(self):
        """Periodic upkeep: workers that ended, stops to confirm, startup timeouts, idle workers, expired leases."""
        t = time.time()
        if self.launcher.launches:
            known = {w.id: w.handle for w in self.workers.values() if w.state != "gone"}
            statuses = await asyncio.to_thread(self.launcher.status_many, known) if known else {}
            for wid, st in statuses.items():
                w = self.workers.get(wid)
                if w is None or st != "gone":
                    continue
                if w.state == "starting":
                    self._start_failed(w, "ended before it was ready")
                self._gone(w, "stopped" if w.state == "stopping" else "its process or job ended")
            la = self.profile.launcher
            for w in list(self.workers.values()):
                if w.state == "stopping" and (w.stop_at is None or t - w.stop_at > STOP_RETRY_S):
                    await self._stop(w)
                elif w.state == "starting" and t - w.since > la.startup_timeout_s:
                    self._start_failed(w, f"was not ready after {la.startup_timeout_s:.0f} s (startup_timeout_s)")
                    await self._drop(w, "startup timeout")
                elif (w.state == "ready" and la.idle_stop_s and w.inflight == 0 and self.load(w) == 0
                      and t - w.last_active > la.idle_stop_s):
                    await self._drop(w, f"idle for {la.idle_stop_s:.0f} s")
        for k, l in list(self.leases.items()):
            w = self.workers.get(l.worker)
            if not self._live(l, t) or w is None or w.state not in ACTIVE:
                del self.leases[k]
        for rid, (_, done, _) in list(self.recent.items()):
            if t - done > DEDUP_TTL_S:
                del self.recent[rid]
        self.changed()

    async def _place(self, key: str | None):
        """Reserve a slot for one request: (worker, how) or (None, None)."""
        async with self.lock:
            lease = self.leases.get(key) if key else None
            if lease is not None:
                w = self.workers.get(lease.worker)
                if w is not None and w.state in ACTIVE and self.load(w, exclude=key) < self.cap(w):
                    lease.inflight += 1
                    w.inflight += 1
                    return w, "sticky"
            free = sorted((w for w in self._active() if self.load(w, exclude=key) < self.cap(w)),
                          key=lambda w: (w.state != "ready", self.load(w, exclude=key) / self.cap(w), -self.cap(w),
                                         w.since))
            w = free[0] if free else None
            if w is None and self._can_launch():
                w = await self._launch()
            if w is None:
                return None, None
            if key:
                self.leases[key] = Lease(w.id, inflight=1)
                how = "moved" if lease is not None else "new"
            else:
                w.anon += 1
                how = "stateless"
            w.inflight += 1
            self.changed()
            return w, how

    def _release(self, key: str | None, w: Worker):
        t = time.time()
        w.inflight = max(0, w.inflight - 1)
        w.last_active = t
        if key:
            lease = self.leases.get(key)
            if lease is not None and lease.worker == w.id:
                lease.inflight = max(0, lease.inflight - 1)
                lease.last = t
        else:
            w.anon = max(0, w.anon - 1)

    def _start_failed_since(self, seq: int | None) -> bool:
        """A worker start failed since `seq` and no worker is ready or starting: nothing can serve a waiting request."""
        return seq is not None and self.start_fail_seq != seq and not self._active()

    async def acquire(self, key: str | None, deadline: float, fail_seq: int | None = None):
        """A ready worker with a slot reserved for this request: (worker, how), or (None, None) after the deadline,
        or (with fail_seq, the pool's start_fail_seq when the request came) once a start failed and no worker is left.
        A cancelled caller never keeps its reservation."""
        while True:
            if self._start_failed_since(fail_seq):
                return None, None
            w, how = await self._place(key)
            if w is not None:
                try:
                    while w.state == "starting" and time.time() < deadline:
                        await asyncio.sleep(0.2)
                except BaseException:
                    self._release(key, w)
                    raise
                if w.state == "ready":
                    return w, how
                self._release(key, w)
                if w.state == "starting":
                    return None, None
                continue
            if time.time() >= deadline:
                return None, None
            await asyncio.sleep(0.2)

    async def _post(self, w: Worker, body: dict, timeout: float):
        r = await self.client.post(w.url + "/chat/completions", json=body, headers=self._auth(),
                                   timeout=httpx.Timeout(timeout, connect=10.0))
        try:
            return r.status_code, r.json()
        except ValueError:
            return r.status_code, None

    def _meta(self, key, request_id, turn) -> dict:
        p = self.profile
        return {"gateway_version": __version__, "profile": p.name, "profile_sha256": p.sha256,
                "revision": p.revision, "engine": p.engine.kind, "engine_model": p.engine.model,
                "launcher": p.launcher.kind, "engine_declared_interventions": list(p.engine.declared_interventions),
                "interventions": [], "episode_id": key, "request_id": request_id, "turn": turn}

    @staticmethod
    def _replay(result) -> tuple:
        status, data = result
        data = dict(data)
        if isinstance(data.get("x_gateway"), dict):
            data["x_gateway"] = {**data["x_gateway"], "replayed": True}
        return status, data

    async def complete(self, body: dict, key: str | None, request_id: str | None, turn=None):
        """(status, reply JSON) for one chat completion request."""
        if not request_id:
            return await self._complete(body, key, None, turn)
        h = body_sha256(body)
        while True:
            entry = self.recent.get(request_id) or self.pending.get(request_id)
            if entry is None:
                break
            if entry[0] != h:
                return 409, normalize.error(409, f"request id {request_id} was already used for another request",
                                            "request_id_reused")
            if request_id in self.recent:
                return self._replay(self.recent[request_id][2])
            result = await asyncio.shield(entry[1])
            if result is not None:
                return self._replay(result)
        fut = asyncio.get_running_loop().create_future()
        self.pending[request_id] = (h, fut)
        result = None
        try:
            result = await self._complete(body, key, request_id, turn)
            return result
        finally:
            self.pending.pop(request_id, None)
            if result is not None and result[0] == 200:
                self.recent[request_id] = (h, time.time(), result)
                while len(self.recent) > MAX_RECENT:
                    self.recent.pop(next(iter(self.recent)))
            if not fut.done():
                fut.set_result(result)

    async def _complete(self, body: dict, key: str | None, request_id: str | None, turn):
        sched = self.profile.scheduler
        t0 = time.time()
        deadline = t0 + sched.request_timeout_s
        meta = self._meta(key, request_id, turn)
        body, applied = with_request_defaults(body, self.profile.engine.request_defaults)
        body = {**body, "model": self.profile.served_name}
        if applied:
            meta["request_defaults"] = applied
        attempts = 0
        seq = self.start_fail_seq
        if self.failing() and not self._active():
            meta.update(wait_s=0.0, attempts=0)
            return self.start_failure_error(meta)
        while True:
            attempts += 1
            w, how = await self.acquire(key, deadline, seq)
            if w is None and self._start_failed_since(seq):
                meta.update(wait_s=round(time.time() - t0, 2), attempts=attempts)
                return self.start_failure_error(meta)
            if w is None:
                msg = f"no {self.name} worker was ready within {sched.request_timeout_s:.0f} s"
                if self.last_launch_error:
                    msg += f" (last start problem: {self.last_launch_error})"
                meta.update(wait_s=round(time.time() - t0, 2), attempts=attempts)
                return 503, normalize.error(503, msg, "no_worker", meta)
            meta.update(worker=w.id, affinity=how, wait_s=round(time.time() - t0, 2), attempts=attempts)
            t1 = time.time()

            async def send(b, w=w):
                return await self._post(w, b, max(1.0, deadline - time.time()))

            try:
                status, data = await send(body)
                if status == 200 and isinstance(data, dict):
                    meta["interventions"] = await interventions.apply(self.profile, body, data, send)
            except httpx.TimeoutException:
                meta.update(inference_s=round(time.time() - t1, 2), total_s=round(time.time() - t0, 2))
                return 504, normalize.error(504, f"{self.name} worker {w.id} did not answer before the request "
                                                 f"timeout ({sched.request_timeout_s:.0f} s)", "inference_timeout", meta)
            except httpx.TransportError as e:
                st = (await asyncio.to_thread(self.launcher.status, w.handle)) if self.launcher.launches else None
                if st == "gone" and w.state != "gone":
                    self._gone(w, "a request to it failed and its process or job ended")
                elif not self.launcher.launches and w.state == "ready":
                    w.state = "starting"
                    self.log(f"{self.name}: attached server {w.url} does not answer; waiting for it")
                    self._spawn(self._watch_start(w))
                if attempts < 3 and time.time() < deadline:
                    self.log(f"{self.name}: request to worker {w.id} failed ({type(e).__name__}); trying again")
                    if st != "gone":
                        await asyncio.sleep(2)
                    continue
                meta.update(total_s=round(time.time() - t0, 2))
                return 502, normalize.error(502, f"{self.name} worker {w.id} could not be reached "
                                                 f"({type(e).__name__})", "worker_unreachable", meta)
            finally:
                self._release(key, w)
                w.requests += 1
            meta.update(inference_s=round(time.time() - t1, 2), total_s=round(time.time() - t0, 2))
            if status != 200 or not isinstance(data, dict):
                return (status if status != 200 else 502), normalize.error(
                    status if status != 200 else 502, f"{self.name} worker {w.id}: {_upstream_message(data, status)}",
                    "worker_error", meta)
            return 200, normalize.finalize(data, self.profile, meta)


def with_request_defaults(body: dict, defaults: dict) -> tuple:
    """(the request with the profile's engine.request_defaults added, {field: value} of what was added). The request's
    own keys win; for a mapping in both (e.g. chat_template_kwargs) its keys are merged and the request's win."""
    out, applied = dict(body), {}
    for k, v in (defaults or {}).items():
        if k not in out:
            out[k] = applied[k] = v
        elif isinstance(v, dict) and isinstance(out[k], dict):
            extra = {kk: vv for kk, vv in v.items() if kk not in out[k]}
            if extra:
                out[k] = {**extra, **out[k]}
                applied[k] = extra
    return out, applied


def _last_log_line(path) -> str | None:
    """The last line of a worker log that the worker wrapper did not write (often the engine's error), at most 300
    characters; None when there is none or the log cannot be read."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line and "[worker]" not in line:
            return line[:300]
    return None


def _upstream_message(data, status) -> str:
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:2000]
        if isinstance(err, str):
            return err[:2000]
        if data.get("message"):
            return str(data["message"])[:2000]
        if data.get("detail"):
            return str(data["detail"])[:2000]
    return f"HTTP {status}" if status != 200 else "the reply was not a JSON object"
