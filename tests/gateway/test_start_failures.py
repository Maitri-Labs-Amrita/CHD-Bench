"""A worker that can never start: the waiting request fails at once with a non-transient error that names the worker
log, and after START_FAILURES_TO_FAIL failed starts in a row the pool starts no worker for a cooldown. The fake engine's
--mode no_start exits before it is ready."""

import asyncio
import time

from chd_client.gateway import scheduler
from chd_client.gateway.scheduler import START_FAILED
from tests.gateway.helpers import Harness, chat, fake_profile


def no_start(tmp_path, *args, **kw):
    kw.setdefault("scheduler", {"request_timeout_s": 120})
    return fake_profile(tmp_path, mode="no_start", args=list(args), **kw)


def launches(pool) -> list:
    started = []
    real = pool.launcher.start

    def counting(wid):
        started.append(wid)
        return real(wid)

    pool.launcher.start = counting
    return started


async def test_a_worker_that_exits_before_ready_fails_the_request_at_once(tmp_path, workdir):
    async with Harness([no_start(tmp_path)], workdir) as h:
        t0 = time.time()
        r = await h.post(chat(), episode="e1")
        took = time.time() - t0
        pool = h.gw.pools["fake"]
    assert r.status_code == 503 and took < 15, (r.status_code, took)
    err = r.json()["error"]
    assert err["code"] == START_FAILED and err["type"] == "server_error"
    assert r.headers["x-should-retry"] == "false"
    logs = list((pool.pool_dir / "logs").glob("w-*.log"))
    assert len(logs) == 1 and f"worker log: {logs[0]}" in err["message"], err["message"]
    assert "ended before it was ready" in err["message"]
    assert "fake_missing_package" in err["message"]
    rec = (workdir / "requests.jsonl").read_text()
    assert START_FAILED in rec


async def test_two_failed_starts_make_the_pool_fail_fast_without_new_workers(tmp_path, workdir, monkeypatch):
    monkeypatch.setattr(scheduler, "LAUNCH_BACKOFF_S", 0.0)
    async with Harness([no_start(tmp_path)], workdir) as h:
        pool = h.gw.pools["fake"]
        started = launches(pool)
        r1 = await h.post(chat(), episode="e1")
        assert r1.status_code == 503 and pool.start_failures == 1 and not pool.failing()
        r2 = await h.post(chat(), episode="e1")
        assert r2.status_code == 503 and pool.start_failures == 2 and pool.failing()
        assert "starts failed in a row" in r2.json()["error"]["message"]
        assert len(started) == 2
        t0 = time.time()
        r3 = await h.post(chat(), episode="e2")
        assert time.time() - t0 < 1.0 and len(started) == 2
        assert r3.status_code == 503 and r3.json()["error"]["code"] == START_FAILED
        assert r3.headers["x-should-retry"] == "false" and r3.json()["x_gateway"]["attempts"] == 0
        health = (await h.client.get("/health")).json()["models"]["fake"]
        assert health["start_failures"] == 2 and health["failing_until"] and "worker log" in health["last_start_failure"]


async def test_after_the_cooldown_a_fixed_worker_starts_and_resets_the_count(tmp_path, workdir, monkeypatch):
    monkeypatch.setattr(scheduler, "LAUNCH_BACKOFF_S", 0.0)
    monkeypatch.setattr(scheduler, "FAILING_COOLDOWN_S", 1.0)
    marker = tmp_path / "fixed"
    async with Harness([no_start(tmp_path, "--start-marker", str(marker))], workdir) as h:
        pool = h.gw.pools["fake"]
        for _ in range(2):
            assert (await h.post(chat(), episode="e1")).status_code == 503
        assert pool.failing()
        marker.write_text("the package is installed now")
        await asyncio.sleep(1.1)
        r = await h.post(chat(), episode="e1")
        assert r.status_code == 200, r.text
        assert pool.start_failures == 0 and not pool.failing()


async def test_a_startup_timeout_is_a_failed_start(tmp_path, workdir):
    p = fake_profile(tmp_path, args=["--startup-delay", "60"], launcher={"startup_timeout_s": 1},
                     scheduler={"request_timeout_s": 120})
    async with Harness([p], workdir) as h:
        t0 = time.time()
        r = await h.post(chat(), episode="e1")
    assert r.status_code == 503 and time.time() - t0 < 15
    assert r.json()["error"]["code"] == START_FAILED and "was not ready after 1 s" in r.json()["error"]["message"]


async def test_a_ready_worker_keeps_serving_while_another_start_fails(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path, launcher={"max_workers": 2})], workdir) as h:
        pool = h.gw.pools["fake"]
        assert (await h.post(chat(), episode="e1")).status_code == 200
        pool.start_failures, pool.failing_until = 5, time.time() + 60
        r = await h.post(chat(), episode="e1")
        assert r.status_code == 200, r.text


def test_the_client_and_the_gateway_agree_on_the_code():
    from chd_client.models import local
    assert local.START_FAILED == START_FAILED
