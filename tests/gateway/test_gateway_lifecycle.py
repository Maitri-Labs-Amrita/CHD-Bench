"""Gateway worker lifecycle: GPU release, restarts, engines that outlive their wrapper, shutdown, saved state, version
changes, lost sbatch replies and the heartbeat under slow upkeep."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import time

import pytest

from tests.gateway.helpers import Harness, chat, fake_profile
from tests.gateway.test_gateway import wait_until
from chd_client.gateway.launchers import local, slurm, stop_orphan
from chd_client.gateway.launchers.local import LocalLauncher, handle_status
from chd_client.gateway.launchers.slurm import SlurmLauncher
from chd_client.gateway.profile import load_profile, parse_profile
from chd_client.gateway.scheduler import Pool


def quiet(*_):
    pass


def local_pool(tmp_path, saved=None, **launcher):
    p = load_profile(fake_profile(tmp_path, launcher={"gpus": ["0", "1"], "max_workers": 2, **launcher}))
    return Pool(p, tmp_path / "pool", tmp_path / "hb", saved, poll_s=0.05, log=quiet)


def test_gpu_release_is_idempotent(tmp_path):
    la = local_pool(tmp_path).launcher
    la.owned = {"a": ["0"]}
    la.free_gpus = ["1"]
    la.forget({"worker_id": "a", "gpus": ["0"], "pid": -1})
    assert sorted(la.free_gpus) == ["0", "1"]
    la.free_gpus.remove("0")
    la.owned["b"] = ["0"]
    la.forget({"worker_id": "a", "gpus": ["0"], "pid": -1})
    assert la.free_gpus == ["1"] and la.owned == {"b": ["0"]}


async def test_a_stopping_worker_keeps_its_gpus_across_a_restart(tmp_path):
    p = load_profile(fake_profile(tmp_path, launcher={"gpus": ["0"], "max_workers": 1}))
    saved = {"workers": {"w-old": {"handle": {"pid": 999999, "gpus": ["0"], "host": socket.gethostname(),
                                              "worker_id": "w-old"},
                                   "profile_sha256": p.sha256, "state": "stopping"}}}
    pool = Pool(p, tmp_path / "pool", tmp_path / "hb", saved, poll_s=0.05, log=quiet)
    alive = {"v": "alive"}
    pool.launcher.status = lambda h: alive["v"]
    pool.launcher.stop = lambda h: None
    await pool.start()
    assert pool.workers["w-old"].state == "stopping" and pool.launcher.free_gpus == []
    assert not pool._can_launch()
    alive["v"] = "gone"
    await pool.tick()
    assert "w-old" not in pool.workers and pool.launcher.free_gpus == ["0"]
    await pool.close(stop_workers=False)


def test_an_engine_outliving_its_wrapper_is_stopped(tmp_path):
    child_file = tmp_path / "child.pid"
    p = load_profile(fake_profile(tmp_path, args=["--child-pid-file", str(child_file)]))
    pool = Pool(p, tmp_path / "pool2", tmp_path / "hb", poll_s=0.05, log=quiet)
    (tmp_path / "hb").write_text("1")
    la = pool.launcher
    h = la.start("w-kill")
    try:
        t = time.time()
        while not (tmp_path / "pool2" / "workers" / "w-kill.json").exists() and time.time() - t < 30:
            time.sleep(0.1)
        os.kill(h["pid"], signal.SIGKILL)
        la.procs[h["pid"]].wait()
        assert la.status(h) == "alive"
        la.stop(h)
        assert la.status(h) == "gone"
        assert local.members("w-kill") == []
    finally:
        for pid in local.members("w-kill") or []:
            os.kill(pid, signal.SIGKILL)


def test_another_hosts_worker_is_unknown_not_gone():
    h = {"pid": 1, "host": "some-other-host", "worker_id": "w-x"}
    assert handle_status(h) is None and local.kill(h) is False and stop_orphan(h) is False


async def test_a_launch_during_shutdown_is_stopped(tmp_path):
    pool = local_pool(tmp_path)
    (tmp_path / "hb").write_text("1")
    await pool.start()
    started = []
    real_start = pool.launcher.start

    def slow_start(wid):
        time.sleep(1.0)
        h = real_start(wid)
        started.append(h)
        return h

    pool.launcher.start = slow_start
    t = asyncio.create_task(pool.acquire(None, time.time() + 4))
    await asyncio.sleep(0.3)
    await pool.close(stop_workers=True)
    assert await t == (None, None)
    assert started and all(handle_status(h) == "gone" for h in started) and pool.workers == {}


def test_saved_state_survives_until_the_pool_starts(tmp_path):
    saved = {"worker_key": "k", "workers": {"w-1": {"handle": {"jobid": "5"}, "profile_sha256": "x"}}}
    assert local_pool(tmp_path, saved).saved() is saved


async def test_other_version_workers_are_stopped_by_handle_type(tmp_path, workdir, monkeypatch):
    calls = []

    def fake_run(cmd, timeout=60):
        calls.append(cmd[0])
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(slurm, "run", fake_run)
    workdir.mkdir(parents=True)
    (workdir / "state.json").write_text(json.dumps({"pools": {"fake": {"workers": {
        "w-slurm": {"handle": {"jobid": "999", "worker_id": "w-slurm"}, "profile_sha256": "an-older-version"}}}}}))
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        assert "w-slurm" not in h.gw.pools["fake"].workers
        assert h.gw.orphans == {} and calls[:2] == ["scancel", "squeue"]


def test_a_lost_sbatch_reply_is_reconciled(tmp_path, monkeypatch):
    def fake_run(cmd, timeout=60):
        if cmd[0] == "sbatch":
            return None
        if cmd[0] == "squeue" and "-n" in cmd:
            assert cmd[cmd.index("-n") + 1] == "chdgw-s-w-lost"
            return subprocess.CompletedProcess(cmd, 0, "777\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(slurm, "run", fake_run)
    p = parse_profile(b'name: s\nengine: {kind: custom, command: [x]}\nlauncher: {kind: slurm, sbatch: {gres: "gpu:1"}}\n')
    la = SlurmLauncher(p, tmp_path, tmp_path / "p.yaml", "k", tmp_path / "hb")
    assert la.start("w-lost")["jobid"] == "777"


async def test_the_heartbeat_does_not_wait_for_slow_upkeep(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        async def slow_tick():
            await asyncio.sleep(5)

        h.gw.pools["fake"].tick = slow_tick
        await asyncio.sleep(0.5)
        first = h.gw.heartbeat.read_text()
        assert await wait_until(lambda: h.gw.heartbeat.read_text() != first, timeout=2)
