"""Gateway robustness: worker crashes, cancelled waiters and launches, lease isolation, Slurm stops, profile checks,
request limits, the worker's heartbeat and engine children, and the engine's build environment."""

import asyncio
import json
import os
import subprocess
import sys
import time

import pytest

from tests.gateway.helpers import Harness, chat, fake_profile
from tests.gateway.test_gateway import pid_alive, wait_until
from chd_client.gateway.engines.hf_server import BadRequest, check_supported
from chd_client.gateway.launchers import slurm
from chd_client.gateway.profile import ProfileError, load_profile, parse_profile
from chd_client.gateway.scheduler import Lease, Pool, Worker


def quiet(*_):
    pass


async def test_a_crashed_local_worker_frees_its_gpu(tmp_path, workdir):
    marker = tmp_path / "crashed"
    p = fake_profile(tmp_path, launcher={"gpus": ["0"], "max_workers": 1})
    async with Harness([p], workdir) as h:
        w1 = (await h.post(chat(), episode="e1")).json()["x_gateway"]["worker"]
        r = await h.post(chat(fake_mode="crash", fake_marker=str(marker)), episode="e1")
        assert r.status_code == 200, r.text
        assert r.json()["x_gateway"]["worker"] != w1
        assert h.gw.pools["fake"].launcher.free_gpus == []


async def test_a_cancelled_waiter_releases_its_slot(tmp_path, workdir):
    p = fake_profile(tmp_path, args=["--startup-delay", "3"])
    async with Harness([p], workdir) as h:
        pool = h.gw.pools["fake"]
        t = asyncio.create_task(pool.acquire("e1", time.time() + 30))
        assert await wait_until(lambda: pool.workers and next(iter(pool.workers.values())).inflight == 1)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        w = next(iter(pool.workers.values()))
        assert w.inflight == 0 and w.anon == 0 and pool.leases["e1"].inflight == 0


async def test_a_cancelled_launch_still_tracks_its_worker(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        pool = h.gw.pools["fake"]
        start = pool.launcher.start

        def slow_start(wid):
            time.sleep(1.0)
            return start(wid)

        pool.launcher.start = slow_start
        t = asyncio.create_task(pool.acquire("e1", time.time() + 30))
        await asyncio.sleep(0.3)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert await wait_until(lambda: len(pool.workers) == 1, timeout=10)
        pid = next(iter(pool.workers.values())).handle["pid"]
    assert await wait_until(lambda: not pid_alive(pid), timeout=10)


async def test_old_requests_do_not_corrupt_a_new_lease(tmp_path):
    p = load_profile(fake_profile(tmp_path, launcher={"max_workers": 2}, scheduler={"episodes_per_worker": 2}))
    pool = Pool(p, tmp_path / "pool", tmp_path / "hb", log=quiet)
    a = Worker("a", {}, p.sha256, state="ready", since=1.0)
    b = Worker("b", {}, p.sha256, state="ready", since=2.0)
    pool.workers = {"a": a, "b": b}
    w, how = await pool._place("ep")
    assert w is a and how == "new"
    a.state = "stopping"
    w2, how2 = await pool._place("ep")
    assert w2 is b and how2 == "moved" and pool.leases["ep"] == Lease("b", 1, pool.leases["ep"].last)
    pool._release("ep", a)
    assert pool.leases["ep"].inflight == 1 and a.inflight == 0
    pool._release("ep", b)
    assert pool.leases["ep"].inflight == 0 and b.inflight == 0


SLURM = """name: s
engine: {kind: custom, command: [x]}
launcher: {kind: slurm, sbatch: {gres: "gpu:1"}, idle_stop_s: 0}
"""


async def test_an_unconfirmed_slurm_stop_keeps_the_worker(tmp_path, monkeypatch):
    queue = {"squeue": "123 RUNNING\n", "calls": []}

    def fake_run(cmd, timeout=60):
        queue["calls"].append(cmd[0])
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 0, "123\n", "")
        if cmd[0] == "squeue":
            return subprocess.CompletedProcess(cmd, 0, queue["squeue"], "")
        return subprocess.CompletedProcess(cmd, 1, "", "scancel: error: Socket timed out")

    monkeypatch.setattr(slurm, "run", fake_run)
    pool = Pool(parse_profile(SLURM.encode()), tmp_path / "pool", tmp_path / "hb", poll_s=0.05, log=quiet)
    await pool.start()
    w, _ = await pool._place(None)
    pool._release(None, w)
    await pool._drop(w, "test")
    assert w.state == "stopping" and pool.saved()["workers"][w.id]["state"] == "stopping"
    assert pool.saved()["workers"][w.id]["handle"]["jobid"] == "123"
    w.stop_at = 0
    await pool.tick()
    assert queue["calls"].count("scancel") == 2 and w.id in pool.workers
    queue["squeue"] = ""
    await pool.tick()
    assert w.id not in pool.workers and pool.saved()["workers"] == {}
    await pool.close(stop_workers=False)


async def test_workers_of_a_removed_profile_are_stopped(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir, keep_workers=True) as h:
        w = (await h.post(chat(), episode="e1")).json()["x_gateway"]["worker"]
        pid = h.gw.pools["fake"].workers[w].handle["pid"]
    assert pid_alive(pid)
    async with Harness([fake_profile(tmp_path, name="other")], workdir):
        assert await wait_until(lambda: not pid_alive(pid), timeout=10)
    state = json.loads((workdir / "state.json").read_text())
    assert state["orphans"] == {} and "fake" not in state["pools"]


@pytest.mark.parametrize("section,key,value", [("interventions", "forced_tool_call", '"false"'),
                                                ("capabilities", "remote_images", '"false"'),
                                                ("engine", "supports_prefill", '"yes"'),
                                                ("capabilities", "max_images", '"8"'),
                                                ("capabilities", "max_output_tokens", '"81920"')])
def test_quoted_values_are_rejected(section, key, value):
    base = {"engine": "kind: custom, command: [x], supports_prefill: true", "launcher": "kind: local"}
    extra = f"{key}: {value}"
    base[section] = f"{base[section]}, {extra}" if section in base else extra
    text = "name: m\n" + "".join(f"{k}: {{{v}}}\n" for k, v in base.items())
    with pytest.raises(ProfileError):
        parse_profile(text.encode())


@pytest.mark.parametrize("body", [{"tool_choice": "required"},
                                  {"tool_choice": {"type": "function", "function": {"name": "f"}}},
                                  {"response_format": {"type": "json_object"}}, {"n": 2}, {"logprobs": True},
                                  {"parallel_tool_calls": False}])
def test_hf_server_refuses_what_it_cannot_honour(body):
    with pytest.raises(BadRequest):
        check_supported(body)


@pytest.mark.parametrize("body", [{}, {"tool_choice": "auto"}, {"tool_choice": "none"},
                                  {"response_format": {"type": "text"}}, {"parallel_tool_calls": True}])
def test_hf_server_accepts_what_it_honours(body):
    check_supported(body)


async def test_requests_over_the_size_limit_are_refused(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir, max_body_mb=0.001) as h:
        r = await h.post(chat("x" * 4000))
        assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


def test_the_worker_stops_the_engines_children(tmp_path):
    child_file = tmp_path / "child.pid"
    p = fake_profile(tmp_path, args=["--child-pid-file", str(child_file)])
    hb = tmp_path / "gateway.alive"
    hb.touch()
    proc = subprocess.Popen([sys.executable, "-m", "chd_client.gateway.worker", "--profile", str(p), "--worker-id", "w-c",
                             "--workdir", str(tmp_path / "pool"), "--heartbeat", str(hb)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t = time.time()
        while not (tmp_path / "pool" / "workers" / "w-c.json").exists() and time.time() - t < 30:
            time.sleep(0.1)
        child = int(child_file.read_text())
        assert pid_alive(child)
        proc.terminate()
        proc.wait(timeout=60)
        t = time.time()
        while pid_alive(child) and time.time() - t < 10:
            time.sleep(0.1)
        assert not pid_alive(child)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            os.kill(int(child_file.read_text()), 9)
        except (OSError, ValueError):
            pass


def test_the_worker_follows_heartbeat_content_not_mtime(tmp_path):
    p = fake_profile(tmp_path)
    hb = tmp_path / "gateway.alive"
    hb.write_text("1")
    proc = subprocess.Popen([sys.executable, "-m", "chd_client.gateway.worker", "--profile", str(p), "--worker-id", "w-h",
                             "--workdir", str(tmp_path / "pool"), "--heartbeat", str(hb), "--orphan-timeout-s", "3"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    old = time.time() - 3600
    try:
        for i in range(8):
            hb.write_text(str(i + 2))
            os.utime(hb, (old, old))
            time.sleep(1)
            assert proc.poll() is None, proc.stdout.read()
        proc.wait(timeout=30)
        assert "has not changed" in proc.stdout.read()
    finally:
        if proc.poll() is None:
            proc.kill()


async def test_the_gateway_heartbeat_changes_content(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        first = h.gw.heartbeat.read_text()
        assert await wait_until(lambda: h.gw.heartbeat.read_text() != first, timeout=5)


def test_python_headers_from_the_engine_environment_reach_c_and_cpp(tmp_path):
    """A node without the Python dev package (no /usr/include/python3.X/Python.h): Triton's C helper and PyTorch
    Inductor's C++ kernels both find Python.h in <engine environment>/include/python3.X."""
    from chd_client.gateway.engines import vllm_env
    inc = tmp_path / "env" / "include" / "python3.99"
    inc.mkdir(parents=True)
    (inc / "Python.h").write_text("")
    (tmp_path / "env" / "bin").mkdir()
    env = vllm_env(str(tmp_path / "env" / "bin" / "python"), "w-1")
    assert env["C_INCLUDE_PATH"].split(":")[0] == str(inc) and env["CPLUS_INCLUDE_PATH"].split(":")[0] == str(inc)
