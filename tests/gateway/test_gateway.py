"""End-to-end: the gateway in this process, real worker subprocesses (local launcher), the fake engine."""

import asyncio
import os
import socket
import subprocess
import sys
import time

import pytest

from tests.gateway.helpers import FAKE, Harness, chat, fake_profile, write_profile
from chd_client.gateway.app import Gateway


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


async def wait_until(cond, timeout=10.0):
    t = time.time()
    while time.time() - t < timeout:
        if cond():
            return True
        await asyncio.sleep(0.1)
    return False


async def test_first_request_starts_a_worker_and_labels_the_reply(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        r = await h.post(chat(), episode="e1", request_id="r1", headers={"X-Turn-Index": "1"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["model"] == "fake"
        assert d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "look"
        assert d["usage"]["cost"] == 0.0 and d["usage"]["cost_source"] == "zero_api_charge"
        x = d["x_gateway"]
        assert x["affinity"] == "new" and x["episode_id"] == "e1" and x["request_id"] == "r1" and x["turn"] == "1"
        assert x["interventions"] == [] and len(x["profile_sha256"]) == 64 and x["worker"].startswith("w-")
        health = (await h.client.get("/health")).json()["models"]["fake"]
        assert [w["state"] for w in health["workers"]] == ["ready"]
        log = (workdir / "requests.jsonl").read_text().splitlines()
        assert len(log) == 1 and '"status": 200' in log[0] and "hello" not in log[0]


async def test_episodes_stick_to_their_worker(tmp_path, workdir):
    p = fake_profile(tmp_path, launcher={"max_workers": 3}, scheduler={"episodes_per_worker": 1})
    async with Harness([p], workdir) as h:
        a1 = (await h.post(chat(), episode="e1")).json()["x_gateway"]
        a2 = (await h.post(chat("again"), episode="e1")).json()["x_gateway"]
        b1 = (await h.post(chat(), episode="e2")).json()["x_gateway"]
        assert a2["worker"] == a1["worker"] and a2["affinity"] == "sticky"
        assert b1["worker"] != a1["worker"] and b1["affinity"] == "new"
        s = (await h.post(chat(), episode=None)).json()["x_gateway"]
        assert s["affinity"] == "stateless" and s["episode_id"] is None


async def test_a_full_pool_waits_then_times_out(tmp_path, workdir):
    p = fake_profile(tmp_path, scheduler={"episodes_per_worker": 1, "request_timeout_s": 2, "lease_s": 600})
    async with Harness([p], workdir) as h:
        assert (await h.post(chat(), episode="e1")).status_code == 200
        r = await h.post(chat(), episode="e2")
        assert r.status_code == 503 and r.json()["error"]["code"] == "no_worker"


async def test_request_id_dedup(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        first = (await h.post(chat(), episode="e1", request_id="r1")).json()
        again = (await h.post(chat(), episode="e1", request_id="r1")).json()
        assert again["id"] == first["id"] and again["x_gateway"]["replayed"] is True
        other = await h.post(chat("different"), episode="e1", request_id="r1")
        assert other.status_code == 409
        slow = chat(fake_mode="slow", fake_delay=1)
        r1, r2 = await asyncio.gather(h.post(slow, episode="e1", request_id="r2"),
                                      h.post(slow, episode="e1", request_id="r2"))
        assert r1.json()["id"] == r2.json()["id"]
        assert sorted([r1.json()["x_gateway"].get("replayed", False), r2.json()["x_gateway"].get("replayed", False)]) \
            == [False, True]


async def test_a_crashed_worker_is_replaced(tmp_path, workdir):
    marker = tmp_path / "crashed"
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        w1 = (await h.post(chat(), episode="e1")).json()["x_gateway"]["worker"]
        r = await h.post(chat(fake_mode="crash", fake_marker=str(marker)), episode="e1")
        assert r.status_code == 200, r.text
        x = r.json()["x_gateway"]
        assert marker.exists() and x["worker"] != w1 and x["attempts"] >= 2


async def test_interventions_are_off_by_default(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path, mode="garbled")], workdir) as h:
        d = (await h.post(chat(), episode="e1")).json()
        msg = d["choices"][0]["message"]
        assert not msg.get("tool_calls") and "<tool_call>" in msg["content"]
        assert d["x_gateway"]["interventions"] == []


async def test_tool_call_repair(tmp_path, workdir):
    p = fake_profile(tmp_path, mode="garbled", interventions={"tool_call_repair": True})
    async with Harness([p], workdir) as h:
        d = (await h.post(chat(), episode="e1")).json()
        msg = d["choices"][0]["message"]
        assert msg["tool_calls"][0]["function"]["name"] == "look" and d["choices"][0]["finish_reason"] == "tool_calls"
        rec = d["x_gateway"]["interventions"][0]
        assert rec["kind"] == "tool_call_repair" and "<tool_call>" in rec["original_content"]


async def test_forced_tool_call(tmp_path, workdir):
    p = fake_profile(tmp_path, mode="text", engine={"supports_prefill": True},
                     interventions={"forced_tool_call": True})
    async with Harness([p], workdir) as h:
        d = (await h.post(chat(), episode="e1")).json()
        msg = d["choices"][0]["message"]
        assert msg["content"] == "I will look first."
        assert msg["tool_calls"][0]["function"]["name"] == "look"
        rec = d["x_gateway"]["interventions"][0]
        assert rec["kind"] == "forced_tool_call" and rec["result"] == "call"
        assert d["usage"]["completion_tokens"] == 10
        d = (await h.post(chat(tool_choice="none"), episode="e1")).json()
        assert d["x_gateway"]["interventions"] == [] and not d["choices"][0]["message"].get("tool_calls")


async def test_requests_the_profile_cannot_serve_are_refused(tmp_path, workdir):
    p = fake_profile(tmp_path, capabilities={"images": False})
    async with Harness([p], workdir) as h:
        img = chat()
        img["messages"][1]["content"] = [{"type": "text", "text": "x"},
                                         {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
        assert (await h.post(img)).status_code == 400
        assert (await h.post({**chat(), "model": "nope"})).status_code == 404
        assert (await h.post(chat(stream=True))).status_code == 400
        assert (await h.post({"model": "fake", "messages": []})).status_code == 400
        assert (await h.client.post("/v1/chat/completions", content=b"{bad")).status_code == 400
        assert (await h.post({**chat(), "model": "local:fake"})).status_code == 200
    p2 = fake_profile(tmp_path, name="fake2")
    async with Harness([p2], tmp_path / "w2") as h:
        img = {**chat(), "model": "fake2"}
        img["messages"][1]["content"] = [{"type": "image_url", "image_url": {"url": "http://169.254.169.254/x.png"}}]
        r = await h.post(img)
        assert r.status_code == 400 and "data: image" in r.json()["error"]["message"]


async def test_engine_errors_pass_through(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir) as h:
        r = await h.post(chat(fake_mode="error"), episode="e1")
        assert r.status_code == 500 and "fake engine error" in r.json()["error"]["message"]


async def test_gateway_token(tmp_path, workdir):
    async with Harness([fake_profile(tmp_path)], workdir, token="t0k") as h:
        assert (await h.post(chat())).status_code == 401
        assert (await h.client.get("/health")).status_code == 401
        r = await h.post(chat(), headers={"Authorization": "Bearer t0k"})
        assert r.status_code == 200


async def test_models_report_each_profiles_capabilities(tmp_path, workdir):
    """GET /v1/models: x_gateway.capabilities per profile; no worker starts."""
    capped = fake_profile(tmp_path, name="capped", capabilities={"max_images": 16, "max_output_tokens": 81920})
    async with Harness([capped, fake_profile(tmp_path)], workdir, token="t0k") as h:
        assert (await h.client.get("/v1/models")).status_code == 401
        d = (await h.client.get("/v1/models", headers={"Authorization": "Bearer t0k"})).json()
        assert {m["id"]: m["x_gateway"]["capabilities"] for m in d["data"]} == {
            "capped": {"images": True, "tools": True, "max_images": 16, "max_output_tokens": 81920},
            "fake": {"images": True, "tools": True, "max_images": None, "max_output_tokens": None}}
        assert all(m["object"] == "model" and m["owned_by"] == "chd-gateway" for m in d["data"])
        assert not h.gw.pools["capped"].workers


def test_local_model_takes_the_cap_from_a_real_gateway_reply(tmp_path, workdir):
    """The openai SDK against the gateway app (no lifespan: nothing starts): LocalModel reads the cap of its own
    profile, and nothing for a profile without one."""
    import openai
    from fastapi.testclient import TestClient

    from chd_client.gateway.app import create_app
    from chd_client.gateway.profile import load_profile
    from chd_client.models.local import LocalModel
    gw = Gateway([load_profile(fake_profile(tmp_path, name="capped",
                                            capabilities={"max_images": 16, "max_output_tokens": 81920})),
                  load_profile(fake_profile(tmp_path))], workdir, token="t0k")
    http = TestClient(create_app(gw))
    client = openai.OpenAI(base_url="http://testserver/v1", api_key="t0k", http_client=http, max_retries=0)
    capped, plain = (LocalModel(n, client=client, base_url="http://testserver/v1") for n in ("capped", "fake"))
    before = plain.request_config()
    capped.resolve_limits(), plain.resolve_limits()
    assert (capped.max_images, capped.max_images_source) == (16, "gateway")
    assert (capped.max_tokens, capped.max_tokens_source) == (81920, "gateway")
    assert capped.request_config()["max_tokens"] == 81920 and capped.request_config()["max_tokens_source"] == "gateway"
    assert plain.max_images is None and "max_images" not in plain.request_config()
    assert plain.max_tokens == 32000 and plain.request_config() == before


async def test_idle_workers_stop_and_shutdown_stops_the_rest(tmp_path, workdir):
    p = fake_profile(tmp_path, launcher={"idle_stop_s": 0.5, "max_workers": 2},
                     scheduler={"lease_s": 0.3, "episodes_per_worker": 1})
    h = Harness([p], workdir)
    async with h:
        await h.post(chat(), episode="e1")
        pool = h.gw.pools["fake"]
        pid = next(iter(pool.workers.values())).handle["pid"]
        assert await wait_until(lambda: not pool.workers, timeout=10)
        assert await wait_until(lambda: not pid_alive(pid), timeout=10)
        await h.post(chat(), episode="e2")
        pid2 = next(iter(pool.workers.values())).handle["pid"]
        assert pid_alive(pid2)
    assert await wait_until(lambda: not pid_alive(pid2), timeout=5)


async def test_restart_takes_workers_back(tmp_path, workdir):
    p = fake_profile(tmp_path)
    async with Harness([p], workdir, keep_workers=True) as h:
        w = (await h.post(chat(), episode="e1")).json()["x_gateway"]["worker"]
        pid = h.gw.pools["fake"].workers[w].handle["pid"]
    assert pid_alive(pid)
    async with Harness([p], workdir) as h:
        x = (await h.post(chat(), episode="e1")).json()["x_gateway"]
        assert x["worker"] == w and x["affinity"] == "sticky"
    assert await wait_until(lambda: not pid_alive(pid), timeout=5)


async def test_a_changed_profile_does_not_reuse_old_workers(tmp_path, workdir):
    p = fake_profile(tmp_path)
    async with Harness([p], workdir, keep_workers=True) as h:
        w = (await h.post(chat(), episode="e1")).json()["x_gateway"]["worker"]
        pid = h.gw.pools["fake"].workers[w].handle["pid"]
    p = fake_profile(tmp_path, scheduler={"lease_s": 601})
    async with Harness([p], workdir) as h:
        assert not pid_alive(pid) or await wait_until(lambda: not pid_alive(pid), timeout=5)
        x = (await h.post(chat(), episode="e1")).json()["x_gateway"]
        assert x["worker"] != w


async def test_a_worker_that_cannot_start_is_reported(tmp_path, workdir):
    p = write_profile(tmp_path, f"""
        name: broken
        engine: {{kind: custom, command: ["{sys.executable}", "-c", "import sys; sys.exit(3)"]}}
        launcher: {{kind: local, startup_timeout_s: 30}}
        scheduler: {{request_timeout_s: 4}}
        """)
    async with Harness([p], workdir) as h:
        r = await h.post({**chat(), "model": "broken"})
        assert r.status_code == 503
        msg = r.json()["error"]["message"]
        assert "ended before it was ready" in msg, msg


async def test_attach_to_a_running_server(tmp_path, workdir):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)])
    try:
        p = write_profile(tmp_path, f"""
            name: attached
            upstream_model: real-name
            launcher: {{kind: attach, urls: ["http://127.0.0.1:{port}/v1"]}}
            cost: {{source: unknown}}
            scheduler: {{request_timeout_s: 10}}
            """)
        async with Harness([p], workdir) as h:
            d = (await h.post({**chat(), "model": "attached"}, episode="e1")).json()
            assert d["model"] == "attached" and d["usage"]["cost"] is None and d["usage"]["cost_source"] == "unknown"
            assert d["system_fingerprint"].startswith("fake-")
        assert srv.poll() is None
    finally:
        srv.terminate()
        srv.wait()


def test_worker_stops_when_the_gateway_heartbeat_is_old(tmp_path):
    p = fake_profile(tmp_path)
    hb = tmp_path / "gateway.alive"
    hb.touch()
    os.utime(hb, (time.time() - 3600, time.time() - 3600))
    t = time.time()
    proc = subprocess.run([sys.executable, "-m", "chd_client.gateway.worker", "--profile", str(p), "--worker-id", "w-x",
                           "--workdir", str(tmp_path / "pool"), "--heartbeat", str(hb), "--orphan-timeout-s", "1"],
                          capture_output=True, text=True, timeout=60)
    assert "heartbeat" in proc.stdout and time.time() - t < 30
    assert not (tmp_path / "pool" / "workers" / "w-x.json").exists()


def test_gateway_refuses_duplicate_names(tmp_path):
    p = fake_profile(tmp_path)
    from chd_client.gateway.profile import load_profile
    with pytest.raises(ValueError, match="same name"):
        Gateway([load_profile(p), load_profile(p)], tmp_path / "w")


def test_local_model_waits_longer_than_the_gateway(tmp_path, workdir):
    """x_gateway.request_timeout_s above the client's own timeout: the client waits that + TIMEOUT_MARGIN_S, so a long
    thinking turn or a worker restart after a spot preemption is not cut by the client and resent."""
    import openai
    from fastapi.testclient import TestClient

    from chd_client.gateway.app import create_app
    from chd_client.gateway.profile import load_profile
    from chd_client.models.local import TIMEOUT_MARGIN_S, LocalModel
    from chd_client.models.openrouter import REQUEST_TIMEOUT_S
    gw = Gateway([load_profile(fake_profile(tmp_path, name="slow", scheduler={"request_timeout_s": 14400})),
                  load_profile(fake_profile(tmp_path))], workdir, token="t0k")
    client = openai.OpenAI(base_url="http://testserver/v1", api_key="t0k", http_client=TestClient(create_app(gw)),
                           max_retries=0)
    slow, plain = (LocalModel(n, client=client, base_url="http://testserver/v1") for n in ("slow", "fake"))
    slow.resolve_limits(), plain.resolve_limits()
    assert slow.timeout_s == 14400 + TIMEOUT_MARGIN_S and slow.client.timeout == 14400 + TIMEOUT_MARGIN_S
    assert slow.request_config()["timeout_s"] == 14400 + TIMEOUT_MARGIN_S
    assert plain.timeout_s == REQUEST_TIMEOUT_S


async def test_request_defaults_reach_the_engine_and_the_request_wins(tmp_path, workdir):
    """engine.request_defaults: fields the gateway adds to each request (e.g. vLLM's thinking_token_budget, which no
    vLLM server flag can set); the request's own keys win, one level deep for mappings; x_gateway records what it added."""
    defaults = {"thinking_token_budget": 17408, "chat_template_kwargs": {"reasoning_budget": 16384, "x": 1}}
    async with Harness([fake_profile(tmp_path, engine={"request_defaults": defaults})], workdir) as h:
        echo = ["thinking_token_budget", "chat_template_kwargs"]
        d = (await h.post(chat(fake_echo=echo), episode="e1")).json()
        assert d["fake_echo"] == defaults and d["x_gateway"]["request_defaults"] == defaults
        d = (await h.post(chat(fake_echo=echo, thinking_token_budget=5, chat_template_kwargs={"x": 2}),
                          episode="e1")).json()
        assert d["fake_echo"] == {"thinking_token_budget": 5, "chat_template_kwargs": {"reasoning_budget": 16384, "x": 2}}
        assert d["x_gateway"]["request_defaults"] == {"chat_template_kwargs": {"reasoning_budget": 16384}}


@pytest.mark.parametrize("defaults,msg", [({"model": "x"}, "may not set model"), (["a"], "must be a mapping"),
                                          ({"messages": [], "stream": True}, "messages, stream")])
def test_bad_request_defaults_are_refused(tmp_path, defaults, msg):
    from chd_client.gateway.profile import ProfileError, load_profile
    with pytest.raises(ProfileError, match=msg):
        load_profile(fake_profile(tmp_path, engine={"request_defaults": defaults}))
