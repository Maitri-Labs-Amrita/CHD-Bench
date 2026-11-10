"""The vllm engine end to end with a stand-in vLLM: a fake environment (bin/python, bin/ninja) and a fake
vllm.entrypoints.openai.api_server module that records what it was started with, then serves the fake engine."""

import json
import os
import sys
import textwrap

import yaml

from tests.gateway.helpers import FAKE, Harness, chat

API_SERVER = """\
import json, os, runpy, shutil, sys
FAKE = os.environ["FAKE_ENGINE"]
args = sys.argv[1:]
keys = ("PATH", "VLLM_USE_FLASHINFER_SAMPLER", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "VLLM_API_KEY",
        "CHD_GATEWAY_WORKER_KEY", "HF_HUB_OFFLINE")
with open(os.environ["FAKE_VLLM_RECORD"], "w") as f:
    json.dump({"argv": args, "env": {k: os.environ.get(k) for k in keys}, "ninja": shutil.which("ninja")}, f)
sys.argv = [FAKE, "--port", args[args.index("--port") + 1], "--host", args[args.index("--host") + 1]]
runpy.run_path(FAKE, run_name="__main__")
"""


def fake_vllm_env(tmp_path):
    """<tmp>/env/bin/{python,ninja} (python runs this interpreter) and a fake vllm package on <tmp>/site."""
    bin_dir = tmp_path / "env" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").write_text(f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    (bin_dir / "ninja").write_text("#!/bin/sh\nexit 0\n")
    for f in ("python", "ninja"):
        os.chmod(bin_dir / f, 0o755)
    pkg = tmp_path / "site" / "vllm" / "entrypoints" / "openai"
    pkg.mkdir(parents=True)
    for d in (pkg, pkg.parent, pkg.parent.parent):
        (d / "__init__.py").write_text("")
    (pkg / "api_server.py").write_text(API_SERVER)
    return bin_dir / "python"


async def test_vllm_worker_gets_its_start_environment(tmp_path, workdir, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "node-tmp"))
    python = fake_vllm_env(tmp_path)
    record = tmp_path / "record.json"
    profile = tmp_path / "qwen-vllm.yaml"
    profile.write_text(yaml.safe_dump({
        "name": "qwen-vllm", "revision": "abc123",
        "engine": {"kind": "vllm", "model": "/models/qwen", "python": str(python), "tool_parser": "qwen3_coder",
                   "args": ["--max-model-len", "131072"],
                   "env": {"PYTHONPATH": str(tmp_path / "site"), "FAKE_VLLM_RECORD": str(record),
                           "FAKE_ENGINE": str(FAKE), "HF_HUB_OFFLINE": "1"}},
        "launcher": {"kind": "local", "startup_timeout_s": 30},
        "scheduler": {"request_timeout_s": 20}}, sort_keys=False))
    async with Harness([profile], workdir) as h:
        r = await h.post(chat(model="qwen-vllm"), episode="e1")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "look"
        assert d["x_gateway"]["engine"] == "vllm"
        worker = d["x_gateway"]["worker"]
    rec = json.loads(record.read_text())
    argv, env = rec["argv"], rec["env"]
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_coder" and "--enable-auto-tool-choice" in argv
    assert argv[argv.index("--revision") + 1] == "abc123" and argv[argv.index("--served-model-name") + 1] == "qwen-vllm"
    assert argv[-2:] == ["--max-model-len", "131072"]
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent) and rec["ninja"] == str(python.parent / "ninja")
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["HF_HUB_OFFLINE"] == "1"
    cache = tmp_path / "node-tmp" / f"chdgw-{worker}"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(cache / "torchinductor")
    assert env["TRITON_CACHE_DIR"] == str(cache / "triton")
    assert env["VLLM_API_KEY"] and env["VLLM_API_KEY"] == env["CHD_GATEWAY_WORKER_KEY"]
    assert env["VLLM_API_KEY"] not in argv


def test_check_prints_the_vllm_command(tmp_path, capsys):
    from chd_client.gateway import cli
    p = tmp_path / "m.yaml"
    p.write_text(textwrap.dedent("""\
        name: m
        engine: {kind: vllm, model: /models/m, python: /envs/vllm/bin/python, tool_parser: hermes}
        launcher: {kind: local}
    """))
    assert cli.main(["check", str(p)]) == 0
    out = capsys.readouterr().out
    assert "engine=vllm" in out and "-m vllm.entrypoints.openai.api_server" in out
    assert "--tool-call-parser hermes" in out
    assert "TORCHINDUCTOR_CACHE_DIR" in out and "VLLM_USE_FLASHINFER_SAMPLER" in out
