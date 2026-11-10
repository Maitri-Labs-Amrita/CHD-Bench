"""Unit tests: profiles, tool-call parsing, cost labels, SLURM scripts and status."""

import json
import subprocess
from pathlib import Path

import pytest

from chd_client.gateway import normalize
from chd_client.gateway.engines import build_command, substitute
from chd_client.gateway.launchers import slurm
from chd_client.gateway.launchers.slurm import SlurmLauncher, clean_env
from chd_client.gateway.profile import ProfileError, load_profile, parse_profile
from chd_client.gateway.toolcalls import FORMATS

ROOT = Path(__file__).resolve().parents[2]
H = FORMATS["hermes"]


@pytest.mark.parametrize("path", sorted((ROOT / "config" / "gateway").rglob("*.yaml")), ids=lambda p: p.name)
def test_example_profiles_are_valid(path):
    p = load_profile(path)
    assert p.name and len(p.sha256) == 64


def prof(text: str):
    return parse_profile(text.encode())


def test_unknown_key_is_an_error():
    with pytest.raises(ProfileError, match="unknown key"):
        prof("name: m\nengine: {kind: transformers, model: x, modle: y}\nlauncher: {kind: local}\n")


def test_attach_needs_an_explicit_cost():
    with pytest.raises(ProfileError, match="cost.source"):
        prof("name: m\nlauncher: {kind: attach, urls: ['http://127.0.0.1:1/v1']}\n")
    p = prof("name: m\nlauncher: {kind: attach, urls: ['http://127.0.0.1:1/v1/']}\ncost: {source: unknown}\n")
    assert p.engine.kind == "external" and p.launcher.urls == ["http://127.0.0.1:1/v1"]


def test_forced_call_needs_prefill():
    with pytest.raises(ProfileError, match="supports_prefill"):
        prof("name: m\nengine: {kind: custom, command: [x]}\nlauncher: {kind: local}\n"
             "interventions: {forced_tool_call: true}\n")
    prof("name: m\nengine: {kind: custom, command: [x], supports_prefill: true}\nlauncher: {kind: local}\n"
         "interventions: {forced_tool_call: true}\n")


@pytest.mark.parametrize("text,match", [
    ("name: m\nengine: {kind: transformers}\nlauncher: {kind: local}\n", "engine.model"),
    ("name: 'bad name'\nengine: {kind: transformers, model: x}\nlauncher: {kind: local}\n", "name must match"),
    ("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: local, gpus: [0], max_workers: 2}\n", "cannot run"),
    ("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: slurm}\n", "sbatch is required"),
    ("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: slurm, sbatch: {'bad key': 1}}\n", "bad entry"),
    ("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: local}\ncost: {source: estimated}\n",
     "input_per_mtok"),
    ("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: local}\n"
     "scheduler: {episodes_per_worker: 0}\n", "episodes_per_worker"),
])
def test_bad_profiles(text, match):
    with pytest.raises(ProfileError, match=match):
        prof(text)


def test_vllm_command_and_key_in_env(monkeypatch):
    monkeypatch.setenv("CHD_GATEWAY_WORKER_KEY", "k")
    p = prof("name: qwen\nrevision: abc\nengine: {kind: vllm, model: Qwen/X, tool_parser: hermes, args: "
             "['--max-model-len', 4096], python: /py}\nlauncher: {kind: local}\n")
    argv, env = build_command(p, host="0.0.0.0", port=1234, worker_id="w", workdir="/w")
    assert argv[:3] == ["/py", "-m", "vllm.entrypoints.openai.api_server"]
    for a in (["--served-model-name", "qwen"], ["--revision", "abc"], ["--tool-call-parser", "hermes"],
              ["--max-model-len", "4096"], ["--port", "1234"]):
        i = argv.index(a[0])
        assert argv[i:i + 2] == a
    assert "--enable-auto-tool-choice" in argv
    assert env["VLLM_API_KEY"] == "k" and "k" not in argv


def test_vllm_without_a_tool_parser_has_no_tool_choice():
    p = prof("name: m\nengine: {kind: vllm, model: x}\nlauncher: {kind: local}\n")
    argv, _ = build_command(p, host="h", port=1, worker_id="w", workdir="/w")
    assert "--enable-auto-tool-choice" not in argv and "--tool-call-parser" not in argv


def test_vllm_engine_profile_rules():
    with pytest.raises(ProfileError, match="engine.model"):
        prof("name: m\nengine: {kind: vllm}\nlauncher: {kind: local}\n")
    with pytest.raises(ProfileError, match="engine.tool_parser"):
        prof("name: m\nengine: {kind: vllm, model: x, tool_parser: [a]}\nlauncher: {kind: local}\n")
    p = prof("name: m\nengine: {kind: vllm, model: x}\nlauncher: {kind: local}\n"
             "interventions: {forced_tool_call: true}\n")
    assert p.engine.prefill_ok()


def test_vllm_start_environment(monkeypatch):
    """vLLM start environment: ninja on PATH (FlashInfer's sampler JIT), and a separate /tmp/torchinductor cache for
    each worker on a node."""
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("TMPDIR", "/node-tmp")
    p = prof("name: m\nengine: {kind: vllm, model: x, python: /envs/vllm/bin/python}\nlauncher: {kind: local}\n")
    _, env = build_command(p, host="h", port=1, worker_id="w-1", workdir="/w")
    assert env["PATH"] == "/envs/vllm/bin:/usr/bin:/bin"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/node-tmp/chdgw-w-1/torchinductor"
    assert env["TRITON_CACHE_DIR"] == "/node-tmp/chdgw-w-1/triton"
    _, env2 = build_command(p, host="h", port=1, worker_id="w-2", workdir="/w")
    assert env2["TORCHINDUCTOR_CACHE_DIR"] != env["TORCHINDUCTOR_CACHE_DIR"]
    assert env2["TRITON_CACHE_DIR"] != env["TRITON_CACHE_DIR"]
    monkeypatch.delenv("TMPDIR")
    _, env = build_command(p, host="h", port=1, worker_id="w-1", workdir="/w")
    assert env["TRITON_CACHE_DIR"] == "/tmp/chdgw-w-1/triton"


def test_vllm_profile_env_wins(monkeypatch):
    monkeypatch.setenv("CHD_GATEWAY_WORKER_KEY", "k")
    p = prof("name: m\nengine: {kind: vllm, model: x, python: /envs/vllm/bin/python, env: {PATH: /mine, "
             "VLLM_USE_FLASHINFER_SAMPLER: 1, TORCHINDUCTOR_CACHE_DIR: /ind, TRITON_CACHE_DIR: /tri, "
             "HF_HUB_OFFLINE: 1}}\n"
             "launcher: {kind: local}\n")
    _, env = build_command(p, host="h", port=1, worker_id="w", workdir="/w")
    assert env == {"PATH": "/mine", "VLLM_USE_FLASHINFER_SAMPLER": "1", "TORCHINDUCTOR_CACHE_DIR": "/ind",
                   "TRITON_CACHE_DIR": "/tri", "HF_HUB_OFFLINE": "1", "VLLM_API_KEY": "k"}


def test_vllm_with_a_bare_python_name_keeps_path():
    p = prof("name: m\nengine: {kind: vllm, model: x, python: python3}\nlauncher: {kind: local}\n")
    _, env = build_command(p, host="h", port=1, worker_id="w", workdir="/w")
    assert "PATH" not in env and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


@pytest.mark.parametrize("engine", ["{kind: transformers, model: x}", "{kind: custom, command: [srv]}"])
def test_other_engines_get_no_vllm_environment(monkeypatch, engine):
    monkeypatch.setenv("CHD_GATEWAY_WORKER_KEY", "k")
    p = prof(f"name: m\nengine: {engine}\nlauncher: {{kind: local}}\n")
    assert build_command(p, host="h", port=1, worker_id="w", workdir="/w")[1] == {}


def test_substitute_leaves_other_braces():
    assert substitute('--x {port} {"a": {b}}', {"port": 7}) == '--x 7 {"a": {b}}'


def test_strict_parses_well_formed_calls_only():
    text = 'a <tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call> b <tool_call>{bad}</tool_call>'
    content, calls = H.strict(text)
    assert [c["function"]["name"] for c in calls] == ["f"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
    assert "<tool_call>{bad}</tool_call>" in content and "f" not in content.replace("{bad}", "")


def test_strict_without_calls_keeps_text():
    assert H.strict("  just text ") == ("just text", [])


@pytest.mark.parametrize("text,names,expect", [
    ('<tool_call>\n{"name": "f", "arguments": {"a": 1}}', None, "f"),
    ('{"name": "f", "arguments": {"a": 1}}</tool_call>', None, "f"),
    ('<tool_call>{"name": "f", "arguments": {"a": 1}}}</tool_call>', None, "f"),
    ('<tool_call>{"name": "f", "arguments": {"a": 1}</tool_call>', None, "f"),
    ('I will call {"name": "f", "arguments": {}}', {"f"}, "f"),
    ('<tool_call>{"name": "f", "arguments": "{\\"a\\": 1}"}</tool_call>', None, "f"),
])
def test_lenient_repairs(text, names, expect):
    _, calls = H.lenient(text, names)
    assert [c["function"]["name"] for c in calls] == [expect]


def test_lenient_ignores_bare_json_for_unknown_tools():
    assert H.lenient('{"name": "g", "arguments": {}}', {"f"})[1] == []


def test_cost_labels():
    p = prof("name: m\nengine: {kind: transformers, model: x}\nlauncher: {kind: local}\n")
    r = normalize.finalize({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, p, {})
    assert r["usage"]["cost"] == 0.0 and r["usage"]["cost_source"] == "zero_api_charge"
    assert r["usage"]["total_tokens"] == 5 and r["x_gateway"]["usage_reported"] is True
    r = normalize.finalize({"usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.5}}, p, {})
    assert r["usage"]["cost"] == 0.5 and r["usage"]["cost_source"] == "reported"
    r = normalize.finalize({"usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0}}, p, {})
    assert r["usage"]["cost"] == 0.0 and r["usage"]["cost_source"] == "zero_api_charge"
    est = prof("name: m\nlauncher: {kind: attach, urls: ['http://h/v1']}\n"
               "cost: {source: estimated, input_per_mtok: 2, output_per_mtok: 10, pricing_revision: '2026-10'}\n")
    r = normalize.finalize({"usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000}}, est, {})
    assert r["usage"]["cost"] == pytest.approx(3.0) and r["usage"]["cost_source"] == "estimated"
    unk = prof("name: m\nlauncher: {kind: attach, urls: ['http://h/v1']}\ncost: {source: unknown}\n")
    r = normalize.finalize({}, unk, {})
    assert r["usage"]["cost"] is None and r["usage"]["cost_source"] == "unknown"
    assert r["x_gateway"]["usage_reported"] is False


SLURM_PROFILE = """name: m
engine: {kind: transformers, model: x}
launcher:
  kind: slurm
  sbatch: {partition: gpu, gres: "gpu:1", exclusive: true, qos: null}
  setup: ["module load cuda"]
"""


def test_slurm_script(tmp_path):
    p = prof(SLURM_PROFILE)
    s = SlurmLauncher(p, tmp_path, tmp_path / "profile.yaml", "se cret'", tmp_path / "hb").script("w-1")
    lines = s.splitlines()
    assert "#SBATCH --partition=gpu" in lines and "#SBATCH --gres=gpu:1" in lines
    assert "#SBATCH --exclusive" in lines and "#SBATCH --no-requeue" in lines
    assert not any("qos" in x for x in lines)
    assert "module load cuda" in lines
    assert "export CHD_GATEWAY_WORKER_KEY='se cret'\"'\"''" in lines
    assert lines[-1].startswith("exec srun --ntasks=1 ") and "--bind all" in lines[-1]
    assert lines.index("module load cuda") < len(lines) - 1


def test_slurm_env_drops_job_variables(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SBATCH_MEM_PER_NODE", "1G")
    monkeypatch.setenv("KEEP_ME", "1")
    env = clean_env()
    assert "SLURM_JOB_ID" not in env and "SBATCH_MEM_PER_NODE" not in env and env["KEEP_ME"] == "1"


def test_slurm_status(monkeypatch, tmp_path):
    l = SlurmLauncher(prof(SLURM_PROFILE), tmp_path, tmp_path / "p.yaml", "k", tmp_path / "hb")
    replies = {}

    def fake_run(cmd, timeout=60):
        return replies["next"]

    monkeypatch.setattr(slurm, "run", fake_run)
    replies["next"] = subprocess.CompletedProcess([], 0, "11 RUNNING\n12 COMPLETING\n", "")
    out = l.status_many({"a": {"jobid": "11"}, "b": {"jobid": "12"}, "c": {"jobid": "13"}})
    assert out == {"a": "alive", "b": "alive", "c": "gone"}
    replies["next"] = None
    assert l.status_many({"a": {"jobid": "11"}}) == {"a": None}
    replies["next"] = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Socket timed out")
    assert l.status({"jobid": "11"}) is None
    replies["next"] = subprocess.CompletedProcess([], 1, "", "slurm_load_jobs error: Invalid job id specified")
    assert l.status({"jobid": "11"}) == "gone"


def test_templates_placeholder_points_at_a_shipped_template():
    p = prof("name: m\nengine: {kind: custom, command: [srv, --chat-template, '{templates}/qwen2_5_vl_tools.jinja']}\n"
             "launcher: {kind: local}\n")
    argv, _ = build_command(p, host="127.0.0.1", port=1, worker_id="w", workdir="/w")
    path = Path(argv[argv.index("--chat-template") + 1])
    assert path.is_file() and "<tool_call>" in path.read_text()


def test_hf_server_converts_openai_messages():
    from chd_client.gateway.engines.hf_server import BadRequest, convert
    msgs, images = convert([
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "tool", "tool_call_id": "c", "content": '{"ok": true}'}])
    assert images == []
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[1] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"a": 1} and msgs[2]["content"] == ""
    assert msgs[3] == {"role": "tool", "content": '{"ok": true}', "tool_call_id": "c"}
    with pytest.raises(BadRequest):
        convert([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "f", "arguments": "[1]"}}]}])
