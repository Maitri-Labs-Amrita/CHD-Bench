"""Engines: the command a worker runs to serve one model on one port.

- vllm: `python -m vllm.entrypoints.openai.api_server` (vLLM installed in engine.python's environment);
- transformers: the built-in server (chd_client.gateway.engines.hf_server), slower, no prefix cache, any OS;
- custom: any command that serves OpenAI /v1/chat/completions and a health route (e.g. a research model's server).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PLACEHOLDERS = ("port", "host", "name", "served_model_name", "model", "revision", "worker_id", "workdir",
                "profile_dir", "templates")
TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def substitute(arg: str, values: dict) -> str:
    """Replace the known {placeholders} only, so other braces (JSON arguments) stay as they are."""
    for k in PLACEHOLDERS:
        arg = arg.replace("{" + k + "}", str(values.get(k, "")))
    return arg


def vllm_env(python: str, worker_id: str) -> dict:
    """The environment a vLLM worker needs to start on a cluster node (tested with vLLM 0.32.0). The profile's
    engine.env overrides each of these.

    - PATH starts with the folder of engine.python, as if its environment were activated: tools that vLLM's
      dependencies start by name (FlashInfer runs `ninja`) are found there.
    - VLLM_USE_FLASHINFER_SAMPLER=0: vLLM's GPU sampler calls FlashInfer's top-k/top-p kernel, which FlashInfer
      compiles at warm-up (nvcc + ninja). Nodes without a CUDA toolkit then fail at start. 0 selects vLLM's own
      PyTorch/Triton sampler (vllm/envs.py VLLM_USE_FLASHINFER_SAMPLER; vllm/v1/sample/ops/topk_topp_sampler.py
      flashinfer_sampler_supported).
    - TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR: node-local folders of this worker only. Without them every
      process of the user on a node shares /tmp/torchinductor_<user>, and two workers that compile at the same
      time break each other's kernels ("Cubin file saved by TritonBundler not found").
    """
    env = {"VLLM_USE_FLASHINFER_SAMPLER": "0"}
    bin_dir = os.path.dirname(os.path.abspath(python)) if os.sep in python else ""
    if bin_dir:
        env["PATH"] = os.pathsep.join(x for x in (bin_dir, os.environ.get("PATH", "")) if x)
    cache = Path(os.environ.get("TMPDIR") or "/tmp") / f"chdgw-{worker_id}"
    env["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "torchinductor")
    env["TRITON_CACHE_DIR"] = str(cache / "triton")
    inc = python_headers(python)
    if inc:
        for var in ("C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
            env[var] = os.pathsep.join(x for x in (inc, os.environ.get(var, "")) if x)
    return env


def python_headers(python: str) -> str | None:
    """A folder with Python.h for Triton's C helper, when the node has none: Triton builds cuda_utils.c at start with
    gcc -I/usr/include/python3.X, and a node without the Python dev package fails ("Python.h: No such file or
    directory"; vLLM 0.32.0). Then <engine environment>/include/python3.X is used, if it has
    Python.h (a self-contained copy: pyconfig.h included). None when the node's own headers exist or there is no copy."""
    if os.sep not in python:
        return None
    env_dir = Path(os.path.abspath(python)).parent.parent
    for d in sorted(env_dir.glob("include/python3.*")):
        if (d / "Python.h").exists() and not (Path("/usr/include") / d.name / "Python.h").exists():
            return str(d)
    return None


def build_command(profile, *, host: str, port: int, worker_id: str, workdir: str, profile_dir: str = "") -> tuple:
    """(argv, extra environment) for one worker's engine. The worker key (CHD_GATEWAY_WORKER_KEY) is passed in the
    environment, never on the command line. profile_dir: the folder of the profile as the user wrote it (the worker
    reads a frozen copy elsewhere)."""
    e = profile.engine
    python = e.python or sys.executable
    env = {str(k): str(v) for k, v in e.env.items()}
    key = os.environ.get("CHD_GATEWAY_WORKER_KEY")
    values = {"port": port, "host": host, "name": profile.name, "served_model_name": profile.served_name,
              "model": e.model or "", "revision": profile.revision or "", "worker_id": worker_id,
              "workdir": workdir, "profile_dir": profile_dir, "templates": TEMPLATES}
    args = [substitute(x, values) for x in e.args]
    if e.kind == "vllm":
        argv = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", e.model, "--host", host,
                "--port", str(port), "--served-model-name", profile.served_name]
        if profile.revision:
            argv += ["--revision", profile.revision]
        if e.tool_parser:
            argv += ["--enable-auto-tool-choice", "--tool-call-parser", e.tool_parser]
        argv += args
        for k, v in vllm_env(python, worker_id).items():
            env.setdefault(k, v)
        if key:
            env["VLLM_API_KEY"] = key
    elif e.kind == "transformers":
        argv = [python, "-m", "chd_client.gateway.engines.hf_server", "--model", e.model, "--host", host,
                "--port", str(port), "--served-model-name", profile.served_name, "--tool-format", profile.tool_format]
        if profile.revision:
            argv += ["--revision", profile.revision]
        argv += args
    elif e.kind == "custom":
        argv = [substitute(a, values) for a in e.command]
    else:
        raise ValueError(f"engine.kind {e.kind} is not started by a worker")
    return argv, env
