"""What a run was made with, in its own folder: enough to say later exactly which prompt, tools,
request settings, code and inputs produced it, without the repos as they are then.

    <run>/provenance/
        system_prompt.txt     the system prompt text the model got (rendered, as sent)
        tools.json            the tool definitions sent to the API (name, description, input_schema; runner-only removed)
        provenance.json       request settings (max_tokens, thinking, tool_choice, effort, image window ...), the
                              provider registry entry (`provider`: name, kind, base URL, key variable name, the
                              registry file's sha256), the provider SDK version, python, the harness git commit and the
                              slicer-eval-mcp checkout's state, the sha256 of config/scoring_v1.json and of the task
                              profile's protocol.json and report.py, the server instructions' sha256;
                              after the run (finish_run): the Slicer / bridge / VTK / Mesa versions the trace recorded
                              and the input manifest (the case files' sha256; ImageCHD: the de-identified spacing row
                              only, never the image-info spreadsheet)

write_provenance runs once the episode id is known; add_run_provenance adds the after-run part.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

from chd_client.run_record import sha256_bytes, sha256_text, write_json_atomic

PROVENANCE_DIR = "provenance"
FILES = ("system_prompt.txt", "tools.json", "provenance.json")
ENV_META_KEYS = ("slicer", "bridge", "server_git", "env_git", "profile", "tools", "protocol", "presets_sha256",
                 "tool_schema_version", "report_schema_version", "report_plugin_sha256", "image", "vtk", "mesa", "opengl", "render", "launch")


def git_info(repo) -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, timeout=10).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def file_sha256(path) -> str | None:
    try:
        return sha256_bytes(Path(path).read_bytes())
    except OSError:
        return None


def scorer_info(repo=None, profile=None) -> dict:
    """The harness (scorer) code and config: CHD-Bench's commit, its scoring config and task profile hashes, and the
    slicer-eval-mcp checkout the environment ran from (chd_client.slicer.PINNED_COMMIT). profile: the episode's task
    profile folder when it is not chd_ct_v1 (the 3d arm's chd_ct_nomask_v1)."""
    from chd_client import slicer
    from chd_client.task.config import PROFILE, REPO
    repo = Path(repo or REPO)
    profile = Path(profile) if profile is not None else PROFILE
    try:
        server = slicer.git_state()
        server["pinned_commit"] = slicer.PINNED_COMMIT
    except Exception as e:  # noqa: BLE001
        server = {"error": f"{type(e).__name__}: {e}"}
    return {"harness_git": git_info(repo),
            "scoring_v1_sha256": file_sha256(repo / "config" / "scoring_v1.json"),
            "profile": profile.name,
            "profile_protocol_sha256": file_sha256(profile / "protocol.json"),
            "profile_report_sha256": file_sha256(profile / "report.py"),
            "slicer_mcp": server}


def tools_json(tools) -> list:
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]


def write_provenance(run_dir, *, system_text: str, tools, model, instructions: str = "", client_meta=None,
                     repo=None, profile=None) -> Path:
    d = Path(run_dir) / PROVENANCE_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / "system_prompt.txt").write_text(system_text)
    tj = tools_json(tools)
    write_json_atomic(d / "tools.json", tj)
    prov = {
        "system_prompt_sha256": sha256_text(system_text),
        "tools_sha256": sha256_text(json.dumps(tj, sort_keys=True, separators=(",", ":"))),
        "server_instructions_sha256": sha256_text(instructions or ""),
        "model": {"id": model.model_id, "version": model.model_version, "adapter": type(model).__name__},
        "request": model.request_config(),
        "sdk": model.sdk_versions(),
        "provider": getattr(model, "provider_info", None),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "client_git": (client_meta or {}).get("client_git"),
        "prompt_version": (client_meta or {}).get("prompt_version"),
        "seed": {"value": (client_meta or {}).get("seed"), "use": "metadata only: not sent to the model API"},
        **scorer_info(repo, profile),
    }
    write_json_atomic(d / "provenance.json", prov)
    return d


def add_run_provenance(run_dir, inputs: dict | None = None) -> dict:
    """After the run: the env trace's render / version fields (trace/meta.json) and the input manifest."""
    d = Path(run_dir)
    p = d / PROVENANCE_DIR / "provenance.json"
    try:
        prov = json.loads(p.read_text())
    except (OSError, ValueError):
        prov = {}
    try:
        meta = json.loads((d / "trace" / "meta.json").read_text())
    except (OSError, ValueError):
        meta = {}
    env = {k: meta[k] for k in ENV_META_KEYS if k in meta}
    missing = [k for k in ("vtk", "mesa", "opengl") if k not in meta]
    if missing:
        env["not_reported"] = missing
    prov["environment"] = env
    if inputs is not None:
        prov["inputs"] = inputs
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(p, prov)
    return prov
