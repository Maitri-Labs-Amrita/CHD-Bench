"""mcp_session: stdio launch parameters and result -> Observation."""

import base64
import json

import mcp.types as types
import pytest

import sys

from chd_client.mcp_session import EnvironmentAbort, server_params, to_observation
from chd_client.task.config import PROFILE, REPO, load_runtime
from tests.fakes import png


def runtime_file(tmp_path):
    """A run's resolved runtime (as dicom_run writes for each run): passed to slicer-mcp as it is."""
    from chd_client.task.config import RESOLVED_MARK
    p = tmp_path / "rt.json"
    p.write_text(json.dumps({**load_runtime(), RESOLVED_MARK: "x"}))
    return p


def test_server_params(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    rt = runtime_file(tmp_path)
    p = server_params(runtime=rt, slicer_url="http://127.0.0.1:2017")
    assert p.command == sys.executable and p.cwd == str(REPO) and "VIRTUAL_ENV" not in p.env
    assert p.args == ["-m", "slicer_mcp.cli", "serve", "--slicer-url", "http://127.0.0.1:2017",
                      "--profile", str(PROFILE), "--runtime", str(rt.resolve())]


def test_a_relative_runtime_is_resolved_before_slicer_mcp_sees_it(tmp_path):
    """slicer-mcp reads relative runtime paths against its own checkout: the server gets an absolute copy."""
    rel = tmp_path / "rel.json"
    rel.write_text(json.dumps({**load_runtime(), "traces_dir": "traces", "cases_dir": "data/cases"}))
    args = server_params(runtime=rel).args
    got = json.loads(open(args[args.index("--runtime") + 1]).read())
    assert got["traces_dir"] == str(REPO / "traces") and got["cases_dir"] == str(REPO / "data" / "cases")


def test_to_observation():
    payload = {"step_index": 3, "state": {}, "result": {}, "warnings": [], "error": None, "frames": ["3d", "axial"]}
    content = [types.TextContent(text=json.dumps(payload))] + [
        types.ImageContent(data=base64.b64encode(png(v)).decode(), mime_type="image/png") for v in (1, 2)]
    o = to_observation("zoom", types.CallToolResult(content=content, is_error=False))
    assert o.step == 3 and o.frames == ["3d", "axial"] and o.images == [png(1), png(2)] and o.error_type is None
    with pytest.raises(EnvironmentAbort):
        to_observation("zoom", types.CallToolResult(content=content[:2], is_error=False))


def test_a_stale_slicer_url_does_not_override_the_runtime(monkeypatch, tmp_path):
    """A per-job runtime names the bridge's port; SLICER_URL left in the shell must not reach the server."""
    monkeypatch.setenv("SLICER_URL", "http://127.0.0.1:2017")
    assert "SLICER_URL" not in server_params(runtime=runtime_file(tmp_path)).env
    assert "SLICER_URL" in server_params().env


def test_identity_comes_from_the_result_meta_only():
    """slicer_mcp/identity and slicer_mcp/case_info are trace-only _meta; the text the model reads is untouched."""
    from chd_client.mcp_session import loaded_volume_sha256
    payload = {"step_index": 0, "state": {}, "result": {"case_info": {"dims": [1, 1, 1]}}, "warnings": [],
               "error": None, "frames": []}
    ident = {"episode_id": "E", "case_id": "c", "volume_sha256": "v" * 64, "seq": 1}
    res = types.CallToolResult(content=[types.TextContent(text=json.dumps(payload))], is_error=False,
                               _meta={"slicer_mcp/identity": ident, "slicer_mcp/case_info": {"volume_sha256": "w" * 64},
                                      "slicer_mcp/episode_id": "E"})
    o = to_observation("reset_episode", res)
    assert o.identity == ident and "v" * 64 not in o.text
    assert loaded_volume_sha256(o) == "w" * 64
    o.meta.pop("slicer_mcp/case_info")
    assert loaded_volume_sha256(o) == "v" * 64
    assert loaded_volume_sha256(to_observation("reset_episode", types.CallToolResult(
        content=[types.TextContent(text=json.dumps(payload))], is_error=False))) is None
